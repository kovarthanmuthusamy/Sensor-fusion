"""Export a trained checkpoint to ONNX and prove the exported file gives the same answers.

See ADR 0007.

The ONNX file holds the network only (photo + depth in, two output grids out). Preparing the input
and decoding the grids into boxes stay in our own tested code, so serving cannot drift away from
training. Next to `model.onnx` goes `model.json`: everything a caller needs to prepare an input the
way this model was trained (channel recipe, normalisation, thickening, decoding settings).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from sensor_fusion.data.labels import CLASS_NAMES
from sensor_fusion.data.transforms import prepare_frame
from sensor_fusion.serve.constants import INPUT_NAME, OUTPUT_NAMES

SIDECAR_FORMAT = 1
IMAGE_SIZE = (900, 1600)  # nuScenes CAM_FRONT, (height, width): the size the model was trained on
PARITY_TOLERANCE = 1e-3


def prepared_input_size(prep: dict[str, Any], in_channels: int) -> tuple[int, int]:
    """Height and width of the network input that `prepare_frame` produces for a full-size photo."""
    frame = prepare_frame(
        np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8),
        np.zeros(IMAGE_SIZE, dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.float32),
        downscale=int(prep["downscale"]),
        depth_max_m=float(prep["depth_max_m"]),
        in_channels=in_channels,
        rgb_mean=prep["rgb_mean"],
        rgb_std=prep["rgb_std"],
        flip=False,
        depth_thicken_px=int(prep.get("depth_thicken_px", 0)),
    )
    return frame.network_input.shape[1], frame.network_input.shape[2]


def export_onnx(
    model: nn.Module, path: Path, *, in_channels: int, input_size: tuple[int, int], opset: int = 17
) -> None:
    """Write the network to `path` with a dynamic batch size and fixed image size."""
    model = model.eval().cpu()
    example = torch.zeros(1, in_channels, *input_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={name: {0: "batch"} for name in (INPUT_NAME, *OUTPUT_NAMES)},
        opset_version=opset,
        dynamo=False,
    )


def check_parity(
    model: nn.Module,
    path: Path,
    *,
    in_channels: int,
    input_size: tuple[int, int],
    atol: float = PARITY_TOLERANCE,
    seed: int = 0,
) -> dict[str, float]:
    """Largest absolute difference per output between PyTorch and the ONNX file on random input.

    Raises if any difference exceeds `atol`: a silent mismatch between the trained model and the
    served one is the failure this check exists to catch. Batch 2 also checks the dynamic batch.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(2, in_channels, *input_size, generator=generator)
    with torch.no_grad():
        expected = model.eval().cpu()(x)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {INPUT_NAME: x.numpy()})
    diffs = {
        name: float(np.abs(g - e.numpy()).max())
        for name, g, e in zip(OUTPUT_NAMES, got, expected, strict=True)
    }
    worst = max(diffs.values())
    if worst > atol:
        raise ValueError(f"ONNX output differs from PyTorch by {worst:.2e} (> {atol:.0e}): {diffs}")
    return diffs


def export_model(
    model: nn.Module,
    out_dir: Path,
    *,
    prep: dict[str, Any],
    in_channels: int,
    model_name: str,
    depth_head: bool,
    epoch: int,
    checkpoint: str,
    decode_score_threshold: float = 0.3,
) -> Path:
    """Export `model` to `out_dir/model.onnx` + `model.json` and verify the parity.

    The sidecar records the parity result, so the file carries its own proof.
    """
    prep = {**prep, "depth_thicken_px": int(prep.get("depth_thicken_px", 0))}
    input_size = prepared_input_size(prep, in_channels)
    onnx_path = out_dir / "model.onnx"
    export_onnx(model, onnx_path, in_channels=in_channels, input_size=input_size)
    diffs = check_parity(model, onnx_path, in_channels=in_channels, input_size=input_size)
    sidecar = {
        "format": SIDECAR_FORMAT,
        "checkpoint": checkpoint,
        "epoch": epoch,
        "model_name": model_name,
        "in_channels": in_channels,
        "depth_head": depth_head,
        "class_names": list(CLASS_NAMES),
        "image_size": list(IMAGE_SIZE),
        "input_size": list(input_size),
        "prep": prep,
        "decode": {"top_k": 100, "score_threshold": decode_score_threshold},
        "onnx_parity_max_abs_diff": diffs,
    }
    (out_dir / "model.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return out_dir


def export_checkpoint(
    checkpoint: str, out_dir: Path, *, decode_score_threshold: float = 0.3
) -> Path:
    """Export a training checkpoint (its stored config says how the model was built)."""
    # Imported here: it pulls in the training stack, which the running service must not need.
    from sensor_fusion.evaluate import load_for_evaluation

    model, config, epoch = load_for_evaluation(checkpoint, torch.device("cpu"))
    prep = {k: (list(v) if k in ("rgb_mean", "rgb_std") else v) for k, v in config.prep.items()}
    return export_model(
        model,
        out_dir,
        prep=prep,
        in_channels=int(config.model.in_channels),
        model_name=str(config.model.name),
        depth_head=bool(config.model.get("depth_head", False)),
        epoch=epoch,
        checkpoint=str(checkpoint),
        decode_score_threshold=decode_score_threshold,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export a training checkpoint to ONNX.")
    parser.add_argument("checkpoint", help="a best.pt / last.pt / epoch-NN.pt written by training")
    parser.add_argument("out_dir", type=Path, help="folder for model.onnx and model.json")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    args = parser.parse_args()
    out = export_checkpoint(
        args.checkpoint, args.out_dir, decode_score_threshold=args.score_threshold
    )
    print(f"wrote {out / 'model.onnx'} and {out / 'model.json'}")


if __name__ == "__main__":
    main()
