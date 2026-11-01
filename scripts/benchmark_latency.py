"""Latency of the detector: PyTorch CPU / GPU, ONNX CPU, and the whole service call (ADR 0007).

    CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_latency.py \\
        --model rgb=outputs/retrain/rgb-8ep/best.pt \\
        --model fusion=outputs/retrain/fusion_dense_head-8ep/best.pt

Each model is exported to ONNX first (with the parity check). Input is a real validation frame, sent
the way a customer would (photo, raw LiDAR scan, calibration). Numbers are for THIS workstation, not
for an embedded device, and are only meaningful when no training job is running.
"""

import argparse
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import onnxruntime as ort
import torch
from nuscenes.nuscenes import NuScenes

from sensor_fusion.data.splits import get_split_scenes
from sensor_fusion.evaluate import load_for_evaluation
from sensor_fusion.serve.constants import INPUT_NAME
from sensor_fusion.serve.export import export_checkpoint
from sensor_fusion.serve.inference import DEFAULT_TORCH_THREADS, OnnxDetector
from sensor_fusion.serve.samples import RequestData, request_from_nuscenes

Stats = dict[str, float]


def summarise(samples: list[float]) -> Stats:
    ordered = sorted(samples)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "mean": statistics.fmean(ordered),
    }


def time_calls(call: Callable[[], object], *, warmup: int, runs: int, sync: bool = False) -> Stats:
    """Median, 95th percentile and mean of `runs` timed calls (milliseconds) after a warm-up."""
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(runs):
        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        if sync:
            torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - start))
    return summarise(samples)


def cpu_name() -> str:
    try:
        out = subprocess.run(
            ["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=20
        ).stdout.split("\n")
        return next(line.strip() for line in out[1:] if line.strip())
    except (OSError, StopIteration, subprocess.SubprocessError):
        return platform.processor() or "unknown CPU"


def benchmark_model(
    name: str,
    checkpoint: str,
    request: RequestData,
    export_dir: Path,
    warmup: int,
    runs: int,
    default_threads: int,
    service_threads: int,
) -> tuple[list[tuple[str, str, Stats]], tuple[str, Stats]]:
    """All timings of one model: the table rows and the stage breakdown of a service call."""
    torch.set_num_threads(default_threads)  # the PyTorch rows use PyTorch's own default
    detector = OnnxDetector(
        export_checkpoint(checkpoint, export_dir / name), torch_threads=default_threads
    )
    network_input, _, _ = detector.prepare(request.image, request.lidar_points, request.calibration)
    batch = network_input[None]
    session = detector.session
    rows: list[tuple[str, str, Stats]] = []

    torch_cpu, _, _ = load_for_evaluation(checkpoint, torch.device("cpu"))
    x_cpu = torch.from_numpy(batch)

    def cpu_call() -> object:
        with torch.no_grad():
            return torch_cpu(x_cpu)

    rows.append(
        (name, "PyTorch CPU (fp32), network only", time_calls(cpu_call, warmup=warmup, runs=runs))
    )
    rows.append(
        (
            name,
            "ONNX Runtime CPU, network only",
            time_calls(lambda: session.run(None, {INPUT_NAME: batch}), warmup=warmup, runs=runs),
        )
    )
    if torch.cuda.is_available():
        torch_gpu, _, _ = load_for_evaluation(checkpoint, torch.device("cuda"))
        x_gpu = torch.from_numpy(batch).cuda()

        def gpu_call() -> object:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return torch_gpu(x_gpu)

        rows.append(
            (
                name,
                "PyTorch GPU (fp16), network only",
                time_calls(gpu_call, warmup=warmup, runs=runs, sync=True),
            )
        )

    torch.set_num_threads(service_threads)  # what the service sets for itself
    totals: list[float] = []
    parts: dict[str, list[float]] = {"prepare_ms": [], "network_ms": [], "decode_ms": []}
    for _ in range(warmup):
        detector.detect(request.image, request.lidar_points, request.calibration)
    for _ in range(runs):
        _, timing = detector.detect(request.image, request.lidar_points, request.calibration)
        totals.append(timing["total_ms"])
        for key, values in parts.items():
            values.append(timing[key])
    rows.append((name, "Service call on CPU (prepare + ONNX + decode)", summarise(totals)))
    return rows, (name, {key: statistics.median(v) for key, v in parts.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--root", default="dataset/v1.0-trainval")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--export-dir", type=Path, default=Path("outputs/serve"))
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--service-threads", type=int, default=DEFAULT_TORCH_THREADS)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    args = parser.parse_args()

    default_threads = torch.get_num_threads()  # captured once, before any setting is changed
    nusc = NuScenes(args.version, dataroot=args.root, verbose=False)
    scene = next(s for s in nusc.scene if s["name"] == get_split_scenes("val")[0])
    request = request_from_nuscenes(nusc, scene["first_sample_token"])

    rows: list[tuple[str, str, Stats]] = []
    breakdown: list[tuple[str, Stats]] = []
    for spec in args.model:
        name, checkpoint = spec.split("=", 1)
        model_rows, model_breakdown = benchmark_model(
            name,
            checkpoint,
            request,
            args.export_dir,
            args.warmup,
            args.runs,
            default_threads,
            args.service_threads,
        )
        rows += model_rows
        breakdown.append(model_breakdown)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none used"
    lines = [
        "# Latency",
        "",
        f"Measured {time.strftime('%Y-%m-%d')} on a shared lab workstation, batch size 1, "
        f"{args.runs} timed runs after {args.warmup} warm-up runs, one real validation frame "
        f"(1600x900 photo, {len(request.lidar_points):,} LiDAR points). **These are workstation "
        "numbers, not edge-device numbers.**",
        "",
        f"- CPU: {cpu_name()} ({platform.system()}); PyTorch rows use PyTorch's default of "
        f"{default_threads} threads, service rows the service's {args.service_threads}; "
        f"ONNX Runtime {ort.__version__} manages its own threads",
        f"- GPU: {gpu} (PyTorch {torch.__version__})",
        "",
        "| model | setup | median ms | p95 ms | mean ms |",
        "|---|---|---|---|---|",
    ]
    for name, setup, t in rows:
        lines.append(f"| {name} | {setup} | {t['median']:.1f} | {t['p95']:.1f} | {t['mean']:.1f} |")
    lines += [
        "",
        "## Where a service call spends its time (median ms, CPU)",
        "",
        "| model | prepare (projection, thickening, normalising) | network (ONNX) | decode |",
        "|---|---|---|---|",
    ]
    for name, part in breakdown:
        cells = " | ".join(f"{part[k]:.1f}" for k in ("prepare_ms", "network_ms", "decode_ms"))
        lines.append(f"| {name} | {cells} |")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
