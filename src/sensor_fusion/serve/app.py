"""Run the detection service:

python -m sensor_fusion.serve.app --model-dir outputs/serve/fusion_dense_head --port 8000
"""

import argparse
import os
from pathlib import Path

from sensor_fusion.serve.api import create_app
from sensor_fusion.serve.inference import DEFAULT_TORCH_THREADS, OnnxDetector


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/models"),
        help="folder with model.onnx and model.json (written by sensor_fusion.serve.export)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=DEFAULT_TORCH_THREADS,
        help="PyTorch threads for preparation and decoding (few small ops: fewer is faster)",
    )
    args = parser.parse_args(argv)

    import uvicorn  # imported here so that importing this module needs no server

    detector = OnnxDetector(Path(args.model_dir), torch_threads=args.torch_threads)
    uvicorn.run(create_app(detector), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
