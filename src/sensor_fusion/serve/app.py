"""Run the detection service:

    python -m sensor_fusion.serve.app --model-dir outputs/serve/fusion_dense_head --port 8000

Settings come from `SF_*` environment variables (see `serve/settings.py` and docs/RUNBOOK.md);
command-line flags override them.
"""

import argparse
import logging
from pathlib import Path

from sensor_fusion.serve.api import create_app
from sensor_fusion.serve.inference import OnnxDetector
from sensor_fusion.serve.observability import configure_logging
from sensor_fusion.serve.settings import LOG_FORMATS, LOG_LEVELS, Settings

log = logging.getLogger("sensor_fusion.serve")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="folder with model.onnx and model.json")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--torch-threads", type=int, help="PyTorch threads for prepare/decode")
    parser.add_argument("--log-level", choices=LOG_LEVELS)
    parser.add_argument("--log-format", choices=LOG_FORMATS)
    args = parser.parse_args(argv)

    settings = Settings.from_env(
        model_dir=args.model_dir,
        host=args.host,
        port=args.port,
        torch_threads=args.torch_threads,
        log_level=args.log_level,
        log_format=args.log_format,
    )
    configure_logging(settings.log_level, settings.log_format)
    log.info("starting", extra={"model": str(settings.model_dir)})

    import uvicorn  # imported here so that importing this module needs no server

    detector = OnnxDetector(settings.model_dir, torch_threads=settings.torch_threads)
    uvicorn.run(
        create_app(detector, settings),
        host=settings.host,
        port=settings.port,
        log_config=None,  # our logging setup stays in charge
        access_log=False,  # the request middleware writes one structured line per request
        timeout_graceful_shutdown=10,  # let running detections finish on SIGTERM
    )


if __name__ == "__main__":
    main()
