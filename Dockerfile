# CPU image of the detection service (ADR 0007, ADR 0014). The model is not baked in: mount a folder
# with model.onnx and model.json (written by `python -m sensor_fusion.serve.export`) at /models.
#
#   docker build --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) -t sensor-fusion-serve .
#   docker run -p 8000:8000 -v "$PWD/outputs/serve/fusion_dense_head:/models:ro" sensor-fusion-serve
#
# Settings are SF_* environment variables (docs/RUNBOOK.md). Runs as a non-root user.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Only what serving needs: no Hydra, W&B, nuScenes devkit or training code paths.
# Versions are the ones tested locally and pinned in requirements.lock.txt.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install "numpy>=1.26,<2" pillow==12.3.0 onnxruntime==1.31.1 fastapi==0.162.0 \
        uvicorn==0.54.0 python-multipart==0.0.33 prometheus_client==0.26.0

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

# An unprivileged user: a bug in the service cannot write to the image or touch the host.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER app

ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.title="sensor-fusion-serve" \
      org.opencontainers.image.description="Night-time camera + LiDAR 2D detector with distance" \
      org.opencontainers.image.revision=$GIT_COMMIT \
      org.opencontainers.image.licenses="MIT"
ENV SF_MODEL_DIR=/models \
    SF_HOST=0.0.0.0 \
    SF_GIT_COMMIT=$GIT_COMMIT

EXPOSE 8000
# Ready (model warm), not just alive: an orchestrator should not send traffic before this passes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/ready', timeout=4)"

CMD ["python", "-m", "sensor_fusion.serve.app"]
