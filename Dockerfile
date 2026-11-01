# CPU image of the detection service (ADR 0007). The model is not baked in: mount a folder with
# model.onnx and model.json (written by `python -m sensor_fusion.serve.export`) at /models.
#
#   docker build -t sensor-fusion-serve .
#   docker run -p 8000:8000 -v "$PWD/outputs/serve/fusion_dense_head:/models:ro" sensor-fusion-serve
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Only what serving needs: no Hydra, W&B, nuScenes devkit or training code paths.
# Versions are the ones tested locally (add them to requirements.lock.txt when it is regenerated).
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install "numpy>=1.26,<2" pillow==12.3.0 onnxruntime==1.31.1 fastapi==0.162.0 \
        uvicorn==0.54.0 python-multipart==0.0.33

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

ENV MODEL_DIR=/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "sensor_fusion.serve.app", "--host", "0.0.0.0", "--port", "8000"]
