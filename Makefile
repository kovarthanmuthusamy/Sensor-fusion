# One command per task. `make help` lists them.
# Variables can be set on the command line, e.g.: make train-fusion EPOCHS=4 GPU=1
PY ?= python
GPU ?= 0
SEED ?= 42
EPOCHS ?= 8
BASELINE ?= outputs/runs/rgb/best.pt
FUSION ?= outputs/runs/fusion_dense_head/best.pt
MODEL_DIR ?= outputs/serve/fusion_dense_head
SAMPLE ?= outputs/sample
GIT_COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)

.DEFAULT_GOAL := help
.PHONY: help install lint format type test check train-baseline train-fusion evaluate export \
        serve docker-build docker-run compose-up load-test clean

help: ## List the tasks
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install: ## Install the package with development, serving and demo extras and the git hooks
	$(PY) -m pip install -e ".[dev,serve,demo]"
	pre-commit install

lint: ## Lint and check formatting (what the pre-commit hooks run)
	pre-commit run --all-files

format: ## Format the code
	ruff format src tests scripts
	ruff check --fix src tests scripts

type: ## Type-check src/
	mypy

test: ## Run the tests with coverage (GPU and dataset tests skip themselves when unavailable)
	pytest --cov --cov-report=term-missing

check: lint type test ## Everything CI runs: lint, types, tests

train-baseline: ## Train the camera-only baseline on one GPU (GPU=0 SEED=42 EPOCHS=8)
	CUDA_VISIBLE_DEVICES=$(GPU) $(PY) -m sensor_fusion.train data=trainval model=rgb seed=$(SEED) trainer.epochs=$(EPOCHS)

train-fusion: ## Train the camera + LiDAR model on one GPU
	CUDA_VISIBLE_DEVICES=$(GPU) $(PY) -m sensor_fusion.train data=trainval model=fusion_dense_head seed=$(SEED) trainer.epochs=$(EPOCHS)

evaluate: ## Compare two checkpoints on the validation split (BASELINE=... FUSION=...)
	$(PY) -m sensor_fusion.evaluate baseline=$(BASELINE) fusion=$(FUSION)

export: ## Export the fusion checkpoint to ONNX with a parity check (FUSION=... MODEL_DIR=...)
	$(PY) -m sensor_fusion.serve.export $(FUSION) $(MODEL_DIR)

serve: ## Run the service locally (MODEL_DIR=...)
	$(PY) -m sensor_fusion.serve.app --model-dir $(MODEL_DIR) --port 8000

docker-build: ## Build the CPU container image, tagged with the git commit
	docker build --build-arg GIT_COMMIT=$(GIT_COMMIT) -t sensor-fusion-serve:$(GIT_COMMIT) -t sensor-fusion-serve:latest .

docker-run: ## Run the container with the model mounted read-only (MODEL_DIR=...)
	docker run --rm -p 8000:8000 --read-only --tmpfs /tmp --cap-drop ALL \
	-v "$(abspath $(MODEL_DIR)):/models:ro" sensor-fusion-serve:latest

compose-up: ## Run the service with docker compose (limits, health check, read-only)
	GIT_COMMIT=$(GIT_COMMIT) MODEL_DIR=$(MODEL_DIR) docker compose up --build

load-test: ## Measure throughput and latency of a running service (SAMPLE=folder with a request)
	$(PY) scripts/load_test.py --url http://localhost:8000 --sample $(SAMPLE) --concurrency 1 2 4 8

clean: ## Remove caches and coverage files (not data, outputs or checkpoints)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
