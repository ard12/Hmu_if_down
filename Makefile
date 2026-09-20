# Fall Detection Hub — Developer Makefile
#
# Usage:
#   make test          Run the full pytest suite
#   make train         Train model on datasets/ (ONNX export if skl2onnx is installed)
#   make docker        Build and start Docker Compose stack
#   make lint          Run flake8 over hub/ and tests/
#   make release TAG=v3.0.0   Tag and push a versioned release

.PHONY: all test train docker lint release clean report model-changelog

PYTHON  ?= python
PYTEST  ?= pytest
COMPOSE ?= docker compose

all: test

# ── Test ────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) -v tests/ --tb=short

# ── Train ───────────────────────────────────────────────────────────────────
train:
	$(PYTHON) hub/train.py --onnx

train-real:
	$(PYTHON) hub/train.py --data-dir datasets/trials/ --output-dir models/v2/ --onnx

# ── Docker ──────────────────────────────────────────────────────────────────
docker:
	$(COMPOSE) -f deploy/docker-compose.yml up --build

docker-down:
	$(COMPOSE) -f deploy/docker-compose.yml down

# ── Lint ────────────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m flake8 hub/ tests/ --max-line-length=120 --extend-ignore=E203,W503

# ── Release ─────────────────────────────────────────────────────────────────
release:
ifndef TAG
	$(error TAG is not set. Usage: make release TAG=v3.0.0)
endif
	git tag $(TAG)
	git push origin $(TAG)
	@echo "[OK] Released $(TAG)"

# ── Clinical Performance Report ──────────────────────────────────────────────
report:
	$(PYTHON) docs/generate_clinical_report.py \
	  --eval-report models/evaluation_report.json \
	  --audit-db audits/audit.db \
	  --output docs/CLINICAL_PERFORMANCE_REPORT.md

# ── Model Changelog ──────────────────────────────────────────────────────────
model-changelog:
	$(PYTHON) docs/generate_model_changelog.py \
	  --db models/registry.db \
	  --output docs/MODEL_CHANGELOG.md

# ── Clean ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ hub/__pycache__/ tests/__pycache__/


