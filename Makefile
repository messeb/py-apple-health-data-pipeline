# Apple Health Data Pipeline: common tasks. `make help` lists them.
# Override any variable on the command line, e.g.
#   make pipeline EXPORT_DIR=~/Downloads/apple_health_export TIMEZONE=America/New_York

PYTHON_VERSION ?= 3.11
VENV           ?= .venv
PY             := $(VENV)/bin/python
CLI            := $(VENV)/bin/health-lakehouse

EXPORT_DIR     ?= apple_health_export
DATA_DIR       ?= data
TIMEZONE       ?= auto
ML_ARGS        ?=
SAMPLE_DIR     ?= sample_data
TARGET         ?= dev
# Prefer a project-local CLI (downloaded to .cache/) over the one on PATH.
DATABRICKS     ?= $(shell test -x .cache/databricks-cli/databricks && echo .cache/databricks-cli/databricks || echo databricks)

.DEFAULT_GOAL := help
.PHONY: help setup sample pipeline ml app demo test test-spark lint format privacy-check \
        bundle-validate bundle-deploy pipeline-databricks app-databricks clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target> [VAR=value ...]\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nVariables: EXPORT_DIR=%s DATA_DIR=%s TIMEZONE=%s TARGET=%s\n" "$(EXPORT_DIR)" "$(DATA_DIR)" "$(TIMEZONE)" "$(TARGET)"

# --- local ------------------------------------------------------------------

setup: ## Create .venv with uv and install the package with all extras
	uv venv --python $(PYTHON_VERSION) $(VENV)
	uv pip install --python $(PY) -e ".[app,ml,dev,spark]"

sample: ## Write a synthetic Apple Health export to $(SAMPLE_DIR)/apple_health_export
	$(CLI) sample --out $(SAMPLE_DIR)/apple_health_export

pipeline: ## Build bronze -> silver -> gold from EXPORT_DIR into DATA_DIR (TIMEZONE=auto|<IANA zone>)
	$(CLI) all --export-dir $(EXPORT_DIR) --data-dir $(DATA_DIR) --timezone $(TIMEZONE)

ml: ## Run the ML experiments on DATA_DIR (ML_ARGS=--quick for a fast run)
	HEALTH_DATA_DIR=$(DATA_DIR) $(PY) ml/run_all.py $(ML_ARGS)

app: ## Start the Streamlit app on DATA_DIR
	HEALTH_DATA_DIR=$(DATA_DIR) $(VENV)/bin/streamlit run app/app.py

demo: ## End to end on synthetic data: sample -> pipeline -> quick ML (no personal data needed)
	$(MAKE) sample SAMPLE_DIR=$(SAMPLE_DIR)
	$(MAKE) pipeline EXPORT_DIR=$(SAMPLE_DIR)/apple_health_export DATA_DIR=$(SAMPLE_DIR)/data
	HEALTH_DATA_DIR=$(SAMPLE_DIR)/data HEALTHML_REPORT_DIR=$(SAMPLE_DIR)/reports $(PY) ml/run_all.py --quick
	@printf "\nDemo data is ready. Start the app with:\n\n    make app DATA_DIR=$(SAMPLE_DIR)/data\n\n"

# --- quality ----------------------------------------------------------------

test: ## Run the test suite (without the Spark parity test)
	$(PY) -m pytest -q --ignore=tests/test_spark_sql.py

test-fast: ## Run the fast unit tests only, as CI does (no torch, Spark or Streamlit needed)
	$(PY) -m pytest -q -rs --ignore=tests/test_spark_sql.py --ignore=tests/test_app.py --ignore=tests/test_ml.py

test-spark: ## Run the Spark parity test: every SQL file on local Spark + Delta vs DuckDB (needs Java 17+)
	$(PY) -m pytest -q tests/test_spark_sql.py

lint: ## Check style with ruff (lint + format check)
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

format: ## Auto-format and auto-fix with ruff
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

privacy-check: ## Make sure no health data, workspace hosts or tokens would be committed
	$(PY) scripts/check_no_health_data.py

# --- Databricks -------------------------------------------------------------
# Auth: `databricks auth login --host <workspace-url>`, then export DATABRICKS_HOST or
# DATABRICKS_CONFIG_PROFILE. The bundle itself contains no workspace host.

bundle-validate: ## Validate the Databricks Asset Bundle (TARGET=dev|prod)
	$(DATABRICKS) bundle validate -t $(TARGET)

bundle-deploy: bundle-validate ## Deploy the bundle: job, notebooks and app (TARGET=dev|prod)
	$(DATABRICKS) bundle deploy -t $(TARGET)

pipeline-databricks: ## Run the bronze -> gold -> ML job on Databricks
	$(DATABRICKS) bundle run -t $(TARGET) health_pipeline

app-databricks: ## Deploy and start the Streamlit app as a Databricks App
	$(DATABRICKS) bundle run -t $(TARGET) health_app

# --- housekeeping -----------------------------------------------------------

clean: ## Remove caches and synthetic sample data (never touches data/ or your export)
	rm -rf $(SAMPLE_DIR) .pytest_cache .ruff_cache build dist *.egg-info spark-warehouse metastore_db derby.log
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
