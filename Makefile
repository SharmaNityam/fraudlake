.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv
RUN := $(UV) run

.PHONY: help setup lint test test-fast db-up db-down data ingest features select train evaluate report all clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install all deps
	$(UV) sync --all-extras --group dev
	$(RUN) pre-commit install || true

lint: ## Ruff lint + format check
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

test: ## Run the full test suite (Spark local + Postgres via testcontainers)
	$(RUN) pytest

test-fast: ## Unit tests only, no Spark/Postgres
	$(RUN) pytest -m "not spark and not postgres"

db-up: ## Start Postgres (docker compose) and wait for healthy
	docker compose up -d --wait postgres

db-down: ## Stop Postgres
	docker compose down

data: ## Download IEEE-CIS raw CSVs from Kaggle into data/raw
	$(RUN) fraudlake data

ingest: ## PySpark: raw CSV -> bronze parquet -> silver parquet -> Postgres raw tables
	$(RUN) fraudlake ingest

features: ## Build SQL feature marts in Postgres
	$(RUN) fraudlake features

select: ## Feature selection (null/variance/corr/adversarial/permutation)
	$(RUN) fraudlake select

train: ## Train baseline + LightGBM/XGBoost/CatBoost with Optuna, log to MLflow
	$(RUN) fraudlake train

evaluate: ## Evaluate best model on time holdout, SHAP, register artifact
	$(RUN) fraudlake evaluate

report: ## Generate model card, feature catalog, README results table
	$(RUN) fraudlake report

all: db-up data ingest features select train evaluate report ## Full pipeline end to end

mlflow-ui: ## Open MLflow UI on the local file store
	$(RUN) mlflow ui --backend-store-uri file:./data/artifacts/mlruns --port 5001

clean: ## Remove derived data (keeps raw download)
	rm -rf data/bronze data/silver data/artifacts
