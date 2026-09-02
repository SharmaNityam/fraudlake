# fraudlake

**A production-shaped fraud-detection pipeline on the IEEE-CIS dataset: PySpark ingestion → Postgres SQL feature marts → gradient-boosted trees, validated the way production would judge them.**

```
kaggle csv ──► spark ingest ──► parquet (bronze / silver)
                                        │
                                        ▼
                         postgres feature marts (versioned SQL)
                                        │
                                        ▼
            time-aware CV · Optuna · LightGBM / XGBoost / CatBoost / RF
                                        │
                     MLflow · SHAP · bootstrap CIs · model card · registry
```

## What this shows

- **Data ingestion & pipelines.** 1.1M transactions × 435 columns go from raw CSV to typed, partitioned Parquet with a data-quality audit, then into Postgres by `COPY`, in under two minutes. Every stage is a CLI command that communicates through files and tables, so each can be re-run and tested alone.
- **PySpark.** Explicit schemas (no `inferSchema`), a reconstructed card-account fingerprint, and rolling velocity features via `Window.rangeBetween` on raw seconds. The same features are rebuilt in SQL and a test asserts the two implementations agree on every row.
- **SQL.** Seven versioned mart files using CTEs, `RANGE BETWEEN INTERVAL` window frames, `FILTER` clauses, `PERCENTILE_CONT`, and `LATERAL` joins, executed by a runner that records each file's SHA-256 in a migrations ledger. Every feature carries a one-line definition that the feature catalog is generated from.
- **Feature engineering & selection.** Card velocity, amount-vs-history z-scores, first-seen flags, device-sharing counts, stationary rarity shares. Selection by null rate, near-constant, correlation, **adversarial validation** (drops calendar proxies), and permutation importance, with a reason recorded for every dropped column.
- **Tree-based models.** RandomForest baseline, LightGBM, XGBoost and CatBoost behind one interface; Optuna tunes each on time folds; final refit uses the folds' early-stopping iterations.
- **Evaluation & validation.** Time-cut holdout, expanding-window folds with a gap, out-of-fold target encoding, bootstrap confidence intervals, cost-optimal threshold, calibration, SHAP, fold-to-fold importance stability, and a measured **optimism gap** between random K-fold and time-based validation.

## Results

<!-- results:start -->
_Run `make all` (or see the committed results below once generated)._
<!-- results:end -->

Detailed numbers, operating point and limitations: [`docs/model_card.md`](docs/model_card.md).
Why the random-split number is not real: [`docs/validation_strategy.md`](docs/validation_strategy.md).
Every engineered feature and its leakage guarantee: [`docs/feature_catalog.md`](docs/feature_catalog.md).

## Quickstart

Requirements: Python 3.12, Java 17 (for Spark), Docker, [`uv`](https://docs.astral.sh/uv/). On macOS LightGBM also needs `brew install libomp`.

```bash
make setup          # uv venv + deps
make db-up          # postgres:16 in docker (port 5439)

# real data (needs ~/.kaggle/kaggle.json and accepted competition rules)
make all            # data → ingest → features → select → train → evaluate → report

# or demo mode, no Kaggle account: synthetic data with the same schema
uv run fraudlake synth --rows 50000
make ingest features select train evaluate report
```

Stage by stage:

| command | what it does | writes |
|---|---|---|
| `fraudlake data` | download the four competition CSVs | `data/raw/` |
| `fraudlake ingest` | Spark: bronze → silver → Postgres `raw.transactions` | `data/bronze/`, `data/silver/`, audit JSON |
| `fraudlake features` | run `sql/*.sql` in order, record checksums | `feat.*`, `mart.training` |
| `fraudlake select` | feature selection on the training window | `artifacts/features/selected.json` |
| `fraudlake train` | Optuna + time-fold CV for each model, MLflow logging | `artifacts/models/<key>/`, `mlflow.db` |
| `fraudlake evaluate` | holdout metrics, random-vs-time comparison, SHAP, registry | `artifacts/evaluation/`, `artifacts/model/vN/` |
| `fraudlake report` | model card, feature catalog, README results | `docs/` |

`make mlflow-ui` opens the experiment tracker; `make test` runs the suite on a synthetic fixture (Spark local, Postgres via testcontainers) without any download.

## Layout

```
sql/                 000_schema · 010_stg · 100_dim_card · 110_fct_velocity · 120_fct_amount_stats
                     130_fct_device_email · 200_mart_training
src/fraudlake/
  ingest/            schema.py · bronze.py · silver.py · load_pg.py · download.py · spark.py
  features/          sql_runner.py · selection.py · encoders.py
  modeling/          splits.py · pipeline.py · models.py · train.py · metrics.py · evaluate.py
                     explain.py · registry.py · data.py
  synth.py           synthetic IEEE-CIS look-alike generator (tests + demo mode)
  report.py          generated docs
tests/               schema, bronze, silver, Spark↔SQL parity, leakage invariants,
                     splits, encoders, selection, end-to-end modelling
notebooks/           01_eda · 02_validation_design · 03_results
docs/                model_card · validation_strategy · feature_catalog
```

## Design decisions

**Why layers instead of one script.** Bronze is a faithful typed copy, silver is joined and enriched, marts are model-ready. Each layer has a contract and a test; a bug in a feature never requires re-reading 1.3 GB of CSV.

**Why the card fingerprint.** The dataset has no customer id. `D1` is "days since the card was first seen", so `txn_day − D1` is constant for one card account; `card1 ‖ addr1 ‖ (txn_day − D1)` recovers an account key that makes velocity and "first time this card did X" features possible.

**Why velocity is implemented twice.** Spark computes it at ingest time, SQL computes it in the warehouse. The parity test (`tests/test_sql_marts.py`) guarantees a feature-store style contract: the feature the model trained on is the feature the warehouse serves.

**Why time-based validation.** Random K-fold on this data leaks card identity across the split and never asks the model to extrapolate forward in time. The measured optimism gap is reported in the results table; the reasoning is in `docs/validation_strategy.md`.

**Why adversarial validation changed the SQL.** The first version of the rarity features counted prior occurrences of an email domain or device. A train-vs-holdout classifier reached AUC 0.99 on them because a cumulative count is a clock. They were redefined as shares of prior traffic, which are stationary. That decision trail is recorded in the SQL header.

**Why PR-AUC.** At 3.5% positives, ROC-AUC barely moves between a good and a great model. Average precision measures the queue a review team actually works.

**Why Postgres and not just DuckDB.** To show the same SQL running on the engine a feature pipeline would run on in production, with a real migrations ledger and indexes that the lateral joins depend on.

## Next steps

Structured Streaming replay of the transaction feed; a feature store service so the Spark and SQL definitions share one registry; drift monitoring on the adversarial-validation AUC; a scoring API around the registry artifact.

## Data

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Vesta Corporation, via Kaggle). The data is not redistributed here; `fraudlake data` downloads it with your own credentials after you accept the competition rules.
