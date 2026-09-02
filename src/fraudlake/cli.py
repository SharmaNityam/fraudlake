"""Typer CLI: one sub-command per pipeline stage.

Stages are deliberately independent processes that communicate through files
(Parquet) and tables (Postgres) so each can be re-run, inspected and tested in
isolation. ``make all`` chains them in order.
"""

from __future__ import annotations

import typer
from rich.console import Console

from fraudlake.config import get_settings

app = typer.Typer(help="fraudlake pipeline", no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def data() -> None:
    """Download the IEEE-CIS competition files from Kaggle into data/raw."""
    from fraudlake.ingest.download import download_raw

    download_raw(get_settings())


@app.command()
def ingest(
    skip_load: bool = typer.Option(False, help="Skip loading silver into Postgres"),
) -> None:
    """PySpark: raw CSV -> bronze -> silver parquet -> Postgres raw tables."""
    from fraudlake.ingest.bronze import build_bronze
    from fraudlake.ingest.load_pg import load_silver_to_postgres
    from fraudlake.ingest.silver import build_silver
    from fraudlake.ingest.spark import get_spark

    settings = get_settings()
    spark = get_spark(settings)
    try:
        build_bronze(spark, settings)
        build_silver(spark, settings)
    finally:
        spark.stop()
    if not skip_load:
        load_silver_to_postgres(settings)


@app.command()
def features() -> None:
    """Run the versioned SQL feature marts against Postgres."""
    from fraudlake.features.sql_runner import run_sql_dir

    run_sql_dir(get_settings())


@app.command()
def select() -> None:
    """Feature selection on the training window; writes artifacts/features/selected.json."""
    from fraudlake.features.selection import run_selection

    run_selection(get_settings())


@app.command()
def train(
    models: str = typer.Option("rf,lightgbm,xgboost,catboost", help="Comma-separated model keys"),
    trials: int | None = typer.Option(None, help="Override Optuna trials per model"),
) -> None:
    """Train models with time-aware CV and Optuna; log to MLflow."""
    from fraudlake.modeling.train import run_training

    run_training(get_settings(), models=models.split(","), trials=trials)


@app.command()
def evaluate() -> None:
    """Evaluate best model on holdout, compute SHAP, register artifact."""
    from fraudlake.modeling.evaluate import run_evaluation

    run_evaluation(get_settings())


@app.command()
def report() -> None:
    """Generate model card, feature catalog and README results table."""
    from fraudlake.report import build_report

    build_report(get_settings())


if __name__ == "__main__":
    app()
