"""Central configuration for the fraudlake pipeline.

Every stage reads from a single ``Settings`` object so that paths, database
credentials and modelling choices are declared once and are overridable via
environment variables (prefix ``FRAUDLAKE_``) or a ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# IEEE-CIS ``TransactionDT`` is a seconds offset from an undisclosed reference
# point. The community consensus (from calendar-effect analysis of the data) is
# 2017-12-01 00:00:00; we adopt it so timestamps look real in the marts.
REFERENCE_EPOCH = "2017-12-01 00:00:00"

KAGGLE_COMPETITION = "ieee-fraud-detection"
RAW_FILES = (
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FRAUDLAKE_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    sql_dir: Path = PROJECT_ROOT / "sql"
    docs_dir: Path = PROJECT_ROOT / "docs"

    # --- postgres ------------------------------------------------------------
    pg_host: str = "localhost"
    pg_port: int = 5433
    pg_db: str = "fraudlake"
    pg_user: str = "fraudlake"
    pg_password: str = "fraudlake"

    # --- spark ---------------------------------------------------------------
    spark_master: str = "local[*]"
    spark_driver_memory: str = "6g"
    spark_shuffle_partitions: int = 16

    # --- modelling -----------------------------------------------------------
    seed: int = 42
    holdout_fraction: float = Field(0.2, description="Last X% of time is the untouched holdout")
    n_folds: int = 5
    fold_gap_seconds: int = 86_400  # 1 day gap between train and validation in each fold
    optuna_trials: int = 30
    optuna_timeout_seconds: int = 1_800

    # feature selection thresholds
    max_null_rate: float = 0.90
    min_variance: float = 1e-3
    max_abs_corr: float = 0.98
    adversarial_auc_threshold: float = 0.75
    adversarial_drop_top_n: int = 15

    # cost-sensitive threshold: what a missed fraud costs vs a manual review
    cost_false_negative: float = 100.0
    cost_false_positive: float = 5.0

    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "fraudlake"

    # --- derived helpers -----------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def mlflow_uri(self) -> str:
        # sqlite: zero-setup like the file store, but still supported by current MLflow
        return self.mlflow_tracking_uri or f"sqlite:///{self.artifacts_dir / 'mlflow.db'}"

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.bronze_dir, self.silver_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
