"""Load ``mart.training`` into pandas and describe its columns.

The mart is the only input to modelling. It is cached to Parquet under
``artifacts/mart`` so the (slow) Postgres pull happens once per feature build;
``fraudlake features`` invalidates the cache.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg
from rich.console import Console

from fraudlake.config import Settings

console = Console()

TARGET = "is_fraud"
TIME_COL = "transaction_dt"
ID_COLS = ("transaction_id", "transaction_dt", "txn_ts", "txn_day", "source", "card_uid")

# raw string columns that need encoding before a tree can consume them
CATEGORICAL = (
    "product_cd",
    "card4",
    "card6",
    "p_emaildomain",
    "r_emaildomain",
    "p_email_norm",
    "r_email_norm",
    *[f"m{i}" for i in range(1, 10)],
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "device_type",
    "device_info",
)
# numeric ids that behave like categoricals: frequency-encode them *as well as* keeping raw
HIGH_CARD_NUMERIC = ("card1", "card2", "card3", "card5", "addr1", "addr2")
# target-encode only where the level count justifies it
TARGET_ENCODE = (
    "product_cd",
    "card4",
    "card6",
    "p_email_norm",
    "r_email_norm",
    "device_info",
    "id_30",
    "id_31",
)


def mart_cache_path(settings: Settings) -> Path:
    return settings.artifacts_dir / "mart" / "training.parquet"


def load_training_frame(settings: Settings, refresh: bool = False) -> pd.DataFrame:
    """Labelled rows only (``source='train'``), sorted by time, cached as Parquet."""
    cache = mart_cache_path(settings)
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    console.print("[cyan]pulling mart.training from Postgres ...[/]")
    sql = (
        "SELECT * FROM mart.training WHERE source = 'train' ORDER BY transaction_dt, transaction_id"
    )
    with psycopg.connect(settings.pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        df = pd.DataFrame.from_records(cur.fetchall(), columns=cols)
    df = _downcast(df)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    console.print(f"[green]cached {len(df):,} rows x {df.shape[1]} cols -> {cache}[/]")
    return df


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype == "object" and c not in ID_COLS and c != "txn_ts":
            df[c] = df[c].astype("string")
    return df


def candidate_features(df: pd.DataFrame) -> list[str]:
    """All model-eligible columns: everything except ids/target and the Spark duplicates
    of the SQL velocity features (``spk_*`` are kept in the mart only for the parity test)."""
    drop = set(ID_COLS) | {TARGET}
    return [c for c in df.columns if c not in drop and not c.startswith("spk_")]


def split_by_type(features: list[str]) -> tuple[list[str], list[str]]:
    cats = [c for c in features if c in CATEGORICAL]
    nums = [c for c in features if c not in CATEGORICAL]
    return nums, cats
