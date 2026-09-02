"""End-to-end warehouse tests: silver parquet -> Postgres -> SQL marts.

Runs against a throwaway Postgres container (testcontainers). Skipped when
Docker is unavailable so the rest of the suite still runs.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from fraudlake.features.sql_runner import fetch_df, run_sql_dir
from fraudlake.ingest.bronze import build_bronze
from fraudlake.ingest.load_pg import load_silver_to_postgres
from fraudlake.ingest.silver import build_silver

pytestmark = [pytest.mark.postgres, pytest.mark.spark]

docker_missing = shutil.which("docker") is None


@pytest.fixture(scope="module")
def pg_dsn():
    if docker_missing:
        pytest.skip("docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine", driver=None) as pg:
        yield pg.get_connection_url()


@pytest.fixture(scope="module")
def warehouse(spark, settings, pg_dsn):
    build_bronze(spark, settings)
    build_silver(spark, settings)
    s = settings.model_copy(update={
        "pg_host": pg_dsn.split("@")[1].split(":")[0],
        "pg_port": int(pg_dsn.split(":")[-1].split("/")[0]),
        "pg_db": pg_dsn.rsplit("/", 1)[1],
        "pg_user": pg_dsn.split("//")[1].split(":")[0],
        "pg_password": pg_dsn.split(":")[2].split("@")[0],
    })
    n = load_silver_to_postgres(s)
    run_sql_dir(s)
    return s, n


def test_load_row_count(warehouse):
    s, n = warehouse
    assert n == 2_400
    df = fetch_df(s.pg_dsn, "SELECT COUNT(*) AS n, COUNT(DISTINCT transaction_id) AS d FROM raw.transactions")
    assert df.loc[0, "n"] == df.loc[0, "d"] == 2_400


def test_migrations_ledger_records_every_file(warehouse):
    s, _ = warehouse
    df = fetch_df(s.pg_dsn, "SELECT filename, sha256 FROM public._fraudlake_migrations ORDER BY id")
    assert list(df["filename"]) == sorted(p.name for p in s.sql_dir.glob("*.sql"))
    assert df["sha256"].str.len().eq(64).all()


def test_mart_has_one_row_per_transaction(warehouse):
    s, _ = warehouse
    df = fetch_df(s.pg_dsn, "SELECT COUNT(*) AS n FROM mart.training")
    assert df.loc[0, "n"] == 2_400


def test_sql_velocity_matches_spark_velocity(warehouse):
    """The Spark window features (spk_*) and the Postgres window features (vel_*)
    implement the same definition and must agree on every single row."""
    s, _ = warehouse
    df = fetch_df(
        s.pg_dsn,
        """SELECT spk_cnt_1h, vel_cnt_1h, spk_cnt_24h, vel_cnt_24h, spk_cnt_7d, vel_cnt_7d,
                  spk_amt_1h, vel_amt_1h, spk_amt_24h, vel_amt_24h, spk_amt_7d, vel_amt_7d,
                  spk_card_txn_idx, card_txn_idx, spk_secs_since_prev, secs_since_prev
           FROM mart.training""",
    )
    for w in ("1h", "24h", "7d"):
        assert (df[f"spk_cnt_{w}"] == df[f"vel_cnt_{w}"]).all(), w
        assert np.allclose(df[f"spk_amt_{w}"], df[f"vel_amt_{w}"]), w
    assert (df["spk_card_txn_idx"] == df["card_txn_idx"]).all()
    pd.testing.assert_series_equal(
        df["spk_secs_since_prev"].astype("float"), df["secs_since_prev"].astype("float"), check_names=False
    )


def test_amount_stats_never_see_current_or_future_rows(warehouse):
    s, _ = warehouse
    df = fetch_df(
        s.pg_dsn,
        "SELECT transaction_id, card_uid, transaction_dt, transaction_amt, prior_n_txn, prior_amt_mean, "
        "amt_over_prior_max, is_first_txn_on_card FROM mart.training ORDER BY transaction_dt, transaction_id",
    )
    rng = np.random.default_rng(1)
    for i in rng.choice(len(df), size=200, replace=False):
        row = df.iloc[i]
        prior = df[(df["card_uid"] == row["card_uid"]) & (df["transaction_dt"] < row["transaction_dt"])]
        assert row["prior_n_txn"] == len(prior)
        assert row["is_first_txn_on_card"] == int(len(prior) == 0)
        if len(prior):
            assert np.isclose(row["prior_amt_mean"], prior["transaction_amt"].mean())
            assert row["amt_over_prior_max"] == int(row["transaction_amt"] > prior["transaction_amt"].max())
        else:
            assert pd.isna(row["prior_amt_mean"])


def test_device_sharing_is_backward_looking(warehouse):
    s, _ = warehouse
    df = fetch_df(
        s.pg_dsn,
        "SELECT transaction_id, card_uid, txn_ts, device_info, device_cards_24h FROM mart.training "
        "WHERE device_info IS NOT NULL ORDER BY txn_ts",
    )
    for i in np.random.default_rng(2).choice(len(df), size=100, replace=False):
        row = df.iloc[i]
        others = df[
            (df["device_info"] == row["device_info"])
            & (df["card_uid"] != row["card_uid"])
            & (df["txn_ts"] >= row["txn_ts"] - pd.Timedelta(hours=24))
            & (df["txn_ts"] < row["txn_ts"])
        ]
        assert row["device_cards_24h"] == others["card_uid"].nunique()


def test_feature_catalog_annotations_present(settings):
    """Every feat.* file documents its features with `-- feat:` lines the report scrapes."""
    for p in settings.sql_dir.glob("1*_fct_*.sql"):
        assert "-- feat:" in p.read_text(), p.name
        assert "leakage:" in p.read_text(), p.name
