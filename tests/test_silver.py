import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from fraudlake.ingest.bronze import build_bronze
from fraudlake.ingest.silver import VELOCITY_WINDOWS, build_silver

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def silver_pdf(spark, settings) -> pd.DataFrame:
    build_bronze(spark, settings)
    df = build_silver(spark, settings)
    cols = ["TransactionID", "TransactionDT", "TransactionAmt", "card_uid", "source", "isFraud",
            "amt_cents", "txn_hour", "p_email_norm", "has_identity", "spk_card_txn_idx",
            "spk_secs_since_prev"]
    cols += [f"spk_cnt_{w}" for w in VELOCITY_WINDOWS] + [f"spk_amt_{w}" for w in VELOCITY_WINDOWS]
    return df.select(*cols).toPandas().sort_values("TransactionDT").reset_index(drop=True)


def test_silver_row_count_and_uniqueness(silver_pdf):
    assert len(silver_pdf) == 2_400
    assert silver_pdf["TransactionID"].is_unique


def test_identity_joined_for_subset(silver_pdf):
    share = silver_pdf["has_identity"].mean()
    assert 0.15 < share < 0.35  # generator uses 25% coverage


def test_derived_columns(silver_pdf):
    assert silver_pdf["amt_cents"].between(0, 100).all()
    assert silver_pdf["txn_hour"].between(0, 23).all()
    assert set(silver_pdf["p_email_norm"].dropna().unique()) <= {"gmail", "yahoo", "hotmail", "anonymous", "outlook"}


def test_spark_velocity_matches_bruteforce_and_never_leaks(silver_pdf):
    """For every row, the window feature must equal a brute-force count over strictly
    earlier rows of the same card within the window. Strictly earlier == no leakage."""
    df = silver_pdf
    rng = np.random.default_rng(0)
    sample = rng.choice(len(df), size=300, replace=False)
    for label, secs in VELOCITY_WINDOWS.items():
        for i in sample:
            row = df.iloc[i]
            mask = (
                (df["card_uid"] == row["card_uid"])
                & (df["TransactionDT"] >= row["TransactionDT"] - secs)
                & (df["TransactionDT"] <= row["TransactionDT"] - 1)
            )
            assert df.loc[mask].shape[0] == row[f"spk_cnt_{label}"], (label, row["TransactionID"])
            assert np.isclose(df.loc[mask, "TransactionAmt"].sum(), row[f"spk_amt_{label}"])


def test_card_txn_index_is_a_zero_based_rank(silver_pdf):
    g = silver_pdf.groupby("card_uid")["spk_card_txn_idx"]
    assert (g.min() == 0).all()
    assert (g.max() == g.size() - 1).all()


def test_hot_cards_have_higher_velocity_when_fraud(silver_pdf):
    # sanity that the planted signal survives: fraud rows should show more prior 24h activity
    train = silver_pdf[silver_pdf["source"] == "train"]
    f = train.groupby("isFraud")["spk_cnt_24h"].mean()
    assert f.loc[1] > f.loc[0]
