import pandas as pd

from fraudlake.ingest.schema import ID_COLUMN_ORDER, TXN_COLUMN_ORDER


def test_synthetic_files_match_kaggle_layout(raw_dir):
    txn = pd.read_csv(raw_dir / "train_transaction.csv", nrows=5)
    ident = pd.read_csv(raw_dir / "train_identity.csv", nrows=5)
    test_id = pd.read_csv(raw_dir / "test_identity.csv", nrows=5)
    assert list(txn.columns) == list(TXN_COLUMN_ORDER)
    assert list(ident.columns) == list(ID_COLUMN_ORDER)
    assert test_id.columns[1] == "id-01"  # kaggle quirk reproduced


def test_synthetic_is_time_ordered_and_imbalanced(raw_dir):
    txn = pd.read_csv(raw_dir / "train_transaction.csv", usecols=["TransactionDT", "isFraud"])
    assert txn["TransactionDT"].is_monotonic_increasing
    assert 0.01 < txn["isFraud"].mean() < 0.15
