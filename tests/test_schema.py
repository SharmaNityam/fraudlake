from fraudlake.ingest.schema import (
    CORRUPT_COL,
    ID_COLUMN_ORDER,
    TXN_COLUMN_ORDER,
    expected_columns,
    identity_schema,
    transaction_schema,
)


def test_transaction_column_count_matches_kaggle():
    assert len(TXN_COLUMN_ORDER) == 394
    assert expected_columns(transaction_schema(True)) == list(TXN_COLUMN_ORDER)


def test_test_transaction_schema_has_no_label():
    cols = expected_columns(transaction_schema(False))
    assert "isFraud" not in cols and len(cols) == 393


def test_identity_column_count_matches_kaggle():
    assert len(ID_COLUMN_ORDER) == 41
    assert expected_columns(identity_schema()) == list(ID_COLUMN_ORDER)


def test_corrupt_record_column_present():
    assert transaction_schema().fieldNames()[-1] == CORRUPT_COL
