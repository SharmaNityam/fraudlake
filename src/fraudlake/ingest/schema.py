"""Explicit Spark schemas for the IEEE-CIS CSV files.

We never use ``inferSchema``: it costs a full extra pass over ~1.3 GB of CSV
and silently changes types between runs (e.g. a column that happens to be all
integers in a sample). Declaring the 394 + 41 columns programmatically keeps
the contract explicit and lets the bronze stage detect malformed rows.

Column families (transaction file):
    TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD,
    card1..card6, addr1, addr2, dist1, dist2, P_emaildomain, R_emaildomain,
    C1..C14, D1..D15, M1..M9, V1..V339
Identity file:
    TransactionID, id_01..id_38, DeviceType, DeviceInfo

The public *test* identity file names columns ``id-01`` (hyphen) instead of
``id_01``; :func:`normalise_identity_columns` handles that.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

CORRUPT_COL = "_corrupt_record"

# --- transaction ------------------------------------------------------------

TXN_CATEGORICAL = ("ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain") + tuple(
    f"M{i}" for i in range(1, 10)
)

# Numeric columns in the CSV that carry integer semantics but are nullable and
# therefore serialised as "1.0" style floats by pandas upstream.
TXN_NUMERIC = (
    ("TransactionAmt",)
    + ("card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2")
    + tuple(f"C{i}" for i in range(1, 15))
    + tuple(f"D{i}" for i in range(1, 16))
    + tuple(f"V{i}" for i in range(1, 340))
)

TXN_COLUMN_ORDER: tuple[str, ...] = (
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
    *[f"M{i}" for i in range(1, 10)],
    *[f"V{i}" for i in range(1, 340)],
)

# --- identity ---------------------------------------------------------------

ID_CATEGORICAL = (
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
    "DeviceType",
    "DeviceInfo",
)

ID_COLUMN_ORDER: tuple[str, ...] = (
    "TransactionID",
    *[f"id_{i:02d}" for i in range(1, 39)],
    "DeviceType",
    "DeviceInfo",
)


def _field(name: str, has_label: bool) -> StructField:
    if name == "TransactionID":
        return StructField(name, LongType(), nullable=False)
    if name == "isFraud":
        return StructField(name, IntegerType(), nullable=not has_label)
    if name == "TransactionDT":
        return StructField(name, LongType(), nullable=False)
    if name in TXN_CATEGORICAL or name in ID_CATEGORICAL:
        return StructField(name, StringType(), nullable=True)
    return StructField(name, DoubleType(), nullable=True)


def transaction_schema(has_label: bool = True) -> StructType:
    """Schema for train_transaction.csv (or test_transaction.csv when ``has_label=False``)."""
    cols = [c for c in TXN_COLUMN_ORDER if has_label or c != "isFraud"]
    fields = [_field(c, has_label) for c in cols]
    fields.append(StructField(CORRUPT_COL, StringType(), nullable=True))
    return StructType(fields)


def identity_schema() -> StructType:
    fields = [_field(c, has_label=False) for c in ID_COLUMN_ORDER]
    fields.append(StructField(CORRUPT_COL, StringType(), nullable=True))
    return StructType(fields)


def normalise_identity_columns(df: DataFrame) -> DataFrame:
    """Rename ``id-01`` style headers (test file) to ``id_01`` (train file)."""
    renames = {c: c.replace("id-", "id_") for c in df.columns if c.startswith("id-")}
    for old, new in renames.items():
        df = df.withColumnRenamed(old, new)
    return df


def expected_columns(schema: StructType) -> list[str]:
    return [f.name for f in schema.fields if f.name != CORRUPT_COL]
