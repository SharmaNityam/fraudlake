import json

import pytest
from pyspark.sql import functions as F

from fraudlake.ingest.bronze import build_bronze

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def bronze(spark, settings):
    return build_bronze(spark, settings)


def test_bronze_writes_partitioned_parquet_with_time_columns(spark, settings, bronze):
    df = spark.read.parquet(str(settings.bronze_dir / "transactions"))
    assert {"txn_ts", "txn_day", "source", "isFraud"} <= set(df.columns)
    assert df.filter(F.col("source") == "train").count() == 2_000
    assert df.filter(F.col("source") == "test").count() == 400
    # partition directories exist
    assert any(p.name.startswith("txn_day=") for p in (settings.bronze_dir / "transactions").iterdir())


def test_bronze_types_are_explicit(spark, settings, bronze):
    df = spark.read.parquet(str(settings.bronze_dir / "transactions"))
    types = dict(df.dtypes)
    assert types["TransactionID"] == "bigint"
    assert types["TransactionAmt"] == "double"
    assert types["ProductCD"] == "string"
    assert types["V339"] == "double"
    assert types["txn_ts"] == "timestamp"


def test_identity_hyphen_headers_normalised(spark, settings, bronze):
    df = spark.read.parquet(str(settings.bronze_dir / "identity"))
    assert "id_01" in df.columns and not any(c.startswith("id-") for c in df.columns)
    assert df.filter(F.col("source") == "test").count() > 0


def test_audit_json_written(settings, bronze):
    payload = json.loads((settings.artifacts_dir / "audit" / "bronze_audit.json").read_text())
    assert payload["transactions"]["rows"] == 2_400
    assert payload["transactions"]["corrupt_rows"] == 0
    assert 0.0 <= payload["transactions"]["null_rate"]["dist2"] <= 1.0
