"""Bronze layer: raw CSV -> typed, partitioned Parquet + data-quality audit.

Bronze is a faithful copy of the source with three changes only:
  * explicit types (see :mod:`fraudlake.ingest.schema`),
  * a ``txn_day`` partition column derived from ``TransactionDT``,
  * a ``source`` column (``train`` / ``test``) so both splits live in one table.
Malformed rows are captured in ``_corrupt_record`` (PERMISSIVE mode) and
counted in the audit rather than silently dropped.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType
from rich.console import Console

from fraudlake.config import REFERENCE_EPOCH, Settings
from fraudlake.ingest.schema import (
    CORRUPT_COL,
    expected_columns,
    identity_schema,
    normalise_identity_columns,
    transaction_schema,
)

console = Console()


def read_csv(spark: SparkSession, path: Path, schema: StructType) -> DataFrame:
    return (
        spark.read.option("header", "true")
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_COL)
        .option("nullValue", "")
        .schema(schema)
        .csv(str(path))
    )


def _validate_header(path: Path, expected: list[str], rename_hyphens: bool = False) -> None:
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
    if rename_hyphens:
        header = [h.replace("id-", "id_") for h in header]
    if header != expected:
        extra = sorted(set(header) - set(expected))
        missing = sorted(set(expected) - set(header))
        raise ValueError(f"{path.name}: header mismatch. missing={missing[:5]} extra={extra[:5]}")


def add_time_columns(df: DataFrame) -> DataFrame:
    """Derive ``txn_ts`` (timestamp) and ``txn_day`` (int day index) from ``TransactionDT``."""
    ref = F.to_timestamp(F.lit(REFERENCE_EPOCH))
    return df.withColumn(
        "txn_ts", (F.unix_timestamp(ref) + F.col("TransactionDT")).cast("timestamp")
    ).withColumn("txn_day", (F.col("TransactionDT") / F.lit(86_400)).cast("int"))


def audit(df: DataFrame, name: str) -> dict:
    """Row counts, corrupt-row count and per-column null rates (single pass)."""
    total = df.count()
    cols = [c for c in df.columns if c != CORRUPT_COL]
    agg_exprs = [F.sum(F.col(c).isNull().cast("long")).alias(c) for c in cols]
    if CORRUPT_COL in df.columns:
        agg_exprs.append(F.sum(F.col(CORRUPT_COL).isNotNull().cast("long")).alias("__corrupt"))
    row = df.agg(*agg_exprs).collect()[0].asDict()
    corrupt = int(row.pop("__corrupt", 0) or 0)
    null_rates = {c: (int(row[c] or 0) / total if total else 0.0) for c in cols}
    return {
        "table": name,
        "rows": total,
        "corrupt_rows": corrupt,
        "columns": len(cols),
        "null_rate": null_rates,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def build_bronze(spark: SparkSession, settings: Settings) -> dict[str, dict]:
    settings.ensure_dirs()
    raw, bronze = settings.raw_dir, settings.bronze_dir
    audits: dict[str, dict] = {}

    # --- transactions (train + test unioned, with source flag) ------------
    parts = []
    for source, has_label in (("train", True), ("test", False)):
        path = raw / f"{source}_transaction.csv"
        if not path.exists():
            console.print(f"[yellow]{path.name} not found; skipping[/]")
            continue
        schema = transaction_schema(has_label=has_label)
        _validate_header(path, expected_columns(schema))
        df = read_csv(spark, path, schema)
        if not has_label:
            df = df.withColumn("isFraud", F.lit(None).cast("int"))
        df = df.select(*expected_columns(transaction_schema(True)), CORRUPT_COL).withColumn(
            "source", F.lit(source)
        )
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no transaction CSVs in {raw}")
    txn = parts[0]
    for p in parts[1:]:
        txn = txn.unionByName(p)
    txn = add_time_columns(txn)
    audits["transactions"] = audit(txn, "transactions")
    (
        txn.filter(F.col(CORRUPT_COL).isNull())
        .drop(CORRUPT_COL)
        .repartition("txn_day")
        .write.mode("overwrite")
        .partitionBy("txn_day")
        .parquet(str(bronze / "transactions"))
    )
    console.print(
        f"[green]bronze/transactions: {audits['transactions']['rows']:,} rows "
        f"({audits['transactions']['corrupt_rows']} corrupt)[/]"
    )

    # --- identity ----------------------------------------------------------
    parts = []
    for source in ("train", "test"):
        path = raw / f"{source}_identity.csv"
        if not path.exists():
            continue
        _validate_header(path, expected_columns(identity_schema()), rename_hyphens=True)
        schema = identity_schema()
        with open(path, encoding="utf-8") as fh:
            if "id-01" in fh.readline():  # kaggle's test file uses hyphens
                schema = StructType(
                    [
                        StructField(f.name.replace("id_", "id-"), f.dataType, f.nullable)
                        for f in schema.fields
                    ]
                )
        df = normalise_identity_columns(read_csv(spark, path, schema))
        parts.append(df.withColumn("source", F.lit(source)))
    if parts:
        ident = parts[0]
        for p in parts[1:]:
            ident = ident.unionByName(p)
        audits["identity"] = audit(ident, "identity")
        (
            ident.filter(F.col(CORRUPT_COL).isNull())
            .drop(CORRUPT_COL)
            .write.mode("overwrite")
            .parquet(str(bronze / "identity"))
        )
        console.print(f"[green]bronze/identity: {audits['identity']['rows']:,} rows[/]")

    audit_dir = settings.artifacts_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "bronze_audit.json").write_text(json.dumps(audits, indent=2))
    return audits
