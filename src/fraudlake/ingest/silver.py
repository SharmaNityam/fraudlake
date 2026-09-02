"""Silver layer: joined, deduplicated, enriched transactions.

What happens here and why:

* **Join** transactions to identity (left; only ~24% of transactions have an
  identity row).
* **Dedupe** on ``TransactionID`` — the source has none, but the layer must be
  idempotent if the raw files are ever re-delivered.
* **Card fingerprint** ``card_uid``: the dataset has no customer id. Following
  the analysis popularised by the competition winners, ``D1`` is "days since
  the card was first seen", so ``txn_day - D1`` is constant for one card
  account. ``card1 || addr1 || (txn_day - D1)`` therefore approximates an
  account id far better than ``card1`` alone.
* **Derived columns**: log amount, the cents part of the amount (a surprisingly
  strong signal in this data), hour / day-of-week, normalised email domains.
* **Spark-side velocity features** (``spk_*``): counts and sums over the card's
  previous 1h / 24h / 7d, computed with ``Window.rangeBetween`` on the raw
  seconds column. The *same* features are rebuilt in SQL in the marts; a test
  asserts the two implementations agree, which is how you keep a Spark feature
  pipeline and a warehouse feature pipeline honest with each other.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from rich.console import Console

from fraudlake.config import Settings

console = Console()

VELOCITY_WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 7 * 86_400}


def normalise_email(col: str) -> F.Column:
    c = F.lower(F.trim(F.col(col)))
    # collapse regional variants: yahoo.co.uk -> yahoo, gmail.com -> gmail
    return F.when(c.isNull(), F.lit(None)).otherwise(F.split(c, r"\.").getItem(0))


def build_card_uid(df: DataFrame) -> DataFrame:
    d1_anchor = F.when(
        F.col("D1").isNotNull(), (F.col("txn_day") - F.col("D1")).cast("int")
    ).otherwise(F.lit(-1))
    return df.withColumn(
        "card_uid",
        F.concat_ws(
            "_",
            F.coalesce(F.col("card1").cast("int").cast("string"), F.lit("na")),
            F.coalesce(F.col("addr1").cast("int").cast("string"), F.lit("na")),
            d1_anchor.cast("string"),
        ),
    )


def add_derived_columns(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("amt_log", F.log1p("TransactionAmt"))
        .withColumn("amt_cents", F.round((F.col("TransactionAmt") % 1) * 100).cast("int"))
        .withColumn("txn_hour", F.hour("txn_ts"))
        .withColumn("txn_dow", F.dayofweek("txn_ts"))
        .withColumn("p_email_norm", normalise_email("P_emaildomain"))
        .withColumn("r_email_norm", normalise_email("R_emaildomain"))
        .withColumn("has_identity", F.col("id_01").isNotNull().cast("int"))
    )


def add_spark_velocity(df: DataFrame) -> DataFrame:
    """Backward-looking, current-row-excluded velocity per card_uid."""
    base = Window.partitionBy("card_uid").orderBy("TransactionDT")
    for label, seconds in VELOCITY_WINDOWS.items():
        w = base.rangeBetween(-seconds, -1)
        df = df.withColumn(f"spk_cnt_{label}", F.count(F.lit(1)).over(w)).withColumn(
            f"spk_amt_{label}", F.coalesce(F.sum("TransactionAmt").over(w), F.lit(0.0))
        )
    # order of this txn within the card's history (0-based), and seconds since previous
    df = df.withColumn("spk_card_txn_idx", F.row_number().over(base) - 1)
    df = df.withColumn(
        "spk_secs_since_prev",
        F.col("TransactionDT") - F.lag("TransactionDT").over(base),
    )
    return df


def build_silver(spark: SparkSession, settings: Settings) -> DataFrame:
    txn = spark.read.parquet(str(settings.bronze_dir / "transactions"))
    ident_path = settings.bronze_dir / "identity"
    if ident_path.exists():
        ident = spark.read.parquet(str(ident_path)).drop("source")
        txn = txn.join(ident, on="TransactionID", how="left")
    else:  # pragma: no cover
        console.print("[yellow]no identity table; silver will have null identity columns[/]")

    before = txn.count()
    txn = txn.dropDuplicates(["TransactionID"])
    dropped = before - txn.count()

    silver = add_spark_velocity(add_derived_columns(build_card_uid(txn)))
    out = settings.silver_dir / "transactions"
    (
        silver.repartition("txn_day")
        .sortWithinPartitions("TransactionDT")
        .write.mode("overwrite")
        .partitionBy("txn_day")
        .parquet(str(out))
    )
    console.print(f"[green]silver/transactions written ({dropped} duplicate rows removed)[/]")
    return spark.read.parquet(str(out))
