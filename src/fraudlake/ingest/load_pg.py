"""Load silver Parquet into Postgres with COPY.

Why COPY and not Spark's JDBC writer: JDBC does row-by-row inserts unless you
tune batch sizes and still ends up 10-50x slower than ``COPY ... FROM STDIN``.
We stream Arrow record batches -> CSV bytes -> COPY inside one transaction, so
a failed load leaves the previous table intact (truncate + load are atomic).

Column names are converted to snake_case so the SQL marts read naturally
(``TransactionAmt`` -> ``transaction_amt``, ``isFraud`` -> ``is_fraud``).
"""

from __future__ import annotations

import re
import time

import psycopg
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
from rich.console import Console

from fraudlake.config import Settings

console = Console()

SCHEMA = "raw"
TABLE = "transactions"

_SPECIAL = {
    "TransactionID": "transaction_id",
    "TransactionDT": "transaction_dt",
    "TransactionAmt": "transaction_amt",
    "isFraud": "is_fraud",
    "ProductCD": "product_cd",
    "P_emaildomain": "p_emaildomain",
    "R_emaildomain": "r_emaildomain",
    "DeviceType": "device_type",
    "DeviceInfo": "device_info",
}


def to_pg_name(col: str) -> str:
    if col in _SPECIAL:
        return _SPECIAL[col]
    return re.sub(r"[^a-z0-9_]", "_", col.lower())


def _pg_type(t: pa.DataType) -> str:
    if pa.types.is_int64(t):
        return "bigint"
    if pa.types.is_integer(t):
        return "integer"
    if pa.types.is_floating(t):
        return "double precision"
    if pa.types.is_boolean(t):
        return "boolean"
    if pa.types.is_timestamp(t):
        return "timestamp"
    if pa.types.is_dictionary(t):
        return _pg_type(t.value_type)
    return "text"


def _ddl(schema: pa.Schema) -> tuple[str, list[str]]:
    cols = [to_pg_name(f.name) for f in schema]
    defs = ", ".join(f'"{c}" {_pg_type(f.type)}' for c, f in zip(cols, schema, strict=True))
    return f'CREATE TABLE {SCHEMA}.{TABLE} ({defs})', cols


def load_silver_to_postgres(settings: Settings, batch_rows: int = 50_000) -> int:
    dataset = ds.dataset(str(settings.silver_dir / "transactions"), format="parquet", partitioning="hive")
    schema = dataset.schema
    create_sql, cols = _ddl(schema)
    col_list = ", ".join(f'"{c}"' for c in cols)
    t0 = time.perf_counter()
    total = 0

    with psycopg.connect(settings.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            cur.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE}")
            cur.execute(create_sql)
            copy_sql = f"COPY {SCHEMA}.{TABLE} ({col_list}) FROM STDIN WITH (FORMAT csv, NULL '')"
            with cur.copy(copy_sql) as copy:
                opts = pcsv.WriteOptions(include_header=False)
                for batch in dataset.to_batches(batch_size=batch_rows):
                    # arrow -> csv keeps int columns as ints (pandas would upcast nullable
                    # ints to float and emit "14490.0", which COPY rejects for bigint)
                    sink = pa.BufferOutputStream()
                    pcsv.write_csv(batch, sink, write_options=opts)
                    copy.write(sink.getvalue().to_pybytes())
                    total += batch.num_rows
                    console.print(f"  copied {total:,} rows", end="\r")
            cur.execute(
                f'CREATE INDEX ON {SCHEMA}.{TABLE} ("card_uid", "transaction_dt")'
            )
            cur.execute(f'CREATE UNIQUE INDEX ON {SCHEMA}.{TABLE} ("transaction_id")')
            cur.execute(f'CREATE INDEX ON {SCHEMA}.{TABLE} ("txn_ts")')
            cur.execute(f"ANALYZE {SCHEMA}.{TABLE}")
        conn.commit()
    console.print(f"\n[green]loaded {total:,} rows into {SCHEMA}.{TABLE} in {time.perf_counter() - t0:.1f}s[/]")
    return total
