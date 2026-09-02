"""Ordered SQL executor with a migrations ledger.

``sql/`` holds numbered files (``000_``, ``010_``, ... ``200_``). Each is run
in its own transaction, in lexical order, and recorded in
``public._fraudlake_migrations`` with its SHA-256 and runtime so a reviewer can
see exactly which version of each mart produced a given model.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg
from rich.console import Console
from rich.table import Table

from fraudlake.config import Settings

console = Console()

LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS public._fraudlake_migrations (
    id          serial PRIMARY KEY,
    filename    text NOT NULL,
    sha256      text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now(),
    duration_ms integer NOT NULL
)
"""


@dataclass(frozen=True)
class SqlFile:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


def list_sql_files(sql_dir: Path) -> list[SqlFile]:
    return [SqlFile(p) for p in sorted(sql_dir.glob("*.sql"))]


def run_sql_dir(settings: Settings, dsn: str | None = None, sql_dir: Path | None = None) -> list[tuple[str, int]]:
    dsn = dsn or settings.pg_dsn
    files = list_sql_files(sql_dir or settings.sql_dir)
    results: list[tuple[str, int]] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(LEDGER_DDL)
        conn.commit()
        for f in files:
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(f.text)
                ms = int((time.perf_counter() - t0) * 1000)
                cur.execute(
                    "INSERT INTO public._fraudlake_migrations (filename, sha256, duration_ms) VALUES (%s, %s, %s)",
                    (f.name, f.sha256, ms),
                )
            conn.commit()
            results.append((f.name, ms))

    table = Table(title="SQL marts", show_lines=False)
    table.add_column("file")
    table.add_column("ms", justify="right")
    for name, ms in results:
        table.add_row(name, f"{ms:,}")
    console.print(table)
    return results


def fetch_df(dsn: str, sql: str):
    """Small helper used by modelling stages: SELECT -> pandas."""
    import pandas as pd

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        return pd.DataFrame.from_records(cur.fetchall(), columns=cols)
