from __future__ import annotations

import os
from pathlib import Path

import pytest

from fraudlake.config import Settings
from fraudlake.synth import write_synthetic_raw

os.environ.setdefault("PYSPARK_PYTHON", os.sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", os.sys.executable)


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    """Deterministic synthetic IEEE-CIS CSVs (train + test), generated once per session."""
    out = tmp_path_factory.mktemp("raw")
    write_synthetic_raw(out, n_rows=2_000, seed=7)
    return out


@pytest.fixture(scope="session")
def settings(tmp_path_factory, raw_dir) -> Settings:
    data_dir = tmp_path_factory.mktemp("data")
    (data_dir / "raw").symlink_to(raw_dir, target_is_directory=True)
    s = Settings(
        data_dir=data_dir,
        spark_master="local[2]",
        spark_driver_memory="2g",
        spark_shuffle_partitions=4,
        optuna_trials=2,
        n_folds=3,
    )
    s.ensure_dirs()
    return s


@pytest.fixture(scope="session")
def spark(settings):
    from fraudlake.ingest.spark import get_spark

    session = get_spark(settings, app_name="fraudlake-tests")
    yield session
    session.stop()


# --- warehouse (Postgres via testcontainers) ---------------------------------------

import shutil  # noqa: E402

docker_missing = shutil.which("docker") is None


@pytest.fixture(scope="session")
def pg_dsn():
    if docker_missing:
        pytest.skip("docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine", driver=None) as pg:
        yield pg.get_connection_url()


@pytest.fixture(scope="session")
def warehouse(spark, settings, pg_dsn):
    """Bronze -> silver -> Postgres -> SQL marts, once per session. Returns (settings, n_rows)."""
    from fraudlake.features.sql_runner import run_sql_dir
    from fraudlake.ingest.bronze import build_bronze
    from fraudlake.ingest.load_pg import load_silver_to_postgres
    from fraudlake.ingest.silver import build_silver

    build_bronze(spark, settings)
    build_silver(spark, settings)
    s = settings.model_copy(
        update={
            "pg_host": pg_dsn.split("@")[1].split(":")[0],
            "pg_port": int(pg_dsn.split(":")[-1].split("/")[0]),
            "pg_db": pg_dsn.rsplit("/", 1)[1],
            "pg_user": pg_dsn.split("//")[1].split(":")[0],
            "pg_password": pg_dsn.split(":")[2].split("@")[0],
        }
    )
    n = load_silver_to_postgres(s)
    run_sql_dir(s)
    return s, n
