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
