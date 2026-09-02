import numpy as np
import pandas as pd

from fraudlake.features.selection import (
    adversarial_validation,
    filter_correlated,
    filter_near_constant,
    filter_null_rate,
)


def test_null_rate_filter():
    df = pd.DataFrame({"a": [1, None, None, None], "b": [1, 2, 3, None]})
    assert set(filter_null_rate(df, ["a", "b"], 0.5)) == {"a"}


def test_near_constant_filter():
    df = pd.DataFrame({"a": [0] * 999 + [1], "b": np.arange(1000), "c": [None] * 1000})
    d = filter_near_constant(df, ["a", "b", "c"], min_variance=1e-3)
    assert "b" not in d and "c" in d and "a" not in d  # a: top share exactly 0.999, not >
    d = filter_near_constant(df, ["a"], min_variance=2e-3)
    assert "a" in d


def test_correlation_prune_keeps_denser_column():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    df = pd.DataFrame({"v1": x, "v2": x * 2 + 1e-6, "v3": rng.standard_normal(500)})
    df.loc[:50, "v1"] = np.nan  # v2 is denser -> v2 kept, v1 dropped
    d = filter_correlated(df, ["v1", "v2", "v3"], 0.98)
    assert set(d) == {"v1"} and "v2" in d["v1"]


def test_adversarial_validation_detects_drift():
    rng = np.random.default_rng(0)
    n = 3000
    stable = rng.standard_normal(n)
    drift_tr, drift_ho = rng.standard_normal(n), rng.standard_normal(n) + 3
    Xtr = pd.DataFrame({"stable": stable, "drift": drift_tr})
    Xho = pd.DataFrame({"stable": rng.standard_normal(n), "drift": drift_ho})
    auc, top = adversarial_validation(Xtr, Xho, seed=0)
    assert auc > 0.9
    assert top[0][0] == "drift"
