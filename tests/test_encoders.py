import numpy as np
import pandas as pd

from fraudlake.features.encoders import FrequencyEncoder, OOFTargetEncoder


def test_frequency_encoder_handles_unseen_and_nan():
    X = pd.DataFrame({"c": ["a", "a", "b", None]})
    enc = FrequencyEncoder(["c"]).fit(X)
    out = enc.transform(pd.DataFrame({"c": ["a", "zzz", None]}))
    assert out["c_freq"].tolist() == [0.5, 0.0, 0.25]


def test_target_encoder_oof_property():
    """A category that appears exactly once with y=1 must NOT get an encoding above the
    prior on its own row: its own label is excluded from its encoding."""
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"c": rng.choice(list("abcdef"), size=n)})
    y = rng.binomial(1, 0.05, size=n)
    X.loc[n - 1, "c"] = "unique"
    y[n - 1] = 1
    enc = OOFTargetEncoder(["c"], n_splits=5, smoothing=1.0)
    oof = enc.fit_transform(X, y)
    # OOF: "unique" is unseen in the fitting folds, so it gets that fold's prior (~0.05),
    # never its own positive label.
    assert oof.loc[n - 1, "c_te"] < 0.15
    # whereas a plain transform (stats from the full data) does see it: (1 + 0.05) / (1 + 1)
    assert enc.transform(X).loc[n - 1, "c_te"] > 0.4


def test_target_encoder_smoothing_pulls_rare_levels_to_prior():
    X = pd.DataFrame({"c": ["rare"] + ["common"] * 999})
    y = np.array([1] + [0] * 999)
    enc = OOFTargetEncoder(["c"], smoothing=50.0).fit(X, y)
    rare = enc.transform(pd.DataFrame({"c": ["rare"]})).iloc[0, 0]
    assert rare < 0.05  # 1 positive / (1 + 50) smoothing -> close to prior 0.001
