import numpy as np
import pandas as pd
import pytest

from fraudlake.modeling.splits import time_holdout_split, time_series_folds


@pytest.fixture
def dt():
    rng = np.random.default_rng(0)
    return pd.Series(np.sort(rng.integers(86_400, 30 * 86_400, size=5_000)))


def test_holdout_is_a_time_cut_not_a_row_cut(dt):
    tr, ho, cut = time_holdout_split(dt, 0.2)
    assert dt.iloc[tr].max() < cut <= dt.iloc[ho].min()
    assert len(tr) + len(ho) == len(dt)
    # 20% of *time*, so roughly 20% of rows for uniform data, but defined by cut
    assert 0.1 < len(ho) / len(dt) < 0.3


def test_folds_are_expanding_and_never_look_forward(dt):
    folds = list(time_series_folds(dt, n_folds=4, gap_seconds=86_400))
    assert len(folds) == 4
    prev_len = 0
    for f in folds:
        assert dt.iloc[f.train_idx].max() < dt.iloc[f.val_idx].min()  # strictly earlier
        assert f.val_start_dt - f.train_end_dt >= 86_400  # gap respected
        assert len(f.train_idx) > prev_len  # expanding
        prev_len = len(f.train_idx)
        assert len(np.intersect1d(f.train_idx, f.val_idx)) == 0


def test_unsorted_input_rejected():
    with pytest.raises(ValueError):
        time_holdout_split(pd.Series([3, 1, 2]), 0.2)
