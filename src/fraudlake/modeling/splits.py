"""Time-aware data splitting.

Random K-fold on transaction data leaks in two ways: (1) the same card's
transactions land on both sides of the split, so the model memorises card
identity; (2) the fraud rate and feature distributions drift over the six
months of data, so a random split reports a score you will never see in
production. The public IEEE-CIS test set is *later in time* than train, which
is exactly the regime this module reproduces.

    |<------------- train window (80% of time) ------------->|<-- holdout -->|
    | fold1 train | gap | fold1 val |
    | fold2 train ..............    | gap | fold2 val |
    ...

* ``holdout``: last ``holdout_fraction`` of *time* (not rows). Touched once, at
  the very end, by ``evaluate``.
* CV folds inside the train window: expanding-window ``TimeSeriesSplit`` with a
  ``gap`` (default 1 day) so features whose windows straddle the boundary
  cannot carry validation-period information into training.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class Fold:
    index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_end_dt: int
    val_start_dt: int
    val_end_dt: int


def time_holdout_split(
    dt: pd.Series, holdout_fraction: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split positional indices into (train_window, holdout) by a time cut, not a row cut."""
    if not dt.is_monotonic_increasing:
        raise ValueError("dt must be sorted ascending; sort the frame by TransactionDT first")
    lo, hi = int(dt.iloc[0]), int(dt.iloc[-1])
    cut = int(lo + (hi - lo) * (1 - holdout_fraction))
    pos = np.arange(len(dt))
    is_hold = dt.to_numpy() >= cut
    return pos[~is_hold], pos[is_hold], cut


def time_series_folds(dt: pd.Series, n_folds: int, gap_seconds: int) -> Iterator[Fold]:
    """Expanding-window folds over a time-sorted ``dt`` (positional indices).

    ``TimeSeriesSplit`` works in row space; we convert the ``gap`` from seconds
    into rows by trimming any training row whose ``dt`` falls within
    ``gap_seconds`` of the fold's validation start. The result is a true
    temporal gap regardless of transaction density.
    """
    if not dt.is_monotonic_increasing:
        raise ValueError("dt must be sorted ascending")
    values = dt.to_numpy()
    tss = TimeSeriesSplit(n_splits=n_folds)
    for i, (tr, va) in enumerate(tss.split(values)):
        val_start = int(values[va[0]])
        tr = tr[values[tr] < val_start - gap_seconds]
        if len(tr) == 0:
            raise ValueError("gap too large for fold size")
        yield Fold(
            index=i,
            train_idx=tr,
            val_idx=va,
            train_end_dt=int(values[tr[-1]]),
            val_start_dt=val_start,
            val_end_dt=int(values[va[-1]]),
        )
