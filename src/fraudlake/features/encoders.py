"""Categorical encoders that respect the time/fold boundary.

Tree models need numbers. For high-cardinality categoricals (card1 has ~13k
levels, device_info ~1.7k) one-hot is hopeless and label encoding invents an
ordering. Two encoders are provided:

* :class:`FrequencyEncoder` — value -> how often it occurs in the *fitting*
  data. Target-free, so it can be fit on the training window and applied
  anywhere.
* :class:`OOFTargetEncoder` — value -> smoothed fraud rate. Target encoding is
  the single most common source of leakage in tabular competitions: if the
  encoding for row *i* includes row *i*'s own label, the model reads the answer
  off the feature. ``fit_transform`` therefore produces out-of-fold encodings
  (each row is encoded by a model fit on the *other* folds) and ``transform``
  (for validation/holdout) uses the full-training-set statistics. The test
  suite asserts the OOF property directly.

Both are sklearn-compatible (``fit``/``transform``/``fit_transform``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

_MISSING = "__nan__"


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype("object").where(s.notna(), _MISSING).astype(str)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], normalize: bool = True):
        self.columns = columns
        self.normalize = normalize

    def fit(self, X: pd.DataFrame, y=None):
        self.maps_: dict[str, pd.Series] = {}
        for c in self.columns:
            counts = _as_str(X[c]).value_counts()
            self.maps_[c] = counts / len(X) if self.normalize else counts
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in self.columns:
            out[f"{c}_freq"] = _as_str(X[c]).map(self.maps_[c]).fillna(0.0).astype("float32")
        return out


class OOFTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], n_splits: int = 5, smoothing: float = 20.0, seed: int = 0):
        self.columns = columns
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.seed = seed

    def _encode(self, train_col: pd.Series, y: np.ndarray, apply_col: pd.Series) -> np.ndarray:
        prior = float(np.mean(y))
        stats = pd.DataFrame({"k": train_col.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        smoothed = (stats["sum"] + prior * self.smoothing) / (stats["count"] + self.smoothing)
        return apply_col.map(smoothed).fillna(prior).to_numpy(dtype="float32")

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.prior_ = float(y.mean())
        self.maps_: dict[str, pd.Series] = {}
        for c in self.columns:
            col = _as_str(X[c])
            stats = pd.DataFrame({"k": col.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
            self.maps_[c] = (stats["sum"] + self.prior_ * self.smoothing) / (stats["count"] + self.smoothing)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in self.columns:
            out[f"{c}_te"] = _as_str(X[c]).map(self.maps_[c]).fillna(self.prior_).astype("float32")
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        y = np.asarray(y)
        self.fit(X, y)
        out = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for c in self.columns:
            col = _as_str(X[c]).reset_index(drop=True)
            enc = np.empty(len(X), dtype="float32")
            for fit_idx, enc_idx in kf.split(col):
                enc[enc_idx] = self._encode(col.iloc[fit_idx], y[fit_idx], col.iloc[enc_idx])
            out[f"{c}_te"] = enc
        return out
