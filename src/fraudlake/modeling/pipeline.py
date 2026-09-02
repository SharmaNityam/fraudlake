"""Turn a mart frame into a numeric matrix, fold-safely.

``FeaturePipeline.fit_transform`` is called on each fold's *training* rows and
``transform`` on its validation rows, so every statistic (frequencies, target
means) is computed only from data the model is allowed to see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlake.features.encoders import FrequencyEncoder, OOFTargetEncoder
from fraudlake.modeling.data import CATEGORICAL, HIGH_CARD_NUMERIC, TARGET_ENCODE


class FeaturePipeline:
    def __init__(self, features: list[str], seed: int = 0, target_encode: bool = True):
        self.features = list(features)
        self.seed = seed
        self.cats = [c for c in self.features if c in CATEGORICAL]
        self.nums = [c for c in self.features if c not in CATEGORICAL]
        self.freq_cols = self.cats + [c for c in HIGH_CARD_NUMERIC if c in self.features]
        self.te_cols = [c for c in TARGET_ENCODE if c in self.cats] if target_encode else []
        self.freq = FrequencyEncoder(self.freq_cols)
        self.te = OOFTargetEncoder(self.te_cols, seed=seed) if self.te_cols else None

    @property
    def output_columns(self) -> list[str]:
        cols = list(self.nums) + [f"{c}_freq" for c in self.freq_cols]
        cols += [f"{c}_te" for c in self.te_cols]
        return cols

    def _numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.nums].astype("float32")

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        parts = [self._numeric(X), self.freq.fit(X).transform(X)]
        if self.te is not None:
            parts.append(self.te.fit_transform(X, y))
        return pd.concat(parts, axis=1)[self.output_columns]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = [self._numeric(X), self.freq.transform(X)]
        if self.te is not None:
            parts.append(self.te.transform(X))
        return pd.concat(parts, axis=1)[self.output_columns]
