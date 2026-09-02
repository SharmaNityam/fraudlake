"""Model zoo with a uniform interface and Optuna search spaces.

Each spec knows how to build, fit (with early stopping where supported),
predict and report its best iteration, so the training loop is model-agnostic.
Class imbalance (~3.5% positives) is handled per library via ``scale_pos_weight``
searched in [1, 10] rather than fixed at the inverse prior: the "textbook"
setting over-weights positives and hurts ranking metrics like PR-AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd


@dataclass
class ModelSpec:
    key: str
    supports_early_stopping: bool

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        raise NotImplementedError

    def build(self, params: dict[str, Any], seed: int, n_estimators: int | None = None):
        raise NotImplementedError

    def fit(self, model, Xtr, ytr, Xva=None, yva=None):
        raise NotImplementedError

    def predict(self, model, X) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    def best_iteration(self, model) -> int | None:
        return None

    def gain_importance(self, model, columns: list[str]) -> pd.Series:
        return pd.Series(model.feature_importances_, index=columns, dtype="float64")


class RandomForestSpec(ModelSpec):
    def __init__(self):
        super().__init__("rf", supports_early_stopping=False)

    def suggest(self, trial):
        return dict(
            max_depth=trial.suggest_int("max_depth", 8, 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100, log=True),
            max_features=trial.suggest_float("max_features", 0.1, 0.6),
        )

    def build(self, params, seed, n_estimators=None):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=n_estimators or 300,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
            **params,
        )

    def fit(self, model, Xtr, ytr, Xva=None, yva=None):
        return model.fit(Xtr.fillna(-999), ytr)

    def predict(self, model, X):
        return model.predict_proba(X.fillna(-999))[:, 1]


class LightGBMSpec(ModelSpec):
    def __init__(self):
        super().__init__("lightgbm", supports_early_stopping=True)

    def suggest(self, trial):
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 300, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.3, 0.9),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True),
        )

    def build(self, params, seed, n_estimators=None):
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=n_estimators or 5000,
            objective="binary",
            bagging_freq=1,
            verbose=-1,
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    def fit(self, model, Xtr, ytr, Xva=None, yva=None):
        import lightgbm as lgb

        if Xva is None:
            return model.fit(Xtr, ytr)
        return model.fit(
            Xtr,
            ytr,
            eval_set=[(Xva, yva)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

    def best_iteration(self, model):
        return int(model.best_iteration_) if getattr(model, "best_iteration_", None) else None

    def gain_importance(self, model, columns):
        return pd.Series(model.booster_.feature_importance("gain"), index=columns, dtype="float64")


class XGBoostSpec(ModelSpec):
    def __init__(self):
        super().__init__("xgboost", supports_early_stopping=True)

    def suggest(self, trial):
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 0.9),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True),
        )

    def build(self, params, seed, n_estimators=None):
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=n_estimators or 5000,
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=100 if n_estimators is None else None,
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    def fit(self, model, Xtr, ytr, Xva=None, yva=None):
        if Xva is None:
            return model.fit(Xtr, ytr)
        return model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    def best_iteration(self, model):
        bi = getattr(model, "best_iteration", None)
        return int(bi) + 1 if bi is not None else None

    def gain_importance(self, model, columns):
        score = model.get_booster().get_score(importance_type="gain")
        return pd.Series({c: score.get(c, 0.0) for c in columns}, dtype="float64")


class CatBoostSpec(ModelSpec):
    def __init__(self):
        super().__init__("catboost", supports_early_stopping=True)

    def suggest(self, trial):
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True),
        )

    def build(self, params, seed, n_estimators=None):
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=n_estimators or 5000,
            eval_metric="PRAUC",
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            thread_count=-1,
            allow_writing_files=False,
            **params,
        )

    def fit(self, model, Xtr, ytr, Xva=None, yva=None):
        if Xva is None:
            return model.fit(Xtr, ytr)
        return model.fit(
            Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=100, use_best_model=True
        )

    def best_iteration(self, model):
        bi = model.get_best_iteration()
        return int(bi) + 1 if bi is not None else None

    def gain_importance(self, model, columns):
        return pd.Series(model.get_feature_importance(), index=columns, dtype="float64")


SPECS: dict[str, ModelSpec] = {
    s.key: s for s in (RandomForestSpec(), LightGBMSpec(), XGBoostSpec(), CatBoostSpec())
}
