"""Feature selection on the training window.

Five filters, applied in order, each recorded with a reason per dropped column
so the decision trail ends up in ``artifacts/features/selected.json``:

1. **Null rate** > ``max_null_rate`` — nothing to learn from.
2. **Near-constant** — one value covers >99.9% of non-null rows.
3. **Correlation prune** among the anonymised ``V`` block (339 columns, many
   exact linear duplicates): |r| > ``max_abs_corr`` keeps the one with more
   non-nulls.
4. **Adversarial validation** — a classifier trained to tell "train window" from
   "holdout window". If it can (AUC > threshold), the features it leans on are
   ones whose distribution shifts over time; the model would learn *when* a
   transaction happened, not *whether* it is fraud. Drop the top-N by gain and
   re-check.
5. **Permutation importance** on a time-ordered validation fold: keep features
   whose importance is positive beyond noise (mean - std > 0).

Only the training window (before the holdout cut) is used for steps 1-3 and 5;
step 4 uses both windows by definition but never uses the target.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudlake.config import Settings
from fraudlake.features.encoders import FrequencyEncoder
from fraudlake.modeling.data import (
    HIGH_CARD_NUMERIC,
    TARGET,
    TIME_COL,
    candidate_features,
    load_training_frame,
    split_by_type,
)
from fraudlake.modeling.splits import time_holdout_split, time_series_folds

console = Console()


@dataclass
class SelectionResult:
    selected: list[str]
    dropped: dict[str, str] = field(default_factory=dict)
    adversarial_auc_before: float | None = None
    adversarial_auc_after: float | None = None
    adversarial_top: list[tuple[str, float]] = field(default_factory=list)
    permutation_importance: dict[str, dict[str, float]] = field(default_factory=dict)
    n_candidates: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# --- individual filters (pure functions, unit-tested) --------------------------


def filter_null_rate(df: pd.DataFrame, cols: list[str], max_null_rate: float) -> dict[str, str]:
    rates = df[cols].isna().mean()
    return {
        c: f"null_rate={rates[c]:.3f}>{max_null_rate}" for c in cols if rates[c] > max_null_rate
    }


def filter_near_constant(
    df: pd.DataFrame, cols: list[str], min_variance: float = 1e-3
) -> dict[str, str]:
    out = {}
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            out[c] = "all_null"
            continue
        top_share = s.value_counts(normalize=True).iloc[0]
        if top_share > 1 - min_variance:
            out[c] = f"near_constant(top_share={top_share:.4f})"
    return out


def filter_correlated(df: pd.DataFrame, cols: list[str], max_abs_corr: float) -> dict[str, str]:
    """Greedy prune: walk columns ordered by non-null count (desc); drop a column if it
    is highly correlated with any already-kept column."""
    if len(cols) < 2:
        return {}
    sub = df[cols].astype("float32")
    order = sub.notna().sum().sort_values(ascending=False).index.tolist()
    corr = sub.corr().abs()
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for c in order:
        partner = next((k for k in kept if corr.loc[c, k] > max_abs_corr), None)
        if partner is None:
            kept.append(c)
        else:
            dropped[c] = f"corr({partner})={corr.loc[c, partner]:.3f}>{max_abs_corr}"
    return dropped


def _lgb_params(seed: int) -> dict:
    return dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        verbose=-1,
        seed=seed,
        n_jobs=-1,
    )


def adversarial_validation(
    X_train: pd.DataFrame, X_hold: pd.DataFrame, seed: int
) -> tuple[float, list[tuple[str, float]]]:
    """AUC of train-vs-holdout classifier and features ranked by gain."""
    X = pd.concat([X_train, X_hold], ignore_index=True)
    y = np.r_[np.zeros(len(X_train)), np.ones(len(X_hold))]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, len(X) // 5)
    va, tr = idx[:n_val], idx[n_val:]
    model = lgb.LGBMClassifier(n_estimators=200, **_lgb_params(seed))
    model.fit(
        X.iloc[tr],
        y[tr],
        eval_set=[(X.iloc[va], y[va])],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    auc = roc_auc_score(y[va], model.predict_proba(X.iloc[va])[:, 1])
    gain = pd.Series(model.booster_.feature_importance("gain"), index=X.columns)
    gain = (gain / gain.sum()).sort_values(ascending=False)
    return float(auc), [(c, float(v)) for c, v in gain.head(30).items()]


def permutation_filter(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    n_repeats: int = 3,
    max_rows: int = 40_000,
) -> dict[str, dict[str, float]]:
    if len(X_val) > max_rows:
        sel = np.random.default_rng(seed).choice(len(X_val), size=max_rows, replace=False)
        X_val, y_val = X_val.iloc[sel], y_val[sel]
    r = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    return {
        c: {"mean": float(m), "std": float(s)}
        for c, m, s in zip(X_val.columns, r.importances_mean, r.importances_std, strict=True)
    }


# --- orchestration ---------------------------------------------------------------


def encode_for_selection(df: pd.DataFrame, fit_mask: np.ndarray, cats: list[str]) -> pd.DataFrame:
    """Frequency-encode categoricals (fit on ``fit_mask`` rows) so trees can use them.
    Kept deliberately simple: selection only needs a reasonable numeric view."""
    fe = FrequencyEncoder(list(cats) + [c for c in HIGH_CARD_NUMERIC if c in df.columns]).fit(
        df[fit_mask]
    )
    return fe.transform(df)


def run_selection(settings: Settings, df: pd.DataFrame | None = None) -> SelectionResult:
    df = df if df is not None else load_training_frame(settings)
    df = df.sort_values([TIME_COL, "transaction_id"]).reset_index(drop=True)
    train_pos, hold_pos, _ = time_holdout_split(df[TIME_COL], settings.holdout_fraction)
    train = df.iloc[train_pos]
    y_train = train[TARGET].to_numpy()

    cands = candidate_features(df)
    res = SelectionResult(selected=[], n_candidates=len(cands))
    nums, cats = split_by_type(cands)

    # 1-2: applied to every candidate (categoricals included)
    res.dropped.update(filter_null_rate(train, cands, settings.max_null_rate))
    remaining = [c for c in cands if c not in res.dropped]
    res.dropped.update(filter_near_constant(train, remaining, settings.min_variance))
    remaining = [c for c in remaining if c not in res.dropped]

    # 3: V block only
    v_cols = [c for c in remaining if c.startswith("v") and c[1:].isdigit()]
    res.dropped.update(filter_correlated(train, v_cols, settings.max_abs_corr))
    remaining = [c for c in remaining if c not in res.dropped]
    console.print(f"after null/constant/corr filters: {len(remaining)} of {len(cands)} features")

    # numeric view for steps 4-5
    nums = [c for c in remaining if c not in cats]
    cats_kept = [c for c in remaining if c in cats]
    fit_mask = np.zeros(len(df), dtype=bool)
    fit_mask[train_pos] = True
    enc = encode_for_selection(df, fit_mask, cats_kept)
    Xall = pd.concat([df[nums].astype("float32"), enc], axis=1)

    # 4: adversarial validation
    auc, top = adversarial_validation(Xall.iloc[train_pos], Xall.iloc[hold_pos], settings.seed)
    res.adversarial_auc_before, res.adversarial_top = auc, top
    console.print(f"adversarial AUC (train vs holdout): {auc:.3f}")
    if auc > settings.adversarial_auc_threshold:
        for c, g in top[: settings.adversarial_drop_top_n]:
            base = c.removesuffix("_freq")
            res.dropped[base] = f"adversarial_drift(gain_share={g:.3f}, auc={auc:.3f})"
        keep_cols = [c for c in Xall.columns if c.removesuffix("_freq") not in res.dropped]
        Xall = Xall[keep_cols]
        auc2, _ = adversarial_validation(Xall.iloc[train_pos], Xall.iloc[hold_pos], settings.seed)
        res.adversarial_auc_after = auc2
        console.print(
            f"adversarial AUC after dropping top {settings.adversarial_drop_top_n}: {auc2:.3f}"
        )
    else:
        res.adversarial_auc_after = auc

    # 5: permutation importance on the last time fold of the training window
    Xtr = Xall.iloc[train_pos].reset_index(drop=True)
    fold = list(
        time_series_folds(
            train[TIME_COL].reset_index(drop=True), settings.n_folds, settings.fold_gap_seconds
        )
    )[-1]
    model = lgb.LGBMClassifier(n_estimators=2000, **_lgb_params(settings.seed))
    model.fit(
        Xtr.iloc[fold.train_idx],
        y_train[fold.train_idx],
        eval_set=[(Xtr.iloc[fold.val_idx], y_train[fold.val_idx])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pi = permutation_filter(model, Xtr.iloc[fold.val_idx], y_train[fold.val_idx], settings.seed)
    res.permutation_importance = pi
    for c, s in pi.items():
        base = c.removesuffix("_freq")
        if s["mean"] - s["std"] <= 0 and base not in res.dropped:
            res.dropped[base] = f"permutation_importance({s['mean']:.5f}±{s['std']:.5f})<=0"

    res.selected = [c for c in remaining if c not in res.dropped]
    val_ap = average_precision_score(
        y_train[fold.val_idx], model.predict_proba(Xtr.iloc[fold.val_idx])[:, 1]
    )
    console.print(
        f"[green]selected {len(res.selected)} features (fold PR-AUC during selection: {val_ap:.4f})[/]"
    )

    out = settings.artifacts_dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected.json").write_text(res.to_json())
    return res


def load_selected(settings: Settings) -> list[str]:
    path = settings.artifacts_dir / "features" / "selected.json"
    if not path.exists():
        raise FileNotFoundError("run `fraudlake select` first")
    return json.loads(path.read_text())["selected"]
