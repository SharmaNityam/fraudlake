"""Evaluation metrics for an imbalanced ranking problem.

Why not accuracy: at 3.5% fraud, predicting "never fraud" scores 96.5%.
Why PR-AUC over ROC-AUC as the headline: ROC-AUC is dominated by the 96.5% of
easy negatives; PR-AUC (average precision) measures exactly the trade-off an
operations team lives with — of the transactions we flag, how many are fraud,
and how much fraud do we catch. ROC-AUC is still reported because it is the
competition metric and lets reviewers compare against the leaderboard.

Operating-point metrics (precision@1%, recall@1% FPR) and the cost-optimal
threshold translate the ranking into decisions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def precision_at_k(y: np.ndarray, p: np.ndarray, k_frac: float = 0.01) -> float:
    k = max(1, int(len(p) * k_frac))
    top = np.argsort(-p)[:k]
    return float(y[top].mean())


def recall_at_fpr(y: np.ndarray, p: np.ndarray, max_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(max_fpr, fpr, tpr))


def cost_optimal_threshold(
    y: np.ndarray, p: np.ndarray, cost_fn: float, cost_fp: float
) -> dict[str, float]:
    """Threshold minimising expected cost = FN * cost_fn + FP * cost_fp."""
    order = np.argsort(-p)
    y_sorted = y[order]
    p_sorted = p[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn = y.sum() - tp
    cost = fn * cost_fn + fp * cost_fp
    i = int(np.argmin(cost))
    baseline = float(y.sum() * cost_fn)  # flag nothing
    return {
        "threshold": float(p_sorted[i]),
        "flag_rate": float((i + 1) / len(p)),
        "precision": float(tp[i] / (i + 1)),
        "recall": float(tp[i] / max(1, y.sum())),
        "expected_cost": float(cost[i]),
        "cost_flag_nothing": baseline,
        "cost_saving_pct": float(100 * (1 - cost[i] / baseline)) if baseline else 0.0,
    }


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append(
                {
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "mean_pred": float(p[m].mean()),
                    "frac_pos": float(y[m].mean()),
                    "n": int(m.sum()),
                }
            )
    return rows


def compute_metrics(
    y: np.ndarray, p: np.ndarray, cost_fn: float = 100.0, cost_fp: float = 5.0
) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision_at_1pct": precision_at_k(y, p, 0.01),
        "recall_at_1pct_fpr": recall_at_fpr(y, p, 0.01),
        "brier": float(brier_score_loss(y, p)),
        "prevalence": float(y.mean()),
        "n": int(len(y)),
        "cost": cost_optimal_threshold(y, p, cost_fn, cost_fp),
        "calibration": calibration_table(y, p),
    }
    prec, rec, _ = precision_recall_curve(y, p)
    out["pr_curve"] = {
        "precision": prec[:: max(1, len(prec) // 200)].tolist(),
        "recall": rec[:: max(1, len(rec) // 200)].tolist(),
    }
    return out


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, metric, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            vals[b] = np.nan
            continue
        vals[b] = metric(y[idx], p[idx])
    vals = vals[~np.isnan(vals)]
    return {
        "point": float(metric(y, p)),
        "lo": float(np.quantile(vals, alpha / 2)),
        "hi": float(np.quantile(vals, 1 - alpha / 2)),
        "n_boot": int(len(vals)),
    }
