"""SHAP explanations for the final model.

TreeExplainer gives exact Shapley values for gradient-boosted trees in
seconds. We compute them on a holdout sample, save global importance
(mean |SHAP|), a beeswarm plot, and the per-row values for the results
notebook's error analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def shap_summary(
    model, X: pd.DataFrame, out_dir: Path, max_rows: int = 5_000, seed: int = 0
) -> pd.Series:
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older API returns [neg, pos]
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    global_imp = pd.Series(np.abs(values).mean(axis=0), index=X.columns).sort_values(
        ascending=False
    )
    global_imp.to_csv(out_dir / "shap_global_importance.csv", header=["mean_abs_shap"])
    (out_dir / "shap_top20.json").write_text(
        json.dumps(global_imp.head(20).round(5).to_dict(), indent=2)
    )
    pd.DataFrame(values, columns=X.columns, index=X.index).astype("float32").to_parquet(
        out_dir / "shap_values.parquet"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(values, X, max_display=25, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_beeswarm.png", dpi=130)
        plt.close()
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        (out_dir / "shap_plot_error.txt").write_text(str(exc))
    return global_imp
