"""Holdout evaluation, the random-vs-time validation comparison, SHAP, registry.

Model *selection* uses the CV score (decided before the holdout is opened);
the holdout is reported for every model but only used to choose the operating
threshold and to quote final numbers with bootstrap confidence intervals.

The random-split comparison retrains the winning configuration with shuffled
K-fold on the training window. The gap between that score and the time-based
CV / holdout score is the headline finding of ``docs/validation_strategy.md``:
it is the amount by which a naive validation setup would have overstated
performance.
"""

from __future__ import annotations

import json
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fraudlake.config import Settings
from fraudlake.features.selection import load_selected
from fraudlake.modeling.data import TARGET, TIME_COL, load_training_frame
from fraudlake.modeling.explain import shap_summary
from fraudlake.modeling.metrics import bootstrap_ci, compute_metrics
from fraudlake.modeling.models import SPECS
from fraudlake.modeling.pipeline import FeaturePipeline
from fraudlake.modeling.registry import register
from fraudlake.modeling.splits import time_holdout_split
from fraudlake.modeling.train import git_sha

console = Console()


def random_split_cv(spec, params, n_estimators, df, y, features, n_folds, seed) -> list[float]:
    """Shuffled stratified K-fold on the training window (the wrong way, for contrast)."""
    scores = []
    for tr, va in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(df, y):
        pipe = FeaturePipeline(features, seed=seed)
        Xtr = pipe.fit_transform(df.iloc[tr], y[tr])
        Xva = pipe.transform(df.iloc[va])
        model = spec.build(params, seed, n_estimators=n_estimators)
        spec.fit(model, Xtr, y[tr])
        scores.append(float(average_precision_score(y[va], spec.predict(model, Xva))))
    return scores


def run_evaluation(
    settings: Settings,
    df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    n_boot: int = 1000,
) -> dict:
    df = df if df is not None else load_training_frame(settings)
    df = df.sort_values([TIME_COL, "transaction_id"]).reset_index(drop=True)
    features = features or load_selected(settings)
    train_pos, hold_pos, cut = time_holdout_split(df[TIME_COL], settings.holdout_fraction)
    train, hold = (
        df.iloc[train_pos].reset_index(drop=True),
        df.iloc[hold_pos].reset_index(drop=True),
    )
    y_tr, y_ho = train[TARGET].to_numpy().astype(int), hold[TARGET].to_numpy().astype(int)

    cv_summary = json.loads((settings.artifacts_dir / "models" / "cv_summary.json").read_text())
    best_key = max(cv_summary, key=lambda k: cv_summary[k]["cv_pr_auc"])
    console.print(
        f"best model by CV PR-AUC: [bold]{best_key}[/] (chosen before opening the holdout)"
    )

    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    results: dict = {
        "holdout_cut_dt": cut,
        "n_holdout": int(len(hold)),
        "holdout_prevalence": float(y_ho.mean()),
        "best_model": best_key,
        "models": {},
    }
    preds: dict[str, np.ndarray] = {}

    with mlflow.start_run(run_name="evaluate"):
        mlflow.set_tags({"stage": "evaluate", "git_sha": git_sha(), "best_model": best_key})
        for key in cv_summary:
            mdir = settings.artifacts_dir / "models" / key
            model, pipe = joblib.load(mdir / "model.joblib"), joblib.load(mdir / "pipeline.joblib")
            spec = SPECS[key]
            t0 = time.perf_counter()
            p = spec.predict(model, pipe.transform(hold))
            preds[key] = p
            m = compute_metrics(y_ho, p, settings.cost_false_negative, settings.cost_false_positive)
            m["score_seconds"] = round(time.perf_counter() - t0, 2)
            m["cv_pr_auc"] = cv_summary[key]["cv_pr_auc"]
            m["cv_pr_auc_std"] = cv_summary[key]["cv_pr_auc_std"]
            if key == best_key:
                m["pr_auc_ci"] = bootstrap_ci(
                    y_ho, p, average_precision_score, n_boot, settings.seed
                )
                m["roc_auc_ci"] = bootstrap_ci(y_ho, p, roc_auc_score, n_boot, settings.seed)
            results["models"][key] = m
            mlflow.log_metrics(
                {
                    f"{key}_holdout_pr_auc": m["pr_auc"],
                    f"{key}_holdout_roc_auc": m["roc_auc"],
                    f"{key}_precision_at_1pct": m["precision_at_1pct"],
                }
            )

        # the validation-strategy comparison, on the best model
        spec = SPECS[best_key]
        best = cv_summary[best_key]
        n_est = best["n_estimators_final"] or 300
        rand = random_split_cv(
            spec, best["best_params"], n_est, train, y_tr, features, settings.n_folds, settings.seed
        )
        results["validation_comparison"] = {
            "random_kfold_pr_auc": float(np.mean(rand)),
            "random_kfold_pr_auc_std": float(np.std(rand)),
            "time_cv_pr_auc": best["cv_pr_auc"],
            "time_cv_pr_auc_std": best["cv_pr_auc_std"],
            "holdout_pr_auc": results["models"][best_key]["pr_auc"],
            "optimism_gap": float(np.mean(rand) - results["models"][best_key]["pr_auc"]),
        }
        mlflow.log_metrics(
            {
                "random_kfold_pr_auc": float(np.mean(rand)),
                "optimism_gap": results["validation_comparison"]["optimism_gap"],
            }
        )

        # SHAP on the best model
        mdir = settings.artifacts_dir / "models" / best_key
        model, pipe = joblib.load(mdir / "model.joblib"), joblib.load(mdir / "pipeline.joblib")
        X_ho = pipe.transform(hold)
        if best_key == "rf":
            X_ho = X_ho.fillna(-999)
        eval_dir = settings.artifacts_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        imp = shap_summary(model, X_ho, eval_dir, seed=settings.seed)
        results["shap_top20"] = imp.head(20).round(5).to_dict()

        # persist holdout predictions for the notebooks
        pd.DataFrame(
            {
                "transaction_id": hold["transaction_id"],
                "y": y_ho,
                **{f"p_{k}": v for k, v in preds.items()},
            }
        ).to_parquet(eval_dir / "holdout_predictions.parquet", index=False)
        (eval_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
        mlflow.log_artifact(str(eval_dir / "metrics.json"))

        threshold = results["models"][best_key]["cost"]["threshold"]
        reg = register(
            settings,
            best_key,
            threshold,
            results["models"][best_key],
            {
                "git_sha": git_sha(),
                "mlflow_run_id": best["mlflow_run_id"],
                "holdout_cut_dt": cut,
                "train_rows": int(len(train)),
                "holdout_rows": int(len(hold)),
            },
            features,
        )
        results["registered_path"] = str(reg)

    _print_table(results)
    return results


def _print_table(results: dict) -> None:
    t = Table(title="holdout (last 20% of time)")
    for col in ("model", "CV PR-AUC", "holdout PR-AUC", "ROC-AUC", "P@1%", "R@1%FPR"):
        t.add_column(col, justify="right" if col != "model" else "left")
    for k, m in results["models"].items():
        star = " *" if k == results["best_model"] else ""
        t.add_row(
            k + star,
            f"{m['cv_pr_auc']:.4f}",
            f"{m['pr_auc']:.4f}",
            f"{m['roc_auc']:.4f}",
            f"{m['precision_at_1pct']:.3f}",
            f"{m['recall_at_1pct_fpr']:.3f}",
        )
    console.print(t)
    vc = results["validation_comparison"]
    console.print(
        f"random K-fold PR-AUC {vc['random_kfold_pr_auc']:.4f} vs holdout {vc['holdout_pr_auc']:.4f} "
        f"-> optimism gap {vc['optimism_gap']:+.4f}"
    )
