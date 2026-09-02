"""Training: time-aware CV, Optuna tuning, MLflow tracking, final refit.

For each model key:
  1. Optuna maximises mean fold PR-AUC over the expanding-window folds inside
     the training window (holdout is never touched here).
  2. Every trial is an MLflow nested run (params + per-fold metrics).
  3. The best configuration is refit on the whole training window with the
     number of trees set from the folds' early-stopping iterations.
  4. Model, fitted feature pipeline, fold gain-importances and CV summary are
     saved under ``artifacts/models/<key>/``.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from rich.console import Console
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudlake.config import Settings
from fraudlake.features.selection import load_selected
from fraudlake.modeling.data import TARGET, TIME_COL, load_training_frame
from fraudlake.modeling.models import SPECS, ModelSpec
from fraudlake.modeling.pipeline import FeaturePipeline
from fraudlake.modeling.splits import Fold, time_holdout_split, time_series_folds

console = Console()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def model_dir(settings: Settings, key: str) -> Path:
    d = settings.artifacts_dir / "models" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def cv_score(
    spec: ModelSpec,
    params: dict,
    df: pd.DataFrame,
    y: np.ndarray,
    folds: list[Fold],
    features: list[str],
    seed: int,
) -> tuple[list[dict], list[pd.Series]]:
    """Fit one config on every fold; return per-fold metrics and gain importances."""
    fold_metrics, importances = [], []
    for f in folds:
        pipe = FeaturePipeline(features, seed=seed)
        Xtr = pipe.fit_transform(df.iloc[f.train_idx], y[f.train_idx])
        Xva = pipe.transform(df.iloc[f.val_idx])
        model = spec.build(params, seed)
        t0 = time.perf_counter()
        spec.fit(model, Xtr, y[f.train_idx], Xva, y[f.val_idx])
        p = spec.predict(model, Xva)
        fold_metrics.append(
            {
                "fold": f.index,
                "pr_auc": float(average_precision_score(y[f.val_idx], p)),
                "roc_auc": float(roc_auc_score(y[f.val_idx], p)),
                "best_iteration": spec.best_iteration(model),
                "n_train": int(len(f.train_idx)),
                "n_val": int(len(f.val_idx)),
                "fit_seconds": round(time.perf_counter() - t0, 1),
            }
        )
        importances.append(spec.gain_importance(model, pipe.output_columns))
    return fold_metrics, importances


def importance_stability(importances: list[pd.Series]) -> float:
    """Mean pairwise Spearman rank correlation of gain importance across folds."""
    if len(importances) < 2:
        return float("nan")
    ranks = pd.concat(importances, axis=1).rank()
    corr = ranks.corr(method="spearman").to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    return float(np.nanmean(corr[iu]))


def tune_model(
    spec: ModelSpec,
    df: pd.DataFrame,
    y: np.ndarray,
    folds: list[Fold],
    features: list[str],
    settings: Settings,
    n_trials: int,
) -> tuple[optuna.Study, dict[int, tuple[list[dict], list[pd.Series]]]]:
    cache: dict[int, tuple[list[dict], list[pd.Series]]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest(trial)
        with mlflow.start_run(run_name=f"{spec.key}-trial-{trial.number}", nested=True):
            fm, imps = cv_score(spec, params, df, y, folds, features, settings.seed)
            mean_ap = float(np.mean([m["pr_auc"] for m in fm]))
            mlflow.log_params(params)
            for m in fm:
                mlflow.log_metric("fold_pr_auc", m["pr_auc"], step=m["fold"])
                mlflow.log_metric("fold_roc_auc", m["roc_auc"], step=m["fold"])
            mlflow.log_metric("cv_pr_auc", mean_ap)
            mlflow.log_metric("cv_pr_auc_std", float(np.std([m["pr_auc"] for m in fm])))
        cache[trial.number] = (fm, imps)
        return mean_ap

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.seed),
        study_name=f"fraudlake-{spec.key}",
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=settings.optuna_timeout_seconds,
        show_progress_bar=False,
    )
    return study, cache


def run_training(
    settings: Settings,
    models: list[str],
    trials: int | None = None,
    df: pd.DataFrame | None = None,
    features: list[str] | None = None,
) -> dict[str, dict]:
    df = df if df is not None else load_training_frame(settings)
    df = df.sort_values([TIME_COL, "transaction_id"]).reset_index(drop=True)
    features = features or load_selected(settings)
    n_trials = trials or settings.optuna_trials

    train_pos, hold_pos, cut = time_holdout_split(df[TIME_COL], settings.holdout_fraction)
    train = df.iloc[train_pos].reset_index(drop=True)
    y = train[TARGET].to_numpy().astype(int)
    folds = list(time_series_folds(train[TIME_COL], settings.n_folds, settings.fold_gap_seconds))
    console.print(
        f"train window: {len(train):,} rows (fraud {y.mean():.3%}), holdout: {len(hold_pos):,} rows, "
        f"{len(folds)} folds, {len(features)} features"
    )

    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    summary: dict[str, dict] = {}

    for key in models:
        spec = SPECS[key]
        console.rule(f"[bold]{key}")
        with mlflow.start_run(run_name=key) as parent:
            mlflow.set_tags({"model": key, "git_sha": git_sha(), "stage": "train"})
            mlflow.log_params(
                {
                    "n_features": len(features),
                    "n_folds": len(folds),
                    "n_trials": n_trials,
                    "holdout_cut_dt": cut,
                    "fold_gap_seconds": settings.fold_gap_seconds,
                }
            )
            t0 = time.perf_counter()
            study, cache = tune_model(spec, train, y, folds, features, settings, n_trials)
            best = study.best_trial
            fm, imps = cache[best.number]
            stability = importance_stability(imps)
            cv_ap = float(np.mean([m["pr_auc"] for m in fm]))
            cv_ap_std = float(np.std([m["pr_auc"] for m in fm]))
            mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
            mlflow.log_metrics(
                {
                    "cv_pr_auc": cv_ap,
                    "cv_pr_auc_std": cv_ap_std,
                    "cv_roc_auc": float(np.mean([m["roc_auc"] for m in fm])),
                    "importance_stability_spearman": stability,
                    "tuning_seconds": time.perf_counter() - t0,
                }
            )

            # final refit on the whole training window
            best_iters = [m["best_iteration"] for m in fm if m["best_iteration"]]
            n_estimators = int(np.mean(best_iters) * 1.1) if best_iters else None
            pipe = FeaturePipeline(features, seed=settings.seed)
            Xtr = pipe.fit_transform(train, y)
            model = spec.build(best.params, settings.seed, n_estimators=n_estimators)
            spec.fit(model, Xtr, y)

            out = model_dir(settings, key)
            joblib.dump(model, out / "model.joblib")
            joblib.dump(pipe, out / "pipeline.joblib")
            imp_df = pd.concat(imps, axis=1)
            imp_df.columns = [f"fold{i}" for i in range(len(imps))]
            imp_df.to_csv(out / "fold_importances.csv")
            summary[key] = {
                "cv_pr_auc": cv_ap,
                "cv_pr_auc_std": cv_ap_std,
                "cv_roc_auc": float(np.mean([m["roc_auc"] for m in fm])),
                "folds": fm,
                "best_params": best.params,
                "n_estimators_final": n_estimators,
                "importance_stability_spearman": stability,
                "mlflow_run_id": parent.info.run_id,
                "n_trials": len(study.trials),
                "tuning_seconds": round(time.perf_counter() - t0, 1),
            }
            (out / "cv.json").write_text(json.dumps(summary[key], indent=2, default=str))
            mlflow.log_artifact(str(out / "cv.json"))
            console.print(
                f"[green]{key}: CV PR-AUC {cv_ap:.4f} ± {cv_ap_std:.4f} "
                f"(importance stability ρ={stability:.2f})[/]"
            )

    (settings.artifacts_dir / "models" / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary
