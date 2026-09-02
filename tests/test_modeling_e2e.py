"""Select -> train -> evaluate on the synthetic warehouse. Small Optuna budget, all four models."""

from __future__ import annotations

import json

import pytest

from fraudlake.features.selection import run_selection
from fraudlake.modeling.data import candidate_features, load_training_frame
from fraudlake.modeling.evaluate import run_evaluation
from fraudlake.modeling.registry import latest
from fraudlake.modeling.train import run_training

pytestmark = [pytest.mark.postgres, pytest.mark.spark]


@pytest.fixture(scope="module")
def frame(warehouse):
    s, _ = warehouse
    df = load_training_frame(s, refresh=True)
    return s, df


def test_training_frame_is_labelled_time_sorted_and_wide(frame):
    _, df = frame
    assert len(df) == 2_000
    assert df["is_fraud"].isin([0, 1]).all()
    assert df["transaction_dt"].is_monotonic_increasing
    assert len(candidate_features(df)) > 380  # raw + engineered, minus ids and spk_ duplicates


@pytest.fixture(scope="module")
def selection(frame):
    s, df = frame
    return run_selection(s, df=df)


def test_selection_keeps_engineered_signal_and_records_reasons(selection):
    assert 5 < len(selection.selected) < 300
    assert selection.adversarial_auc_before is not None
    assert all(isinstance(r, str) and r for r in selection.dropped.values())
    # the planted velocity signal must survive
    assert any(c.startswith("vel_cnt") for c in selection.selected)


@pytest.fixture(scope="module")
def trained(frame, selection):
    s, df = frame
    return s, run_training(
        s,
        models=["rf", "lightgbm", "xgboost", "catboost"],
        trials=2,
        df=df,
        features=selection.selected,
    )


def test_training_produces_cv_summary_and_artifacts(trained):
    s, summary = trained
    assert set(summary) == {"rf", "lightgbm", "xgboost", "catboost"}
    for key, m in summary.items():
        assert 0 <= m["cv_pr_auc"] <= 1
        assert len(m["folds"]) == s.n_folds
        assert (s.artifacts_dir / "models" / key / "model.joblib").exists()
        assert (s.artifacts_dir / "models" / key / "pipeline.joblib").exists()
    assert (s.artifacts_dir / "models" / "cv_summary.json").exists()


def test_evaluation_reports_holdout_ci_gap_and_registers(trained, selection):
    s, _ = trained
    res = run_evaluation(s, features=selection.selected, n_boot=50)
    best = res["best_model"]
    m = res["models"][best]
    assert m["pr_auc_ci"]["lo"] <= m["pr_auc_ci"]["point"] <= m["pr_auc_ci"]["hi"]
    assert "optimism_gap" in res["validation_comparison"]
    assert 0 < m["cost"]["threshold"] < 1
    reg = latest(s)
    assert reg is not None and (reg / "manifest.json").exists()
    manifest = json.loads((reg / "manifest.json").read_text())
    assert manifest["model"] == best and manifest["n_features"] == len(selection.selected)
    assert (s.artifacts_dir / "evaluation" / "shap_top20.json").exists()
    assert (s.artifacts_dir / "evaluation" / "holdout_predictions.parquet").exists()
