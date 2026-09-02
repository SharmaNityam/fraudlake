"""Versioned model artifacts.

``artifacts/model/v<N>/`` holds everything needed to score new data and to
audit the decision: model, fitted feature pipeline, feature list with its
hash, chosen threshold, holdout metrics, and the training provenance
(git sha, MLflow run, data cut). ``latest`` is a pointer file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from fraudlake.config import Settings


def _root(settings: Settings) -> Path:
    r = settings.artifacts_dir / "model"
    r.mkdir(parents=True, exist_ok=True)
    return r


def next_version(settings: Settings) -> int:
    existing = [int(p.name[1:]) for p in _root(settings).glob("v*") if p.name[1:].isdigit()]
    return max(existing, default=0) + 1


def register(
    settings: Settings,
    key: str,
    threshold: float,
    metrics: dict,
    provenance: dict,
    features: list[str],
) -> Path:
    src = settings.artifacts_dir / "models" / key
    v = next_version(settings)
    dst = _root(settings) / f"v{v}"
    dst.mkdir()
    for name in ("model.joblib", "pipeline.joblib", "cv.json"):
        shutil.copy(src / name, dst / name)
    feat_hash = hashlib.sha256("\n".join(features).encode()).hexdigest()[:16]
    (dst / "features.json").write_text(
        json.dumps({"hash": feat_hash, "features": features}, indent=2)
    )
    (dst / "threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2))
    (dst / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    manifest = {
        "version": v,
        "model": key,
        "registered_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "feature_hash": feat_hash,
        "n_features": len(features),
        **provenance,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (_root(settings) / "latest").write_text(f"v{v}")
    return dst


def latest(settings: Settings) -> Path | None:
    p = _root(settings) / "latest"
    return _root(settings) / p.read_text().strip() if p.exists() else None
