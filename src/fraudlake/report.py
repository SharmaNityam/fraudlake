"""Generate documentation from artifacts, never by hand.

* ``docs/feature_catalog.md`` — scraped from the SQL mart headers and
  ``-- feat:`` annotations, so the catalog cannot drift from the SQL.
* ``docs/model_card.md`` — from evaluation metrics, CV summary, selection
  report and the registry manifest.
* ``README.md`` — the results block between the ``results`` markers is
  rewritten; everything else is left alone.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from rich.console import Console

from fraudlake.config import PROJECT_ROOT, Settings
from fraudlake.modeling.registry import latest

console = Console()

HEADER_RE = re.compile(r"/\*(.*?)\*/", re.S)
FEAT_RE = re.compile(r"--\s*feat:\s*([a-zA-Z0-9_]+)\s*\|\s*(.+)")


def parse_sql_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    m = HEADER_RE.search(text)
    if m:
        key = None
        for line in m.group(1).splitlines():
            kv = re.match(r"\s*(name|grain|purpose|leakage):\s*(.*)", line)
            if kv:
                key = kv.group(1)
                meta[key] = kv.group(2).strip()
            elif key and line.strip():
                meta[key] += " " + line.strip()
    feats = [(n, d.strip()) for n, d in FEAT_RE.findall(text)]
    return {"file": path.name, **meta, "features": feats}


def build_feature_catalog(settings: Settings) -> Path:
    files = [parse_sql_file(p) for p in sorted(settings.sql_dir.glob("*.sql"))]
    out = [
        "# Feature catalog",
        "",
        "Generated from the `sql/` headers and `-- feat:` annotations by `fraudlake report`. "
        "Edit the SQL, not this file.",
        "",
    ]
    for f in files:
        out += [f"## `{f['file']}` — {f.get('name', '')}", ""]
        for k in ("grain", "purpose", "leakage"):
            if f.get(k):
                out.append(f"- **{k}:** {f[k]}")
        if f["features"]:
            out += ["", "| feature | definition |", "|---|---|"]
            out += [f"| `{n}` | {d} |" for n, d in f["features"]]
        out.append("")
    path = settings.docs_dir / "feature_catalog.md"
    path.write_text("\n".join(out))
    return path


def _load(settings: Settings):
    a = settings.artifacts_dir
    ev = json.loads((a / "evaluation" / "metrics.json").read_text())
    cv = json.loads((a / "models" / "cv_summary.json").read_text())
    sel = json.loads((a / "features" / "selected.json").read_text())
    reg = latest(settings)
    manifest = json.loads((reg / "manifest.json").read_text()) if reg else {}
    audit_path = a / "audit" / "bronze_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    return ev, cv, sel, manifest, audit


def results_table(ev: dict) -> str:
    rows = [
        "| model | CV PR-AUC (time folds) | holdout PR-AUC | holdout ROC-AUC | precision@1% | recall@1% FPR |",
        "|---|---|---|---|---|---|",
    ]
    order = sorted(ev["models"], key=lambda k: ev["models"][k]["cv_pr_auc"], reverse=True)
    for k in order:
        m = ev["models"][k]
        name = f"**{k}** (selected)" if k == ev["best_model"] else k
        ci = m.get("pr_auc_ci")
        hold = f"{m['pr_auc']:.4f}" + (f" [{ci['lo']:.4f}, {ci['hi']:.4f}]" if ci else "")
        rows.append(
            f"| {name} | {m['cv_pr_auc']:.4f} ± {m['cv_pr_auc_std']:.4f} | {hold} | "
            f"{m['roc_auc']:.4f} | {m['precision_at_1pct']:.3f} | {m['recall_at_1pct_fpr']:.3f} |"
        )
    return "\n".join(rows)


def validation_block(ev: dict) -> str:
    vc = ev["validation_comparison"]
    return "\n".join(
        [
            "| validation scheme | PR-AUC |",
            "|---|---|",
            f"| shuffled stratified K-fold (the wrong way) | {vc['random_kfold_pr_auc']:.4f} ± {vc['random_kfold_pr_auc_std']:.4f} |",
            f"| expanding time folds with 1-day gap | {vc['time_cv_pr_auc']:.4f} ± {vc['time_cv_pr_auc_std']:.4f} |",
            f"| untouched time holdout (last 20%) | {vc['holdout_pr_auc']:.4f} |",
            "",
            f"**Optimism gap of a random split: {vc['optimism_gap']:+.4f} PR-AUC.**",
        ]
    )


def build_model_card(settings: Settings) -> Path:
    ev, cv, sel, manifest, audit = _load(settings)
    best = ev["best_model"]
    m = ev["models"][best]
    cost = m["cost"]
    ci_pr, ci_roc = m.get("pr_auc_ci", {}), m.get("roc_auc_ci", {})
    top_shap = list(ev.get("shap_top20", {}).items())[:15]
    drop_reasons: dict[str, int] = {}
    for r in sel["dropped"].values():
        drop_reasons[r.split("(")[0].split("=")[0]] = (
            drop_reasons.get(r.split("(")[0].split("=")[0], 0) + 1
        )
    n_rows = audit.get("transactions", {}).get("rows", "n/a")

    lines = [
        "# Model card — fraudlake fraud scorer",
        "",
        f"Generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC by `fraudlake report`. Registry version "
        f"`v{manifest.get('version', '?')}`, git `{manifest.get('git_sha', '?')}`, MLflow run "
        f"`{manifest.get('mlflow_run_id', '?')}`.",
        "",
        "## Intended use",
        "",
        "Rank card-not-present e-commerce transactions by probability of fraud so a review team can "
        "work the top of the queue. Not a stand-alone decision system; the operating threshold below "
        "assumes a human review step and the stated cost ratio.",
        "",
        "## Data",
        "",
        f"- IEEE-CIS Fraud Detection (Vesta Corporation), {n_rows:,} transactions across train + test files"
        if isinstance(n_rows, int)
        else "- IEEE-CIS Fraud Detection (Vesta Corporation)",
        f"- Labelled rows: {manifest.get('train_rows', 0) + manifest.get('holdout_rows', 0):,}; "
        f"holdout = last 20% of time ({manifest.get('holdout_rows', 0):,} rows, "
        f"fraud prevalence {ev['holdout_prevalence']:.2%})",
        f"- Features: {manifest.get('n_features', '?')} selected from {sel['n_candidates']} candidates "
        f"(feature-set hash `{manifest.get('feature_hash', '?')}`)",
        "",
        "## Model",
        "",
        f"- Family: `{best}` chosen by time-fold CV PR-AUC **before** the holdout was opened",
        f"- CV PR-AUC {cv[best]['cv_pr_auc']:.4f} ± {cv[best]['cv_pr_auc_std']:.4f} over {len(cv[best]['folds'])} "
        f"expanding folds with a {settings.fold_gap_seconds // 3600}h gap",
        f"- Fold-to-fold importance stability (Spearman ρ): {cv[best]['importance_stability_spearman']:.2f}",
        f"- Optuna trials: {cv[best]['n_trials']}; final trees: {cv[best]['n_estimators_final']}",
        "",
        "## Holdout performance",
        "",
        results_table(ev),
        "",
        f"Bootstrap 95% CI (n={ci_pr.get('n_boot', 0)}): PR-AUC [{ci_pr.get('lo', 0):.4f}, {ci_pr.get('hi', 0):.4f}], "
        f"ROC-AUC [{ci_roc.get('lo', 0):.4f}, {ci_roc.get('hi', 0):.4f}]. Brier {m['brier']:.4f}.",
        "",
        "### Operating point",
        "",
        f"With a missed fraud costing {settings.cost_false_negative:.0f} units and a manual review "
        f"{settings.cost_false_positive:.0f}, the cost-optimal threshold is **{cost['threshold']:.4f}**: flag "
        f"{cost['flag_rate']:.2%} of traffic, precision {cost['precision']:.1%}, recall {cost['recall']:.1%}, "
        f"expected cost {cost['cost_saving_pct']:.1f}% below the flag-nothing baseline.",
        "",
        "## Validation strategy",
        "",
        validation_block(ev),
        "",
        "See `docs/validation_strategy.md` for why the random split number is not real.",
        "",
        "## Feature selection",
        "",
        "| filter | features dropped |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(drop_reasons.items(), key=lambda kv: -kv[1])],
        "",
        f"Adversarial validation AUC (train window vs holdout window): {sel['adversarial_auc_before']:.3f} before "
        f"dropping drift features, {sel['adversarial_auc_after']:.3f} after.",
        "",
        "## What the model uses (mean |SHAP| on holdout sample)",
        "",
        "| feature | mean abs SHAP |",
        "|---|---|",
        *[f"| `{k}` | {v:.4f} |" for k, v in top_shap],
        "",
        "## Limitations",
        "",
        "- Six months of one merchant platform's traffic; fraud patterns drift and the model needs retraining on a cadence.",
        "- `card_uid` is a reconstruction (card1 + addr1 + D1 anchor), not a true account id.",
        "- Anonymised `V` columns are used as-is; their meaning is unknown, which limits recourse explanations.",
        "- Holdout is later in time than training but still from the same platform; expect lower numbers on a new merchant.",
        "- Bootstrap CIs capture sampling noise only, not drift.",
        "",
    ]
    path = settings.docs_dir / "model_card.md"
    path.write_text("\n".join(lines))
    return path


def update_readme(settings: Settings) -> Path:
    ev, cv, sel, manifest, audit = _load(settings)
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text()
    block = "\n".join(
        [
            "<!-- results:start -->",
            results_table(ev),
            "",
            validation_block(ev),
            "",
            f"_Holdout = last 20% of time, {ev['n_holdout']:,} transactions, {ev['holdout_prevalence']:.2%} fraud. "
            f"Model selected by CV before the holdout was opened. Bootstrap 95% CI in brackets. "
            f"Registry `v{manifest.get('version', '?')}`, git `{manifest.get('git_sha', '?')}`._",
            "<!-- results:end -->",
        ]
    )
    if "<!-- results:start -->" in text:
        text = re.sub(r"<!-- results:start -->.*?<!-- results:end -->", block, text, flags=re.S)
    else:
        text += "\n\n" + block + "\n"
    readme.write_text(text)
    return readme


def update_validation_doc(settings: Settings) -> Path:
    ev, *_ = _load(settings)
    path = settings.docs_dir / "validation_strategy.md"
    text = path.read_text()
    block = "<!-- validation:start -->\n" + validation_block(ev) + "\n<!-- validation:end -->"
    text = re.sub(r"<!-- validation:start -->.*?<!-- validation:end -->", block, text, flags=re.S)
    path.write_text(text)
    return path


def build_report(settings: Settings) -> None:
    settings.docs_dir.mkdir(exist_ok=True)
    console.print(f"[green]updated {update_validation_doc(settings)}[/]")
    console.print(f"[green]wrote {build_feature_catalog(settings)}[/]")
    console.print(f"[green]wrote {build_model_card(settings)}[/]")
    console.print(f"[green]updated {update_readme(settings)}[/]")
