# Model card — fraudlake fraud scorer

Generated 2026-09-03 14:12 UTC by `fraudlake report`. Registry version `v1`, git `746b5e3`, MLflow run `3312f60d094e40479dca8005a3ffa7c5`.

## Intended use

Rank card-not-present e-commerce transactions by probability of fraud so a review team can work the top of the queue. Not a stand-alone decision system; the operating threshold below assumes a human review step and the stated cost ratio.

## Data

- IEEE-CIS Fraud Detection (Vesta Corporation), 1,097,231 transactions across train + test files
- Labelled rows: 590,540; holdout = last 20% of time (104,637 rows, fraud prevalence 3.40%)
- Features: 216 selected from 466 candidates (feature-set hash `490756c4b753ee94`)

## Model

- Family: `lightgbm` chosen by time-fold CV PR-AUC **before** the holdout was opened
- CV PR-AUC 0.6066 ± 0.0201 over 5 expanding folds with a 24h gap
- Fold-to-fold importance stability (Spearman ρ): 0.94
- Optuna trials: 9; final trees: 929

## Holdout performance

| model | CV PR-AUC (time folds) | holdout PR-AUC | holdout ROC-AUC | precision@1% | recall@1% FPR |
|---|---|---|---|---|---|
| **lightgbm** (selected) | 0.6066 ± 0.0201 | 0.5932 [0.5768, 0.6078] | 0.9251 | 0.915 | 0.504 |
| xgboost | 0.6003 ± 0.0197 | 0.6020 | 0.9163 | 0.924 | 0.529 |
| catboost | 0.5948 ± 0.0181 | 0.5672 | 0.9151 | 0.896 | 0.492 |
| rf | 0.5295 ± 0.0255 | 0.5257 | 0.9073 | 0.872 | 0.450 |

Bootstrap 95% CI (n=1000): PR-AUC [0.5768, 0.6078], ROC-AUC [0.9204, 0.9300]. Brier 0.0204.

### Operating point

With a missed fraud costing 100 units and a manual review 5, the cost-optimal threshold is **0.0834**: flag 10.17% of traffic, precision 25.7%, recall 76.8%, expected cost 65.7% below the flag-nothing baseline.

## Validation strategy

| validation scheme | PR-AUC |
|---|---|
| shuffled stratified K-fold (the wrong way) | 0.8004 ± 0.0048 |
| expanding time folds with 1-day gap | 0.6066 ± 0.0201 |
| untouched time holdout (last 20%) | 0.5932 |

**Optimism gap of a random split: +0.2071 PR-AUC.**

See `docs/validation_strategy.md` for why the random split number is not real.

## Feature selection

| filter | features dropped |
|---|---|
| permutation_importance | 189 |
| corr | 20 |
| adversarial_drift | 15 |
| near_constant | 14 |
| null_rate | 12 |

Adversarial validation AUC (train window vs holdout window): 0.998 before dropping drift features, 0.865 after.

## What the model uses (mean |SHAP| on holdout sample)

| feature | mean abs SHAP |
|---|---|
| `c13` | 0.3230 |
| `v69` | 0.1782 |
| `transaction_amt` | 0.1778 |
| `card1_freq` | 0.1712 |
| `c14` | 0.1250 |
| `d2` | 0.1173 |
| `d4` | 0.1143 |
| `c1` | 0.1121 |
| `v294` | 0.1087 |
| `card2_freq` | 0.0989 |
| `c11` | 0.0935 |
| `card1` | 0.0904 |
| `c5` | 0.0898 |
| `m4_freq` | 0.0886 |
| `p_emaildomain_freq` | 0.0804 |

## Limitations

- Six months of one merchant platform's traffic; fraud patterns drift and the model needs retraining on a cadence.
- `card_uid` is a reconstruction (card1 + addr1 + D1 anchor), not a true account id.
- Anonymised `V` columns are used as-is; their meaning is unknown, which limits recourse explanations.
- Holdout is later in time than training but still from the same platform; expect lower numbers on a new merchant.
- Bootstrap CIs capture sampling noise only, not drift.
