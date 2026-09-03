# Feature catalog

Generated from the `sql/` headers and `-- feat:` annotations by `fraudlake report`. Edit the SQL, not this file.

## `000_schema.sql` — schemas

- **purpose:** Namespaces for the layered warehouse. raw = as loaded from silver, stg = typed/renamed staging, feat = feature tables (one grain each), mart = model-ready wide tables.

## `010_stg_transactions.sql` — stg.transactions

- **grain:** one row per transaction (train + test), time-ordered
- **purpose:** Thin staging view over raw.transactions. Keeps the raw layer untouched and gives the feature layer a stable contract to build against.
- **leakage:** none (no aggregation)

## `100_dim_card.sql` — feat.dim_card

- **grain:** one row per card_uid
- **purpose:** Descriptive card-account dimension. ONLY first_seen_ts is safe to join into training features (it is known at the time of every later transaction). n_txn_total / last_seen_ts use the full history and are provided for EDA and the model card, never for features.
- **leakage:** n_txn_total and last_seen_ts look into the future -- not joined to mart.

## `110_fct_velocity.sql` — feat.fct_velocity

- **grain:** one row per transaction
- **purpose:** Backward-looking activity of the same card in the 1h / 24h / 7d before this transaction, plus position in the card's history. These are the same definitions as the Spark spk_* columns; tests assert parity.
- **leakage:** windows end at "1 second PRECEDING", so the current row and any row in the same second are excluded. Nothing after txn_ts is visible.

| feature | definition |
|---|---|
| `vel_cnt_1h` | # txns on this card in the previous hour |
| `vel_amt_1h` | amount spent on this card in the previous hour |
| `vel_cnt_24h` | # txns on this card in the previous 24 hours |
| `vel_amt_24h` | amount spent on this card in the previous 24 hours |
| `vel_cnt_7d` | # txns on this card in the previous 7 days |
| `vel_amt_7d` | amount spent on this card in the previous 7 days |
| `card_txn_idx` | 0-based index of this txn in the card's history |
| `secs_since_prev` | seconds since the card's previous txn (null if first) |
| `secs_to_prev_same_amt` | seconds since same card spent exactly this amount (repeat-charge signal) |
| `n_products_7d` | distinct product codes on this card in prior 7 days (via lateral) |

## `120_fct_amount_stats.sql` — feat.fct_amount_stats

- **grain:** one row per transaction
- **purpose:** How unusual is this amount *for this card*, using only the card's prior transactions (expanding window ending at the previous row). Also "first time" flags: first txn on card, first time this email / address / product appears on the card.
- **leakage:** all windows are ROWS ... 1 PRECEDING; the current row never sees itself or anything later. Card age uses first_seen_ts which is <= txn_ts by construction.

| feature | definition |
|---|---|
| `card_age_days` | days since the card account was first seen |
| `is_first_txn_on_card` | 1 if this is the card's first transaction |
| `is_new_email_for_card` | first time this purchaser email domain is used on this card |
| `is_new_addr_for_card` | first time this billing addr2 (country) is used on this card |
| `is_new_product_for_card` | first time this product code is bought on this card |
| `is_new_device_for_card` | first time this device_info appears on this card (null device counts as a device) |
| `amt_ratio_prior_mean` | amount / mean of the card's prior amounts |
| `amt_z_prior` | z-score of amount vs the card's prior amounts (null if <2 prior) |
| `amt_over_prior_max` | 1 if amount exceeds every prior amount on the card |

## `130_fct_device_email.sql` — feat.fct_device_email

- **grain:** one row per transaction
- **purpose:** Population-level, backward-looking rarity of the categorical values on this transaction (email domain, device, OS/browser, billing region), expressed as the SHARE of all prior traffic carrying the same value. Raw prior *counts* were tried first and rejected by adversarial validation: a cumulative count grows with the calendar and lets the model learn "when" instead of "whether". A share is stationary. Also "how many distinct other cards used this device in the last 24h" (device sharing is a classic fraud-ring signal).
- **leakage:** numerators and denominators are ROW_NUMBER over time (strictly prior rows only). The lateral card-sharing count bounds txn_ts strictly below the current row. No target is used anywhere.

| feature | definition |
|---|---|
| `p_email_prior_share` | share of all prior txns with this purchaser email domain |
| `r_email_prior_share` | share of all prior txns with this recipient email domain |
| `device_prior_share` | share of all prior txns with this exact device_info string |
| `os_browser_prior_share` | share of all prior txns with this OS + browser combination |
| `addr1_prior_share` | share of all prior txns from this billing region |
| `email_mismatch` | purchaser and recipient email domains both present and different |
| `device_cards_24h` | distinct OTHER cards that used this device in the prior 24h |

## `200_mart_training.sql` — mart.training

- **grain:** one row per transaction, ordered by transaction_dt
- **purpose:** Model-ready wide table: every raw column the model may use plus all feat.* columns, joined on transaction_id. The modelling code selects from this table only, so "what did the model see" is answerable with one query.
- **leakage:** inherits the guarantees of each feat.* table; dim_card's future-looking columns are deliberately NOT joined.
