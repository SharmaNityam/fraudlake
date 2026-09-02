/*
name: feat.fct_amount_stats
grain: one row per transaction
purpose: How unusual is this amount *for this card*, using only the card's
         prior transactions (expanding window ending at the previous row).
         Also "first time" flags: first txn on card, first time this email /
         address / product appears on the card.
leakage: all windows are ROWS ... 1 PRECEDING; the current row never sees itself
         or anything later. Card age uses first_seen_ts which is <= txn_ts by
         construction.
*/
DROP TABLE IF EXISTS feat.fct_amount_stats;
CREATE TABLE feat.fct_amount_stats AS
WITH base AS (
    SELECT t.*, c.first_seen_ts
    FROM stg.transactions t
    JOIN feat.dim_card c USING (card_uid)
),
stats AS (
    SELECT
        transaction_id,
        transaction_amt,
        AVG(transaction_amt)    OVER prior AS prior_amt_mean,
        STDDEV(transaction_amt) OVER prior AS prior_amt_std,
        MAX(transaction_amt)    OVER prior AS prior_amt_max,
        COUNT(*)                OVER prior AS prior_n,
        -- feat: card_age_days | days since the card account was first seen
        EXTRACT(EPOCH FROM (txn_ts - first_seen_ts)) / 86400.0 AS card_age_days,
        -- feat: is_first_txn_on_card | 1 if this is the card's first transaction
        (ROW_NUMBER() OVER (PARTITION BY card_uid ORDER BY transaction_dt) = 1)::int AS is_first_txn_on_card,
        -- feat: is_new_email_for_card | first time this purchaser email domain is used on this card
        (ROW_NUMBER() OVER (PARTITION BY card_uid, p_email_norm ORDER BY transaction_dt) = 1)::int AS is_new_email_for_card,
        -- feat: is_new_addr_for_card | first time this billing addr2 (country) is used on this card
        (ROW_NUMBER() OVER (PARTITION BY card_uid, addr2 ORDER BY transaction_dt) = 1)::int AS is_new_addr_for_card,
        -- feat: is_new_product_for_card | first time this product code is bought on this card
        (ROW_NUMBER() OVER (PARTITION BY card_uid, product_cd ORDER BY transaction_dt) = 1)::int AS is_new_product_for_card,
        -- feat: is_new_device_for_card | first time this device_info appears on this card (null device counts as a device)
        (ROW_NUMBER() OVER (PARTITION BY card_uid, device_info ORDER BY transaction_dt) = 1)::int AS is_new_device_for_card
    FROM base
    WINDOW prior AS (PARTITION BY card_uid ORDER BY transaction_dt
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    transaction_id,
    prior_n                                                        AS prior_n_txn,
    prior_amt_mean,
    -- feat: amt_ratio_prior_mean | amount / mean of the card's prior amounts
    transaction_amt / NULLIF(prior_amt_mean, 0)                    AS amt_ratio_prior_mean,
    -- feat: amt_z_prior | z-score of amount vs the card's prior amounts (null if <2 prior)
    CASE WHEN prior_n >= 2 AND prior_amt_std > 0
         THEN (transaction_amt - prior_amt_mean) / prior_amt_std END AS amt_z_prior,
    -- feat: amt_over_prior_max | 1 if amount exceeds every prior amount on the card
    (transaction_amt > prior_amt_max)::int                          AS amt_over_prior_max,
    card_age_days,
    is_first_txn_on_card,
    is_new_email_for_card,
    is_new_addr_for_card,
    is_new_product_for_card,
    is_new_device_for_card
FROM stats;

CREATE UNIQUE INDEX ON feat.fct_amount_stats (transaction_id);
