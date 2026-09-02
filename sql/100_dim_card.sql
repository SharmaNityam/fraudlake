/*
name: feat.dim_card
grain: one row per card_uid
purpose: Descriptive card-account dimension. ONLY first_seen_ts is safe to join
         into training features (it is known at the time of every later
         transaction). n_txn_total / last_seen_ts use the full history and are
         provided for EDA and the model card, never for features.
leakage: n_txn_total and last_seen_ts look into the future -- not joined to mart.
*/
DROP TABLE IF EXISTS feat.dim_card;
CREATE TABLE feat.dim_card AS
SELECT
    card_uid,
    MIN(txn_ts)                                AS first_seen_ts,
    MAX(txn_ts)                                AS last_seen_ts,
    COUNT(*)                                   AS n_txn_total,
    COUNT(*) FILTER (WHERE source = 'train')   AS n_txn_train,
    AVG(is_fraud) FILTER (WHERE source = 'train') AS fraud_rate_train,
    MODE() WITHIN GROUP (ORDER BY product_cd)  AS modal_product,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY transaction_amt) AS median_amt
FROM stg.transactions
GROUP BY card_uid;

CREATE UNIQUE INDEX ON feat.dim_card (card_uid);
