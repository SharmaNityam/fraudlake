/*
name: stg.transactions
grain: one row per transaction (train + test), time-ordered
purpose: Thin staging view over raw.transactions. Keeps the raw layer untouched
         and gives the feature layer a stable contract to build against.
leakage: none (no aggregation)
*/
DROP VIEW IF EXISTS stg.transactions CASCADE;
CREATE VIEW stg.transactions AS
SELECT
    transaction_id,
    source,
    is_fraud,
    transaction_dt,
    txn_ts,
    txn_day,
    txn_hour,
    txn_dow,
    transaction_amt,
    amt_log,
    amt_cents,
    product_cd,
    card_uid,
    card1, card2, card3, card4, card5, card6,
    addr1, addr2, dist1, dist2,
    p_emaildomain, r_emaildomain, p_email_norm, r_email_norm,
    device_type, device_info, id_30 AS os_name, id_31 AS browser_name, id_33 AS screen_res,
    has_identity
FROM raw.transactions;
