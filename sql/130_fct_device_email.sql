/*
name: feat.fct_device_email
grain: one row per transaction
purpose: Population-level, backward-looking rarity of the categorical values on
         this transaction (email domain, device, OS/browser, billing region).
         "How many times has this device been seen before now?" and "how many
         distinct cards used this device in the last 24h?" (device sharing is a
         classic fraud-ring signal).
leakage: counts are ROW_NUMBER over time (strictly prior rows only). The lateral
         card-sharing count bounds txn_ts strictly below the current row.
         No target is used anywhere.
*/
DROP TABLE IF EXISTS feat.fct_device_email;
CREATE TABLE feat.fct_device_email AS
WITH t AS (
    SELECT transaction_id, txn_ts, transaction_dt, card_uid,
           p_email_norm, r_email_norm, device_info, os_name, browser_name, addr1, screen_res
    FROM stg.transactions
)
SELECT
    t.transaction_id,
    -- feat: p_email_prior_cnt | prior txns (all cards) with this purchaser email domain
    CASE WHEN p_email_norm IS NULL THEN NULL
         ELSE ROW_NUMBER() OVER (PARTITION BY p_email_norm ORDER BY transaction_dt, transaction_id) - 1 END AS p_email_prior_cnt,
    -- feat: r_email_prior_cnt | prior txns with this recipient email domain
    CASE WHEN r_email_norm IS NULL THEN NULL
         ELSE ROW_NUMBER() OVER (PARTITION BY r_email_norm ORDER BY transaction_dt, transaction_id) - 1 END AS r_email_prior_cnt,
    -- feat: device_prior_cnt | prior txns with this exact device_info string
    CASE WHEN device_info IS NULL THEN NULL
         ELSE ROW_NUMBER() OVER (PARTITION BY device_info ORDER BY transaction_dt, transaction_id) - 1 END AS device_prior_cnt,
    -- feat: os_browser_prior_cnt | prior txns with this OS + browser combination
    CASE WHEN os_name IS NULL AND browser_name IS NULL THEN NULL
         ELSE ROW_NUMBER() OVER (PARTITION BY os_name, browser_name ORDER BY transaction_dt, transaction_id) - 1 END AS os_browser_prior_cnt,
    -- feat: addr1_prior_cnt | prior txns from this billing region
    CASE WHEN addr1 IS NULL THEN NULL
         ELSE ROW_NUMBER() OVER (PARTITION BY addr1 ORDER BY transaction_dt, transaction_id) - 1 END AS addr1_prior_cnt,
    -- feat: email_mismatch | purchaser and recipient email domains both present and different
    (p_email_norm IS NOT NULL AND r_email_norm IS NOT NULL AND p_email_norm <> r_email_norm)::int AS email_mismatch,
    -- feat: device_cards_24h | distinct OTHER cards that used this device in the prior 24h
    ds.device_cards_24h
FROM t
LEFT JOIN LATERAL (
    SELECT COUNT(DISTINCT d.card_uid) AS device_cards_24h
    FROM t d
    WHERE t.device_info IS NOT NULL
      AND d.device_info = t.device_info
      AND d.card_uid <> t.card_uid
      AND d.txn_ts >= t.txn_ts - INTERVAL '24 hours'
      AND d.txn_ts <  t.txn_ts
) ds ON TRUE;

CREATE UNIQUE INDEX ON feat.fct_device_email (transaction_id);
