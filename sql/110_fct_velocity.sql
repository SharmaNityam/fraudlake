/*
name: feat.fct_velocity
grain: one row per transaction
purpose: Backward-looking activity of the same card in the 1h / 24h / 7d before
         this transaction, plus position in the card's history. These are the
         same definitions as the Spark spk_* columns; tests assert parity.
leakage: windows end at "1 second PRECEDING", so the current row and any row in
         the same second are excluded. Nothing after txn_ts is visible.
*/
DROP TABLE IF EXISTS feat.fct_velocity;
CREATE TABLE feat.fct_velocity AS
WITH ordered AS (
    SELECT transaction_id, card_uid, txn_ts, transaction_dt, transaction_amt, product_cd
    FROM stg.transactions
)
SELECT
    transaction_id,
    -- feat: vel_cnt_1h  | # txns on this card in the previous hour
    COUNT(*) OVER w1h                                        AS vel_cnt_1h,
    -- feat: vel_amt_1h  | amount spent on this card in the previous hour
    COALESCE(SUM(transaction_amt) OVER w1h, 0)               AS vel_amt_1h,
    -- feat: vel_cnt_24h | # txns on this card in the previous 24 hours
    COUNT(*) OVER w24h                                       AS vel_cnt_24h,
    -- feat: vel_amt_24h | amount spent on this card in the previous 24 hours
    COALESCE(SUM(transaction_amt) OVER w24h, 0)              AS vel_amt_24h,
    -- feat: vel_cnt_7d  | # txns on this card in the previous 7 days
    COUNT(*) OVER w7d                                        AS vel_cnt_7d,
    -- feat: vel_amt_7d  | amount spent on this card in the previous 7 days
    COALESCE(SUM(transaction_amt) OVER w7d, 0)               AS vel_amt_7d,
    -- feat: card_txn_idx | 0-based index of this txn in the card's history
    ROW_NUMBER() OVER wall - 1                               AS card_txn_idx,
    -- feat: secs_since_prev | seconds since the card's previous txn (null if first)
    transaction_dt - LAG(transaction_dt) OVER wall           AS secs_since_prev,
    -- feat: secs_to_prev_same_amt | seconds since same card spent exactly this amount (repeat-charge signal)
    transaction_dt - LAG(transaction_dt) OVER (PARTITION BY card_uid, transaction_amt ORDER BY transaction_dt) AS secs_since_prev_same_amt,
    -- feat: n_products_7d | distinct product codes on this card in prior 7 days (via lateral)
    lp.n_products_7d
FROM ordered o
CROSS JOIN LATERAL (
    SELECT COUNT(DISTINCT p.product_cd) AS n_products_7d
    FROM stg.transactions p
    WHERE p.card_uid = o.card_uid
      AND p.transaction_dt >= o.transaction_dt - 7 * 86400
      AND p.transaction_dt <  o.transaction_dt
) lp
WINDOW
    wall AS (PARTITION BY card_uid ORDER BY transaction_dt),
    w1h  AS (PARTITION BY card_uid ORDER BY txn_ts
             RANGE BETWEEN INTERVAL '1 hour'  PRECEDING AND INTERVAL '1 second' PRECEDING),
    w24h AS (PARTITION BY card_uid ORDER BY txn_ts
             RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 second' PRECEDING),
    w7d  AS (PARTITION BY card_uid ORDER BY txn_ts
             RANGE BETWEEN INTERVAL '7 days'  PRECEDING AND INTERVAL '1 second' PRECEDING);

CREATE UNIQUE INDEX ON feat.fct_velocity (transaction_id);
