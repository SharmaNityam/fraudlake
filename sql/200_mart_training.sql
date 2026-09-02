/*
name: mart.training
grain: one row per transaction, ordered by transaction_dt
purpose: Model-ready wide table: every raw column the model may use plus all
         feat.* columns, joined on transaction_id. The modelling code selects
         from this table only, so "what did the model see" is answerable with
         one query.
leakage: inherits the guarantees of each feat.* table; dim_card's future-looking
         columns are deliberately NOT joined.
*/
DROP TABLE IF EXISTS mart.training;
CREATE TABLE mart.training AS
SELECT
    r.*,
    v.vel_cnt_1h, v.vel_amt_1h, v.vel_cnt_24h, v.vel_amt_24h, v.vel_cnt_7d, v.vel_amt_7d,
    v.card_txn_idx, v.secs_since_prev, v.secs_since_prev_same_amt, v.n_products_7d,
    a.prior_n_txn, a.prior_amt_mean, a.amt_ratio_prior_mean, a.amt_z_prior, a.amt_over_prior_max,
    a.card_age_days, a.is_first_txn_on_card, a.is_new_email_for_card, a.is_new_addr_for_card,
    a.is_new_product_for_card, a.is_new_device_for_card,
    d.p_email_prior_share, d.r_email_prior_share, d.device_prior_share, d.os_browser_prior_share,
    d.addr1_prior_share, d.email_mismatch, d.device_cards_24h
FROM raw.transactions r
JOIN feat.fct_velocity      v USING (transaction_id)
JOIN feat.fct_amount_stats  a USING (transaction_id)
JOIN feat.fct_device_email  d USING (transaction_id)
ORDER BY r.transaction_dt, r.transaction_id;

CREATE UNIQUE INDEX ON mart.training (transaction_id);
CREATE INDEX ON mart.training (transaction_dt);
