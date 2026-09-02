"""Synthetic IEEE-CIS look-alike generator.

Used for (a) the test-suite fixtures and (b) ``fraudlake synth`` demo mode so
the whole pipeline can be exercised without Kaggle credentials. The generator
reproduces the *shape* of the real data — column names, dtypes, null patterns,
time ordering, class imbalance, and card-level repeat behaviour — but makes no
claim to reproduce its statistics. Fraud is planted with a simple mechanism
(velocity bursts on a subset of cards) so that a correct pipeline can find it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudlake.ingest.schema import ID_COLUMN_ORDER, TXN_COLUMN_ORDER

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", None]
CARD4 = ["visa", "mastercard", "american express", "discover", None]
CARD6 = ["debit", "credit", None]
PRODUCT = ["W", "C", "R", "H", "S"]
DEVICE_TYPE = ["desktop", "mobile", None]
DEVICE_INFO = ["Windows", "iOS Device", "MacOS", "SAMSUNG", "rv:11.0", None]
OS = ["Windows 10", "iOS 12.1.0", "Mac OS X 10_13_6", "Android 8.0.0", None]
BROWSER = ["chrome 70.0", "safari 12.0", "ie 11.0 for desktop", "edge 17.0", None]


def make_transactions(
    n_rows: int = 2_000,
    n_cards: int = 300,
    fraud_rate: float = 0.035,
    seed: int = 7,
    span_days: int = 30,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # time: sorted seconds offsets, mimicking TransactionDT (starts at 86400 in real data)
    dt = np.sort(rng.integers(86_400, 86_400 + span_days * 86_400, size=n_rows)).astype(np.int64)

    # cards with a heavy tail of activity, like real data
    card_weights = rng.pareto(1.5, size=n_cards) + 1
    card_idx = rng.choice(n_cards, size=n_rows, p=card_weights / card_weights.sum())
    card1 = 1_000 + card_idx * 7
    card2 = np.where(rng.random(n_rows) < 0.02, np.nan, 100 + (card_idx % 500)).astype(float)
    card3 = np.where(rng.random(n_rows) < 0.01, np.nan, 150.0)
    card5 = np.where(rng.random(n_rows) < 0.01, np.nan, 226.0)
    addr1 = np.where(rng.random(n_rows) < 0.1, np.nan, 100 + (card_idx % 400)).astype(float)
    addr2 = np.where(np.isnan(addr1), np.nan, 87.0)

    amt = np.round(np.exp(rng.normal(3.8, 1.0, size=n_rows)), 3)

    # fraud: some cards go "hot" — their transactions cluster into a burst of a few
    # hours (velocity signal) — plus a random background rate
    hot_cards = rng.choice(n_cards, size=max(1, n_cards // 20), replace=False)
    is_hot = np.isin(card_idx, hot_cards)
    burst = is_hot & (rng.random(n_rows) < 0.6)
    burst_center = {c: rng.integers(dt.min(), dt.max()) for c in hot_cards}
    dt = dt.copy()
    dt[burst] = np.array(
        [burst_center[c] + rng.integers(0, 3 * 3_600) for c in card_idx[burst]], dtype=np.int64
    )
    is_fraud = (burst | (rng.random(n_rows) < fraud_rate / 2)).astype(int)
    amt = np.where(is_fraud == 1, np.round(amt * rng.uniform(1.5, 4.0, size=n_rows), 3), amt)

    # re-establish time order (TransactionID is assigned in time order, as on Kaggle)
    order = np.argsort(dt, kind="stable")
    dt, card_idx, card1, card2, card3, card5, addr1, addr2, amt, is_fraud = (
        a[order] for a in (dt, card_idx, card1, card2, card3, card5, addr1, addr2, amt, is_fraud)
    )
    txn_day = dt // 86_400
    # D1 = days since the card was first seen: constant anchor per card (this is what
    # makes card1||addr1||(day - D1) a usable account fingerprint on the real data)
    first_day = pd.Series(txn_day).groupby(card_idx).transform("min").to_numpy()
    d1 = (txn_day - first_day).astype(float)
    d1[rng.random(n_rows) < 0.05] = np.nan

    df = pd.DataFrame(
        {
            "TransactionID": np.arange(2_987_000, 2_987_000 + n_rows, dtype=np.int64),
            "isFraud": is_fraud,
            "TransactionDT": dt,
            "TransactionAmt": amt,
            "ProductCD": rng.choice(PRODUCT, size=n_rows, p=[0.74, 0.12, 0.06, 0.06, 0.02]),
            "card1": card1.astype(float),
            "card2": card2,
            "card3": card3,
            "card4": rng.choice(CARD4, size=n_rows, p=[0.65, 0.32, 0.015, 0.01, 0.005]),
            "card5": card5,
            "card6": rng.choice(CARD6, size=n_rows, p=[0.74, 0.25, 0.01]),
            "addr1": addr1,
            "addr2": addr2,
            "dist1": np.where(rng.random(n_rows) < 0.6, np.nan, rng.exponential(50, n_rows).round()),
            "dist2": np.where(rng.random(n_rows) < 0.93, np.nan, rng.exponential(80, n_rows).round()),
            "P_emaildomain": rng.choice(EMAIL_DOMAINS, size=n_rows),
            "R_emaildomain": rng.choice(EMAIL_DOMAINS, size=n_rows, p=[0.1, 0.05, 0.03, 0.02, 0.02, 0.78]),
        }
    )
    for i in range(1, 15):
        df[f"C{i}"] = rng.poisson(1.5 + is_fraud * 3, size=n_rows).astype(float)
    df["D1"] = d1
    for i in range(2, 16):
        df[f"D{i}"] = np.where(rng.random(n_rows) < 0.5, np.nan, rng.integers(0, 600, n_rows)).astype(float)
    for i in range(1, 10):
        df[f"M{i}"] = rng.choice(["T", "F", None], size=n_rows, p=[0.45, 0.35, 0.2])
    v_null = rng.random(339) * 0.9  # per-column null rate
    v_block = rng.standard_normal((n_rows, 339)) + is_fraud[:, None] * rng.random(339) * 0.5
    v_block = np.where(rng.random((n_rows, 339)) < v_null, np.nan, np.round(v_block, 4))
    v_df = pd.DataFrame(v_block, columns=[f"V{i}" for i in range(1, 340)])
    df = pd.concat([df, v_df], axis=1)
    return df[list(TXN_COLUMN_ORDER)]


def make_identity(txn: pd.DataFrame, coverage: float = 0.25, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(txn)
    mask = rng.random(n) < coverage
    ids = txn.loc[mask, "TransactionID"].to_numpy()
    m = len(ids)
    df = pd.DataFrame({"TransactionID": ids})
    for i in range(1, 39):
        name = f"id_{i:02d}"
        if name in ("id_12",):
            df[name] = rng.choice(["Found", "NotFound"], size=m)
        elif name in ("id_15", "id_16", "id_23", "id_27", "id_28", "id_29"):
            df[name] = rng.choice(["Found", "New", "NotFound", None], size=m)
        elif name == "id_30":
            df[name] = rng.choice(OS, size=m)
        elif name == "id_31":
            df[name] = rng.choice(BROWSER, size=m)
        elif name == "id_33":
            df[name] = rng.choice(["1920x1080", "1366x768", "2208x1242", None], size=m)
        elif name == "id_34":
            df[name] = rng.choice(["match_status:2", "match_status:1", None], size=m)
        elif name in ("id_35", "id_36", "id_37", "id_38"):
            df[name] = rng.choice(["T", "F", None], size=m)
        else:
            df[name] = np.where(rng.random(m) < 0.3, np.nan, rng.standard_normal(m).round(2))
    df["DeviceType"] = rng.choice(DEVICE_TYPE, size=m, p=[0.55, 0.4, 0.05])
    df["DeviceInfo"] = rng.choice(DEVICE_INFO, size=m)
    return df[list(ID_COLUMN_ORDER)]


def write_synthetic_raw(out_dir: Path, n_rows: int = 2_000, seed: int = 7) -> dict[str, Path]:
    """Write train_transaction.csv / train_identity.csv (and empty-label test files) to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    txn = make_transactions(n_rows=n_rows, seed=seed)
    ident = make_identity(txn, seed=seed + 4)
    paths = {
        "train_transaction.csv": out_dir / "train_transaction.csv",
        "train_identity.csv": out_dir / "train_identity.csv",
    }
    txn.to_csv(paths["train_transaction.csv"], index=False)
    ident.to_csv(paths["train_identity.csv"], index=False)

    # test files: later time window, no label, hyphenated identity headers (as on Kaggle)
    test_txn = make_transactions(n_rows=max(200, n_rows // 5), seed=seed + 1)
    test_txn["TransactionDT"] += int(txn["TransactionDT"].max())
    test_txn["TransactionID"] += n_rows
    test_id = make_identity(test_txn, seed=seed + 5)
    test_id.columns = [c.replace("id_", "id-") for c in test_id.columns]
    paths["test_transaction.csv"] = out_dir / "test_transaction.csv"
    paths["test_identity.csv"] = out_dir / "test_identity.csv"
    test_txn.drop(columns=["isFraud"]).to_csv(paths["test_transaction.csv"], index=False)
    test_id.to_csv(paths["test_identity.csv"], index=False)
    return paths
