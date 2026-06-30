"""
Phase 9 — XGBoost Feature Engineering

Transforms Phase 8 outputs into a single account-level feature matrix.

Input files (all from Phase 8 analytics/ directory):
  analytics_transactions.csv   — every transaction with all Phase 8 flags
  risk_scores.csv              — per-account risk tier + boolean flag columns

Output:
  pd.DataFrame with one row per account_id, columns = ALL_FEATURES

Design principle:
  ALL features are RELATIVE / behavioural — no hardcoded rupee thresholds.
  This makes the model generalise to any statement the judges upload,
  regardless of the dataset's scale.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from xgb_config import ALL_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    txn_df:  pd.DataFrame,
    risk_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full feature matrix from Phase 8 outputs.

    Parameters
    ----------
    txn_df  : analytics_transactions.csv loaded as DataFrame
    risk_df : risk_scores.csv loaded as DataFrame

    Returns
    -------
    feature_df : DataFrame indexed by account_id with columns = ALL_FEATURES
    """
    txn_df  = _coerce_txn(txn_df)
    risk_df = risk_df.set_index("account_id")

    account_ids = txn_df["account_id"].unique()
    rows = []

    for acc_id in account_ids:
        acc_txns = txn_df[txn_df["account_id"] == acc_id]
        risk_row = risk_df.loc[acc_id] if acc_id in risk_df.index else None
        row = _build_account_features(acc_id, acc_txns, risk_row)
        rows.append(row)

    feature_df = pd.DataFrame(rows).set_index("account_id")

    # Ensure all expected columns exist (fill 0 if any group was empty)
    for col in ALL_FEATURES:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    # Clip extreme outliers (winsorise at 1st / 99th percentile)
    feature_df = _winsorise(feature_df)

    return feature_df[ALL_FEATURES]


# ─────────────────────────────────────────────────────────────────────────────
# Per-account feature builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_account_features(
    acc_id: str,
    txns:   pd.DataFrame,
    risk:   pd.Series | None,
) -> dict:
    feats: dict = {"account_id": acc_id}

    # ── VOLUME features ──────────────────────────────────────────────────
    feats["txn_count"]    = len(txns)
    feats["debit_count"]  = int((txns["debit"]  > 0).sum())
    feats["credit_count"] = int((txns["credit"] > 0).sum())
    feats["total_debit"]  = float(txns["debit"].sum())
    feats["total_credit"] = float(txns["credit"].sum())
    feats["net_flow"]     = feats["total_credit"] - feats["total_debit"]

    # ── AMOUNT features ──────────────────────────────────────────────────
    debits  = txns[txns["debit"]  > 0]["debit"]
    credits = txns[txns["credit"] > 0]["credit"]

    feats["mean_debit"]  = float(debits.mean())  if len(debits)  > 0 else 0.0
    feats["std_debit"]   = float(debits.std())   if len(debits)  > 1 else 0.0
    feats["max_debit"]   = float(debits.max())   if len(debits)  > 0 else 0.0
    feats["mean_credit"] = float(credits.mean()) if len(credits) > 0 else 0.0
    feats["std_credit"]  = float(credits.std())  if len(credits) > 1 else 0.0
    feats["max_credit"]  = float(credits.max())  if len(credits) > 0 else 0.0

    # Ratio of outflow to inflow — near 1.0 is a classic mule signal
    total_credit = feats["total_credit"]
    total_debit  = feats["total_debit"]
    feats["debit_credit_ratio"] = (
        total_debit / total_credit if total_credit > 0 else 0.0
    )

    # Largest single transaction as ratio of total flow
    max_single = max(feats["max_debit"], feats["max_credit"])
    total_flow = total_debit + total_credit
    feats["max_single_txn_ratio"] = (
        max_single / total_flow if total_flow > 0 else 0.0
    )

    # ── TEMPORAL features ────────────────────────────────────────────────
    dates = pd.to_datetime(txns["date"], errors="coerce").dropna()
    if len(dates) > 1:
        span_days = max((dates.max() - dates.min()).days, 1)
        feats["active_days"]  = int(span_days)
        feats["txn_per_day"]  = round(len(txns) / span_days, 4)
    else:
        feats["active_days"] = 1
        feats["txn_per_day"] = float(len(txns))

    # Odd-hour ratio (00:00–05:00)
    hours = txns["time"].apply(_extract_hour)
    odd_mask = (hours >= 0) & (hours < 5)
    feats["odd_hour_ratio"] = float(odd_mask.sum() / len(txns)) if len(txns) > 0 else 0.0

    # Weekend ratio
    weekend_mask = dates.dt.dayofweek >= 5   # Saturday=5, Sunday=6
    feats["weekend_ratio"] = float(weekend_mask.sum() / len(dates)) if len(dates) > 0 else 0.0

    # Velocity burst count (from is_velocity_flag)
    vel_col = "is_velocity_flag"
    feats["burst_count"] = int(txns[vel_col].sum()) if vel_col in txns.columns else 0

    # ── STRUCTURAL features ──────────────────────────────────────────────
    cp_col = "counterparty_name"
    if cp_col in txns.columns:
        valid_cp = txns[cp_col].dropna().astype(str).str.strip()
        valid_cp = valid_cp[valid_cp.str.lower() != "nan"]
        n_cp = len(valid_cp.unique())
    else:
        n_cp = 0
    feats["unique_counterparties"] = n_cp

    # Counterparty diversity = unique counterparties / total transactions
    # High diversity = touches many accounts = potential hub / mule
    feats["counterparty_diversity"] = n_cp / len(txns) if len(txns) > 0 else 0.0

    # Fan-in score: credits from many distinct senders in a window
    # Proxy: (credit_count / unique_counterparties) if high credit side
    credit_rows = txns[txns["credit"] > 0]
    if n_cp > 0 and len(credit_rows) > 0:
        cp_credit_diversity = (
            credit_rows[cp_col].dropna().astype(str)
            .str.strip().nunique() if cp_col in credit_rows.columns else 0
        )
        feats["fan_in_score"] = round(cp_credit_diversity / n_cp, 4)
    else:
        feats["fan_in_score"] = 0.0

    # Fan-out score: debits to many distinct receivers
    debit_rows = txns[txns["debit"] > 0]
    if n_cp > 0 and len(debit_rows) > 0:
        cp_debit_diversity = (
            debit_rows[cp_col].dropna().astype(str)
            .str.strip().nunique() if cp_col in debit_rows.columns else 0
        )
        feats["fan_out_score"] = round(cp_debit_diversity / n_cp, 4)
    else:
        feats["fan_out_score"] = 0.0

    # Binary structural flags from Phase 8 pattern detectors
    feats["round_trip_flag"] = int(
        txns.get("is_round_trip", pd.Series([False] * len(txns))).any()
    )
    feats["layering_flag"] = int(
        txns.get("is_layering", pd.Series([False] * len(txns))).any()
    )

    # ── PHASE 8 FLAG features ────────────────────────────────────────────
    phase8_flags = [
        "flag_round_trip", "flag_layering", "flag_fan_in", "flag_fan_out",
        "flag_smurfing", "flag_odd_hour", "flag_velocity",
        "flag_high_value", "flag_balance_breach",
    ]
    if risk is not None:
        for flag in phase8_flags:
            feats[flag] = int(bool(risk.get(flag, False)))
    else:
        # Fall back to inferring from transaction flags
        flag_map = {
            "flag_round_trip":    "is_round_trip",
            "flag_layering":      "is_layering",
            "flag_fan_in":        "is_fan_in",
            "flag_fan_out":       "is_fan_out",
            "flag_smurfing":      "is_smurfing",
            "flag_odd_hour":      "is_odd_hour",
            "flag_velocity":      "is_velocity_flag",
            "flag_high_value":    "is_high_value_flag",
            "flag_balance_breach":"is_balance_breach",
        }
        for flag, txn_col in flag_map.items():
            feats[flag] = int(
                txns[txn_col].any() if txn_col in txns.columns else False
            )

    # ── CHANNEL features ─────────────────────────────────────────────────
    channel_col = "channel"
    if channel_col in txns.columns:
        ch = txns[channel_col].fillna("").str.upper()
        n  = len(txns)
        feats["upi_ratio"]  = float((ch == "UPI").sum()  / n) if n > 0 else 0.0
        feats["neft_ratio"] = float((ch == "NEFT").sum() / n) if n > 0 else 0.0
        feats["rtgs_ratio"] = float((ch == "RTGS").sum() / n) if n > 0 else 0.0
        feats["cash_ratio"] = float(
            (ch.str.contains("CASH|ATM", regex=True)).sum() / n
        ) if n > 0 else 0.0
        feats["imps_ratio"] = float((ch == "IMPS").sum() / n) if n > 0 else 0.0
    else:
        for key in ("upi_ratio", "neft_ratio", "rtgs_ratio", "cash_ratio", "imps_ratio"):
            feats[key] = 0.0

    # ── Z-SCORE features ─────────────────────────────────────────────────
    all_amounts = pd.concat([
        txns[txns["debit"]  > 0]["debit"],
        txns[txns["credit"] > 0]["credit"],
    ])
    if len(all_amounts) > 1:
        mu  = float(all_amounts.mean())
        sig = float(all_amounts.std())
        if sig > 0:
            z_scores = (all_amounts - mu) / sig
            feats["amount_zscore_max"]  = float(z_scores.abs().max())
            feats["amount_zscore_mean"] = float(z_scores.abs().mean())
        else:
            feats["amount_zscore_max"]  = 0.0
            feats["amount_zscore_mean"] = 0.0
    else:
        feats["amount_zscore_max"]  = 0.0
        feats["amount_zscore_mean"] = 0.0

    # Balance volatility = std(balance) / mean(balance), robust to scale
    balances = txns["balance"].replace(0, np.nan).dropna()
    if len(balances) > 1 and balances.mean() != 0:
        feats["balance_volatility"] = float(balances.std() / abs(balances.mean()))
    else:
        feats["balance_volatility"] = 0.0

    # Replace any NaN / inf introduced above
    for k, v in feats.items():
        if k == "account_id":
            continue
        if not isinstance(v, (int, float, np.integer, np.floating)):
            feats[k] = 0.0
        elif np.isnan(v) or np.isinf(v):
            feats[k] = 0.0

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_txn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("debit", "credit", "balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    bool_cols = [
        "is_round_trip", "is_layering", "is_fan_in", "is_fan_out",
        "is_smurfing", "is_odd_hour", "is_velocity_flag",
        "is_high_value_flag", "is_balance_breach", "is_duplicate",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                {"True": True, "False": False, True: True, False: False}
            ).fillna(False)
    return df


def _extract_hour(time_str: str) -> int:
    try:
        return int(str(time_str).split(":")[0])
    except (ValueError, IndexError, AttributeError):
        return 12


def _winsorise(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip continuous features at lower/upper percentile to suppress outlier distortion."""
    binary_cols = {c for c in df.columns if df[c].nunique() <= 2}
    for col in df.columns:
        if col in binary_cols:
            continue
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        if hi > lo:
            df[col] = df[col].clip(lo, hi)
    return df
