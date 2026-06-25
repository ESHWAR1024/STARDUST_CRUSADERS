"""
Phase 8 — Beneficiary Analysis & Risk Scorer

BENEFICIARY ANALYSIS
  For each account, profiles every counterparty:
    - total sent/received
    - transaction count
    - first/last seen date
    - whether it's a new high-value beneficiary (first txn, large amount)
    - z-score of the transaction amount relative to this account's distribution

RISK SCORER
  Aggregates all Phase 8 pattern flags + Phase 7 flags into a 0–100
  risk score per account with a named risk tier (CRITICAL / HIGH / MEDIUM / LOW).

  Weights are defined in analytics_config.RISK_WEIGHTS.
  Score is NOT a probability — it is an ordinal investigator-priority rank.
  A CRITICAL account should be investigated first; LOW is routine.
"""

from __future__ import annotations
from collections import defaultdict

import numpy as np
import pandas as pd

from analytics_config import (
    RISK_WEIGHTS, RISK_TIERS,
    BENE_HIGH_VALUE_ZSCORE, BENE_NEW_HIGH_VALUE_RATIO,
    ODD_HOUR_START, ODD_HOUR_END,
)


# ─────────────────────────────────────────────────────────────────────────────
# BENEFICIARY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_beneficiaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a flat DataFrame with one row per (account_id, counterparty)
    pair, enriched with behavioural statistics.

    This is used by:
      - The reporting engine (beneficiary table in the investigation report)
      - The risk scorer (new_high_value_bene flag)
      - The frontend dashboard (beneficiary explorer panel)
    """
    records = []

    # Per-account stats for z-score computation
    acct_stats = _compute_account_stats(df)

    # Build per-account first-seen date for each counterparty
    first_seen: dict[tuple, str] = {}
    debit_df = df[df["debit"] > 0].copy()
    debit_df["_date_parsed"] = pd.to_datetime(debit_df["date"], errors="coerce")

    for (acc_id, cpname), group in debit_df.groupby(
        ["account_id", "counterparty_name"], sort=False
    ):
        if not cpname or str(cpname).strip() in ("", "nan"):
            continue
        first_seen[(acc_id, cpname)] = str(group["_date_parsed"].min().date())

    # Aggregate
    for (acc_id, cpname), group in debit_df.groupby(
        ["account_id", "counterparty_name"], sort=False
    ):
        if not cpname or str(cpname).strip() in ("", "nan"):
            continue

        total_sent  = group["debit"].sum()
        txn_count   = len(group)
        max_single  = group["debit"].max()
        last_date   = str(group["_date_parsed"].max().date())
        f_date      = first_seen.get((acc_id, cpname), "")
        is_new      = (f_date == last_date) and txn_count == 1  # only ever one txn

        # Z-score of max_single relative to this account's debit distribution
        acct_mean = acct_stats.get(acc_id, {}).get("debit_mean", 0)
        acct_std  = acct_stats.get(acc_id, {}).get("debit_std", 1)
        zscore    = (max_single - acct_mean) / acct_std if acct_std > 0 else 0

        acct_max  = acct_stats.get(acc_id, {}).get("debit_max", 1)
        new_hv    = is_new and (max_single >= BENE_NEW_HIGH_VALUE_RATIO * acct_max)

        records.append({
            "account_id":               acc_id,
            "counterparty_name":        cpname,
            "total_sent":               round(float(total_sent), 2),
            "txn_count":                int(txn_count),
            "max_single_txn":           round(float(max_single), 2),
            "first_date":               f_date,
            "last_date":                last_date,
            "amount_zscore":            round(float(zscore), 3),
            "is_new_high_value_bene":   bool(new_hv),
            "is_high_value_bene":       bool(zscore >= BENE_HIGH_VALUE_ZSCORE),
        })

    bene_df = pd.DataFrame(records) if records else pd.DataFrame(columns=[
        "account_id", "counterparty_name", "total_sent", "txn_count",
        "max_single_txn", "first_date", "last_date",
        "amount_zscore", "is_new_high_value_bene", "is_high_value_bene",
    ])

    return bene_df.sort_values(
        ["account_id", "total_sent"], ascending=[True, False]
    ).reset_index(drop=True)


def _compute_account_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for acc_id, group in df[df["debit"] > 0].groupby("account_id"):
        debits = group["debit"].values
        stats[acc_id] = {
            "debit_mean": float(np.mean(debits)),
            "debit_std":  float(np.std(debits)) if len(debits) > 1 else 1.0,
            "debit_max":  float(np.max(debits)),
        }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORER
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_scores(
    df:             pd.DataFrame,
    round_trips:    list[dict],
    layering:       list[dict],
    fan_in:         list[dict],
    fan_out:        list[dict],
    smurfing:       list[dict],
    odd_hours:      list[dict],
    bene_df:        pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per account_id:
        account_id, account_holder, bank_name,
        risk_score (0–100), risk_tier,
        flag_round_trip, flag_layering, flag_fan_in, flag_fan_out,
        flag_smurfing, flag_odd_hour, flag_velocity, flag_high_value,
        flag_balance_breach, flag_new_hv_bene,
        active_patterns, risk_reasoning
    """
    accounts = df[["account_id", "account_holder", "bank_name"]].drop_duplicates(
        subset=["account_id"]
    ).set_index("account_id")

    # Collect which accounts appear in each pattern
    rt_accts  = {f["account_a"] for f in round_trips} | {f["account_b"] for f in round_trips}
    lay_accts = {acc for f in layering for acc in f.get("accounts", [])}
    fi_accts  = {f["collector"] for f in fan_in}
    fo_accts  = {f["distributor"] for f in fan_out}
    sm_accts  = {f["account"] for f in smurfing}
    oh_accts  = {f["account"] for f in odd_hours}

    # Phase 7 per-account flag rates
    vel_accts  = set(df[df["is_velocity_flag"]  == True]["account_id"])
    hv_accts   = set(df[df["is_high_value_flag"] == True]["account_id"])
    bb_accts   = set(df[df["is_balance_breach"]  == True]["account_id"])

    # New high-value beneficiary accounts (from beneficiary analysis)
    nhv_accts = set()
    if not bene_df.empty and "is_new_high_value_bene" in bene_df.columns:
        nhv_accts = set(bene_df[bene_df["is_new_high_value_bene"]]["account_id"])

    rows = []
    for acc_id, meta in accounts.iterrows():
        flags = {
            "round_trip":     acc_id in rt_accts,
            "layering":       acc_id in lay_accts,
            "fan_in":         acc_id in fi_accts,
            "fan_out":        acc_id in fo_accts,
            "smurfing":       acc_id in sm_accts,
            "odd_hour":       acc_id in oh_accts,
            "velocity":       acc_id in vel_accts,
            "high_value":     acc_id in hv_accts,
            "balance_breach": acc_id in bb_accts,
        }

        # Weighted score (each flag contributes its weight × 100 to the score)
        raw_score = sum(
            RISK_WEIGHTS.get(k, 0) * 100
            for k, v in flags.items() if v
        )
        # New HV beneficiary adds a small bonus
        if acc_id in nhv_accts:
            raw_score += 5

        score = min(round(raw_score, 1), 100.0)

        tier = "LOW"
        for t, threshold in sorted(RISK_TIERS.items(), key=lambda x: -x[1]):
            if score >= threshold:
                tier = t
                break

        active = [k.upper() for k, v in flags.items() if v]
        if acc_id in nhv_accts:
            active.append("NEW_HV_BENE")

        reasoning = _build_reasoning(acc_id, flags, acc_id in nhv_accts,
                                      round_trips, layering, fan_in,
                                      fan_out, smurfing, odd_hours)

        rows.append({
            "account_id":         acc_id,
            "account_holder":     meta.get("account_holder", ""),
            "bank_name":          meta.get("bank_name", ""),
            "risk_score":         score,
            "risk_tier":          tier,
            "flag_round_trip":    flags["round_trip"],
            "flag_layering":      flags["layering"],
            "flag_fan_in":        flags["fan_in"],
            "flag_fan_out":       flags["fan_out"],
            "flag_smurfing":      flags["smurfing"],
            "flag_odd_hour":      flags["odd_hour"],
            "flag_velocity":      flags["velocity"],
            "flag_high_value":    flags["high_value"],
            "flag_balance_breach":flags["balance_breach"],
            "flag_new_hv_bene":   acc_id in nhv_accts,
            "active_patterns":    " | ".join(active) if active else "NONE",
            "risk_reasoning":     reasoning,
        })

    risk_df = pd.DataFrame(rows).sort_values(
        "risk_score", ascending=False
    ).reset_index(drop=True)

    return risk_df


def _build_reasoning(
    acc_id, flags, is_nhv,
    round_trips, layering, fan_in, fan_out, smurfing, odd_hours,
) -> str:
    parts = []

    if flags["round_trip"]:
        rt = [f for f in round_trips if acc_id in (f["account_a"], f["account_b"])]
        if rt:
            r = rt[0]
            parts.append(
                f"Round-trip: ₹{r['outflow_amount']:,.0f} sent to {r['account_b']}, "
                f"₹{r['return_amount']:,.0f} returned in {r['gap_days']}d"
            )

    if flags["layering"]:
        lay = [f for f in layering if acc_id in f.get("accounts", [])]
        if lay:
            l = lay[0]
            parts.append(
                f"Layering: {l['chain_length']}-hop chain "
                f"₹{l['start_amount']:,.0f}→₹{l['end_amount']:,.0f}"
            )

    if flags["fan_in"]:
        fi = [f for f in fan_in if f["collector"] == acc_id]
        if fi:
            f_ = fi[0]
            parts.append(
                f"Fan-in collector: ₹{f_['total_inflow']:,.0f} from "
                f"{f_['sender_count']} senders"
            )

    if flags["fan_out"]:
        fo = [f for f in fan_out if f["distributor"] == acc_id]
        if fo:
            f_ = fo[0]
            parts.append(
                f"Fan-out: ₹{f_['total_outflow']:,.0f} to "
                f"{f_['receiver_count']} receivers"
            )

    if flags["smurfing"]:
        sm = [f for f in smurfing if f["account"] == acc_id]
        if sm:
            s = sm[0]
            parts.append(
                f"Smurfing: {s['txn_count']} structured transfers "
                f"below ₹{s['threshold_used']:,.0f}, "
                f"total ₹{s['total_amount']:,.0f}"
            )

    if flags["odd_hour"]:
        oh = [f for f in odd_hours if f["account"] == acc_id]
        if oh:
            o = oh[0]
            parts.append(
                f"Odd-hour activity: {o['odd_hour_txns']} transactions "
                f"between {ODD_HOUR_START:02d}:00–{ODD_HOUR_END:02d}:00"
            )

    if flags["velocity"]:
        parts.append("Velocity burst: rapid successive debits flagged in Phase 7")
    if flags["high_value"]:
        parts.append("High-value outlier: IQR outlier flagged in Phase 7")
    if flags["balance_breach"]:
        parts.append("Balance breach: statement balance does not reconcile")
    if is_nhv:
        parts.append("New high-value beneficiary: first-time large transfer to unknown recipient")

    return " | ".join(parts) if parts else "No suspicious patterns detected"



