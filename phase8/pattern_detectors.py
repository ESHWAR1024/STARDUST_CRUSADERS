"""
Phase 8 — Pattern Detectors

Six detectors, each returning a list of Finding dicts and a set of
flagged transaction indices (for stamping into the main DataFrame).

  1. detect_round_trips   — money leaves and returns to same account
  2. detect_layering      — rapid hop-chain through 3+ accounts
  3. detect_fan_in        — many senders → one collector in short window
  4. detect_fan_out       — one account → many receivers in short window
  5. detect_smurfing      — structured deposits just below threshold
  6. detect_odd_hours     — repeated transactions between 00:00-05:00

All detectors are STATISTICAL or STRUCTURAL — no hardcoded rupee values
except SMURF_THRESHOLD (which is a real regulatory concept, not arbitrary).
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import networkx as nx

from analytics_config import (
    ROUND_TRIP_MAX_DAYS, ROUND_TRIP_MIN_RATIO, ROUND_TRIP_MIN_AMOUNT,
    LAYERING_MIN_CHAIN, LAYERING_MAX_HOP_HOURS, LAYERING_MIN_AMOUNT,
    FAN_IN_MIN_SENDERS, FAN_IN_WINDOW_HOURS, FAN_IN_MIN_TOTAL,
    FAN_OUT_MIN_RECEIVERS, FAN_OUT_WINDOW_HOURS, FAN_OUT_MIN_TOTAL,
    SMURF_THRESHOLD, SMURF_BAND_LOW, SMURF_MIN_TXNS,
    SMURF_WINDOW_DAYS, SMURF_MIN_UNIQUE_DEST,
    ODD_HOUR_START, ODD_HOUR_END, ODD_HOUR_MIN_TXNS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ROUND-TRIP DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_round_trips(
    df: pd.DataFrame,
    txn_graph: nx.MultiDiGraph,
) -> tuple[list[dict], set]:
    """
    A round trip is: Account A sends money to Account B, and within
    ROUND_TRIP_MAX_DAYS money flows back to A (≥ ROUND_TRIP_MIN_RATIO of
    the original amount).

    Detection: for every debit from A to B, look for a credit from B to A
    within the window. Uses the txn_graph for fast neighbour lookup.
    """
    findings = []
    flagged_idx = set()

    # Build debit and credit maps: {(from, to): [(amount, timestamp, row_idx)]}
    debit_map  = defaultdict(list)
    credit_map = defaultdict(list)

    for idx, row in df.iterrows():
        ts  = _ts(row)
        src = row["account_id"]
        dst = str(row.get("counterparty_name", "")).strip()
        if not dst or dst == "nan":
            continue

        if row["debit"] > ROUND_TRIP_MIN_AMOUNT:
            debit_map[(src, dst)].append((float(row["debit"]), ts, idx))
        if row["credit"] > ROUND_TRIP_MIN_AMOUNT:
            credit_map[(dst, src)].append((float(row["credit"]), ts, idx))

    for (src, dst), debits in debit_map.items():
        reverse_credits = credit_map.get((dst, src), [])
        for (d_amt, d_ts, d_idx) in debits:
            for (c_amt, c_ts, c_idx) in reverse_credits:
                if c_ts <= d_ts:
                    continue   # return must come AFTER original send
                gap_days = (c_ts - d_ts).total_seconds() / 86400
                if gap_days > ROUND_TRIP_MAX_DAYS:
                    continue
                ratio = c_amt / d_amt
                if ratio < ROUND_TRIP_MIN_RATIO:
                    continue

                findings.append({
                    "pattern":        "ROUND_TRIP",
                    "account_a":      src,
                    "account_b":      dst,
                    "outflow_amount": round(d_amt, 2),
                    "return_amount":  round(c_amt, 2),
                    "return_ratio":   round(ratio, 4),
                    "outflow_date":   str(d_ts.date()),
                    "return_date":    str(c_ts.date()),
                    "gap_days":       round(gap_days, 1),
                    "severity":       "HIGH" if gap_days < 3 else "MEDIUM",
                    "description":    (
                        f"{src} sent ₹{d_amt:,.0f} to {dst} on {d_ts.date()}, "
                        f"₹{c_amt:,.0f} ({ratio*100:.0f}%) returned {gap_days:.1f}d later"
                    ),
                })
                flagged_idx.update([d_idx, c_idx])

    return findings, flagged_idx


# ─────────────────────────────────────────────────────────────────────────────
# 2. LAYERING DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_layering(
    df: pd.DataFrame,
    txn_graph: nx.MultiDiGraph,
) -> tuple[list[dict], set]:
    """
    Layering: money moves through a chain of 3+ accounts, each hop within
    LAYERING_MAX_HOP_HOURS of the previous, with each hop amount ≥
    LAYERING_MIN_AMOUNT.

    Uses DFS on the time-ordered txn_graph starting from every node.
    """
    findings = []
    flagged_idx = set()

    # Build time-ordered edge list per source node
    # {node: [(timestamp, target, amount, row_idx)]}
    edge_index = defaultdict(list)
    for idx, row in df.iterrows():
        src = row["account_id"]
        dst = str(row.get("counterparty_name", "")).strip()
        if not dst or dst == "nan":
            continue
        amt = float(row.get("debit", 0) or 0)
        if amt < LAYERING_MIN_AMOUNT:
            continue
        ts = _ts(row)
        edge_index[src].append((ts, dst, amt, idx))

    # Sort each source's edges by timestamp
    for src in edge_index:
        edge_index[src].sort(key=lambda x: x[0])

    # DFS for chains
    def dfs(node, chain, last_ts, last_amt, visited, chain_idxs):
        if len(chain) >= LAYERING_MIN_CHAIN:
            findings.append(_make_layering_finding(chain, chain_idxs))
            flagged_idx.update(chain_idxs)

        if len(chain) >= 8:   # hard cap on chain length
            return

        for (ts, dst, amt, row_idx) in edge_index.get(node, []):
            if ts < last_ts:
                continue
            gap_hours = (ts - last_ts).total_seconds() / 3600
            if gap_hours > LAYERING_MAX_HOP_HOURS:
                break   # sorted by time; no point continuing
            if dst in visited:
                continue   # no cycles

            dfs(
                dst,
                chain + [(node, dst, amt, str(ts.date()))],
                ts, amt,
                visited | {dst},
                chain_idxs + [row_idx],
            )

    for start_node in list(edge_index.keys()):
        dfs(start_node, [], datetime(2000, 1, 1), float("inf"), {start_node}, [])

    # Deduplicate: if a longer chain subsumes a shorter, keep the longer
    findings = _dedup_layering(findings)
    return findings, flagged_idx


def _make_layering_finding(chain, chain_idxs) -> dict:
    accounts = [c[0] for c in chain] + [chain[-1][1]]
    amounts  = [c[2] for c in chain]
    dates    = [c[3] for c in chain]
    skim     = 1 - (amounts[-1] / amounts[0]) if amounts[0] > 0 else 0
    return {
        "pattern":     "LAYERING",
        "chain":       " → ".join(accounts),
        "chain_length":len(chain),
        "start_amount":round(amounts[0], 2),
        "end_amount":  round(amounts[-1], 2),
        "skim_ratio":  round(skim, 4),
        "start_date":  dates[0],
        "end_date":    dates[-1],
        "accounts":    accounts,
        "severity":    "CRITICAL" if len(chain) >= 5 else "HIGH",
        "description": (
            f"Layering chain {len(chain)} hops: "
            f"₹{amounts[0]:,.0f} → ₹{amounts[-1]:,.0f} "
            f"({skim*100:.1f}% skimmed) via {' → '.join(accounts)}"
        ),
    }


def _dedup_layering(findings: list[dict]) -> list[dict]:
    """Remove findings whose chain is a strict subset of another finding's chain."""
    chains = [set(f["accounts"]) for f in findings]
    keep   = []
    for i, f in enumerate(findings):
        subsumed = any(
            i != j and chains[i].issubset(chains[j])
            for j in range(len(findings))
        )
        if not subsumed:
            keep.append(f)
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# 3. FAN-IN DETECTION (collector account)
# ─────────────────────────────────────────────────────────────────────────────

def detect_fan_in(df: pd.DataFrame) -> tuple[list[dict], set]:
    """
    Fan-in: an account receives credits from FAN_IN_MIN_SENDERS+ distinct
    senders within FAN_IN_WINDOW_HOURS, with total inflow ≥ FAN_IN_MIN_TOTAL.
    This is the hallmark of a COLLECTOR account.
    """
    findings = []
    flagged_idx = set()

    credit_df = df[df["credit"] > 0].copy()
    credit_df["_ts"] = credit_df.apply(_ts, axis=1)

    for acc_id, group in credit_df.groupby("account_id"):
        group = group.sort_values("_ts")
        rows  = group.to_dict("records")
        idxs  = group.index.tolist()

        # Sliding window
        for i, (row_i, idx_i) in enumerate(zip(rows, idxs)):
            window_end = row_i["_ts"] + timedelta(hours=FAN_IN_WINDOW_HOURS)
            window_rows = []
            window_idxs = []

            for j in range(i, len(rows)):
                if rows[j]["_ts"] > window_end:
                    break
                window_rows.append(rows[j])
                window_idxs.append(idxs[j])

            senders = {
                str(r.get("counterparty_name", "")).strip()
                for r in window_rows
                if r.get("counterparty_name") and str(r.get("counterparty_name", "")).strip() != "nan"
            }
            total = sum(r["credit"] for r in window_rows)

            if len(senders) < FAN_IN_MIN_SENDERS or total < FAN_IN_MIN_TOTAL:
                continue

            findings.append({
                "pattern":         "FAN_IN",
                "collector":       acc_id,
                "sender_count":    len(senders),
                "senders":         list(senders),
                "total_inflow":    round(total, 2),
                "window_start":    str(window_rows[0]["_ts"].date()),
                "window_end":      str(window_rows[-1]["_ts"].date()),
                "txn_count":       len(window_rows),
                "severity":        "CRITICAL" if len(senders) >= 8 else "HIGH",
                "description": (
                    f"Collector account {acc_id}: ₹{total:,.0f} received from "
                    f"{len(senders)} senders in {FAN_IN_WINDOW_HOURS}h window "
                    f"({window_rows[0]['_ts'].date()} – {window_rows[-1]['_ts'].date()})"
                ),
            })
            flagged_idx.update(window_idxs)
            break   # one finding per account per run (avoid combinatorial explosion)

    return findings, flagged_idx


# ─────────────────────────────────────────────────────────────────────────────
# 4. FAN-OUT DETECTION (distribution account / mule)
# ─────────────────────────────────────────────────────────────────────────────

def detect_fan_out(df: pd.DataFrame) -> tuple[list[dict], set]:
    """
    Fan-out: an account sends debits to FAN_OUT_MIN_RECEIVERS+ distinct
    receivers within FAN_OUT_WINDOW_HOURS, total ≥ FAN_OUT_MIN_TOTAL.
    Hallmark of a DISTRIBUTION or MULE account.
    """
    findings = []
    flagged_idx = set()

    debit_df = df[df["debit"] > 0].copy()
    debit_df["_ts"] = debit_df.apply(_ts, axis=1)

    for acc_id, group in debit_df.groupby("account_id"):
        group = group.sort_values("_ts")
        rows  = group.to_dict("records")
        idxs  = group.index.tolist()

        for i, (row_i, idx_i) in enumerate(zip(rows, idxs)):
            window_end  = row_i["_ts"] + timedelta(hours=FAN_OUT_WINDOW_HOURS)
            window_rows = []
            window_idxs = []

            for j in range(i, len(rows)):
                if rows[j]["_ts"] > window_end:
                    break
                window_rows.append(rows[j])
                window_idxs.append(idxs[j])

            receivers = {
                str(r.get("counterparty_name", "")).strip()
                for r in window_rows
                if r.get("counterparty_name") and str(r.get("counterparty_name", "")).strip() != "nan"
            }
            total = sum(r["debit"] for r in window_rows)

            if len(receivers) < FAN_OUT_MIN_RECEIVERS or total < FAN_OUT_MIN_TOTAL:
                continue

            findings.append({
                "pattern":         "FAN_OUT",
                "distributor":     acc_id,
                "receiver_count":  len(receivers),
                "receivers":       list(receivers),
                "total_outflow":   round(total, 2),
                "window_start":    str(window_rows[0]["_ts"].date()),
                "window_end":      str(window_rows[-1]["_ts"].date()),
                "txn_count":       len(window_rows),
                "severity":        "CRITICAL" if len(receivers) >= 8 else "HIGH",
                "description": (
                    f"Distribution account {acc_id}: ₹{total:,.0f} sent to "
                    f"{len(receivers)} receivers in {FAN_OUT_WINDOW_HOURS}h window "
                    f"({window_rows[0]['_ts'].date()} – {window_rows[-1]['_ts'].date()})"
                ),
            })
            flagged_idx.update(window_idxs)
            break

    return findings, flagged_idx


# ─────────────────────────────────────────────────────────────────────────────
# 5. SMURFING / STRUCTURING DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_smurfing(df: pd.DataFrame) -> tuple[list[dict], set]:
    """
    Smurfing: multiple transactions in the band
    [SMURF_THRESHOLD * SMURF_BAND_LOW, SMURF_THRESHOLD)
    within SMURF_WINDOW_DAYS, directed to SMURF_MIN_UNIQUE_DEST+ distinct
    destinations.

    This mirrors how AML analysts spot structuring: the amount is
    deliberately kept just below the CTR-like reporting line.
    """
    findings = []
    flagged_idx = set()

    band_lo = SMURF_THRESHOLD * SMURF_BAND_LOW
    band_hi = SMURF_THRESHOLD

    debit_df = df[
        (df["debit"] >= band_lo) & (df["debit"] < band_hi)
    ].copy()
    debit_df["_ts"] = debit_df.apply(_ts, axis=1)
    debit_df["_date"] = pd.to_datetime(debit_df["date"], errors="coerce")

    for acc_id, group in debit_df.groupby("account_id"):
        group = group.sort_values("_ts")
        rows  = group.to_dict("records")
        idxs  = group.index.tolist()

        # Rolling SMURF_WINDOW_DAYS window
        for i, (row_i, idx_i) in enumerate(zip(rows, idxs)):
            window_end  = row_i["_ts"] + timedelta(days=SMURF_WINDOW_DAYS)
            window_rows = []
            window_idxs = []

            for j in range(i, len(rows)):
                if rows[j]["_ts"] > window_end:
                    break
                window_rows.append(rows[j])
                window_idxs.append(idxs[j])

            if len(window_rows) < SMURF_MIN_TXNS:
                continue

            destinations = {
                str(r.get("counterparty_name", "")).strip()
                for r in window_rows
                if r.get("counterparty_name") and str(r.get("counterparty_name", "")).strip() != "nan"
            }

            if len(destinations) < SMURF_MIN_UNIQUE_DEST:
                continue

            total = sum(r["debit"] for r in window_rows)
            amounts = [r["debit"] for r in window_rows]

            findings.append({
                "pattern":        "SMURFING",
                "account":        acc_id,
                "txn_count":      len(window_rows),
                "total_amount":   round(total, 2),
                "amounts":        [round(a, 2) for a in amounts],
                "unique_destinations": len(destinations),
                "destinations":   list(destinations),
                "window_start":   str(window_rows[0]["_ts"].date()),
                "window_end":     str(window_rows[-1]["_ts"].date()),
                "threshold_used": SMURF_THRESHOLD,
                "severity":       "HIGH",
                "description": (
                    f"Structuring by {acc_id}: {len(window_rows)} transfers "
                    f"between ₹{band_lo:,.0f}–₹{band_hi:,.0f} "
                    f"to {len(destinations)} destinations, "
                    f"total ₹{total:,.0f} over {SMURF_WINDOW_DAYS}d window"
                ),
            })
            flagged_idx.update(window_idxs)
            break   # one finding per account per run

    return findings, flagged_idx


# ─────────────────────────────────────────────────────────────────────────────
# 6. ODD-HOUR DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_odd_hours(df: pd.DataFrame) -> tuple[list[dict], set]:
    """
    Flag accounts with ODD_HOUR_MIN_TXNS+ transactions between
    ODD_HOUR_START:00 and ODD_HOUR_END:00 (exclusive).
    This is a soft signal — combined with other patterns it elevates risk.
    """
    findings = []
    flagged_idx = set()

    df = df.copy()
    df["_hour"] = df["time"].apply(_extract_hour)
    odd_mask = (df["_hour"] >= ODD_HOUR_START) & (df["_hour"] < ODD_HOUR_END)
    odd_df   = df[odd_mask]

    for acc_id, group in odd_df.groupby("account_id"):
        count = len(group)
        if count < ODD_HOUR_MIN_TXNS:
            continue

        total_debit  = group["debit"].sum()
        total_credit = group["credit"].sum()
        hours_seen   = sorted(group["_hour"].unique().tolist())

        findings.append({
            "pattern":           "ODD_HOUR",
            "account":           acc_id,
            "odd_hour_txns":     count,
            "total_debit":       round(float(total_debit), 2),
            "total_credit":      round(float(total_credit), 2),
            "hours_active":      hours_seen,
            "first_date":        str(group["date"].min()),
            "last_date":         str(group["date"].max()),
            "severity":          "MEDIUM",
            "description": (
                f"{acc_id} has {count} transactions between "
                f"{ODD_HOUR_START:02d}:00–{ODD_HOUR_END:02d}:00 "
                f"(₹{total_debit:,.0f} debit, ₹{total_credit:,.0f} credit)"
            ),
        })
        flagged_idx.update(group.index.tolist())

    return findings, flagged_idx


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts(row) -> datetime:
    date_str = str(row.get("date", "")).strip()
    time_str = str(row.get("time", "00:00:00")).strip()
    if not time_str or time_str in ("nan", ""):
        time_str = "00:00:00"
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return datetime(2000, 1, 1)


def _extract_hour(time_str: str) -> int:
    try:
        return int(str(time_str).split(":")[0])
    except (ValueError, IndexError):
        return 12   # default to noon if unparseable
