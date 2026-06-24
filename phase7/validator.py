"""
Phase 7 — Validator

Six passes in sequence. Every pass appends to clean_flags and also
appends a row to the ACTION LOG so there is a complete per-row audit
trail of exactly what was done and why.

  Pass 1 : Date validation          — null, bad format, out-of-range
  Pass 2 : Amount cleaning          — brackets, CR/DR, lakh commas, currency
  Pass 3 : Balance continuity       — tamper detection (forensic priority)
  Pass 4 : Statistical outliers     — per-account IQR, not global threshold
  Pass 5 : Velocity flagging        — N+ debits in a short window (fraud signal)
  Pass 6 : Narration / channel      — OCR noise strip, canonical channel names

Nothing is silently dropped. Every modification is logged.
The action log is returned to clean.py and written to all_actions.csv.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from cleaning_config import (
    DATE_FORMAT, DATE_VALID_YEAR_MIN, DATE_VALID_YEAR_MAX,
    AMOUNT_BRACKET_NEGATIVE, AMOUNT_STRIP_CURRENCY, AMOUNT_HANDLE_CR_DR,
    BALANCE_TOLERANCE, BALANCE_BREACH_ACCOUNT_FLAG_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER, OUTLIER_MIN_TXN_COUNT,
    VELOCITY_WINDOW_MINUTES, VELOCITY_MIN_TXNS, VELOCITY_MIN_AMOUNT,
    NARRATION_STRIP_CHARS, CHANNEL_NORMALISE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 — Date Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    report = {"null_dates": 0, "bad_format_dates": 0, "out_of_range_dates": 0}
    flags  = [""] * len(df)
    actions = []

    for i, val in enumerate(df["date"]):
        s = str(val).strip() if pd.notna(val) else ""
        if not s:
            flags[i] = _add(flags[i], "NULL_DATE")
            report["null_dates"] += 1
            actions.append(_action(df, i, "NULL_DATE",
                "Date column is empty — row kept but date-dependent checks skipped"))
            continue

        try:
            parsed = datetime.strptime(s, DATE_FORMAT)
            if not (DATE_VALID_YEAR_MIN <= parsed.year <= DATE_VALID_YEAR_MAX):
                flags[i] = _add(flags[i], "DATE_OUT_OF_RANGE")
                report["out_of_range_dates"] += 1
                actions.append(_action(df, i, "DATE_OUT_OF_RANGE",
                    f"Date {s} is outside valid range "
                    f"{DATE_VALID_YEAR_MIN}–{DATE_VALID_YEAR_MAX}"))
        except ValueError:
            flags[i] = _add(flags[i], "BAD_DATE_FORMAT")
            report["bad_format_dates"] += 1
            actions.append(_action(df, i, "BAD_DATE_FORMAT",
                f"'{s}' could not be parsed as YYYY-MM-DD"))

    df = df.copy()
    df["_date_flags"] = flags
    return df, report, actions


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2 — Amount Cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_amounts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    report  = {"amount_corrections": 0, "zero_debit_credit_rows": 0}
    actions = []
    df      = df.copy()

    for col in ("debit", "credit", "balance"):
        for i, val in enumerate(df[col]):
            original   = _safe_float(val)
            cleaned    = _parse_amount_cell(val)
            df.at[df.index[i], col] = cleaned
            if abs(cleaned - original) > 1e-6 and not (
                original == 0.0 and str(val).strip() in ("", "-", "nil", "n/a", "nan", None)
            ):
                report["amount_corrections"] += 1
                actions.append(_action(df, i, f"AMOUNT_CORRECTED_{col.upper()}",
                    f"{col}: '{val}' → {cleaned}"))

    zero_mask = (df["debit"] == 0.0) & (df["credit"] == 0.0)
    report["zero_debit_credit_rows"] = int(zero_mask.sum())
    for i in df.index[zero_mask]:
        actions.append(_action(df, i, "ZERO_DEBIT_AND_CREDIT",
            "Both debit and credit are zero — likely a summary/header row from PDF"))

    return df, report, actions


def _parse_amount_cell(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    s = str(val).strip()
    if s.lower() in ("", "-", "--", "nil", "n/a", "nan", "none"):
        return 0.0

    negative = False

    if AMOUNT_BRACKET_NEGATIVE and s.startswith("(") and s.endswith(")"):
        s, negative = s[1:-1], True

    if AMOUNT_STRIP_CURRENCY:
        s = re.sub(r"[₹$£€]|INR|Rs\.?", "", s, flags=re.IGNORECASE).strip()

    if AMOUNT_HANDLE_CR_DR:
        su = s.upper()
        if re.search(r"(?i)(cr|c)\s*$", su):
            s = re.sub(r"(?i)(cr|c)\s*$", "", s).strip()
        elif re.search(r"(?i)(dr|d)\s*$", su):
            s = re.sub(r"(?i)(dr|d)\s*$", "", s).strip()
            negative = True
        # Leading CR/DR (some banks prefix)
        if re.search(r"(?i)^\s*(cr|c)\b", s):
            s = re.sub(r"(?i)^\s*(cr|c)\b", "", s).strip()
        elif re.search(r"(?i)^\s*(dr|d)\b", s):
            s = re.sub(r"(?i)^\s*(dr|d)\b", "", s).strip()
            negative = True

    s = s.replace(",", "")
    s = re.sub(r"[^\d.\-]", "", s)
    if s.count(".") > 1:
        first, *rest = s.split(".")
        s = first + "." + "".join(rest)

    if not s:
        return 0.0
    try:
        return -abs(float(s)) if negative else abs(float(s))
    except ValueError:
        return 0.0


def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pass 3 — Balance Continuity
# ─────────────────────────────────────────────────────────────────────────────
def validate_balance_continuity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    """
    Forensic tamper-detection: prior_balance + credit - debit ≈ current_balance.
    A statement that doesn't reconcile is either OCR-corrupted or tampered.
    Both need investigator attention.
    """
    report = {
        "accounts_checked":       0,
        "accounts_with_breaches": 0,
        "total_breach_rows":      0,
        "suspect_accounts":       [],
    }
    actions = []
    df = df.copy()
    df["is_balance_breach"] = False

    for account_id, group in df.groupby("account_id"):
        valid = group[group["date"].notna() & (group["date"] != "")].copy()
        valid = valid.sort_values(["date", "time"]).reset_index(drop=True)
        if len(valid) < 2:
            continue

        report["accounts_checked"] += 1
        breach_indices = []

        for i in range(1, len(valid)):
            prev_bal  = valid.loc[i - 1, "balance"]
            curr_bal  = valid.loc[i,     "balance"]
            debit     = valid.loc[i,     "debit"]
            credit    = valid.loc[i,     "credit"]

            if prev_bal == 0 and curr_bal == 0:
                continue

            expected = round(prev_bal + credit - debit, 2)
            actual   = round(curr_bal, 2)
            diff     = abs(expected - actual)

            if diff > BALANCE_TOLERANCE:
                idx = valid.index[i]
                breach_indices.append(idx)
                actions.append(_action(df, idx, "BALANCE_BREACH",
                    f"Expected balance {expected} but got {actual} "
                    f"(diff ₹{diff:.2f}) — possible tamper or OCR error"))

        if breach_indices:
            breach_ratio = len(breach_indices) / len(valid)
            df.loc[breach_indices, "is_balance_breach"] = True
            report["total_breach_rows"]      += len(breach_indices)
            report["accounts_with_breaches"] += 1

            if breach_ratio > BALANCE_BREACH_ACCOUNT_FLAG_THRESHOLD:
                report["suspect_accounts"].append({
                    "account_id":   account_id,
                    "breach_ratio": round(breach_ratio, 3),
                    "breach_rows":  len(breach_indices),
                    "total_rows":   len(valid),
                    "severity":     "HIGH" if breach_ratio > 0.5 else "MEDIUM",
                })

    return df, report, actions


# ─────────────────────────────────────────────────────────────────────────────
# Pass 4 — Statistical Outlier Flagging
# ─────────────────────────────────────────────────────────────────────────────
def flag_statistical_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    """
    Per-account IQR outlier detection — not global rupee thresholds.
    A ₹10L transaction is normal for a business account but an outlier
    for a dormant student account. The ML layer uses this as a feature.
    """
    report  = {"accounts_analysed": 0, "outlier_rows_flagged": 0}
    actions = []
    df      = df.copy()
    df["is_high_value_flag"] = False

    for account_id, group in df.groupby("account_id"):
        if len(group) < OUTLIER_MIN_TXN_COUNT:
            continue
        report["accounts_analysed"] += 1

        for col in ("debit", "credit"):
            nonzero = group[group[col] > 0][col]
            if len(nonzero) < 4:
                continue
            q1, q3 = nonzero.quantile(0.25), nonzero.quantile(0.75)
            iqr    = q3 - q1
            if iqr == 0:
                continue
            fence = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            mask  = group[col] > fence

            for idx in group[mask].index:
                amount = group.loc[idx, col]
                df.loc[idx, "is_high_value_flag"] = True
                actions.append(_action(df, idx, f"HIGH_VALUE_{col.upper()}",
                    f"{col} ₹{amount:,.2f} exceeds IQR fence ₹{fence:,.2f} "
                    f"(Q3={q3:,.2f}, IQR={iqr:,.2f}) for account {account_id}"))
                report["outlier_rows_flagged"] += 1

    return df, report, actions


# ─────────────────────────────────────────────────────────────────────────────
# Pass 5 — Velocity Flagging (new)
# ─────────────────────────────────────────────────────────────────────────────
def flag_velocity_bursts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    """
    Flag accounts where N+ debit transactions occur within a short window.
    This is a classic mule account signal — money arrives, then multiple
    rapid outbound transfers follow within minutes.
    Requires both date AND time columns to be meaningful.
    """
    report  = {"velocity_accounts_flagged": 0, "velocity_rows_flagged": 0}
    actions = []
    df      = df.copy()
    df["is_velocity_flag"] = False

    for account_id, group in df.groupby("account_id"):
        timed = group[
            group["date"].notna() & (group["date"] != "") &
            group["time"].notna() & (group["time"] != "00:00:00")
        ].copy()

        if len(timed) < VELOCITY_MIN_TXNS:
            continue

        try:
            timed["_dt"] = pd.to_datetime(
                timed["date"] + " " + timed["time"], errors="coerce"
            )
            timed = timed.dropna(subset=["_dt"]).sort_values("_dt")
        except Exception:
            continue

        debits = timed[timed["debit"] >= VELOCITY_MIN_AMOUNT].copy()
        if len(debits) < VELOCITY_MIN_TXNS:
            continue

        window = timedelta(minutes=VELOCITY_WINDOW_MINUTES)
        flagged_set = set()

        for i in range(len(debits)):
            t0    = debits.iloc[i]["_dt"]
            burst = debits[(debits["_dt"] >= t0) & (debits["_dt"] <= t0 + window)]
            if len(burst) >= VELOCITY_MIN_TXNS:
                for idx in burst.index:
                    if idx not in flagged_set:
                        flagged_set.add(idx)
                        total = burst["debit"].sum()
                        actions.append(_action(df, idx, "VELOCITY_BURST",
                            f"{len(burst)} debits totalling ₹{total:,.2f} "
                            f"within {VELOCITY_WINDOW_MINUTES} min for account {account_id}"))

        if flagged_set:
            df.loc[list(flagged_set), "is_velocity_flag"] = True
            report["velocity_accounts_flagged"] += 1
            report["velocity_rows_flagged"] += len(flagged_set)

    return df, report, actions


# ─────────────────────────────────────────────────────────────────────────────
# Pass 6 — Narration & Channel Normalisation
# ─────────────────────────────────────────────────────────────────────────────
def normalise_text_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    report  = {"narrations_cleaned": 0, "channels_normalised": 0}
    actions = []
    df      = df.copy()

    original_narrations = df["narration"].copy()
    df["narration"] = (
        df["narration"]
        .fillna("")
        .astype(str)
        .str.replace(NARRATION_STRIP_CHARS, " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.upper()
    )
    changed_narr = df["narration"] != original_narrations.fillna("").str.upper()
    report["narrations_cleaned"] = int(changed_narr.sum())
    for i in df.index[changed_narr]:
        actions.append(_action(df, i, "NARRATION_CLEANED",
            f"OCR noise stripped: '{original_narrations.iloc[i]}' → '{df.loc[i,'narration']}'"))

    def _norm_ch(ch):
        ch = str(ch).strip().upper()
        return CHANNEL_NORMALISE.get(ch, ch)

    original_channels = df["channel"].copy()
    df["channel"]     = df["channel"].apply(_norm_ch)
    changed_ch        = df["channel"] != original_channels
    report["channels_normalised"] = int(changed_ch.sum())
    for i in df.index[changed_ch]:
        actions.append(_action(df, i, "CHANNEL_NORMALISED",
            f"'{original_channels.iloc[i]}' → '{df.loc[i,'channel']}'"))

    return df, report, actions


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _add(existing: str, flag: str) -> str:
    existing = str(existing).strip() if existing else ""
    return flag if not existing else existing + " | " + flag


def _action(df: pd.DataFrame, row_idx, action_type: str, detail: str) -> dict:
    """Build one row for the all_actions.csv audit log."""
    try:
        row = df.iloc[row_idx] if isinstance(row_idx, int) else df.loc[row_idx]
        return {
            "row_index":    row_idx,
            "account_id":   row.get("account_id", ""),
            "date":         row.get("date", ""),
            "narration":    str(row.get("narration", ""))[:80],
            "debit":        row.get("debit", ""),
            "credit":       row.get("credit", ""),
            "balance":      row.get("balance", ""),
            "source_file":  row.get("source_file", ""),
            "action_type":  action_type,
            "detail":       detail,
        }
    except Exception:
        return {
            "row_index": row_idx, "account_id": "", "date": "",
            "narration": "", "debit": "", "credit": "", "balance": "",
            "source_file": "", "action_type": action_type, "detail": detail,
        }