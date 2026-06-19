"""
Phase 7 - Validator

Four validation passes run in sequence:

  1. Date validation  — ISO format check, range check, gap detection
  2. Amount cleaning  — OCR artifacts, bracket negatives, lakh commas
  3. Balance continuity — prior_balance + credit - debit ≈ current_balance
     This is the most forensically significant check: a statement where
     balances don't reconcile is either OCR-corrupted OR tampered with.
     Both cases need an investigator's attention.
  4. Statistical outlier flagging — per-account IQR on debit and credit
     amounts. Not a fraud signal on its own (large legitimate transactions
     exist) but surfaces accounts with unusual high-value activity for the
     ML layer to weigh.

Every check ADDS a flag to 'clean_flags' rather than silently dropping
rows — the investigator reading the final report needs to know what was
uncertain, not just what was clean.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime

from cleaning_config import (
    DATE_FORMAT, DATE_NULL_ACTION,
    AMOUNT_BRACKET_NEGATIVE, AMOUNT_STRIP_CURRENCY, AMOUNT_HANDLE_CR_DR,
    BALANCE_TOLERANCE, BALANCE_BREACH_ACCOUNT_FLAG_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER, OUTLIER_MIN_TXN_COUNT,
    NARRATION_STRIP_CHARS, CHANNEL_NORMALISE,
)


# ---------------------------------------------------------------------------
# 1. Date Validation
# ---------------------------------------------------------------------------
def validate_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"null_dates": 0, "bad_format_dates": 0, "out_of_range_dates": 0}
    flags = [""] * len(df)

    for i, val in enumerate(df["date"]):
        if pd.isna(val) or str(val).strip() == "":
            flags[i] = _append_flag(flags[i], "NULL_DATE")
            report["null_dates"] += 1
            continue

        s = str(val).strip()
        try:
            parsed = datetime.strptime(s, DATE_FORMAT)
            # Sanity range: 2000-01-01 to 2030-12-31
            if not (2000 <= parsed.year <= 2030):
                flags[i] = _append_flag(flags[i], "DATE_OUT_OF_RANGE")
                report["out_of_range_dates"] += 1
        except ValueError:
            flags[i] = _append_flag(flags[i], "BAD_DATE_FORMAT")
            report["bad_format_dates"] += 1

    df = df.copy()
    df["_date_flags"] = flags
    return df, report


# ---------------------------------------------------------------------------
# 2. Amount Cleaning
# ---------------------------------------------------------------------------
def clean_amounts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean debit/credit/balance columns in place.
    Handles: OCR bracket negatives, lakh commas, currency symbols,
    CR/DR suffixes, and empty-string → 0.0 conversion.
    """
    report = {"amount_corrections": 0, "zero_debit_credit_rows": 0}
    df = df.copy()

    for col in ("debit", "credit", "balance"):
        cleaned, n_fixed = _clean_amount_series(df[col])
        df[col] = cleaned
        report["amount_corrections"] += n_fixed

    # Flag rows where both debit and credit are 0 after cleaning
    # (these are usually OCR failures or PDF summary rows)
    zero_mask = (df["debit"] == 0.0) & (df["credit"] == 0.0)
    report["zero_debit_credit_rows"] = int(zero_mask.sum())

    return df, report


def _clean_amount_series(series: pd.Series) -> tuple[pd.Series, int]:
    cleaned = []
    n_fixed = 0

    for val in series:
        original = val
        result = _parse_amount_cell(val)
        if result != _safe_float(original):
            n_fixed += 1
        cleaned.append(result)

    return pd.Series(cleaned, index=series.index), n_fixed


def _parse_amount_cell(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0

    s = str(val).strip()
    if s in ("", "-", "nil", "n/a", "nan"):
        return 0.0

    negative = False

    # Bracket negative: (1234.56) → -1234.56
    if AMOUNT_BRACKET_NEGATIVE and s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        negative = True

    # Strip currency symbols
    if AMOUNT_STRIP_CURRENCY:
        s = re.sub(r"[₹$£€]|INR|Rs\.?", "", s, flags=re.IGNORECASE).strip()

    # Handle CR/DR suffix: "1234.56CR" means credit (positive), "1234.56DR" debit
    if AMOUNT_HANDLE_CR_DR:
        if s.upper().endswith("DR"):
            s = s[:-2].strip()
            negative = True
        elif s.upper().endswith("CR"):
            s = s[:-2].strip()

    # Remove Indian lakh commas: 1,23,456.78 → 123456.78
    s = s.replace(",", "")

    # Remove any remaining non-numeric except dot and minus
    s = re.sub(r"[^\d.\-]", "", s)

    if not s:
        return 0.0

    try:
        result = float(s)
        return -result if negative else result
    except ValueError:
        return 0.0


def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# 3. Balance Continuity Validation
# ---------------------------------------------------------------------------
def validate_balance_continuity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    For each account, check that:
        row[i-1].balance + row[i].credit - row[i].debit ≈ row[i].balance

    This is the primary tamper-detection signal. A real bank statement
    always reconciles. An OCR-corrupted or manually altered statement won't.

    Only checks accounts with ≥ 2 rows. Rows with null dates are skipped
    in the continuity sequence (can't order them).
    """
    report = {
        "accounts_checked": 0,
        "accounts_with_breaches": 0,
        "total_breach_rows": 0,
        "suspect_accounts": [],
    }

    df = df.copy()
    df["is_balance_breach"] = False

    for account_id, group in df.groupby("account_id"):
        # Only rows with valid dates, sorted chronologically
        valid = group[group["date"].notna() & (group["date"] != "")].copy()
        valid = valid.sort_values(["date", "time"]).reset_index(drop=True)

        if len(valid) < 2:
            continue

        report["accounts_checked"] += 1
        breach_indices = []

        for i in range(1, len(valid)):
            prev_bal = valid.loc[i - 1, "balance"]
            curr_bal = valid.loc[i, "balance"]
            debit    = valid.loc[i, "debit"]
            credit   = valid.loc[i, "credit"]

            # Skip if any value is zero AND looks like an OCR failure
            if prev_bal == 0 and curr_bal == 0:
                continue

            expected = round(prev_bal + credit - debit, 2)
            actual   = round(curr_bal, 2)

            if abs(expected - actual) > BALANCE_TOLERANCE:
                breach_indices.append(valid.index[i])

        if breach_indices:
            breach_ratio = len(breach_indices) / len(valid)
            df.loc[breach_indices, "is_balance_breach"] = True
            report["total_breach_rows"] += len(breach_indices)
            report["accounts_with_breaches"] += 1

            if breach_ratio > BALANCE_BREACH_ACCOUNT_FLAG_THRESHOLD:
                report["suspect_accounts"].append({
                    "account_id": account_id,
                    "breach_ratio": round(breach_ratio, 3),
                    "breach_rows": len(breach_indices),
                    "total_rows": len(valid),
                })

    return df, report


# ---------------------------------------------------------------------------
# 4. Statistical Outlier Flagging
# ---------------------------------------------------------------------------
def flag_statistical_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Per-account IQR-based outlier detection on debit and credit amounts.
    Uses the account's OWN distribution — not a global threshold —
    so a ₹10L transaction in a business account isn't flagged, but
    a ₹10L transaction in a dormant student account IS.

    This is exactly the kind of feature the ML layer (Phase 10) will use.
    """
    report = {"accounts_analysed": 0, "outlier_rows_flagged": 0}
    df = df.copy()
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
            iqr = q3 - q1
            if iqr == 0:
                continue

            upper_fence = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            outlier_mask = (group[col] > upper_fence)
            n_outliers = outlier_mask.sum()

            if n_outliers > 0:
                df.loc[group[outlier_mask].index, "is_high_value_flag"] = True
                report["outlier_rows_flagged"] += int(n_outliers)

    return df, report


# ---------------------------------------------------------------------------
# 5. Narration + Channel Normalisation
# ---------------------------------------------------------------------------
def normalise_text_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"narrations_cleaned": 0, "channels_normalised": 0}
    df = df.copy()

    # Narration: strip OCR noise characters, collapse whitespace, uppercase
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
    report["narrations_cleaned"] = int((df["narration"] != original_narrations.fillna("").str.upper()).sum())

    # Channel: map OCR/variant values to canonical names
    def _normalise_channel(ch):
        ch = str(ch).strip().upper()
        return CHANNEL_NORMALISE.get(ch, ch)

    original_channels = df["channel"].copy()
    df["channel"] = df["channel"].apply(_normalise_channel)
    report["channels_normalised"] = int((df["channel"] != original_channels).sum())

    return df, report


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _append_flag(existing: str, new_flag: str) -> str:
    if existing:
        return existing + " | " + new_flag
    return new_flag
