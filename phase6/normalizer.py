"""
Phase 6 - Normalizer

Takes (raw_df, bank_code, bank_fmt, header_text, source_format, source_file)
and produces a clean DataFrame matching UNIFIED_SCHEMA exactly.

This is the core mapping layer: it handles date parsing, amount cleaning,
channel inference, counterparty extraction, and missing-value defaults.
"""

import os
import re
import pandas as pd
from ingestion_config import UNIFIED_SCHEMA
from schema_detector import (
    detect_bank_from_text, detect_bank_from_columns,
    detect_header_row, parse_amount, parse_date,
    infer_channel, extract_counterparty
)


def normalize(raw_df: pd.DataFrame, header_text: str,
              source_file: str, source_format: str,
              provided_account_id: str = None) -> tuple[pd.DataFrame, list]:
    """
    Main normalization entry point.
    Returns (normalized_df, warnings_list).
    """
    warnings = []

    if raw_df.empty:
        return pd.DataFrame(columns=UNIFIED_SCHEMA), ["Empty input DataFrame"]

    # --- 1. Detect which bank format this is ---
    bank_code, bank_fmt = detect_bank_from_columns(raw_df.columns.tolist())

    if bank_fmt is None:
        # Try re-scanning for the true header row (common in CSV with metadata rows)
        header_row_idx = detect_header_row(raw_df)
        if header_row_idx > 0:
            new_cols = raw_df.iloc[header_row_idx].values
            raw_df = raw_df.iloc[header_row_idx + 1:].copy()
            raw_df.columns = new_cols
            raw_df = raw_df.reset_index(drop=True)
            bank_code, bank_fmt = detect_bank_from_columns(raw_df.columns.tolist())

    if bank_fmt is None:
        # Last resort: detect from header text
        bank_code = detect_bank_from_text(header_text)
        if bank_code:
            from ingestion_config import BANK_FORMAT_REGISTRY
            bank_fmt = BANK_FORMAT_REGISTRY[bank_code]
        else:
            warnings.append("Could not detect bank format — columns may not match any known bank")
            return pd.DataFrame(columns=UNIFIED_SCHEMA), warnings

    bank_name = bank_fmt["bank_name"]

    # --- 2. Extract account metadata from header text ---
    account_holder = _extract_account_holder(header_text)
    account_id = provided_account_id or _extract_account_id_from_filename(source_file)

    # --- 3. Drop empty rows ---
    raw_df = raw_df.dropna(how="all").reset_index(drop=True)

    # --- 4. Map columns row by row ---
    output_rows = []
    for _, row in raw_df.iterrows():
        row_warnings = []

        # Date
        date_raw = _get_col(row, bank_fmt["date_col"])
        date_parsed = parse_date(date_raw, bank_fmt["date_formats"])
        if not date_parsed or date_parsed == str(date_raw):
            row_warnings.append(f"unparseable date: {date_raw}")

        # Amounts
        debit = parse_amount(_get_col(row, bank_fmt["debit_col"]))
        credit = parse_amount(_get_col(row, bank_fmt["credit_col"]))
        balance = parse_amount(_get_col(row, bank_fmt["balance_col"]))

        if debit == 0.0 and credit == 0.0:
            row_warnings.append("both debit and credit are zero")

        # Narration + derived fields
        narration = str(_get_col(row, bank_fmt["narration_col"]) or "").strip()
        channel = infer_channel(narration)
        counterparty = extract_counterparty(narration)
        utr_ref = str(_get_col(row, bank_fmt["ref_col"]) or "").strip()

        output_rows.append({
            "account_id": account_id,
            "account_holder": account_holder,
            "bank_name": bank_name,
            "date": date_parsed,
            "time": "00:00:00",
            "narration": narration,
            "channel": channel,
            "debit": round(debit, 2),
            "credit": round(credit, 2),
            "balance": round(balance, 2),
            "utr_ref": utr_ref,
            "counterparty_name": counterparty,
            "source_file": os.path.basename(source_file),
            "source_format": source_format,
            "ingestion_warnings": " | ".join(row_warnings) if row_warnings else "",
        })

    normalized = pd.DataFrame(output_rows, columns=UNIFIED_SCHEMA)

    # --- 5. Drop rows with no date and no amounts (total/summary rows) ---
    before = len(normalized)
    normalized = normalized[
        ~((normalized["date"] == "") &
          (normalized["debit"] == 0.0) &
          (normalized["credit"] == 0.0))
    ].reset_index(drop=True)
    dropped = before - len(normalized)
    if dropped:
        warnings.append(f"Dropped {dropped} summary/blank rows")

    warnings_with_counts = _count_row_warnings(normalized, warnings)
    return normalized, warnings_with_counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_col(row, col_name: str):
    """Safely get a value from a row by column name."""
    try:
        val = row[col_name]
        if pd.isna(val):
            return None
        return val
    except (KeyError, TypeError):
        return None


def _extract_account_holder(header_text: str) -> str:
    """Extract account holder name from file header block."""
    patterns = [
        r"Account Holder[:\s]+([A-Za-z\s\.]+?)(?:\n|IFSC|Account Number|$)",
        r"Name[:\s]+([A-Za-z\s\.]+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, header_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _extract_account_id_from_filename(filename: str) -> str:
    """
    Try to extract account_id from filename.
    Expected format: statement_ACC000042_SBI.csv
    """
    match = re.search(r"(ACC\d+)", os.path.basename(filename))
    if match:
        return match.group(1)
    return os.path.splitext(os.path.basename(filename))[0]


def _count_row_warnings(df: pd.DataFrame, existing_warnings: list) -> list:
    """Count rows with warnings and add a summary to the warnings list."""
    warn_rows = (df["ingestion_warnings"] != "").sum()
    if warn_rows:
        existing_warnings.append(f"{warn_rows} rows had parsing warnings")
    return existing_warnings
