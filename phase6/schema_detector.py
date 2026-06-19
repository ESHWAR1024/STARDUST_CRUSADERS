"""
Phase 6 - Schema Detector

Given a DataFrame (freshly parsed from any format), identifies:
1. Which bank's format this matches
2. Which row is the actual header (CSV files often have 2-4 metadata
   rows above the real column headers)
3. Returns the matched bank code + format config, or None if unknown
"""

import re
import pandas as pd
from ingestion_config import BANK_FORMAT_REGISTRY, BANK_NAME_KEYWORDS


def detect_bank_from_text(text: str) -> str | None:
    """Scan free text (file header block, PDF title) for bank name keywords."""
    text_lower = text.lower()
    for keyword, bank_code in BANK_NAME_KEYWORDS.items():
        if keyword in text_lower:
            return bank_code
    return None


def detect_header_row(raw_df: pd.DataFrame, max_scan_rows: int = 10) -> int:
    """
    Find the row index in raw_df that contains the actual column headers.
    Bank CSV exports often have 2-4 metadata lines (bank name, account
    holder, IFSC, blank line) before the table headers.
    Returns the 0-based row index of the header row, or 0 if not found.
    """
    all_format_cols = set()
    for fmt in BANK_FORMAT_REGISTRY.values():
        all_format_cols.update([
            fmt["date_col"], fmt["narration_col"], fmt["ref_col"],
            fmt["debit_col"], fmt["credit_col"], fmt["balance_col"],
        ])

    for row_idx in range(min(max_scan_rows, len(raw_df))):
        row_values = [str(v).strip() for v in raw_df.iloc[row_idx].values if pd.notna(v)]
        # Check if this row contains at least 3 known column header strings
        matches = sum(1 for v in row_values if v in all_format_cols)
        if matches >= 3:
            return row_idx

    return 0  # fallback: assume row 0 is the header


def detect_bank_from_columns(columns: list) -> tuple[str | None, dict | None]:
    """
    Given a list of column names (after header row is found), find which
    bank format it matches. Returns (bank_code, format_config) or (None, None).
    """
    col_set = set(str(c).strip() for c in columns)

    best_bank = None
    best_score = 0

    for bank_code, fmt in BANK_FORMAT_REGISTRY.items():
        required_cols = {
            fmt["date_col"], fmt["narration_col"],
            fmt["debit_col"], fmt["credit_col"], fmt["balance_col"]
        }
        score = len(required_cols & col_set)
        if score > best_score:
            best_score = score
            best_bank = bank_code

    if best_score >= 3:
        return best_bank, BANK_FORMAT_REGISTRY[best_bank]
    return None, None


def parse_amount(val) -> float:
    """Safely convert a bank statement amount cell to float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    s = str(val).strip().replace(",", "").replace("₹", "").replace("INR", "").strip()
    if s in ("", "-", "nil", "n/a"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_date(val, date_formats: list) -> str:
    """Try each date format, return ISO date string or original string on failure."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    for fmt in date_formats:
        try:
            return pd.to_datetime(s, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    # Fallback: let pandas infer
    try:
        return pd.to_datetime(s, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return s  # return as-is with a warning


def infer_channel(narration: str) -> str:
    """Infer payment channel from narration text."""
    from ingestion_config import CHANNEL_KEYWORDS
    narration_lower = narration.lower()
    for channel, keywords in CHANNEL_KEYWORDS:
        if any(kw in narration_lower for kw in keywords):
            return channel
    return "OTHER"


def extract_counterparty(narration: str) -> str:
    """
    Best-effort counterparty extraction from narration.
    Most bank narrations follow patterns like:
      UPI-JOHN DOE-PAYMENT or NEFT CR-HDFC BANK-SALARY
    """
    parts = re.split(r"[-/]", narration)
    if len(parts) >= 2:
        # Second segment is usually the counterparty name
        candidate = parts[1].strip().title()
        if len(candidate) > 2 and not candidate.replace(" ", "").isnumeric():
            return candidate
    return ""
