"""
Phase 6 — Normalizer
Maps any parsed DataFrame to the UNIFIED_SCHEMA.
"""

import os, re
import pandas as pd
from ingestion_config import UNIFIED_SCHEMA, BANK_FORMAT_REGISTRY
from schema_detector import (
    detect_bank_from_text, detect_bank_from_columns, detect_generic_format,
    detect_header_row, parse_amount, parse_date,
    infer_channel, extract_counterparty,
)

NON_TX_ROW_KEYWORDS = [
    "opening balance", "closing balance", "b/f", "balance forward",
    "total credit", "total debit", "total deposits", "total withdrawals",
    "statement generated", "computer generated", "registered office",
    "customer care", "this is a computer", "page ", "balance as on",
    "brought forward", "carried forward",
]


def normalize(
    raw_df: pd.DataFrame,
    header_text: str,
    source_file: str,
    source_format: str,
    provided_account_id: str = None,
) -> tuple[pd.DataFrame, list]:

    warnings = []

    if raw_df.empty:
        return pd.DataFrame(columns=UNIFIED_SCHEMA), ["Empty input DataFrame"]

    raw_df = raw_df.copy()
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    # ── Promote header if columns look unnamed ─────────────────────────────
    current_cols = [str(c) for c in raw_df.columns]
    looks_unheadered = all(
        c.startswith("Unnamed:") or c.strip() == "" or c.isdigit()
        for c in current_cols
    )
    if looks_unheadered:
        hi = detect_header_row(raw_df)
        raw_df = raw_df.iloc[hi + 1:].copy()
        raw_df.columns = [str(c).strip() for c in raw_df.iloc[hi].values
                          if pd.notna(c)]  # already sliced above, use detected
        raw_df = raw_df.reset_index(drop=True)

    # ── Detect bank format (4 fallback levels) ─────────────────────────────
    bank_code = detect_bank_from_text(header_text)
    bank_fmt  = BANK_FORMAT_REGISTRY.get(bank_code) if bank_code else None

    # Level 1 — header text matched, but validate columns exist
    if bank_fmt is not None:
        required = {
            bank_fmt["date_col"], bank_fmt["narration_col"],
            bank_fmt["debit_col"], bank_fmt["credit_col"], bank_fmt["balance_col"],
        }
        if len(required & set(raw_df.columns)) < 3:
            bank_fmt = None

    # Level 2 — column-name exact match
    if bank_fmt is None:
        detected, col_fmt = detect_bank_from_columns(raw_df.columns.tolist())
        if col_fmt:
            bank_code, bank_fmt = detected, col_fmt

    # Level 3 — re-scan for header row deeper in the data
    if bank_fmt is None:
        hi = detect_header_row(raw_df)
        if hi > 0:
            new_cols = raw_df.iloc[hi].values
            raw_df   = raw_df.iloc[hi + 1:].copy()
            raw_df.columns = [str(c).strip() for c in new_cols]
            raw_df   = raw_df.reset_index(drop=True)
            detected, col_fmt = detect_bank_from_columns(raw_df.columns.tolist())
            if col_fmt:
                bank_code, bank_fmt = detected, col_fmt

    # Level 4 — generic role matching
    if bank_fmt is None:
        bank_fmt = detect_generic_format(raw_df.columns.tolist())
        if bank_fmt is not None:
            if bank_code and bank_code not in BANK_FORMAT_REGISTRY:
                bank_fmt["bank_name"] = bank_code.title()
                warnings.append(f"Unlisted bank '{bank_code}' — generic column match")
            else:
                warnings.append("Bank not in registry — used generic column match")

    if bank_fmt is None:
        warnings.append("Could not detect bank format — skipping file")
        return pd.DataFrame(columns=UNIFIED_SCHEMA), warnings

    bank_name      = bank_fmt.get("bank_name", "Unknown / Generic")
    account_holder = _extract_account_holder(header_text)
    account_id     = provided_account_id or _extract_account_id(source_file)

    raw_df = raw_df.dropna(how="all").reset_index(drop=True)

    # Collect date-like fallback columns (for when primary date_col is garbled)
    date_fallback_cols = [
        c for c in raw_df.columns
        if c != bank_fmt.get("date_col", "")
        and any(kw in str(c).lower() for kw in ["date", "value", "post", "tran"])
    ]

    output_rows = []

    for _, row in raw_df.iterrows():
        row_warnings = []

        # Skip non-transaction rows (summaries, footers)
        row_text = " ".join(
            str(_get_col(row, c) or "")
            for c in [bank_fmt.get("date_col", ""), bank_fmt.get("narration_col", "")]
        ).lower()
        if any(kw in row_text for kw in NON_TX_ROW_KEYWORDS):
            continue

        narration_raw = str(
            _get_col(row, bank_fmt.get("narration_col", "")) or ""
        ).strip()

        # ── Date (3 levels) ───────────────────────────────────────────────
        date_raw    = _get_col(row, bank_fmt.get("date_col", ""))
        date_parsed = parse_date(date_raw, bank_fmt.get("date_formats", []))

        # Fallback: try other date-like columns (e.g. "Value Date" when
        #           "Txn Date" was garbled by OCR)
        if not date_parsed:
            for fb_col in date_fallback_cols:
                fb_date = parse_date(
                    _get_col(row, fb_col), bank_fmt.get("date_formats", [])
                )
                if fb_date:
                    date_parsed = fb_date
                    row_warnings.append(f"date from '{fb_col}'")
                    break

        # Rescue: date embedded in the narration text
        if not date_parsed and narration_raw:
            dm = re.search(
                r'\b\d{1,2}[-/\.\s]+[A-Za-z]{3,}(?:\s*,?\s*\d{4})?\b'
                r'|\b\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?\b',
                narration_raw,
            )
            if dm:
                extracted = dm.group(0).replace(",", "").strip()
                if len(extracted) <= 7 and not re.search(r'\d{4}', extracted):
                    extracted += " 2024"
                rescue_fmts = (
                    bank_fmt.get("date_formats", []) +
                    ["%d %b %Y", "%d %b %y", "%d/%m/%Y", "%d-%m-%Y"]
                )
                rescued = parse_date(extracted, rescue_fmts)
                if rescued:
                    date_parsed   = rescued
                    narration_raw = narration_raw.replace(dm.group(0), "").strip()
                    row_warnings.append("date rescued from narration")

        # ── Amounts ───────────────────────────────────────────────────────
        debit   = parse_amount(_get_col(row, bank_fmt.get("debit_col",   "")))
        credit  = parse_amount(_get_col(row, bank_fmt.get("credit_col",  "")))
        balance = parse_amount(_get_col(row, bank_fmt.get("balance_col", "")))

        # Rescue: amounts embedded in narration when both columns are zero
        if debit == 0.0 and credit == 0.0 and narration_raw:
            am = re.search(
                r'(?:₹|rs\.?|inr)?[\s]*(\d+(?:,\d+)*\.\d{2})\b',
                narration_raw, re.IGNORECASE,
            )
            if am:
                rescued_amt   = float(am.group(1).replace(",", ""))
                is_credit_kw  = any(
                    kw in narration_raw.lower()
                    for kw in ["received", "added", "credited", "+", "refund"]
                )
                credit        = rescued_amt if is_credit_kw else 0.0
                debit         = 0.0         if is_credit_kw else rescued_amt
                narration_raw = narration_raw.replace(am.group(0), "").strip()
                row_warnings.append("amount rescued from narration")

        # ── Ref / UTR ─────────────────────────────────────────────────────
        utr_ref = str(_get_col(row, bank_fmt.get("ref_col", "")) or "").strip()
        if utr_ref.lower() in ("nan", "none", "null", ""):
            um = re.search(
                r'\b(?:txn|ref|id|utr)?[\s:-]*([A-Za-z0-9]{8,22})\b',
                narration_raw, re.IGNORECASE,
            )
            if um:
                c = um.group(1)
                if any(ch.isdigit() for ch in c) and len(c) >= 8:
                    utr_ref = c

        if not date_parsed:
            row_warnings.append("unparseable date")

        narration_clean = re.sub(
            r'(?i)\b(?:rs\.?|inr|₹|txn id|ref no)\b\s*', '', narration_raw
        ).strip()



        # ── Time Rescue (Extract from Narration) ──────────────────────────
        time_parsed = "00:00:00"
        if narration_raw:
            # Matches formats: 14:30:05, 14:30, 02:15 PM, 2:15PM, 14.30.05
            tm = re.search(
                r'\b([01]?\d|2[0-3])[:.]([0-5]\d)(?:[:.]([0-5]\d))?\s*([APap][. ]?[Mm]\.?)?\b',
                narration_raw
            )
            if tm:
                raw_t = tm.group(0)
                try:
                    # Replace dot separators between digits (14.30.00 -> 14:30:00) but keep AM/PM tokens intact.
                    cleaned_t = re.sub(r'(?<=\d)\.(?=\d)', ':', raw_t)
                    cleaned_t = cleaned_t.replace('.', '')
                    time_parsed = pd.to_datetime(cleaned_t).strftime("%H:%M:%S")
                    
                    # Optional: Remove the time from the narration to keep it clean
                    narration_raw = narration_raw.replace(raw_t, "").strip()
                except Exception:
                    pass

        if not date_parsed:
            row_warnings.append("unparseable date")

        narration_clean = re.sub(
            r'(?i)\b(?:rs\.?|inr|₹|txn id|ref no)\b\s*', '', narration_raw
        ).strip()



        output_rows.append({
            "account_id":        account_id,
            "account_holder":    account_holder,
            "bank_name":         bank_name,
            "date":              date_parsed,
            "time":              time_parsed,
            "narration":         narration_clean,
            "channel":           infer_channel(narration_clean),
            "debit":             round(debit, 2),
            "credit":            round(credit, 2),
            "balance":           round(balance, 2),
            "utr_ref":           utr_ref,
            "counterparty_name": extract_counterparty(narration_clean),
            "source_file":       os.path.basename(source_file),
            "source_format":     source_format,
            "ingestion_warnings": " | ".join(row_warnings) if row_warnings else "",
        })

    normalized = pd.DataFrame(output_rows, columns=UNIFIED_SCHEMA)

    # ── Drop rows that have NO date AND NO monetary amounts ───────────────
    # (less aggressive than the original OR condition — a row with valid
    #  amounts but a garbled date is still useful for investigators)
    before = len(normalized)
    normalized = normalized[
        ~(
            (normalized["date"]   == "")  &
            (normalized["debit"]  == 0.0) &
            (normalized["credit"] == 0.0)
        )
    ].reset_index(drop=True)
    dropped = before - len(normalized)
    if dropped:
        warnings.append(f"Dropped {dropped} junk rows (no date AND no amounts)")

    warnings = _count_row_warnings(normalized, warnings)
    return normalized, warnings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_col(row, col_name: str):
    try:
        val = row[col_name]
        return None if pd.isna(val) else val
    except (KeyError, TypeError):
        return None


def _extract_account_holder(header_text: str) -> str:
    patterns = [
        r"Account\s+(?:Holder|Name)\s*[:\s]+([A-Za-z\s\.]{3,39})"
        r"(?:\n|IFSC|Account|$)",
        r"Customer\s+Name\s*[:\s]+([A-Za-z\s\.]{3,39})(?:\n|$)",
        r"Statement\s+of\s*[:\s]+([A-Za-z\s\.]{3,39})(?:\n|$)",
        r"Hello,\s*([A-Za-z\s\.]{3,39})(?:\n|$)",
        r"Name\s*[:\s]+([A-Za-z\s\.]{3,39})(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, header_text, re.IGNORECASE)
        if m:
            c = m.group(1).strip()
            if 3 <= len(c) < 40:
                return c
    return ""


def _extract_account_id(filename: str) -> str:
    m = re.search(r'(ACC\d+)', os.path.basename(filename))
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(filename))[0]


def _count_row_warnings(df: pd.DataFrame, existing: list) -> list:
    n = (df["ingestion_warnings"] != "").sum()
    if n:
        existing.append(f"{n} rows had parsing warnings")
    return existing