"""
Phase 6 - Normalizer

Handles any financial dataset — not just bank statements.
Uses three-tier column detection and gracefully handles:
  - Missing balance column (computed from running totals)
  - Single-amount column with +/- or CR/DR notation
  - Hindi/regional language headers
  - Completely unknown column names (Col1, A, B...)
  - Tally, QuickBooks, ERP, court/police formats
"""

import os
import re
import pandas as pd
from ingestion_config import UNIFIED_SCHEMA
from schema_detector import (
    detect_bank, find_header_row, assign_column_roles,
    compute_running_balance, parse_date, parse_amount,
    infer_channel, extract_counterparty,
)


def normalize(
    raw_df: pd.DataFrame,
    header_text: str,
    source_file: str,
    source_format: str,
    provided_account_id: str = None,
) -> tuple[pd.DataFrame, list]:
    warnings = []

    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=UNIFIED_SCHEMA), ["Empty input"]

    # -----------------------------------------------------------------------
    # Step 1: Find real header row
    # OCR (image) sources skip this — the OCR bbox parser already handles
    # header detection. Running find_header_row on garbled OCR column names
    # makes things worse, not better.
    # -----------------------------------------------------------------------
    if source_format not in ("image",):
        header_row_idx = find_header_row(raw_df)
        if header_row_idx > 0:
            new_cols = [str(v).strip() for v in raw_df.iloc[header_row_idx].values]
            raw_df = raw_df.iloc[header_row_idx + 1:].copy()
            raw_df.columns = new_cols
            raw_df = raw_df.reset_index(drop=True)

    raw_df = raw_df.dropna(how="all").reset_index(drop=True)
    raw_df = raw_df.loc[:, raw_df.notna().any()]

    if raw_df.empty:
        return pd.DataFrame(columns=UNIFIED_SCHEMA), ["No data rows after header detection"]

    # -----------------------------------------------------------------------
    # Step 2: Detect bank name
    # -----------------------------------------------------------------------
    bank_name, bank_detection_method = detect_bank(header_text)
    if bank_name == "Unknown Bank":
        bank_name, bank_detection_method = detect_bank(os.path.basename(source_file))

    # -----------------------------------------------------------------------
    # Step 3: Three-tier column role assignment
    # -----------------------------------------------------------------------
    try:
        roles = assign_column_roles(raw_df.columns.tolist(), df=raw_df)
    except Exception as e:
        warnings.append(f"Column role detection error: {e} — attempting keyword-only fallback")
        try:
            from schema_detector import assign_column_roles_by_keywords
            roles = assign_column_roles_by_keywords(raw_df.columns.tolist())
        except Exception:
            return pd.DataFrame(columns=UNIFIED_SCHEMA), warnings + ["Column detection failed completely"]

    # -----------------------------------------------------------------------
    # Step 3b: Validate date column — OCR often splits "Post Date" into
    # "Post" + "Date" columns. Pick the column with actual date values.
    # -----------------------------------------------------------------------
    date_col = roles.get("date")
    if date_col:
        from schema_detector import parse_date as _parse_date
        sample = raw_df[date_col].dropna().astype(str).head(10)
        date_hits = sum(1 for v in sample if _parse_date(v) and not _parse_date(v) == v)
        if date_hits < 2:
            # Try all other columns to find one with actual date content
            for col in raw_df.columns:
                if col == date_col:
                    continue
                s = raw_df[col].dropna().astype(str).head(10)
                hits = sum(1 for v in s if _parse_date(v) and not _parse_date(v) == v)
                if hits > date_hits:
                    date_hits = hits
                    date_col = col
            roles["date"] = date_col

    # Log what was detected
    warnings.append(
        f"Bank: {bank_name} (via {bank_detection_method}) | "
        f"date={roles.get('date')} | narration={roles.get('narration')} | "
        f"debit={roles.get('debit') or roles.get('amount')} | "
        f"credit={roles.get('credit') or roles.get('amount')} | "
        f"balance={roles.get('balance')}"
    )

    # Critical check: need at least a date column
    if not roles.get("date"):
        warnings.append(f"CRITICAL: Could not identify date column. Columns: {raw_df.columns.tolist()}")
        return pd.DataFrame(columns=UNIFIED_SCHEMA), warnings

    # -----------------------------------------------------------------------
    # Step 4: Compute balance if missing
    # -----------------------------------------------------------------------
    balance_col = roles.get("balance")
    balance_computed = False
    if not balance_col:
        debit_c  = roles.get("debit")  or roles.get("amount")
        credit_c = roles.get("credit") or roles.get("amount")
        if debit_c and credit_c:
            raw_df["_computed_balance"] = compute_running_balance(
                raw_df, debit_c, credit_c, start_balance=0.0
            )
            balance_col = "_computed_balance"
            balance_computed = True
            warnings.append("Balance column not found — computed from running debit/credit totals")

    # -----------------------------------------------------------------------
    # Step 5: Metadata
    # -----------------------------------------------------------------------
    account_holder = _extract_account_holder(header_text)
    account_id = (
        provided_account_id
        or _extract_account_id_from_text(header_text)
        or _extract_account_id_from_filename(source_file)
    )

    # -----------------------------------------------------------------------
    # Step 6: Row-by-row normalization
    # -----------------------------------------------------------------------
    output_rows = []
    skipped = 0

    for _, row in raw_df.iterrows():

        # Date — skip rows with no parseable date
        date_str = parse_date(_get(row, roles.get("date")))
        if not date_str:
            skipped += 1
            continue

        # Narration — if no narration column was identified, fall back to
        # concatenating whatever unmapped columns this row has, rather
        # than leaving every transaction's narration blank.
        narration = str(_get(row, roles.get("narration")) or "").strip()
        if not narration:
            narration = _build_narration_from_unmapped(row, roles)
        narration = narration.upper()

        # Amounts
        debit, credit = 0.0, 0.0

        if roles.get("debit") and roles.get("credit"):
            # Standard two-column format
            debit,  d_sign = parse_amount(_get(row, roles["debit"]))
            credit, c_sign = parse_amount(_get(row, roles["credit"]))
            # If debit column had CR sign, it's actually a credit reversal
            if d_sign == "+":
                credit, debit = debit, 0.0
            if c_sign == "-":
                debit, credit = credit, 0.0

        elif roles.get("amount"):
            # Single amount column
            amt, sign = parse_amount(_get(row, roles["amount"]))
            if sign == "+":
                credit = amt
            elif sign == "-":
                debit = amt
            else:
                # Infer from narration context
                nar_lower = narration.lower()
                if any(w in nar_lower for w in [
                    "cr-", " cr ", "credit", "deposit", "received",
                    "salary", "refund", "cashback", "interest credit",
                ]):
                    credit = amt
                else:
                    debit = amt

        # Balance
        balance = 0.0
        if balance_col:
            b, _ = parse_amount(_get(row, balance_col))
            balance = b

        # Skip pure zero rows (header repeats, subtotal rows)
        if debit == 0.0 and credit == 0.0 and balance == 0.0:
            skipped += 1
            continue

        utr_ref = str(_get(row, roles.get("ref")) or "").strip()
        time_str = _extract_time(row, roles)



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
            "counterparty_name": extract_counterparty(narration),
            "source_file":       os.path.basename(source_file),
            "source_format":     source_format,
            "ingestion_warnings": "balance_computed" if balance_computed else "",
        })

    if skipped:
        warnings.append(f"Skipped {skipped} non-transaction rows")

    if not output_rows:
        warnings.append("No transaction rows could be extracted")
        return pd.DataFrame(columns=UNIFIED_SCHEMA), warnings

    return pd.DataFrame(output_rows, columns=UNIFIED_SCHEMA), warnings


def _get(row, col_name):
    if not col_name:
        return None
    try:
        val = row[col_name]
        return None if (isinstance(val, float) and pd.isna(val)) else val
    except (KeyError, TypeError):
        return None


def _build_narration_from_unmapped(row, roles: dict) -> str:
    """
    Fallback when no narration column was identified: concatenate
    whatever columns this row has that weren't already mapped to a known
    role, so a transaction's narration is never silently left blank just
    because the file's narration column had an unrecognized header name.
    """
    mapped = {col for col in roles.values() if col}
    parts = []
    for col, val in row.items():
        if col in mapped or pd.isna(val):
            continue
        text = str(val).strip()
        if text and text.lower() not in ("nan", "none"):
            parts.append(text)
    return " ".join(parts[:4])


def _extract_time(row, roles: dict) -> str:
    """
    Pull a real HH:MM:SS out of a dedicated time/timestamp column, or out
    of the date column itself if it embeds a time component (e.g.
    "2026-01-15 14:32:00"). Falls back to "00:00:00" only if neither
    yields anything - most formats genuinely have no time information,
    but plenty do and shouldn't be flattened to midnight.
    """
    for col in row.index:
        col_lower = str(col).lower()
        if "time" not in col_lower and "timestamp" not in col_lower:
            continue
        val = _get(row, col)
        if not val:
            continue
        m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(val))
        if m:
            hour, minute, second = m.group(1), m.group(2), m.group(3) or "00"
            return f"{int(hour):02d}:{minute}:{second}"

    date_val = _get(row, roles.get("date"))
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(date_val or ""))
    if m:
        hour, minute, second = m.group(1), m.group(2), m.group(3) or "00"
        return f"{int(hour):02d}:{minute}:{second}"
    return "00:00:00"


def _extract_account_holder(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"(?:account holder|account name|name|customer)[:\s]+([A-Za-z][A-Za-z\s\.]{2,40}?)(?:\n|$|account|ifsc)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()
    return ""


def _extract_account_id_from_text(text: str) -> str:
    """
    Pull the actual account number out of the statement header text
    (e.g. "Account Number: 14729595422743"), checked BEFORE falling back
    to the filename. Most real statements carry the account number
    somewhere in their header; relying on the filename alone misses that
    whenever the file is named something generic like "statement.csv".
    """
    if not text:
        return ""
    patterns = [
        r"(?:account\s*(?:no|number|id)|a/c\s*(?:no|number)|acct\s*(?:no|number))\s*[:\-]?\s*([A-Z0-9X*]{6,24})",
        r"\b(\d{9,18})\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return re.sub(r"[^A-Za-z0-9X*]", "", m.group(1))[:24]
    return ""


def _extract_account_id_from_filename(filename: str) -> str:
    m = re.search(r"(ACC\d+)", os.path.basename(filename))
    if m:
        return m.group(1)
    return re.sub(r"[^A-Za-z0-9_]", "_",
                  os.path.splitext(os.path.basename(filename))[0])[:20]