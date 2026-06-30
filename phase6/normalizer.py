"""
Phase 6 — Normalizer  (v2)

Converts any raw parsed DataFrame into the unified schema.

New in v2:
  • Paytm XLSX: COD_DRCR direction column, COD_ACCT_NO_FROM/TO graph edges
  • BOB/Federal XLSX: BENEF/REMIT ACCT NO + IFSC extraction
  • IDFC balance: strips "Cr"/"Dr" suffix from balance column
  • counterparty_account + counterparty_ifsc now populated for all formats
  • Improved account_id extraction from statement header text
  • Balance delta inference for BOB 3-column PDFs
"""

import os
import re
import pandas as pd
try:
    from phase6.ingestion_config import (
        UNIFIED_SCHEMA,
        PAYTM_DRCR_COL, PAYTM_FROM_COL, PAYTM_TO_COL, PAYTM_AMOUNT_COL,
        BOB_BENEF_ACCT_COL, BOB_BENEF_IFSC_COL, BOB_BENEF_NAME_COL,
    )
except ImportError:
    from ingestion_config import (
        UNIFIED_SCHEMA,
        PAYTM_DRCR_COL, PAYTM_FROM_COL, PAYTM_TO_COL, PAYTM_AMOUNT_COL,
        BOB_BENEF_ACCT_COL, BOB_BENEF_IFSC_COL, BOB_BENEF_NAME_COL,
    )
try:
    from phase6.schema_detector import (
        detect_bank, find_header_row, assign_column_roles,
        assign_column_roles_by_keywords,
        compute_running_balance, parse_date, parse_amount,
        infer_channel, extract_counterparty,
        extract_counterparty_account, extract_counterparty_ifsc,
    )
except ImportError:
    from schema_detector import (
        detect_bank, find_header_row, assign_column_roles,
        assign_column_roles_by_keywords,
        compute_running_balance, parse_date, parse_amount,
        infer_channel, extract_counterparty,
        extract_counterparty_account, extract_counterparty_ifsc,
    )


def normalize(
    raw_df: pd.DataFrame,
    header_text: str,
    source_file: str,
    source_format: str,
    provided_account_id: str = None,
) -> tuple[pd.DataFrame, list]:
    """
    Main entry point. Returns (normalized_df, warnings_list).
    """
    warnings: list[str] = []

    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=UNIFIED_SCHEMA), ["Empty input"]

    # ── Step 1: Find real header row (skip for OCR/image — already handled) ──
    if source_format not in ("image",):
        header_row_idx = find_header_row(raw_df)
        if header_row_idx > 0:
            new_cols = [str(v).strip() for v in raw_df.iloc[header_row_idx].values]
            raw_df = raw_df.iloc[header_row_idx + 1:].copy()
            raw_df.columns = new_cols
            raw_df = raw_df.reset_index(drop=True)

    # Normalize embedded newlines in headers and cell values (fix PDFs with split cells)
    raw_df.columns = [str(c).replace('\n', ' ').strip() for c in raw_df.columns]
    # Clean cell values per-column (avoid applymap issues on some pandas builds)
    for c in raw_df.columns:
        try:
            raw_df[c] = raw_df[c].astype(str).str.replace('\n', ' ').str.replace(r'-\s+', '-', regex=True).str.replace(r'\s+-\s+', '-', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
        except Exception:
            raw_df[c] = raw_df[c].astype(str).map(lambda v: str(v).replace('\n', ' ').replace(' - ', '-').replace('- ', '-').replace(' -', '-').strip())

    raw_df = raw_df.dropna(how="all").reset_index(drop=True)
    raw_df = raw_df.loc[:, raw_df.notna().any()]

    if raw_df.empty:
        return pd.DataFrame(columns=UNIFIED_SCHEMA), [
            "No data rows after header detection"
        ]

    # ── Step 2: Detect bank ───────────────────────────────────────────────────
    bank_name, bank_method = detect_bank(header_text)
    if bank_name == "Unknown Bank":
        bank_name, bank_method = detect_bank(os.path.basename(source_file))

    # ── Step 3: Check for special-schema formats ──────────────────────────────
    # Paytm: has COD_DRCR column that encodes direction
    if PAYTM_DRCR_COL in raw_df.columns:
        return _normalize_paytm(
            raw_df, header_text, source_file, source_format,
            bank_name, provided_account_id, warnings
        )

    # BOB / Federal XLSX: has BENEF/REMIT columns
    if BOB_BENEF_ACCT_COL in raw_df.columns or "BENEF/REMIT ACCT NO" in raw_df.columns:
        return _normalize_bob_xlsx(
            raw_df, header_text, source_file, source_format,
            bank_name, provided_account_id, warnings
        )

    # ── Step 4: Column role detection (3-tier) ────────────────────────────────
    try:
        roles = assign_column_roles(raw_df.columns.tolist(), df=raw_df)
    except Exception as e:
        warnings.append(f"Column role detection error: {e}")
        try:
            roles = assign_column_roles_by_keywords(raw_df.columns.tolist())
        except Exception:
            return pd.DataFrame(columns=UNIFIED_SCHEMA), warnings + [
                "Column detection failed completely"
            ]

    # Validate date column
    date_col = roles.get("date")
    if date_col:
        sample = raw_df[date_col].dropna().astype(str).head(10)
        date_hits = sum(1 for v in sample if parse_date(v))
        if date_hits < 2:
            for col in raw_df.columns:
                if col == date_col:
                    continue
                s2 = raw_df[col].dropna().astype(str).head(10)
                hits2 = sum(1 for v in s2 if parse_date(v))
                if hits2 > date_hits:
                    date_hits, date_col = hits2, col
            roles["date"] = date_col

    warnings.append(
        f"Bank: {bank_name} ({bank_method}) | "
        f"date={roles.get('date')} | nar={roles.get('narration')} | "
        f"dr={roles.get('debit') or roles.get('amount')} | "
        f"cr={roles.get('credit') or roles.get('amount')} | "
        f"bal={roles.get('balance')}"
    )

    if not roles.get("date"):
        warnings.append(
            f"CRITICAL: No date column found. Columns: {raw_df.columns.tolist()}"
        )
        return pd.DataFrame(columns=UNIFIED_SCHEMA), warnings

    # ── Step 5: Compute balance if missing ───────────────────────────────────
    balance_col = roles.get("balance")
    balance_computed = False
    if not balance_col:
        dc = roles.get("debit") or roles.get("amount")
        cc = roles.get("credit") or roles.get("amount")
        if dc and cc:
            raw_df["_computed_balance"] = compute_running_balance(raw_df, dc, cc)
            balance_col = "_computed_balance"
            balance_computed = True
            warnings.append("Balance computed from running debit/credit")

    # ── Step 6: Metadata ─────────────────────────────────────────────────────
    account_holder = _extract_account_holder(header_text)
    account_id = (
        provided_account_id
        or _extract_account_id_from_text(header_text)
        or _extract_account_id_from_filename(source_file)
    )

    # ── Step 7: Row-by-row normalization ─────────────────────────────────────
    output_rows = []
    prev_balance = None
    skipped = 0

    for _, row in raw_df.iterrows():
        date_str = parse_date(_get(row, roles.get("date")))
        # Fallback: if assigned date column didn't parse, try finding
        # a date-like token anywhere in the row (some PDFs split columns)
        if not date_str:
            all_text = " ".join(str(_get(row, c) or "") for c in raw_df.columns)
            m = re.search(r"\d{1,2}[\/.\-]\d{1,2}[\/.\-]\d{2,4}", all_text)
            if m:
                date_str = parse_date(m.group(0))
        if not date_str:
            skipped += 1
            continue

        narration = str(_get(row, roles.get("narration")) or "").strip()
        if not narration:
            narration = _build_narration_from_unmapped(row, roles)
        narration = narration.upper()

        debit, credit = 0.0, 0.0

        if roles.get("debit") and roles.get("credit"):
            debit,  d_sign = parse_amount(_get(row, roles["debit"]))
            credit, c_sign = parse_amount(_get(row, roles["credit"]))
            if d_sign == "+":
                credit, debit = debit, 0.0
            if c_sign == "-":
                debit, credit = credit, 0.0
        elif roles.get("amount"):
            amt, sign = parse_amount(_get(row, roles["amount"]))
            if sign == "+":
                credit = amt
            elif sign == "-":
                debit = amt
            else:
                nl = narration.lower()
                if any(w in nl for w in [
                    "cr-", " cr ", "credit", "deposit", "received",
                    "salary", "refund", "cashback", "interest credit",
                ]):
                    credit = amt
                else:
                    debit = amt
        elif roles.get("drcr"):
            # Generic DR/CR direction column
            drcr_val = str(_get(row, roles["drcr"]) or "").strip().upper()
            amt_col = roles.get("debit") or roles.get("credit") or roles.get("amount")
            if amt_col:
                amt, _ = parse_amount(_get(row, amt_col))
                if drcr_val in ("C", "CR", "CREDIT"):
                    credit = amt
                else:
                    debit = amt

        balance = 0.0
        if balance_col:
            balance, b_sign = parse_amount(_get(row, balance_col))
            # IDFC: balance with "Dr" suffix means overdrawn (negative)
            if b_sign == "-":
                balance = -balance

        # Infer debit/credit from balance delta for BOB 3-col PDFs
        if debit == 0.0 and credit == 0.0 and balance != 0.0 and prev_balance is not None:
            delta = round(balance - prev_balance, 2)
            if delta > 0:
                credit = delta
            elif delta < 0:
                debit = abs(delta)

        if debit == 0.0 and credit == 0.0 and balance == 0.0:
            skipped += 1
            prev_balance = balance
            continue

        prev_balance = balance

        utr_ref = str(_get(row, roles.get("ref")) or "").strip()
        time_str = _extract_time(row, roles)

        # Counterparty extraction
        cp_name    = extract_counterparty(narration)
        cp_account = extract_counterparty_account(narration)
        cp_ifsc    = extract_counterparty_ifsc(narration)

        output_rows.append({
            "account_id":          account_id,
            "account_holder":      account_holder,
            "bank_name":           bank_name,
            "date":                date_str,
            "time":                time_str,
            "narration":           narration,
            "channel":             infer_channel(narration),
            "debit":               round(debit, 2),
            "credit":              round(credit, 2),
            "balance":             round(balance, 2),
            "utr_ref":             utr_ref,
            "counterparty_name":   cp_name,
            "counterparty_account": cp_account,
            "counterparty_ifsc":   cp_ifsc,
            "source_file":         os.path.basename(source_file),
            "source_format":       source_format,
            "ingestion_warnings":  "balance_computed" if balance_computed else "",
        })

    if skipped:
        warnings.append(f"Skipped {skipped} non-transaction rows")
    if not output_rows:
        warnings.append("No transaction rows could be extracted")
        return pd.DataFrame(columns=UNIFIED_SCHEMA), warnings

    return pd.DataFrame(output_rows, columns=UNIFIED_SCHEMA), warnings


# ===========================================================================
# Special normalizer: Paytm Payments Bank XLSX
# COD_DRCR='C' → credit, 'D' → debit
# COD_ACCT_NO_FROM / COD_ACCT_NO_TO → direct graph edges
# ===========================================================================
def _normalize_paytm(raw_df, header_text, source_file, source_format,
                      bank_name, provided_account_id, warnings) -> tuple:
    warnings.append("Paytm XLSX: using COD_DRCR direction + FROM/TO accounts")
    bank_name = bank_name if bank_name != "Unknown Bank" else "Paytm Payments Bank"

    account_id = (
        provided_account_id
        or _extract_account_id_from_text(header_text)
        or _extract_account_id_from_filename(source_file)
    )

    # Detect account ID from the ACCT_NO column itself
    if "COD_ACCT_NO" in raw_df.columns:
        acct_series = raw_df["COD_ACCT_NO"].dropna()
        if not acct_series.empty:
            account_id = str(acct_series.iloc[0]).strip()

    output_rows = []
    for _, row in raw_df.iterrows():
        date_str = parse_date(_get(row, "DAT_TXN_PROCESSING")
                              or _get(row, "DAT_TXN_VALUE"))
        if not date_str:
            continue

        drcr = str(_get(row, PAYTM_DRCR_COL) or "").strip().upper()
        try:
            amt = abs(float(str(_get(row, PAYTM_AMOUNT_COL) or "0")
                            .replace(",", "")))
        except (ValueError, TypeError):
            amt = 0.0

        if amt == 0.0:
            continue

        debit  = amt if drcr == "D" else 0.0
        credit = amt if drcr == "C" else 0.0

        narration_raw = (
            _get(row, "TXT_TXN_DESC")
            or _get(row, "TXT_TRAN_PARTICULAR")
            or _get(row, "TXT_TXN_NARRATIVE_TO")
            or ""
        )
        narration = str(narration_raw).upper().strip()

        # Direct counterparty from FROM/TO columns
        if drcr == "D":
            cp_account = str(_get(row, "COD_ACCT_NO_TO") or "").strip()
        else:
            cp_account = str(_get(row, "COD_ACCT_NO_FROM") or "").strip()

        utr_ref = str(_get(row, "REF_TXN_NO") or "").strip()

        output_rows.append({
            "account_id":          account_id,
            "account_holder":      _extract_account_holder(header_text),
            "bank_name":           bank_name,
            "date":                date_str,
            "time":                "00:00:00",
            "narration":           narration,
            "channel":             infer_channel(narration),
            "debit":               round(debit, 2),
            "credit":              round(credit, 2),
            "balance":             0.0,  # Paytm format has no running balance
            "utr_ref":             utr_ref,
            "counterparty_name":   extract_counterparty(narration),
            "counterparty_account": cp_account,
            "counterparty_ifsc":   str(_get(row, "COD_PROD") or "").strip(),
            "source_file":         os.path.basename(source_file),
            "source_format":       source_format,
            "ingestion_warnings":  "paytm_no_balance",
        })

    if not output_rows:
        warnings.append("Paytm: no rows extracted")
        return pd.DataFrame(columns=UNIFIED_SCHEMA), warnings

    warnings.append(f"Paytm: {len(output_rows)} transactions")
    return pd.DataFrame(output_rows, columns=UNIFIED_SCHEMA), warnings


# ===========================================================================
# Special normalizer: BOB / Federal / Bandhan XLSX (rich BENEF columns)
# ===========================================================================
def _normalize_bob_xlsx(raw_df, header_text, source_file, source_format,
                         bank_name, provided_account_id, warnings) -> tuple:
    warnings.append("BOB/Federal XLSX: extracting BENEF/REMIT counterparty data")

    # Find actual header row
    header_row_idx = find_header_row(raw_df)
    if header_row_idx > 0:
        new_cols = [str(v).strip() for v in raw_df.iloc[header_row_idx].values]
        raw_df = raw_df.iloc[header_row_idx + 1:].copy()
        raw_df.columns = new_cols
        raw_df = raw_df.reset_index(drop=True)
    raw_df = raw_df.dropna(how="all").reset_index(drop=True)

    roles = assign_column_roles(raw_df.columns.tolist(), df=raw_df)

    account_id = (
        provided_account_id
        or _extract_account_id_from_text(header_text)
        or _extract_account_id_from_filename(source_file)
    )
    # BOB XLSX typically has "ACCOUNT NO." column
    for acct_col in ("ACCOUNT NO.", "ACCOUNT NO", "ACCT_NO"):
        if acct_col in raw_df.columns:
            s = raw_df[acct_col].dropna()
            if not s.empty:
                account_id = str(s.iloc[0]).strip()
                break

    # Find BENEF columns (flexible naming)
    benef_acct_col = None
    benef_ifsc_col = None
    benef_name_col = None
    for col in raw_df.columns:
        cl = col.upper()
        if "BENEF" in cl and "ACCT" in cl and "NAME" not in cl:
            benef_acct_col = col
        if "BENEF" in cl and "IFSC" in cl:
            benef_ifsc_col = col
        if "BENEF" in cl and "NAME" in cl:
            benef_name_col = col
        if "REMIT" in cl and "ACCT" in cl and "NAME" not in cl:
            benef_acct_col = benef_acct_col or col
        if "REMIT" in cl and "IFSC" in cl:
            benef_ifsc_col = benef_ifsc_col or col

    account_holder = _extract_account_holder(header_text)
    if not account_holder and "ACCOUNT NAME" in raw_df.columns:
        s = raw_df["ACCOUNT NAME"].dropna()
        account_holder = str(s.iloc[0]).strip() if not s.empty else ""

    output_rows = []
    skipped = 0
    for _, row in raw_df.iterrows():
        date_str = parse_date(_get(row, roles.get("date")))
        if not date_str:
            skipped += 1
            continue

        narration = str(_get(row, roles.get("narration")) or "").strip().upper()
        if not narration:
            narration = _build_narration_from_unmapped(row, roles)

        debit, credit = 0.0, 0.0
        if roles.get("debit") and roles.get("credit"):
            debit,  ds = parse_amount(_get(row, roles["debit"]))
            credit, cs = parse_amount(_get(row, roles["credit"]))
            if ds == "+":
                credit, debit = debit, 0.0
            if cs == "-":
                debit, credit = credit, 0.0

            # BOB uses BALANCE INDICATOR 'C'/'D' — override if present
            bi_col = roles.get("drcr")
            if bi_col:
                bi = str(_get(row, bi_col) or "").strip().upper()
                # If both debit & credit are non-zero, use indicator to zero one
                if bi in ("C", "CR") and debit > 0:
                    credit, debit = debit + credit, 0.0
                elif bi in ("D", "DR") and credit > 0:
                    debit, credit = debit + credit, 0.0

        elif roles.get("amount"):
            amt, sign = parse_amount(_get(row, roles["amount"]))
            bi_col = roles.get("drcr")
            if bi_col:
                bi = str(_get(row, bi_col) or "").strip().upper()
                if bi in ("C", "CR"):
                    credit = amt
                else:
                    debit = amt
            elif sign == "+":
                credit = amt
            elif sign == "-":
                debit = amt

        balance, _ = parse_amount(_get(row, roles.get("balance")))

        if debit == 0.0 and credit == 0.0 and balance == 0.0:
            skipped += 1
            continue

        utr_ref = str(_get(row, roles.get("ref")) or "").strip()

        # Extract counterparty from BENEF columns (definitive)
        cp_account = str(_get(row, benef_acct_col) or "").strip() if benef_acct_col else ""
        cp_ifsc    = str(_get(row, benef_ifsc_col) or "").strip() if benef_ifsc_col else ""
        cp_name    = str(_get(row, benef_name_col) or "").strip() if benef_name_col else ""

        # Fall back to narration extraction
        if not cp_account:
            cp_account = extract_counterparty_account(narration)
        if not cp_ifsc:
            cp_ifsc = extract_counterparty_ifsc(narration)
        if not cp_name:
            cp_name = extract_counterparty(narration)

        output_rows.append({
            "account_id":          account_id,
            "account_holder":      account_holder,
            "bank_name":           bank_name,
            "date":                date_str,
            "time":                _extract_time(row, roles),
            "narration":           narration,
            "channel":             infer_channel(narration),
            "debit":               round(debit, 2),
            "credit":              round(credit, 2),
            "balance":             round(balance, 2),
            "utr_ref":             utr_ref,
            "counterparty_name":   cp_name,
            "counterparty_account": cp_account,
            "counterparty_ifsc":   cp_ifsc,
            "source_file":         os.path.basename(source_file),
            "source_format":       source_format,
            "ingestion_warnings":  "",
        })

    if skipped:
        warnings.append(f"BOB/Federal: skipped {skipped} non-transaction rows")
    if not output_rows:
        warnings.append("BOB/Federal: no rows extracted")
        return pd.DataFrame(columns=UNIFIED_SCHEMA), warnings

    warnings.append(f"BOB/Federal: {len(output_rows)} transactions")
    return pd.DataFrame(output_rows, columns=UNIFIED_SCHEMA), warnings


# ===========================================================================
# Helpers
# ===========================================================================
def _get(row, col_name):
    if not col_name:
        return None
    try:
        val = row[col_name]
        return None if (isinstance(val, float) and pd.isna(val)) else val
    except (KeyError, TypeError):
        return None


def _build_narration_from_unmapped(row, roles: dict) -> str:
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
    import re
    for col in row.index:
        col_lower = str(col).lower()
        if "time" not in col_lower and "timestamp" not in col_lower:
            continue
        val = _get(row, col)
        if val is None:
            continue
        m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(val))
        if m:
            h, mi, s = m.group(1), m.group(2), m.group(3) or "00"
            return f"{int(h):02d}:{mi}:{s}"

    date_val = _get(row, roles.get("date"))
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(date_val or ""))
    if m:
        h, mi, s = m.group(1), m.group(2), m.group(3) or "00"
        return f"{int(h):02d}:{mi}:{s}"
    return "00:00:00"


def _extract_account_holder(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"(?:account holder|account name|name|customer)[:\s]+([A-Za-z][A-Za-z\s\.]{2,40}?)(?:\n|$|account|ifsc)",
        r"(?:A/C Name|Ac Name)\s*:\s*([A-Za-z\s\.MR\.MRS\.]{3,40})",
        r"(?:Accountholder Name)\s*:\s*(MR\.?\s*[A-Za-z\s\.]{3,40})",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()
    return ""


def _extract_account_id_from_text(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"(?:account\s*(?:no|number|id)|a/c\s*(?:no|number)|acct\s*(?:no|number)|ECS A/c No\.?)\s*[:\-]?\s*([A-Z0-9X*]{6,24})",
        r"\b(\d{9,18})\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return re.sub(r"[^A-Za-z0-9X*]", "", m.group(1))[:24]
    return ""


def _extract_account_id_from_filename(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    # If filename IS the account number (most common in dataset)
    m = re.match(r"^(\d{8,18})", base)
    if m:
        return m.group(1)
    m = re.search(r"(ACC\d+)", base)
    if m:
        return m.group(1)
    return re.sub(r"[^A-Za-z0-9_]", "_", base)[:20]