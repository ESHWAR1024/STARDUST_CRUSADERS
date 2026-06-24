"""
Phase 6 - Schema Detector

Three-tier column detection:
  Tier 1: Keyword matching (column header contains known fragment)
  Tier 2: Content sampling (column data looks like dates/amounts)
  Tier 3: Positional fallback (first col = date, last = balance, etc.)

Also handles:
  - Balance column missing entirely (compute from running total)
  - Hindi/regional language column names
  - Positive/negative single amount column
  - Tally, QuickBooks, ERP, custom formats
"""

import re
import json
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime
from ingestion_config import (
    COLUMN_ROLE_KEYWORDS, CONTENT_PATTERNS,
    HINDI_COLUMN_MAP, BANK_NAME_KEYWORDS,
    DATE_FORMATS, CHANNEL_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Bank name detection
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Bank detection — IFSC lookup (primary, no hardcoded bank list) with
# keyword/regex text scanning as a fallback for when no IFSC is found or
# the lookup service can't be reached (e.g. offline investigation
# environment with no internet access).
# ---------------------------------------------------------------------------
_IFSC_LOOKUP_CACHE = {}

# Offline last-resort cache, used ONLY when the live IFSC API can't be
# reached (no internet) - covers the major Indian banks so detection
# still works offline for the common case. This is fundamentally safer
# than the keyword/narration-text fallback below: it's keyed off the
# SAME already-anchored, validated IFSC code (never a blind text scan),
# so it can't accidentally match a counterparty mentioned in a
# transaction narration. Deliberately small and only a fallback - the
# live API is authoritative and covers every bank/branch automatically,
# this just keeps things working when there's no network.
_OFFLINE_IFSC_PREFIX_MAP = {
    "SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank", "CNRB": "Canara Bank", "PUNB": "Punjab National Bank",
    "KKBK": "Kotak Mahindra Bank", "YESB": "Yes Bank", "INDB": "IndusInd Bank",
    "BKID": "Bank of India", "BARB": "Bank of Baroda", "UBIN": "Union Bank of India",
    "CBIN": "Central Bank of India", "IDIB": "Indian Bank", "IBKL": "IDBI Bank",
    "FDRL": "Federal Bank", "SIBL": "South Indian Bank", "KARB": "Karnataka Bank",
    "BDBL": "Bandhan Bank", "RATN": "RBL Bank", "IDFB": "IDFC First Bank",
    "AUBL": "AU Small Finance Bank", "MAHB": "Bank of Maharashtra",
    "UCBA": "UCO Bank", "IOBA": "Indian Overseas Bank", "PSIB": "Punjab & Sind Bank",
}


def lookup_bank_via_ifsc(ifsc_code: str, timeout: float = 3.0):
    """
    Look up the REAL bank name for an IFSC code via the public, RBI-
    sourced Razorpay IFSC API (https://ifsc.razorpay.com/<code>) - no
    hardcoded bank-name registry needed for the common path. Works
    automatically for every bank/branch in India, including ones never
    explicitly listed anywhere in this codebase.

    Falls back to a small offline prefix cache ONLY if the live API is
    unreachable (no internet) - still keyed off this same already-
    anchored IFSC code, never a blind text scan, so it can't misfire the
    way narration-keyword matching can.

    Returns None only if the code is malformed or genuinely unrecognized
    by both the live API and the offline cache.
    """
    code = (ifsc_code or "").strip().upper()
    if not re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", code):
        return None
    if code in _IFSC_LOOKUP_CACHE:
        return _IFSC_LOOKUP_CACHE[code]
    try:
        req = urllib.request.Request(
            f"https://ifsc.razorpay.com/{code}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        bank_name = data.get("BANK") or None
        _IFSC_LOOKUP_CACHE[code] = bank_name
        return bank_name
    except Exception:
        offline_name = _OFFLINE_IFSC_PREFIX_MAP.get(code[:4])
        _IFSC_LOOKUP_CACHE[code] = offline_name
        return offline_name


def extract_account_ifsc(text: str) -> str:
    """
    Find the statement's OWN IFSC code - anchored to an explicit "IFSC"
    label nearby, so a counterparty's IFSC embedded inside a transaction
    narration (which never carries an "IFSC" label, just the raw code as
    part of a UPI/NEFT reference string) isn't picked up by mistake.
    Takes the FIRST such labeled match in the document: the account-info
    block reliably precedes the transaction history in every real bank
    statement layout, so "first labeled occurrence" is a safe proxy for
    "the account's own IFSC", even when scanning the full document.
    """
    if not text:
        return ""
    m = re.search(r"IFSC\s*(?:Code)?\s*[:\-]?\s*([A-Za-z]{4}0[A-Za-z0-9]{6})", text)
    return m.group(1).upper() if m else ""


def detect_bank(text: str) -> tuple:
    """
    Primary bank-identification entry point. Tries a live IFSC lookup
    first - this is the part that genuinely "sees and detects" rather
    than matching against a hardcoded name list, and works for any bank
    automatically. Only falls back to keyword/regex text scanning
    (detect_bank_from_text) when no IFSC label is found in the document,
    or the lookup service can't be reached.

    Returns (bank_name, method) where method is one of
    "ifsc_lookup" / "keyword_fallback" / "not_found" - surfaced in the
    ingestion warnings so it's visible in the terminal output which path
    actually fired, rather than a black box.
    """
    ifsc = extract_account_ifsc(text)
    if ifsc:
        bank = lookup_bank_via_ifsc(ifsc)
        if bank:
            return bank, f"ifsc_lookup:{ifsc}"
    bank = detect_bank_from_text(text)
    return bank, ("not_found" if bank == "Unknown Bank" else "keyword_fallback")


def detect_bank_from_text(text: str) -> str:
    if not text:
        return "Unknown Bank"
    text_lower = text.lower()

    # Find EVERY keyword that matches, with its position, then take the
    # earliest one — not just whichever happens to come first in
    # BANK_NAME_KEYWORDS' dict order. A bank's own name reliably appears
    # at/near the top of its own statement; a keyword matching deeper in
    # the text (e.g. a counterparty mentioned in an early transaction
    # line) shouldn't be able to outrank that just because of dict
    # ordering. Word-boundary matching avoids short keywords like "axis"
    # or "rbl" matching inside an unrelated longer word.
    best_pos, best_name = None, None
    for keyword, name in BANK_NAME_KEYWORDS.items():
        m = re.search(r'\b' + re.escape(keyword) + r'\b', text_lower)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos = m.start()
            best_name = name
    if best_name:
        return best_name

    # Generic fallback: catches banks not in our hardcoded registry, e.g.
    # smaller regional/cooperative banks, by pattern-matching "<Name> Bank"
    # in the free text. Without this, anything not in BANK_NAME_KEYWORDS
    # falls all the way through to "Unknown Bank" even when the bank's
    # name is sitting right there in the statement header.
    generic = re.search(
        r"([A-Z][A-Za-z&.\s]{2,80}?(?:Bank|BANK)(?:\s+Ltd\.?|\s+Limited)?)",
        text,
    )
    if generic:
        return re.sub(r"\s+", " ", generic.group(1)).strip().title()
    return "Unknown Bank"


# ---------------------------------------------------------------------------
# Header row detection
# ---------------------------------------------------------------------------
# Header-ONLY keywords — multi-word phrases that appear in column headers
# but NOT in transaction narrations. Single words like "withdrawal", "credit",
# "deposit" are intentionally EXCLUDED because they appear in narrations too.
_HEADER_ONLY_KEYWORDS = [
    # Date column headers
    "txn date", "tran date", "transaction date", "post date", "value date",
    "posting date", "entry date", "process date", "effective date",
    "date of transaction", "booking date", "txn_date", "trn_date",
    # Narration column headers
    "transaction details", "transaction particulars", "transaction remarks",
    "transaction description", "entry description", "particulars", "narration",
    "txn_desc", "trn_desc",
    # Amount column headers
    "withdrawal amount", "debit amount", "credit amount", "deposit amount",
    "amount debited", "amount credited", "closing balance", "running balance",
    "balance amount", "debit amt", "credit amt", "withdrawal amt", "deposit amt",
    "dr amount", "cr amount", "amount (dr)", "amount (cr)",
    "debit_amt", "credit_amt",
    # Reference column headers
    "cheque number", "reference number", "transaction id", "tran id",
    "cheque no", "ref no", "vch no", "txn id",
]


def _col_looks_like_data_value(col_name: str) -> bool:
    """
    Returns True if a column name looks like a DATA VALUE not a proper header.
    Catches four cases:
      1. Date values used as column name (pandas missed the real header row)
      2. Numeric values used as column name
      3. Pandas placeholder "Unnamed: N" (file had no header in row 0)
      4. Long garbled OCR text in column name position (OCR metadata bleed-through)
    """
    s = str(col_name).strip()

    # Case 1: Looks like a date (dd/mm/yyyy, yyyy-mm-dd, dd-Mon-yyyy)
    if re.match(r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4}$", s):
        return True
    if re.match(r"^\d{1,2}[- ][A-Za-z]{3}[- ]\d{2,4}$", s):
        return True

    # Case 2: Looks like a pure number / amount
    if re.match(r"^[\d,]+\.?\d*$", s):
        return True

    # Case 3: Pandas unnamed placeholder — header row not in row 0
    if re.match(r"^Unnamed:\s*\d+$", s):
        return True

    # Case 4: Long text (> 25 chars) with multiple spaces or colons —
    # typical OCR metadata bleed: "Account Holder: Name IFSC: PUNB0769987"
    # or bank title row: "HDFC Bank - Account Statement"
    # Real column headers are short and clean (< 35 chars, no colon)
    if len(s) > 35 or (len(s) > 15 and (":" in s or s.count(" ") > 4)):
        return True

    return False


def find_header_row(raw_df: pd.DataFrame, max_scan: int = 15) -> int:
    """
    Determine if pandas already found the correct header (return 0)
    or if the true header row is buried in the data (return its index).

    Two independent guards decide "don't bother rescanning":
      1. Data-value-ratio: do current column names look like DATA
         (dates, raw numbers, "Unnamed: N", garbled OCR text)? If so,
         pandas read the file without finding a real header — keep
         scanning.
      2. Keyword-usability: do current column names ALREADY match known
         role keywords (date/amount/balance/etc.)? If so, they're a
         perfectly usable header even if they don't look like "data" by
         guard #1's definition - skip rescanning regardless.
    Either guard passing is enough to short-circuit to row 0; scanning is
    only attempted when BOTH say the current header looks unusable.
    """
    current_cols = raw_df.columns.tolist()

    # Guard 1: do columns look like data values (dates/numbers/Unnamed/garbled)?
    data_value_cols = sum(1 for c in current_cols if _col_looks_like_data_value(str(c)))
    data_value_ratio = data_value_cols / max(len(current_cols), 1)

    # Guard 2: do columns already match known role keywords?
    quick_roles = assign_column_roles_by_keywords(current_cols)
    has_usable_header = bool(
        quick_roles.get("date")
        and (quick_roles.get("balance") or quick_roles.get("amount")
             or quick_roles.get("debit") or quick_roles.get("credit"))
    )

    if data_value_ratio < 0.4 or has_usable_header:
        return 0  # ← DO NOT scan data rows, current header is already usable

    # Both guards say the current header looks unusable - scan rows for
    # the real header. Use HEADER-ONLY keywords (NOT single words like
    # "withdrawal"/"credit" that also appear in transaction narrations).
    best_row, best_score = 0, 0
    for i in range(min(max_scan, len(raw_df))):
        row_text = " ".join(
            str(v).lower() for v in raw_df.iloc[i].values if pd.notna(v)
        )
        # Score using header-only multi-word phrases
        score = sum(1 for kw in _HEADER_ONLY_KEYWORDS if kw in row_text)
        # Hindi header bonus
        hindi_hits = sum(1 for h in HINDI_COLUMN_MAP if h in row_text)
        score += hindi_hits * 3
        if score > best_score:
            best_score = score
            best_row = i

    return best_row if best_score >= 1 else 0


# ---------------------------------------------------------------------------
# Tier 1: Keyword-based column role assignment
# ---------------------------------------------------------------------------
def _normalise_header(col) -> str:
    """
    Normalize separator characters to spaces BEFORE keyword matching, so
    machine-style headers match the same way human-readable ones do:
      TXN_DATE        -> "txn date"     (matches "txn date" keyword)
      Withdrawal_Amt  -> "withdrawal amt"
      DEBIT.AMOUNT    -> "debit amount"
    Without this, _keyword_score's substring matching misses any header
    that uses underscores/dots/slashes instead of spaces as word
    separators - common in ERP/database-style exports.
    """
    text = str(col).lower().strip()
    text = re.sub(r"[_\-/().:#]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _keyword_score(col_name: str, keywords: list) -> int:
    col_lower = _normalise_header(col_name)
    score = 0
    for kw in keywords:
        if len(kw) <= 3:
            # Short keywords need word-boundary match
            if re.search(r'\b' + re.escape(kw) + r'\b', col_lower):
                score += len(kw) * 3
        else:
            if kw in col_lower:
                score += len(kw)
    return score


def assign_column_roles_by_keywords(columns: list) -> dict:
    """
    Returns {role: column_name} mapping using keyword scoring.
    None if no match found for a role.
    """
    # Map Hindi column names first
    translated = {}
    for col in columns:
        col_str = str(col).strip()
        if col_str in HINDI_COLUMN_MAP:
            translated[col_str] = HINDI_COLUMN_MAP[col_str]

    role_order = ["date", "balance", "debit", "credit", "narration", "ref", "amount"]
    assigned = {}
    used_cols = set()

    for role in role_order:
        keywords = COLUMN_ROLE_KEYWORDS[role]
        best_col, best_score = None, 0

        for col in columns:
            if col in used_cols:
                continue
            # Check Hindi translation
            col_str = str(col).strip()
            if col_str in translated and translated[col_str] == role:
                best_col = col
                best_score = 999
                break
            score = _keyword_score(col_str, keywords)
            if score > best_score:
                best_score = score
                best_col = col

        if best_col and best_score > 0:
            assigned[role] = best_col
            used_cols.add(best_col)
        else:
            assigned[role] = None

    return assigned


# ---------------------------------------------------------------------------
# Tier 2: Content-based column detection
# ---------------------------------------------------------------------------
def _col_matches_pattern(series, pattern_list: list,
                          min_match_ratio: float = 0.6) -> bool:
    """Check if at least min_match_ratio of non-null values match any pattern."""
    try:
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        sample = series.dropna().astype(str).head(20)
        if len(sample) == 0:
            return False
        hits = sum(
            1 for v in sample
            if any(p.match(v.strip()) for p in pattern_list)
        )
        return hits / len(sample) >= min_match_ratio
    except Exception:
        return False


def _is_row_number_column(series: pd.Series) -> bool:
    """
    Detect if a column is just a row number / serial number.
    Row numbers: sequential integers starting from 0 or 1, no decimals,
    all values <= total row count * 2.
    """
    try:
        nums = pd.to_numeric(series.dropna(), errors='coerce').dropna()
        if len(nums) < 2:
            return False
        # Must be integers (no decimal part)
        if not (nums == nums.round()).all():
            return False
        # Must be small sequential-ish values (max <= 2x row count)
        if nums.max() > len(nums) * 3:
            return False
        # Must be monotonically increasing or at least mostly so
        diffs = nums.diff().dropna()
        if (diffs > 0).mean() > 0.7 and nums.max() <= len(nums) * 2:
            return True
        return False
    except Exception:
        return False


def assign_column_roles_by_content(df: pd.DataFrame, already_assigned: dict) -> dict:
    """
    For roles still unassigned after keyword pass, look at actual column
    data to infer role. Handles Col1/Col2/A/B style column names.
    Skips columns that look like row numbers (Sl, ID, S.No, etc.)

    Two-pass design: first IDENTIFY every financial-looking column
    without committing a role yet, then assign roles with the rightmost
    financial column reserved for "balance" when balance is still
    unknown and 2+ financial columns exist. A single-pass, left-to-right
    "claim debit then credit then amount" approach (the original design)
    greedily consumed every numeric column for debit/credit, leaving none
    available for "balance" - which only Tier 3's positional fallback
    knows how to find (rightmost numeric column convention), and which
    Tier 3 never gets a chance to run on if Tier 2 already claimed
    everything.
    """
    assigned = dict(already_assigned)
    used_cols = set(v for v in assigned.values() if v is not None)
    unassigned_roles = [r for r, v in assigned.items() if v is None]

    if not unassigned_roles:
        return assigned

    unassigned_cols = [c for c in df.columns if c not in used_cols]
    if not unassigned_cols:
        return assigned

    # Pass 1: classify columns, committing "date" immediately (no
    # ambiguity there) but only COLLECTING financial-looking columns —
    # role assignment among them happens in pass 2, with positional
    # awareness.
    financial_cols = []
    for col in unassigned_cols:
        try:
            raw = df[col]
            if not isinstance(raw, pd.Series):
                raw = pd.Series(raw)
            series = raw.dropna()
        except Exception:
            continue
        if len(series) == 0:
            continue

    # Convert OCR slash misreads to standard dashes
    s = re.sub(r'[|l\\]', '-', s)

    # ── NEW: Fix Tesseract 0 -> 6 misreads in dates ─────────────
    # If a standalone 2-digit component starts with 6 (e.g., 65, 68), 
    # it is impossible for it to be a valid day or month. Convert to 0x.
    s = re.sub(r'\b6(\d)\b', r'0\1', s)
    
    # Fix the year: 2623 -> 2023, 2624 -> 2024
    s = re.sub(r'\b26(\d{2})\b', r'20\1', s)
    # ────────────────────────────────────────────────────────────

    # Normalise dot or space-separated to dash: "25 03 2025" -> "25-03-2025"
    s = re.sub(r'^(\d{1,2})[\s.]+(\d{1,2})[\s.]+(\d{2,4})$', r'\1-\2-\3', s)

        # Check date pattern
        if "date" in unassigned_roles:
            try:
                if _col_matches_pattern(series, CONTENT_PATTERNS["date"]):
                    assigned["date"] = col
                    used_cols.add(col)
                    unassigned_roles.remove("date")
                    continue
            except Exception:
                pass

        # Check amount pattern — collect, don't assign yet
        if any(r in unassigned_roles for r in ("debit", "credit", "amount", "balance")):
            try:
                nums = pd.to_numeric(
                    series.astype(str).str.replace(",", ""), errors="coerce"
                ).dropna()
                is_financial = (
                    _col_matches_pattern(series, CONTENT_PATTERNS["amount"])
                    and (len(nums) == 0 or float(nums.mean()) > 10 or float(nums.max()) > 100)
                )
            except Exception:
                is_financial = False
            if is_financial:
                financial_cols.append(col)

    # Pass 2: assign roles among the financial columns found, reserving
    # the rightmost one for "balance" when it's still unknown and there's
    # more than one candidate to choose from.
    if financial_cols:
        if "balance" in unassigned_roles and len(financial_cols) >= 2:
            balance_col = financial_cols[-1]
            assigned["balance"] = balance_col
            used_cols.add(balance_col)
            unassigned_roles.remove("balance")
            financial_cols = financial_cols[:-1]

        for col in financial_cols:
            if "debit" in unassigned_roles and "credit" in unassigned_roles:
                # Neither known yet — if this is the LAST remaining
                # ambiguous column with no companion, it's more likely a
                # single signed "amount" column than an arbitrary pick of
                # debit vs credit.
                if len(financial_cols) == 1 and "amount" in unassigned_roles:
                    assigned["amount"] = col
                    used_cols.add(col)
                    unassigned_roles.remove("amount")
                    continue
                assigned["debit"] = col
                used_cols.add(col)
                unassigned_roles.remove("debit")
            elif "credit" in unassigned_roles:
                assigned["credit"] = col
                used_cols.add(col)
                unassigned_roles.remove("credit")
            elif "debit" in unassigned_roles:
                assigned["debit"] = col
                used_cols.add(col)
                unassigned_roles.remove("debit")
            elif "amount" in unassigned_roles:
                assigned["amount"] = col
                used_cols.add(col)
                unassigned_roles.remove("amount")

    return assigned


# ---------------------------------------------------------------------------
# Tier 3: Positional fallback
# ---------------------------------------------------------------------------
def assign_column_roles_by_position(df: pd.DataFrame, assigned: dict) -> dict:
    """
    Last resort: positional heuristics.
    - First column → date (if still unassigned)
    - Last numeric column → balance
    - Second-to-last and third-to-last numeric → credit, debit
    - Widest text column → narration
    """
    assigned = dict(assigned)
    used = set(v for v in assigned.values() if v is not None)
    cols = [c for c in df.columns if c not in used]

    if not cols:
        return assigned

    numeric_cols = [c for c in cols if pd.to_numeric(
        df[c].dropna().head(10), errors='coerce'
    ).notna().mean() > 0.5]

    text_cols = [c for c in cols if c not in numeric_cols]

    if "date" not in assigned or not assigned["date"]:
        # First column with date-like content OR first text column
        if text_cols:
            assigned["date"] = text_cols[0]
            used.add(text_cols[0])

    if "balance" not in assigned or not assigned["balance"]:
        if numeric_cols:
            assigned["balance"] = numeric_cols[-1]
            used.add(numeric_cols[-1])
            numeric_cols = numeric_cols[:-1]

    if ("debit" not in assigned or not assigned["debit"]) and len(numeric_cols) >= 2:
        assigned["debit"] = numeric_cols[-2]
        used.add(numeric_cols[-2])
        numeric_cols = numeric_cols[:-2]

    if ("credit" not in assigned or not assigned["credit"]) and numeric_cols:
        assigned["credit"] = numeric_cols[-1]
        used.add(numeric_cols[-1])

    if "narration" not in assigned or not assigned["narration"]:
        # Widest text column (most characters) = narration
        remaining_text = [c for c in text_cols if c not in used]
        if remaining_text:
            widest = max(
                remaining_text,
                key=lambda c: df[c].dropna().astype(str).str.len().mean()
            )
            assigned["narration"] = widest

    return assigned


# ---------------------------------------------------------------------------
# Combined role detection (all 3 tiers)
# ---------------------------------------------------------------------------
def assign_column_roles(columns: list, df: pd.DataFrame = None) -> dict:
    """
    Run all 3 detection tiers in sequence.
    Each tier fills in what the previous couldn't find.
    """
    # Tier 1: keywords
    roles = assign_column_roles_by_keywords(columns)

    # Tier 2: content-based (needs actual data)
    if df is not None:
        missing = [r for r, v in roles.items() if v is None and r != "ref"]
        if missing:
            roles = assign_column_roles_by_content(df, roles)

    # Tier 3: positional fallback
    if df is not None:
        still_missing = [r for r, v in roles.items()
                         if v is None and r in ("date", "narration", "balance")]
        if still_missing:
            roles = assign_column_roles_by_position(df, roles)

    return roles


# ---------------------------------------------------------------------------
# Computed balance fallback
# ---------------------------------------------------------------------------
def compute_running_balance(df: pd.DataFrame,
                             debit_col: str, credit_col: str,
                             start_balance: float = 0.0) -> list:
    """
    When balance column is missing entirely, compute it from
    running debit/credit totals.

    NOTE: must check missingness with pd.notna(), NOT Python truthiness
    ("value or 0"). NaN is truthy in Python (bool(float('nan')) is True),
    so "row.get(col, 0) or 0" does NOT substitute 0 for a missing cell -
    it silently keeps the NaN. That NaN then poisons the running-sum
    accumulator for every subsequent row (anything + nan = nan), and
    parse_amount's own NaN guard downstream quietly converts the
    resulting NaN into 0.0 - so the visible symptom was every row
    showing a flat, wrong balance of 0.0 instead of an accumulating
    running total, with no error or warning anywhere in the chain.
    """
    balance = start_balance
    balances = []
    for _, row in df.iterrows():
        debit_raw = row.get(debit_col)
        credit_raw = row.get(credit_col)
        try:
            debit = float(str(debit_raw).replace(",", "")) if pd.notna(debit_raw) else 0.0
        except (ValueError, TypeError):
            debit = 0.0
        try:
            credit = float(str(credit_raw).replace(",", "")) if pd.notna(credit_raw) else 0.0
        except (ValueError, TypeError):
            credit = 0.0
        balance = round(balance + credit - debit, 2)
        balances.append(balance)
    return balances


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
# Common OCR character substitutions in dates
_OCR_DATE_FIXES = str.maketrans({
    "@": "0", "O": "0", "o": "0",   # O/@ misread as 0
    "l": "1", "I": "1",              # l/I misread as 1
    "S": "5",                          # S misread as 5
    "B": "8",                          # B misread as 8
})


def parse_date(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "date", "-", "n/a"):
        return ""

    # Try original string first
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

    # Apply OCR character substitutions and retry
    s_fixed = s.translate(_OCR_DATE_FIXES)
    if s_fixed != s:
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(s_fixed, fmt).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue

    # Pandas fallback - MUST use errors="coerce" and verify the result,
    # otherwise an invalid date (e.g. "62-01-2026") raises inside
    # strptime/dateutil and was previously caught by `except Exception: return s`,
    # which let the raw, unparsed garbage string flow straight into the
    # final dataset instead of being filtered out by the caller's
    # `if not date_str: skip` check. Every failure path below now
    # returns "" so unparseable dates are correctly dropped, not silently
    # passed through. The 1990-2100 sanity range guards against pandas
    # occasionally coercing nonsense into a technically-valid but
    # nonsensical date.
    for candidate in (s, s_fixed):
        try:
            parsed = pd.to_datetime(candidate, dayfirst=True, errors="coerce")
        except Exception:
            continue
        if pd.notna(parsed) and 1990 <= parsed.year <= 2100:
            return parsed.strftime("%Y-%m-%d")
    return ""


# ---------------------------------------------------------------------------
# Amount parsing
# ---------------------------------------------------------------------------
def parse_amount(val) -> tuple[float, str]:
    """Returns (abs_amount, sign) where sign is '+', '-', or ''."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0, ""
    s = str(val).strip()
    if s in ("", "-", "--", "nil", "n/a", "nan"):
        return 0.0, ""

    sign = ""

    # Leading CR/DR prefix: "CR 1234.56", "Cr.1500", "DR999" — some banks
    # put the sign marker BEFORE the number instead of after. Checked
    # first since it consumes its own slice of the string; the trailing-
    # suffix check below is skipped if a sign was already found here.
    prefix_match = re.match(r"(?i)^\s*(cr|dr)\b\.?\s*", s)
    if prefix_match:
        sign = "+" if prefix_match.group(1).lower() == "cr" else "-"
        s = s[prefix_match.end():].strip()

    s_upper = s.upper()

    # Negative number: -1234
    if s.startswith("-"):
        sign = sign or "-"
        s = s[1:]

    # Bracket negative: (1234)
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        sign = sign or "-"

    # CR/DR suffix — only checked if no prefix sign was already found
    if not sign:
        if s_upper.endswith("CR") or s_upper.endswith(" CR"):
            s = re.sub(r'(?i)\s*cr$', '', s).strip()
            sign = "+"
        elif s_upper.endswith("DR") or s_upper.endswith(" DR"):
            s = re.sub(r'(?i)\s*dr$', '', s).strip()
            sign = "-"

    # Strip currency
    s = re.sub(r'[₹$£€]|INR|Rs\.?', '', s, flags=re.IGNORECASE).strip()

    # Remove commas
    s = s.replace(",", "")

    # Remove non-numeric except dot
    s = re.sub(r"[^\d.]", "", s)

    if not s:
        return 0.0, sign
    try:
        return abs(float(s)), sign
    except ValueError:
        return 0.0, sign


# ---------------------------------------------------------------------------
# Channel inference
# ---------------------------------------------------------------------------
def infer_channel(narration: str) -> str:
    if not narration:
        return "OTHER"
    n = str(narration).lower()
    for channel, keywords in CHANNEL_KEYWORDS:
        if any(kw in n for kw in keywords):
            return channel
    return "OTHER"


def extract_counterparty(narration: str) -> str:
    if not narration:
        return ""
    parts = re.split(r"[-/|]", str(narration))
    for part in parts[1:3]:
        candidate = part.strip()
        if re.match(r'^[A-Z0-9]{8,}$', candidate):
            continue
        if re.match(r'^\d{10}$', candidate):
            continue
        if len(candidate) > 2:
            return candidate.title()
    return ""


def is_likely_transaction_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df.columns) < 2:
        return False
    # Needs at least some numeric column
    has_numeric = any(
        pd.to_numeric(df[c].dropna().head(5), errors='coerce').notna().any()
        for c in df.columns
    )
    return has_numeric