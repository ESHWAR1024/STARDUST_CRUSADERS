"""
Phase 6 — Schema Detector  (v2)

Three-tier column detection:
  Tier 1 : Keyword matching (column header contains known fragment)
  Tier 2 : Content sampling (column data looks like dates / amounts)
  Tier 3 : Positional fallback

New in v2:
  • IDFC "Cr/Dr" suffix balance parsing
  • Paytm COD_DRCR single-direction column
  • BOB / Federal BENEF/REMIT account extraction
  • Improved counterparty extraction (UPI VPA, IFSC, IMPS, NEFT patterns)
  • Richer channel inference (NACH, BBPS, reversal, interest, charges)
  • UPI VPA → counterparty_account extraction
  • IFSC extraction from narration for graph edges
"""

import re
import json
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime
try:
    from phase6.ingestion_config import (
        COLUMN_ROLE_KEYWORDS, CONTENT_PATTERNS,
        HINDI_COLUMN_MAP, BANK_NAME_KEYWORDS,
        DATE_FORMATS, CHANNEL_KEYWORDS,
    )
except ImportError:
    from ingestion_config import (
        COLUMN_ROLE_KEYWORDS, CONTENT_PATTERNS,
        HINDI_COLUMN_MAP, BANK_NAME_KEYWORDS,
        DATE_FORMATS, CHANNEL_KEYWORDS,
    )


# ===========================================================================
# Bank detection
# ===========================================================================
_IFSC_LOOKUP_CACHE: dict = {}

_OFFLINE_IFSC_PREFIX_MAP = {
    "SBIN": "State Bank of India",  "HDFC": "HDFC Bank",
    "ICIC": "ICICI Bank",           "UTIB": "Axis Bank",
    "CNRB": "Canara Bank",          "PUNB": "Punjab National Bank",
    "KKBK": "Kotak Mahindra Bank",  "YESB": "Yes Bank",
    "INDB": "IndusInd Bank",        "BKID": "Bank of India",
    "BARB": "Bank of Baroda",       "UBIN": "Union Bank of India",
    "CBIN": "Central Bank of India","IDIB": "Indian Bank",
    "IBKL": "IDBI Bank",            "FDRL": "Federal Bank",
    "SIBL": "South Indian Bank",    "KARB": "Karnataka Bank",
    "BDBL": "Bandhan Bank",         "RATN": "RBL Bank",
    "IDFB": "IDFC First Bank",      "AUBL": "AU Small Finance Bank",
    "PYTM": "Paytm Payments Bank",  "MAHB": "Bank of Maharashtra",
    "UCBA": "UCO Bank",             "IOBA": "Indian Overseas Bank",
    "PSIB": "Punjab & Sind Bank",   "CRGB": "Kerala Gramin Bank",
    "APGB": "Andhra Pradesh Gramin Vikas Bank",
    "KVGB": "Karnataka Vikas Grameen Bank",
    "PKGB": "Pragathi Krishna Gramin Bank",
}


def lookup_bank_via_ifsc(ifsc_code: str, timeout: float = 3.0) -> str | None:
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
        offline = _OFFLINE_IFSC_PREFIX_MAP.get(code[:4])
        _IFSC_LOOKUP_CACHE[code] = offline
        return offline


def extract_account_ifsc(text: str) -> str:
    """Pull the statement's OWN IFSC from an anchored label."""
    if not text:
        return ""
    m = re.search(
        r"IFSC\s*(?:Code|/RTGS/NEFT)?\s*[:\-]?\s*([A-Za-z]{4}0[A-Za-z0-9]{6})",
        text, re.IGNORECASE
    )
    return m.group(1).upper() if m else ""


def detect_bank(text: str) -> tuple[str, str]:
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
    best_pos, best_name = None, None
    for keyword, name in BANK_NAME_KEYWORDS.items():
        m = re.search(r'\b' + re.escape(keyword) + r'\b', text_lower)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos = m.start()
            best_name = name
    if best_name:
        return best_name
    generic = re.search(
        r"([A-Z][A-Za-z&.\s]{2,80}?(?:Bank|BANK)(?:\s+Ltd\.?|\s+Limited)?)",
        text,
    )
    if generic:
        return re.sub(r"\s+", " ", generic.group(1)).strip().title()
    return "Unknown Bank"


# ===========================================================================
# Header row detection
# ===========================================================================
_HEADER_ONLY_KEYWORDS = [
    "txn date", "tran date", "transaction date", "post date", "value date",
    "posting date", "entry date", "process date", "effective date",
    "date of transaction", "booking date", "txn_date", "trn_date",
    "transaction details", "transaction particulars", "transaction remarks",
    "transaction description", "entry description", "particulars", "narration",
    "txn_desc", "trn_desc", "tran particular", "tran_rmks",
    "withdrawal amount", "debit amount", "credit amount", "deposit amount",
    "amount debited", "amount credited", "closing balance", "running balance",
    "balance amount", "debit amt", "credit amt", "withdrawal amt", "deposit amt",
    "dr amount", "cr amount", "amount (dr)", "amount (cr)",
    "cheque number", "reference number", "transaction id", "tran id",
    "cheque no", "ref no", "vch no", "txn id", "instrument no",
]


def _col_looks_like_data_value(col_name: str) -> bool:
    s = str(col_name).strip()
    if re.match(r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4}$", s):
        return True
    if re.match(r"^\d{1,2}[- ][A-Za-z]{3}[- ]\d{2,4}$", s):
        return True
    if re.match(r"^[\d,]+\.?\d*$", s):
        return True
    if re.match(r"^Unnamed:\s*\d+$", s):
        return True
    if len(s) > 35 or (len(s) > 15 and (":" in s or s.count(" ") > 4)):
        return True
    return False


def find_header_row(raw_df: pd.DataFrame, max_scan: int = 20) -> int:
    current_cols = raw_df.columns.tolist()
    data_value_cols = sum(1 for c in current_cols if _col_looks_like_data_value(str(c)))
    data_value_ratio = data_value_cols / max(len(current_cols), 1)
    quick_roles = assign_column_roles_by_keywords(current_cols)
    has_usable_header = (
        pd.notna(quick_roles.get("date"))
        and (
            pd.notna(quick_roles.get("balance"))
            or pd.notna(quick_roles.get("amount"))
            or pd.notna(quick_roles.get("debit"))
            or pd.notna(quick_roles.get("credit"))
        )
    )
    if data_value_ratio < 0.4 or has_usable_header:
        return 0
    best_row, best_score = 0, 0
    for i in range(min(max_scan, len(raw_df))):
        row_text = " ".join(
            str(v).lower() for v in raw_df.iloc[i].values if pd.notna(v)
        )
        score = sum(1 for kw in _HEADER_ONLY_KEYWORDS if kw in row_text)
        hindi_hits = sum(1 for h in HINDI_COLUMN_MAP if h in row_text)
        score += hindi_hits * 3
        if score > best_score:
            best_score = score
            best_row = i
    return best_row if best_score >= 1 else 0


# ===========================================================================
# Tier 1 — Keyword-based column role assignment
# ===========================================================================
def _normalise_header(col) -> str:
    text = str(col).lower().strip()
    text = re.sub(r"[_\-/().:#]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _keyword_score(col_name: str, keywords: list) -> int:
    col_lower = _normalise_header(col_name)
    score = 0
    for kw in keywords:
        if len(kw) <= 3:
            if re.search(r'\b' + re.escape(kw) + r'\b', col_lower):
                score += len(kw) * 3
        else:
            if kw in col_lower:
                score += len(kw)
    return score


def assign_column_roles_by_keywords(columns: list) -> dict:
    translated = {}
    for col in columns:
        col_str = str(col).strip()
        if col_str in HINDI_COLUMN_MAP:
            translated[col_str] = HINDI_COLUMN_MAP[col_str]

    role_order = ["date", "balance", "debit", "credit", "narration", "ref",
                  "amount", "drcr"]
    assigned = {}
    used_cols: set = set()

    for role in role_order:
        keywords = COLUMN_ROLE_KEYWORDS.get(role, [])
        best_col, best_score = None, 0
        for col in columns:
            if col in used_cols:
                continue
            col_str = str(col).strip()
            if col_str in translated and translated[col_str] == role:
                best_col, best_score = col, 999
                break
            score = _keyword_score(col_str, keywords)
            if score > best_score:
                best_score, best_col = score, col
        if best_col and best_score > 0:
            assigned[role] = best_col
            used_cols.add(best_col)
        else:
            assigned[role] = None

    return assigned


# ===========================================================================
# Tier 2 — Content-based detection
# ===========================================================================
def _col_matches_pattern(series, pattern_list: list,
                          min_match_ratio: float = 0.6) -> bool:
    try:
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        sample = series.dropna().astype(str).head(20)
        if len(sample) == 0:
            return False
        hits = sum(
            1 for v in sample if any(p.match(v.strip()) for p in pattern_list)
        )
        return hits / len(sample) >= min_match_ratio
    except Exception:
        return False


def _is_row_number_column(series: pd.Series) -> bool:
    try:
        nums = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(nums) < 2:
            return False
        if not (nums == nums.round()).all():
            return False
        diffs = nums.diff().dropna()
        if (diffs > 0).mean() > 0.7 and nums.max() <= len(nums) * 2:
            return True
        return False
    except Exception:
        return False


def assign_column_roles_by_content(df: pd.DataFrame,
                                    already_assigned: dict) -> dict:
    assigned = dict(already_assigned)
    used_cols = set(v for v in assigned.values() if v is not None)
    unassigned_roles = [r for r, v in assigned.items() if v is None]
    if not unassigned_roles:
        return assigned

    unassigned_cols = [c for c in df.columns if c not in used_cols]
    if not unassigned_cols:
        return assigned

    financial_cols = []
    for col in unassigned_cols:
        try:
            raw = df[col]
            series = (raw if isinstance(raw, pd.Series) else pd.Series(raw)).dropna()
        except Exception:
            continue
        if len(series) == 0 or _is_row_number_column(series):
            continue
        if "date" in unassigned_roles:
            try:
                if _col_matches_pattern(series, CONTENT_PATTERNS["date"]):
                    assigned["date"] = col
                    used_cols.add(col)
                    unassigned_roles.remove("date")
                    continue
            except Exception:
                pass
        if any(r in unassigned_roles for r in ("debit", "credit", "amount", "balance")):
            try:
                nums = pd.to_numeric(
                    series.astype(str).str.replace(",", ""), errors="coerce"
                ).dropna()
                is_financial = (
                    _col_matches_pattern(series, CONTENT_PATTERNS["amount"])
                    and (len(nums) == 0 or float(nums.mean()) > 10
                         or float(nums.max()) > 100)
                )
            except Exception:
                is_financial = False
            if is_financial:
                financial_cols.append(col)

    if financial_cols:
        if "balance" in unassigned_roles and len(financial_cols) >= 2:
            assigned["balance"] = financial_cols[-1]
            used_cols.add(financial_cols[-1])
            unassigned_roles.remove("balance")
            financial_cols = financial_cols[:-1]
        for col in financial_cols:
            if "debit" in unassigned_roles and "credit" in unassigned_roles:
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


# ===========================================================================
# Tier 3 — Positional fallback
# ===========================================================================
def assign_column_roles_by_position(df: pd.DataFrame, assigned: dict) -> dict:
    assigned = dict(assigned)
    used = set(v for v in assigned.values() if v is not None)
    cols = [c for c in df.columns if c not in used]
    if not cols:
        return assigned

    numeric_cols = [
        c for c in cols
        if pd.to_numeric(df[c].dropna().head(10), errors="coerce").notna().mean() > 0.5
    ]
    text_cols = [c for c in cols if c not in numeric_cols]

    if not (assigned.get("date")):
        if text_cols:
            assigned["date"] = text_cols[0]
            used.add(text_cols[0])
    if not assigned.get("balance"):
        if numeric_cols:
            assigned["balance"] = numeric_cols[-1]
            used.add(numeric_cols[-1])
            numeric_cols = numeric_cols[:-1]
    if not assigned.get("debit") and len(numeric_cols) >= 2:
        assigned["debit"] = numeric_cols[-2]
        used.add(numeric_cols[-2])
        numeric_cols = numeric_cols[:-2]
    if not assigned.get("credit") and numeric_cols:
        assigned["credit"] = numeric_cols[-1]
        used.add(numeric_cols[-1])
    if not assigned.get("narration"):
        remaining_text = [c for c in text_cols if c not in used]
        if remaining_text:
            widest = max(
                remaining_text,
                key=lambda c: df[c].dropna().astype(str).str.len().mean()
            )
            assigned["narration"] = widest
    return assigned


# ===========================================================================
# Combined role detection (all 3 tiers)
# ===========================================================================
def assign_column_roles(columns: list, df: pd.DataFrame = None) -> dict:
    roles = assign_column_roles_by_keywords(columns)
    if df is not None:
        missing = [r for r, v in roles.items() if v is None and r != "ref"]
        if missing:
            roles = assign_column_roles_by_content(df, roles)
    if df is not None:
        still_missing = [
            r for r, v in roles.items()
            if v is None and r in ("date", "narration", "balance")
        ]
        if still_missing:
            roles = assign_column_roles_by_position(df, roles)
    return roles


# ===========================================================================
# Computed balance fallback
# ===========================================================================
def compute_running_balance(df: pd.DataFrame,
                             debit_col: str, credit_col: str,
                             start_balance: float = 0.0) -> list:
    balance = start_balance
    balances = []
    for _, row in df.iterrows():
        debit_raw  = row.get(debit_col)
        credit_raw = row.get(credit_col)
        try:
            debit  = float(str(debit_raw).replace(",", "")) if pd.notna(debit_raw)  else 0.0
        except (ValueError, TypeError):
            debit = 0.0
        try:
            credit = float(str(credit_raw).replace(",", "")) if pd.notna(credit_raw) else 0.0
        except (ValueError, TypeError):
            credit = 0.0
        balance = round(balance + credit - debit, 2)
        balances.append(balance)
    return balances


# ===========================================================================
# Date parsing
# ===========================================================================
_OCR_DATE_FIXES = str.maketrans({
    "@": "0", "O": "0", "o": "0",
    "l": "1", "I": "1",
    "S": "5", "B": "8",
})

# IDFC timestamps embed date+time: "15/05/25 10:20"
_IDFC_DT_RE = re.compile(
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+\d{1,2}:\d{2}"
)


def parse_date(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "date", "-", "n/a", ""):
        return ""

    s = re.sub(r'[\r\n]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # Strip embedded time component (IDFC: "15/05/25 10:20 15/05/25")
    m = _IDFC_DT_RE.match(s)
    if m:
        s = m.group(1)

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

    s_fixed = s.translate(_OCR_DATE_FIXES)
    if s_fixed != s:
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(s_fixed, fmt).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue

    # Try each whitespace-separated token in noisy cells
    for candidate in re.split(r'[\s,;|]+', s):
        if not candidate or candidate.lower() in ("cr", "dr", "cr.", "dr."):
            continue
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue

    # Extract the first date-like substring from longer values
    for match in re.finditer(r'\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}', s):
        token = match.group(0)
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(token, fmt).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue

    for candidate in (s, s_fixed):
        try:
            parsed = pd.to_datetime(candidate, dayfirst=True, errors="coerce", utc=False)
        except Exception:
            continue
        if pd.notna(parsed) and 1990 <= parsed.year <= 2100:
            return parsed.strftime("%Y-%m-%d")
    return ""


# ===========================================================================
# Amount parsing  (handles "Cr" / "Dr" suffix used by IDFC & Ratnakar)
# ===========================================================================
def parse_amount(val) -> tuple[float, str]:
    """Returns (abs_amount, sign) where sign is '+', '-', or ''."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0, ""
    s = str(val).strip()
    if s in ("", "-", "--", "nil", "n/a", "nan"):
        return 0.0, ""

    sign = ""

    # Leading CR/DR prefix
    prefix_match = re.match(r"(?i)^\s*(cr|dr)\b\.?\s*", s)
    if prefix_match:
        sign = "+" if prefix_match.group(1).lower() == "cr" else "-"
        s = s[prefix_match.end():].strip()

    if s.startswith("-"):
        sign = sign or "-"
        s = s[1:]

    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        sign = sign or "-"

    # Trailing CR / Dr suffix  (IDFC: "5,195.00Cr", "804,695.00Cr")
    if not sign:
        cr_match = re.search(r'(?i)(cr|dr)\s*$', s)
        if cr_match:
            sign = "+" if cr_match.group(1).lower() == "cr" else "-"
            s = s[:cr_match.start()].strip()

    # Strip currency symbols
    s = re.sub(r'[₹$£€]|INR|Rs\.?', '', s, flags=re.IGNORECASE).strip()
    s = s.replace(",", "")
    s = re.sub(r"[^\d.]", "", s)

    if not s:
        return 0.0, sign
    try:
        return abs(float(s)), sign
    except ValueError:
        return 0.0, sign


# ===========================================================================
# Channel inference
# ===========================================================================
def infer_channel(narration: str) -> str:
    if not narration:
        return "OTHER"
    n = str(narration).lower()
    for channel, keywords in CHANNEL_KEYWORDS:
        if any(kw in n for kw in keywords):
            return channel
    return "OTHER"


# ===========================================================================
# Counterparty extraction
# New patterns for each channel type found in dataset
# ===========================================================================

# UPI VPA pattern: name@bankcode
_UPI_VPA_RE = re.compile(
    r'([A-Za-z0-9._\-]+@(?:oksbi|okhdfc|okaxis|okhdfcbank|ybl|axl|ibl|'
    r'kotak|naviax|fbl|upi|paytm|mbk|cnrb|idfb|fdrl|bdbl|ratn|aubl|'
    r'okicici|icicipay|sbi|hdfcbank|axisbank|[a-zA-Z0-9]+))',
    re.IGNORECASE
)

# IMPS/NEFT reference: extract name after UTR
_IMPS_NAME_RE = re.compile(
    r'(?:IMPS|NEFT|RTGS)[/\-](?:P2A|P2M)?[/\-]?\d+[/\-]([A-Za-z][A-Za-z\s]{2,30})',
    re.IGNORECASE
)

# UPI/xxx/CR|DR/NAME/BANK pattern (IDFC, Bandhan, BOB)
_UPI_NAME_RE = re.compile(
    r'UPI/\d+/(?:CR|DR)/([A-Za-z\s]{2,25})/',
    re.IGNORECASE
)

# IFSC code inside narration (counterparty bank)
_NARRATION_IFSC_RE = re.compile(
    r'\b([A-Z]{4}0[A-Z0-9]{6})\b'
)


def extract_counterparty(narration: str) -> str:
    if not narration:
        return ""
    nar = str(narration).strip()

    # 1. Try UPI/xxx/CR/NAME pattern
    m = _UPI_NAME_RE.search(nar)
    if m:
        name = m.group(1).strip()
        if len(name) > 1 and not re.match(r'^[A-Z0-9]{8,}$', name):
            return name.title()

    # 2. Try IMPS/NEFT name extraction
    m = _IMPS_NAME_RE.search(nar)
    if m:
        name = m.group(1).strip()
        if len(name) > 2:
            return name.title()

    # 3. Try VPA extraction as counterparty ID
    m = _UPI_VPA_RE.search(nar)
    if m:
        return m.group(1)

    # 4. Generic slash-split fallback
    parts = re.split(r"[-/|]", nar)
    for part in parts[1:4]:
        candidate = part.strip()
        if re.match(r'^[A-Z0-9]{8,}$', candidate):
            continue
        if re.match(r'^\d{10}$', candidate):
            continue
        if len(candidate) > 2 and not re.match(r'^\d+$', candidate):
            return candidate.title()
    return ""


def extract_counterparty_account(narration: str) -> str:
    """Extract counterparty account number from UPI VPA or NEFT/IMPS ref."""
    if not narration:
        return ""
    # UPI VPA
    m = _UPI_VPA_RE.search(narration)
    if m:
        return m.group(1).lower()
    # Account number pattern (10–18 digits not part of UTR)
    parts = re.split(r"[/\-|]", narration)
    for part in parts:
        p = part.strip()
        if re.match(r'^\d{10,18}$', p):
            return p
    return ""


def extract_counterparty_ifsc(narration: str) -> str:
    """Extract the FIRST IFSC code from narration (counterparty bank)."""
    if not narration:
        return ""
    matches = _NARRATION_IFSC_RE.findall(narration)
    # Return first one that's NOT the account's own IFSC (can't know here,
    # so just return the first found; normalizer will filter if needed)
    return matches[0] if matches else ""


def is_likely_transaction_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df.columns) < 2:
        return False
    has_numeric = any(
        pd.to_numeric(df[c].dropna().head(5), errors="coerce").notna().any()
        for c in df.columns
    )
    return has_numeric