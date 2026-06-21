"""
Phase 6 — Schema Detector
Bank detection, column mapping, amount/date parsing, channel/counterparty.
"""

import re
import pandas as pd
from ingestion_config import BANK_FORMAT_REGISTRY, BANK_NAME_KEYWORDS


# ── Bank detection ────────────────────────────────────────────────────────────

def detect_bank_from_text(text: str) -> str | None:
    text_lower = text.lower()
    for keyword, bank_code in BANK_NAME_KEYWORDS.items():
        if keyword in text_lower:
            return bank_code
    m = re.search(
        r"([A-Za-z\s]+(?:Bank|Sahakari|Grameen|Gramin)(?:\s+Ltd|\s+Limited)?)\b",
        text, re.IGNORECASE,
    )
    if m:
        candidate = m.group(1).strip()
        noise = {"internet", "mobile", "phone", "sms", "online"}
        if len(candidate) > 4 and not any(n in candidate.lower() for n in noise):
            return candidate.upper()
    return None


def detect_bank_from_columns(columns: list) -> tuple[str | None, dict | None]:
    col_set = {str(c).strip() for c in columns}
    best_bank, best_score = None, 0
    for bank_code, fmt in BANK_FORMAT_REGISTRY.items():
        required = {
            fmt["date_col"], fmt["narration_col"],
            fmt["debit_col"], fmt["credit_col"], fmt["balance_col"],
        }
        score = len(required & col_set)
        if score > best_score:
            best_score, best_bank = score, bank_code
    if best_score >= 4:
        return best_bank, BANK_FORMAT_REGISTRY[best_bank]
    return None, None


def detect_header_row(raw_df: pd.DataFrame, max_scan_rows: int = 12) -> int:
    all_fmt_cols = set()
    for fmt in BANK_FORMAT_REGISTRY.values():
        all_fmt_cols.update([
            fmt["date_col"], fmt["narration_col"], fmt["ref_col"],
            fmt["debit_col"], fmt["credit_col"], fmt["balance_col"],
        ])
    for row_idx in range(min(max_scan_rows, len(raw_df))):
        row_vals = [str(v).strip() for v in raw_df.iloc[row_idx].values if pd.notna(v)]
        if sum(1 for v in row_vals if v in all_fmt_cols) >= 3:
            return row_idx
    return 0


def detect_generic_format(columns: list) -> dict | None:
    from ingestion_config import GENERIC_COLUMN_ROLES, GENERIC_MIN_REQUIRED_ROLES
    cols = [str(c).strip() for c in columns]
    cols_lower = {c: c.lower() for c in cols}
    role_matches, used = {}, set()

    for role in ["date_col", "narration_col", "debit_col",
                 "credit_col", "balance_col", "ref_col"]:
        keywords = GENERIC_COLUMN_ROLES[role]
        best_col, best_len = None, -1
        for col in cols:
            if col in used:
                continue
            for kw in keywords:
                if kw in cols_lower[col] and len(kw) > best_len:
                    best_col, best_len = col, len(kw)
        if best_col:
            role_matches[role] = best_col
            used.add(best_col)

    if "narration_col" not in role_matches:
        return None

    return {
        "bank_name":     "Unknown / Generic",
        "date_col":      role_matches.get("date_col", ""),
        "narration_col": role_matches["narration_col"],
        "ref_col":       role_matches.get("ref_col", ""),
        "debit_col":     role_matches.get("debit_col", ""),
        "credit_col":    role_matches.get("credit_col", ""),
        "balance_col":   role_matches.get("balance_col", ""),
        "date_formats": [
            "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y",
            "%d-%b-%Y", "%d %b %Y", "%Y-%m-%d",
            "%d %b %y", "%d-%b-%y", "%d.%m.%Y", "%d.%m.%y",
        ],
    }


# ── Amount parsing ────────────────────────────────────────────────────────────

def parse_amount(val) -> float:
    """
    Robustly parse a money amount from any OCR / CSV / Excel cell.

    Handles:
      • CR / DR / C / D balance indicators  "7,03,350.00CR" → 703350.0
      • OCR bracket / pipe artifacts         "]", "}", "|", "¢"
      • Spaces inside numbers               "7,03 350.00"   → 703350.0
      • OCR digit substitutions             Z→2, O→0, l→1
      • Dash-only / nil / blank             → 0.0
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    s = str(val).strip()
    if not s or s in ("<NA>", "nan", "NaN", "None"):
        return 0.0

    # Strip CR/DR/C/D balance indicator at end
    s = re.sub(r'\s*(?:CR?|DR?)\s*$', '', s, flags=re.IGNORECASE).strip()

    # Strip OCR bracket / punctuation artifacts
    s = re.sub(r'^[\[({|¢!]+', '', s).strip()
    s = re.sub(r'[\])}|¢!]+$', '', s).strip()

    # Remove currency symbols and commas
    s = s.replace('₹', '').replace('$', '').replace('INR', '').replace(',', '').strip()

    # Collapse spaces inside numbers: "7 03 350.00" → "703350.00"
    s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)

    # Fix common OCR digit substitutions inside numeric context
    s = re.sub(r'(?<!\w)Z(?=\d)', '2', s)
    s = re.sub(r'(?<=\d)O(?!\w)', '0', s)
    s = re.sub(r'(?<!\w)[Ol](?=\d{2,})', '0', s)

    if s in ('', '-', '.', 'nil', 'n/a', 'na', '--'):
        return 0.0

    try:
        return round(float(s), 2)
    except ValueError:
        m = re.search(r'\d+\.\d{2}', s)
        if m:
            try:
                return round(float(m.group(0)), 2)
            except ValueError:
                pass
        return 0.0


# ── Date parsing ─────────────────────────────────────────────────────────────

_EXTRA_FMTS = [
    "%d.%m.%Y", "%d.%m.%y",
    "%d %b %Y", "%d %b %y",
    "%d-%b-%Y", "%d-%b-%y",
    "%d/%b/%Y",
    "%Y/%m/%d",
]


def parse_date(val, date_formats: list) -> str:
    """
    Parse a date from any raw cell value.

    Handles:
      • Multiple separator chars  (-, /, ., space)
      • OCR noise chars           (O→0, l→1, I→1)
      • Dot-separated dates       "25.03.2025"
      • Month abbreviations       "01-Apr-2025"
      • Pandas dayfirst fallback  (guarded: only accepts years 2000-2040)
    """
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""

    s = str(val).strip()
    if not s or s in ("<NA>", "nan", "NaN", "None", "-", ""):
        return ""

    # OCR character noise
    s = s.replace('O', '0').replace('o', '0')
    s = s.replace('l', '1').replace('I', '1')

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

    for fmt in list(date_formats) + _EXTRA_FMTS:
        try:
            result = pd.to_datetime(s, format=fmt)
            if pd.notna(result):
                return result.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

    # Pandas dayfirst fallback — guard against implausible years
    try:
        if not any(ch.isdigit() for ch in s):
            return ""
        result = pd.to_datetime(s, dayfirst=True, errors='coerce')
        if pd.notna(result) and 2000 <= result.year <= 2040:
            return result.strftime("%Y-%m-%d")
    except Exception:
        pass

    return ""


# ── Channel & counterparty ────────────────────────────────────────────────────

def infer_channel(narration: str) -> str:
    from ingestion_config import CHANNEL_KEYWORDS
    nl = narration.lower()
    for channel, keywords in CHANNEL_KEYWORDS:
        if any(kw in nl for kw in keywords):
            return channel
    return "OTHER"


def extract_counterparty(narration: str) -> str:
    parts = re.split(r'[-/]', narration)
    if len(parts) >= 2:
        candidate = parts[1].strip().title()
        if len(candidate) > 2 and not candidate.replace(" ", "").isnumeric():
            return candidate
    return ""