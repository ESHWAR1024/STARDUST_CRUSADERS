"""
Phase 6 - Format-specific parsers

Each parser takes a file path and returns:
  (raw_df, header_text, source_format, warnings)

- raw_df      : DataFrame with raw content (pre-normalization)
- header_text : free text from the file header (bank name, account info)
- source_format: "csv" | "xlsx" | "pdf" | "image"
- warnings    : list of issues encountered during parsing
"""

import io
import os
import re
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Shared regex constants
#
# IMPORTANT — NO \b anchors on _AMOUNT_RE:
#   \b fails on "7,03,350.00C" because between digit '0' and letter 'C'
#   both are \w chars, so there is no word-boundary. Without \b the pattern
#   correctly matches "7,03,350.00" and stops before "C".
#
# IMPORTANT — _DATE_RE uses only [-/] (not .) for 2-component dates:
#   Using [-/.] would match "00.00", "09.00" etc. as date fragments.
#   Dot-separated dates require all 3 components (dd.mm.yyyy).
# ─────────────────────────────────────────────────────────────────────────────

_AMOUNT_RE = re.compile(
    r'(?<!\d)\d{1,3}(?:[,\s]\d{2,3})*[.,]\d{2}(?!\d)'
    r'|(?<!\d)\d{1,8}[.,]\d{2}(?!\d)'
)

_DATE4_RE = re.compile(
    r'(?<!\d)\d{1,2}\s*[-/.\s|l\\]+\s*\d{1,2}\s*[-/.\s|l\\]+\s*\d{4}(?!\d)'
    r'|(?<!\d)\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}(?!\d)'
)

_DATE_RE = re.compile(
    r'(?<!\d)\d{1,2}\s*[-/.\s|l\\]+\s*\d{1,2}(?:\s*[-/.\s|l\\]+\s*\d{2,4})?(?!\d)'
    r'|(?<!\d)\d{1,2}\s+[A-Za-z]{3,}\s+\d{2,4}(?!\d)'
)

_NON_TX = [
    "opening balance", "closing balance", "b/f", "balance forward",
    "total credit", "total debit", "total deposits", "total withdrawals",
    "statement generated", "computer generated", "registered office",
    "customer care", "this is a computer", "page ", "balance as on",
    "brought forward", "carried forward", "c/f",
]
_CREDIT_KWS = [
    "interest credit", "credit interest", "by transfer", "salary",
    "deposit", "credited", "refund", "received", "neft cr", "imps cr",
    "inward", "p2p cr", "neft-", "imps-p2p",
]
_DEBIT_KWS = [
    "withdrawal", "wdl", "paid", "payment", "debit", "charge", "atm",
    "ecs", "nach", "towards", " db", "outward", "p2p dr", "pos ",
]


# ═══════════════════════════════════════════════════════════════════════════
# CSV
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# CSV Parser
# ---------------------------------------------------------------------------
def parse_csv(file_path: str) -> tuple:
    warnings = []
    header_text = ""

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = [f.readline() for _ in range(8)]
        header_text = " ".join(raw_lines)
    except Exception as e:
        warnings.append(f"Could not read header lines: {e}")

    # Find the real header row by scanning for known column keywords
    from ingestion_config import COLUMN_ROLE_KEYWORDS
    all_col_keywords = set()
    for keywords in COLUMN_ROLE_KEYWORDS.values():
        all_col_keywords.update(keywords)

    skip = 0
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                # Count how many known column names appear in this line
                hits = sum(1 for kw in all_col_keywords if kw.lower() in line.lower())
                if hits >= 3:
                    skip = i
                    break
    except Exception:
        pass

    df = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                skiprows=skip,
                skip_blank_lines=True,
                dtype=str,
                on_bad_lines="skip",
            )
            if len(df.columns) >= 4:
                break
        except Exception as e:
            warnings.append(f"CSV read error (enc={encoding}): {e}")
            continue

    if df is None or df.empty:
        warnings.append("CSV parsing failed entirely")
        return pd.DataFrame(), header_text, "csv", warnings

    return df, header_text, "csv", warnings


# ---------------------------------------------------------------------------
# Excel Parser
# ---------------------------------------------------------------------------
def parse_xlsx(file_path: str) -> tuple:
    warnings = []
    header_text = ""

    try:
        # Read raw with no header to capture bank metadata rows
        raw = pd.read_excel(file_path, header=None, dtype=str, nrows=6)
        header_text = " ".join(
            str(v) for row in raw.values for v in row if pd.notna(v)
        )
    except Exception as e:
        warnings.append(f"Could not read Excel header block: {e}")

    # Read with header starting at different rows
    df = None
    for header_row in (0, 1, 2, 3, 4):
        try:
            df = pd.read_excel(
                file_path,
                header=header_row,
                dtype=str,
                engine="openpyxl",
            )
            if len(df.columns) >= 4 and not all(str(c).startswith("Unnamed") for c in df.columns):
                break
        except Exception as e:
            warnings.append(f"Excel read failed at header_row={header_row}: {e}")
            continue

    if df is None:
        warnings.append("Excel parsing failed entirely")
        return pd.DataFrame(), header_text, "xlsx", warnings

    return df, header_text, "xlsx", warnings


# ---------------------------------------------------------------------------
# PDF Parser (pdfplumber - handles text-layer PDFs)
# ---------------------------------------------------------------------------
def parse_pdf(file_path: str, password: str = None, password_candidates: list = None) -> tuple:
    """
    password: a single password to try if the PDF turns out to be
    encrypted.
    password_candidates: an optional list of passwords to try in order
    (useful for investigators who have a few likely candidates - DOB,
    PAN, account-number-based formulas banks commonly use - rather than
    one confirmed password). `password`, if given, is tried first.
    """
    try:
        import pdfplumber
        from pdfplumber.utils.exceptions import PdfminerException
    except ImportError:
        return pd.DataFrame(), "", "pdf", ["pdfplumber not installed"]

    warnings = []
    header_text = ""
    all_rows = []
    headers = None

    candidates = ([password] if password else []) + list(password_candidates or [])

    def _open_pdf():
        """
        Try opening unprotected first (the common case), then each
        candidate password in order. Raises the LAST encryption-related
        exception if every attempt fails, so the caller can distinguish
        "this PDF is password-protected and none of the supplied
        passwords worked" from any other kind of parsing failure.
        """
        try:
            return pdfplumber.open(file_path)
        except PdfminerException as e:
            if not candidates:
                raise PdfminerException(
                    "PDF appears to be password-protected. Re-run with "
                    "--pdf-password (or --pdf-passwords for multiple "
                    "candidates) to supply one."
                ) from e
            last_err = e
            for pw in candidates:
                try:
                    return pdfplumber.open(file_path, password=pw)
                except PdfminerException as e2:
                    last_err = e2
                    continue
            raise PdfminerException(
                f"PDF is password-protected and none of the "
                f"{len(candidates)} supplied password(s) worked."
            ) from last_err

    try:
        with _open_pdf() as pdf:
            full_text_chunks = []
            for page_num, page in enumerate(pdf.pages):
                # Accumulate text from EVERY page for bank detection, not
                # just a narrow slice of page 1. Safety against picking up
                # a counterparty's bank instead of the statement's own
                # comes from detect_bank()'s anchoring (an explicit "IFSC"
                # label for the IFSC-lookup path, earliest-occurrence
                # preference for the keyword-name fallback path) — not
                # from artificially restricting how much text is searched.
                page_text = page.extract_text() or ""
                if page_text:
                    full_text_chunks.append(page_text)
            header_text = "\n".join(full_text_chunks)

            for page_num, page in enumerate(pdf.pages):
                # Extract tables with explicit settings for better accuracy
                tables = page.extract_tables({
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                })

                if not tables:
                    # Fallback: try default extraction
                    tables = page.extract_tables()

                for table in tables:
                    if not table:
                        continue
                    # Need at least 2 rows (header + 1 data row) only when
                    # we're still LOOKING for the table that has the
                    # header. Once headers is already established, a
                    # continuation page can legitimately have just 1 data
                    # row and no header of its own - rejecting it here
                    # would (and did) drop short continuation pages.
                    if headers is None and len(table) < 2:
                        continue

                    # Once the real transaction table is already found on
                    # an earlier page, a later page's table is a
                    # CONTINUATION of it - it won't repeat the header row,
                    # so re-validating its row 0 against header keywords
                    # rejected every page after the first. Accept it as a
                    # continuation purely by column count instead.
                    if headers is not None:
                        if len(table[0]) != n_cols:
                            continue
                    elif not _is_transaction_table(table):
                        continue

                    # Clean table: strip whitespace, replace None with ""
                    cleaned = []
                    for row in table:
                        clean_row = [
                            str(cell).strip() if cell is not None else ""
                            for cell in row
                        ]
                        if any(c for c in clean_row):
                            cleaned.append(clean_row)

                    if not cleaned:
                        continue

                    if headers is None:
                        header_row_idx = _find_pdf_header_row(cleaned)
                        raw_headers = cleaned[header_row_idx]
                        # Merge adjacent cells that are continuation of a split header
                        # e.g. ["Ref No./Cheque No", ". Debit"] → keep as separate cols
                        # but fix obvious splits where cell starts with ". " or "/ "
                        headers = _merge_split_headers(raw_headers)
                        n_cols = len(headers)
                        data_rows = cleaned[header_row_idx + 1:]
                    else:
                        data_rows = cleaned
                        if cleaned[0] == headers or _merge_split_headers(cleaned[0]) == headers:
                            data_rows = cleaned[1:]

                    for row in data_rows:
                        if len(row) < n_cols:
                            row += [""] * (n_cols - len(row))
                        all_rows.append(row[:n_cols])

    except PdfminerException as e:
        return pd.DataFrame(), header_text, "pdf", [f"PASSWORD_PROTECTED: {e}"]
    except Exception as e:
        return pd.DataFrame(), header_text, "pdf", [f"PDF parsing error: {e}"]

    if not all_rows or headers is None:
        warnings.append("No tables found in PDF — trying text fallback")
        return _pdf_text_fallback(file_path, header_text, warnings)

    df = pd.DataFrame(all_rows, columns=headers)
    # Replace empty strings with NaN for downstream compatibility
    df = df.replace("", pd.NA)
    warnings.append(f"PDF: extracted {len(df)} rows, {len(headers)} columns")

    return df, header_text, "pdf", warnings


def _merge_split_headers(headers: list) -> list:
    """
    pdfplumber sometimes splits a single header cell across two columns
    when a line break falls inside a cell (e.g. "Ref No./Cheque No" + ". Debit").
    Merge cells where the next cell starts with punctuation or lowercase.
    """
    merged = []
    skip_next = False
    for i, cell in enumerate(headers):
        if skip_next:
            skip_next = False
            continue
        cell = str(cell).strip()
        if i + 1 < len(headers):
            next_cell = str(headers[i + 1]).strip()
            # Merge if next cell starts with punctuation or is a continuation fragment
            if next_cell and next_cell[0] in (".", ",", "/", " ") and len(next_cell) < 10:
                merged.append((cell + next_cell).strip())
                skip_next = True
                continue
        merged.append(cell)
    return merged


def _find_pdf_header_row(rows: list) -> int:
    """Find which row index contains the actual column headers."""
    from ingestion_config import COLUMN_ROLE_KEYWORDS
    all_col_keywords = set()
    for kws in COLUMN_ROLE_KEYWORDS.values():
        all_col_keywords.update(kw.lower() for kw in kws)

    best_idx = 0
    best_score = 0
    for i, row in enumerate(rows[:6]):
        score = sum(
            1 for cell in row
            if any(kw in str(cell).lower() for kw in all_col_keywords)
        )
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _is_transaction_table(table: list) -> bool:
    """
    Return True only if this table looks like a transaction table
    (has date + amount columns), not an account info block.
    """
    if not table or len(table) < 2:
        return False
    # Must have at least 5 columns (date, narration, ref, debit/credit, balance)
    if len(table[0]) < 5:
        return False
    # Header row must contain at least one date/amount keyword
    header_text = " ".join(str(c).lower() for c in table[0])
    date_keywords = {"date", "txn", "tran", "post"}
    amount_keywords = {"debit", "credit", "withdrawal", "deposit", "balance", "amount"}
    has_date = any(kw in header_text for kw in date_keywords)
    has_amount = any(kw in header_text for kw in amount_keywords)
    return has_date and has_amount


def _pdf_text_fallback(file_path: str, header_text: str, warnings: list) -> tuple:
    """
    Last resort for PDFs where table extraction fails completely:
    extract all text and try to parse line by line.
    """
    try:
        import pdfplumber
        all_lines = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                all_lines.extend(text.splitlines())
        df = _ocr_text_to_dataframe(all_lines, warnings)
        warnings.append("PDF: used text-line fallback (table extraction failed)")
        return df, header_text, "pdf", warnings
    except Exception as e:
        return pd.DataFrame(), header_text, "pdf", warnings + [f"PDF text fallback failed: {e}"]


# ---------------------------------------------------------------------------
# Image / Scanned Statement Parser (Tesseract OCR)
# ---------------------------------------------------------------------------
def parse_image(file_path: str) -> tuple:
    try:
        import pytesseract
        from PIL import Image
        import PIL.ImageEnhance as IE
    except ImportError:
        return pd.DataFrame(), "", "image", ["pytesseract or pillow not installed"]

    warnings = []
    header_text = ""

    try:
        img = Image.open(file_path).convert("L")
        w, h = img.size

        # Scale to at least 2400px wide for accuracy
        scale = max(1, 2400 // max(w, 1))
        if scale > 1:
            img = img.resize((w * scale, h * scale), Image.LANCZOS)

        img = IE.Contrast(img).enhance(1.5)
        img = IE.Sharpness(img).enhance(2.0)

        # Get word-level bounding boxes — this preserves column position info
        # that plain text output loses (critical for reconstructing table structure)
        tsv_data = pytesseract.image_to_data(
            img, config="--psm 6 --oem 3",
            output_type=pytesseract.Output.DATAFRAME
        )

        # Also get raw text for header extraction
        raw_text = pytesseract.image_to_string(img, config="--psm 6 --oem 3 -c preserve_interword_spaces=1")
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        header_text = " ".join(lines[:5])

    except Exception as e:
        return pd.DataFrame(), "", "image", [f"Image load/OCR error: {e}"]

    df = _ocr_bbox_to_dataframe(tsv_data, warnings)

    # Fallback to line-split parser if bbox reconstruction fails
    if df.empty and lines:
        warnings.append("OCR bbox reconstruction failed, trying line-split fallback")
        df = _ocr_linefallback_to_dataframe(lines, warnings)

    if df.empty:
        warnings.append("OCR produced no parseable table rows")

    return df, header_text, "image", warnings


def _ocr_bbox_to_dataframe(tsv: "pd.DataFrame", warnings: list) -> "pd.DataFrame":
    """
    OpenCV pipeline for clean OCR:
      1. Greyscale
      2. Upscale to ≥3000 px wide  (Tesseract sweet-spot)
      3. Gentle denoising           (preserves thin strokes)
      4. Adaptive thresholding      (handles uneven lighting / shadows)
    """
    import cv2
    from PIL import Image as PILImage

    img = cv2.imread(file_path)
    gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None
            else np.array(PILImage.open(file_path).convert("L")))

    h, w = gray.shape
    if w < 3000:
        gray = cv2.resize(gray, None, fx=3000 / w, fy=3000 / w,
                          interpolation=cv2.INTER_CUBIC)

    gray   = cv2.fastNlMeansDenoising(gray, h=8,
                                       templateWindowSize=7,
                                       searchWindowSize=21)
    binary = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    blockSize=51, C=15, 
)
    return PILImage.fromarray(binary)


# ═══════════════════════════════════════════════════════════════════════════
# Core line-based transaction parser  (shared by image + PDF fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _text_line_parser(
    lines: list,
    header_text: str,
    warnings: list,
    source: str = "image",
) -> pd.DataFrame:
    """Parse OCR/PDF text lines into a transaction table via header detection and balance-delta debit/credit inference."""
    from schema_detector import detect_bank_from_text, parse_amount
    from ingestion_config import BANK_FORMAT_REGISTRY

    bank_code = detect_bank_from_text(header_text) if header_text else None
    bfmt      = BANK_FORMAT_REGISTRY.get(bank_code) if bank_code else None

    date_col  = bfmt["date_col"]           if bfmt else "Date"
    narr_col  = bfmt["narration_col"]      if bfmt else "Narration"
    ref_col   = bfmt.get("ref_col", "Ref") if bfmt else "Ref"
    debit_col = bfmt["debit_col"]          if bfmt else "Debit"
    cred_col  = bfmt["credit_col"]         if bfmt else "Credit"
    bal_col   = bfmt["balance_col"]        if bfmt else "Balance"

    HDR = [
        "date", "debit", "credit", "balance", "narration", "description",
        "withdrawal", "deposit", "particulars", "remarks", "ref", "txn",
        "cheque", "post", "tran", "amount",
    ]
    header_idx = None
    for i, ll in enumerate(logical_lines):
        hits = sum(1 for kw in header_keywords if kw in ll["text"].lower())
        if hits >= 3:
            header_idx = i
            break
            
    if hdr_idx is None:
        warnings.append(f"Line parser ({source}): no header found")
        return pd.DataFrame()

    def _ocr_clean(s):
        return s.translate(str.maketrans("OolI", "0011"))

    def _has_amt(s):  return bool(_AMOUNT_RE.search(_ocr_clean(s)))
    def _has_date(s): return bool(_DATE_RE.search(_ocr_clean(s)))
    def _is_non(s):   return any(p in s.lower() for p in _NON_TX)

    def _is_cont(s):
        s = s.strip()
        if not s: return True
        if re.match(r'^[\s|({!]*[CDRcdR]{1,2}[\s|)}\]!]*$', s): return True
        if re.match(r'^[\s|({!\[]*\d{2,4}[\s|)}\]!]*(?:[CDRcdR][\s|)}\]!]*)?$', s): return True
        return len(s) <= 3 and not _has_amt(s) and not _has_date(s)

    def _is_main(s):
        if _is_non(s): 
            return False
            
        cln = _ocr_clean(s).lstrip()
        m = _DATE_RE.search(cln)
        
        # Increased to 50 to account for wide left margins or empty first columns
        return bool(m and m.start() < 50 and _has_amt(cln))

    blocks, cur = [], None
    for ln in lines[hdr_idx + 1:]:
        if _is_non(ln):
            if cur:
                blocks.append(cur)
                cur = None
        elif _is_cont(ln):
            if cur:
                cur.append(ln)
        elif _is_main(ln):
            if cur:
                blocks.append(cur)
            cur = [ln]
        else:
            if cur and ln.strip():
                cur.append(ln)
    if cur:
        blocks.append(cur)

    if not blocks:
        warnings.append(f"Line parser ({source}): no transaction blocks found")
        return pd.DataFrame()

    def _build_date(partial: str, year: str) -> str:
        parts = re.split(r'[-/.\s]+', partial.strip())
        if len(parts) >= 3:
            return '-'.join(parts[:2]) + '-' + year
        elif len(parts) == 2:
            return '-'.join(parts) + '-' + year
        return partial

    def _best_date(blk: list) -> str:
        full, partial = [], []
        for ln in blk:
            cln = _ocr_clean(ln)
            full    += _DATE4_RE.findall(cln)
            partial += _DATE_RE.findall(cln)
        if full:
            return full[-1]
        if partial:
            best = sorted(partial, key=len)[-1]
            for ln in blk[1:]:
                m = re.search(r'\b(20\d{2})\b', ln)
                if m:
                    return _build_date(best, m.group(1))
            return best
        return ""

    def _desc(blk: list) -> str:
        s = ' '.join(blk)
        s = _AMOUNT_RE.sub(' ', s)
        s = _DATE4_RE.sub(' ', s)
        s = _DATE_RE.sub(' ', s)
        s = re.sub(r'\b20\d{2}\b', ' ', s)
        s = re.sub(r'[|(){}\[\]~=!¢]', ' ', s)
        s = re.sub(r'\b[A-Za-z]\b', ' ', s)
        s = re.sub(r'^[-|\s]+', '', s)
        s = re.sub(r'[-|\s]+$', '', s)
        return re.sub(r'\s+', ' ', s).strip().upper()

    def _ref(main_line: str) -> str:
        m = re.search(r'\b([A-Z0-9]{8,22})\b', main_line)
        if m and any(c.isdigit() for c in m.group(1)):
            return m.group(1)
        return ''

    parsed = []
    for blk in blocks:
        all_raw  = _AMOUNT_RE.findall(' '.join(blk))
        bal_str  = all_raw[-1] if all_raw else ''
        rest_raw = all_raw[:-1] if len(all_raw) > 1 else []

        bal  = parse_amount(bal_str)
        rest = [parse_amount(a) for a in rest_raw if parse_amount(a) > 0]

        parsed.append({
            "date": _best_date(blk),
            "desc": _desc(blk),
            "ref":  _ref(blk[0]),
            "bal":  bal,
            "rest": rest,
        })

    rows, prev = [], None
    for i, p in enumerate(parsed):
        bal, rest  = p["bal"], p["rest"]
        debit = credit = 0.0

        if prev is None or prev == 0.0:
            amt  = rest[0] if rest else 0.0
            dl   = p["desc"].lower()
            is_c = any(k in dl for k in _CREDIT_KWS)
            is_d = any(k in dl for k in _DEBIT_KWS) and not is_c
            if is_c:
                credit = amt
            elif is_d:
                debit = amt
            elif i + 1 < len(parsed) and parsed[i + 1]["bal"] > 0:
                nxt = parsed[i + 1]["bal"]
                if   nxt > bal: credit = amt
                elif nxt < bal: debit  = amt
                else:           credit = amt
            else:
                credit = amt
        else:
            delta = round(bal - prev, 2)
            if   delta < 0: debit  = abs(delta)
            elif delta > 0: credit = delta
            else:
                amt  = rest[0] if rest else 0.0
                dl   = p["desc"].lower()
                credit = amt if any(k in dl for k in _CREDIT_KWS) else 0.0
                debit  = amt if credit == 0.0 else 0.0

        prev = bal
        rows.append({
            date_col:  p["date"],
            narr_col:  p["desc"],
            ref_col:   p["ref"],
            debit_col: debit,
            cred_col:  credit,
            bal_col:   bal,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=col_names)


# ---------------------------------------------------------------------------
# JSON parser (for flattened transaction exports from ERPs/systems)
# ---------------------------------------------------------------------------
def parse_json(file_path: str) -> tuple:
    import json
    warnings = []
    header_text = ""

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list of records and {"transactions": [...]} wrapper
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Find the first list value
            for key, val in data.items():
                if isinstance(val, list) and len(val) > 0:
                    records = val
                    header_text = str(key)
                    break
            else:
                return pd.DataFrame(), "", "json", ["No list of records found in JSON"]
        else:
            return pd.DataFrame(), "", "json", ["Unsupported JSON structure"]

        df = pd.json_normalize(records)
        return df, header_text, "json", warnings

    except Exception as e:
        return pd.DataFrame(), "", "json", [f"JSON parse error: {e}"]


# ---------------------------------------------------------------------------
# TSV / pipe-delimited parser
# ---------------------------------------------------------------------------
def parse_tsv(file_path: str) -> tuple:
    warnings = []
    header_text = ""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(500)
        # Detect delimiter
        delim = "\t" if "\t" in sample else "|" if "|" in sample else ","
        df = pd.read_csv(file_path, sep=delim, dtype=str,
                         encoding="utf-8", errors="replace")
        return df, header_text, "tsv", warnings
    except Exception as e:
        return pd.DataFrame(), "", "tsv", [f"TSV parse error: {e}"]