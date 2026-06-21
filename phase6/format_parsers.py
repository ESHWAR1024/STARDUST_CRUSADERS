"""
Phase 6 — Format-specific parsers
Supports: CSV, XLSX, PDF (pdfplumber), Image (Tesseract + OpenCV)
"""

import os, re
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

def parse_csv(file_path: str) -> tuple:
    warnings = []
    header_text = ""

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = [f.readline() for _ in range(12)]
        header_text = "".join(raw_lines)
    except Exception as e:
        warnings.append(f"Header read error: {e}")

    from ingestion_config import BANK_FORMAT_REGISTRY
    kws = set()
    for fmt in BANK_FORMAT_REGISTRY.values():
        kws.update([fmt["date_col"], fmt["narration_col"],
                    fmt["debit_col"], fmt["credit_col"], fmt["balance_col"]])

    skip = 0
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if sum(1 for k in kws if k.lower() in line.lower()) >= 3:
                    skip = i
                    break
    except Exception:
        pass

    df = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            c = pd.read_csv(
                file_path, encoding=enc, skiprows=skip,
                skip_blank_lines=True, dtype=str, on_bad_lines="skip",
            )
            if len(c.columns) >= 4:
                df = c
                break
        except Exception:
            continue

    if df is None or df.empty:
        warnings.append("CSV parsing failed entirely")
        return pd.DataFrame(), header_text, "csv", warnings

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df, header_text, "csv", warnings


# ═══════════════════════════════════════════════════════════════════════════
# XLSX
# ═══════════════════════════════════════════════════════════════════════════

def parse_xlsx(file_path: str) -> tuple:
    warnings = []
    header_text = ""

    try:
        raw = pd.read_excel(file_path, header=None, dtype=str, nrows=8)
        header_text = " ".join(
            str(v) for row in raw.values for v in row if pd.notna(v)
        )
    except Exception as e:
        warnings.append(f"Excel header read error: {e}")

    df = None
    for hr in range(6):
        try:
            c = pd.read_excel(file_path, header=hr, dtype=str, engine="openpyxl")
            if len(c.columns) >= 4 and not all(
                str(col).startswith("Unnamed") for col in c.columns
            ):
                df = c
                break
        except Exception:
            continue

    if df is None:
        warnings.append("Excel parsing failed entirely")
        return pd.DataFrame(), header_text, "xlsx", warnings

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df, header_text, "xlsx", warnings


# ═══════════════════════════════════════════════════════════════════════════
# PDF
# ═══════════════════════════════════════════════════════════════════════════

def parse_pdf(file_path: str) -> tuple:
    try:
        import pdfplumber
    except ImportError:
        return pd.DataFrame(), "", "pdf", ["pdfplumber not installed"]

    warnings = []
    header_text = ""
    all_rows, headers, n_cols = [], None, 0

    try:
        with pdfplumber.open(file_path) as pdf:
            for pnum, page in enumerate(pdf.pages):
                if pnum == 0:
                    header_text = (page.extract_text() or "")[:800]

                tables = None
                for strat in (
                    {"vertical_strategy": "lines", "horizontal_strategy": "lines",
                     "snap_tolerance": 3, "join_tolerance": 3},
                    {},
                ):
                    tables = (page.extract_tables(strat)
                              if strat else page.extract_tables())
                    if tables:
                        break

                for table in (tables or []):
                    if not table or len(table) < 2 or not _is_tx_table(table):
                        continue
                    cleaned = [
                        [str(c).strip() if c else "" for c in row]
                        for row in table if any(c for c in row)
                    ]
                    if not cleaned:
                        continue
                    if headers is None:
                        hi = _find_hdr(cleaned)
                        headers = _merge_hdrs(cleaned[hi])
                        n_cols = len(headers)
                        data = cleaned[hi + 1:]
                    else:
                        data = cleaned
                        if _merge_hdrs(cleaned[0]) == headers:
                            data = cleaned[1:]
                    for row in data:
                        all_rows.append((row + [""] * n_cols)[:n_cols])

    except Exception as e:
        return pd.DataFrame(), header_text, "pdf", [f"PDF error: {e}"]

    if not all_rows or headers is None:
        warnings.append("No tables found — trying text fallback")
        return _pdf_text_fallback(file_path, header_text, warnings)

    df = pd.DataFrame(all_rows, columns=headers).replace("", pd.NA)
    warnings.append(f"PDF table: {len(df)} rows, {len(headers)} cols")
    return df, header_text, "pdf", warnings


def _is_tx_table(t):
    if not t or len(t) < 2 or len(t[0]) < 4:
        return False
    h = " ".join(str(c).lower() for c in t[0])
    return (
        any(w in h for w in ("date", "txn", "tran", "post")) and
        any(w in h for w in ("debit", "credit", "balance",
                             "withdrawal", "deposit", "amount"))
    )


def _find_hdr(rows):
    from ingestion_config import BANK_FORMAT_REGISTRY
    kws = set()
    for fmt in BANK_FORMAT_REGISTRY.values():
        kws.update([fmt["date_col"].lower(), fmt["narration_col"].lower(),
                    fmt["debit_col"].lower(), fmt["credit_col"].lower(),
                    fmt["balance_col"].lower()])
    best_i, best_s = 0, 0
    for i, row in enumerate(rows[:8]):
        s = sum(1 for c in row if any(k in str(c).lower() for k in kws))
        if s > best_s:
            best_s, best_i = s, i
    return best_i


def _merge_hdrs(hdrs):
    out, skip = [], False
    for i, cell in enumerate(hdrs):
        if skip:
            skip = False
            continue
        cell = str(cell).strip()
        if i + 1 < len(hdrs):
            nxt = str(hdrs[i + 1]).strip()
            if nxt and nxt[0] in (".", ",", "/") and len(nxt) < 14:
                out.append(cell + nxt)
                skip = True
                continue
        out.append(cell)
    return out


def _pdf_text_fallback(file_path, header_text, warnings):
    try:
        import pdfplumber
        lines = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                lines.extend(
                    (page.extract_text(x_tolerance=2, y_tolerance=2) or "")
                    .splitlines()
                )
        df = _text_line_parser(lines, header_text, warnings, source="pdf")
        warnings.append("PDF: used text-line fallback")
        return df, header_text, "pdf", warnings
    except Exception as e:
        return (pd.DataFrame(), header_text, "pdf",
                warnings + [f"PDF fallback failed: {e}"])


# ═══════════════════════════════════════════════════════════════════════════
# Image
# ═══════════════════════════════════════════════════════════════════════════

def parse_image(file_path: str) -> tuple:
    """
    Always uses the line-based parser — never bbox.

    Why no bbox:
      Scanned image OCR gives noisy word coordinates.  Column anchors
      shift by ±20 px per word, causing debit and credit to collapse
      into a single "merged" column.  The line-based approach avoids
      columns entirely and uses balance-delta to split debit/credit.
    """
    try:
        import pytesseract
    except ImportError:
        return pd.DataFrame(), "", "image", ["pytesseract not installed"]

    warnings = []

    try:
        img = _preprocess(file_path)
    except Exception as e:
        return pd.DataFrame(), "", "image", [f"Preprocessing failed: {e}"]

    try:
        raw_text = pytesseract.image_to_string(
        img, 
        config="--psm 6 --oem 3 -c preserve_interword_spaces=1"
    )
    
    # ── DEBUG INJECTION: Save the raw OCR text to a file ──
        with open("debug_ocr_output.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)
        print("\n--- SAVED RAW OCR TO debug_ocr_output.txt ---")
    # ──────────────────────────────────────────────────────
    except Exception as e:
        return pd.DataFrame(), "", "image", [f"OCR failed: {e}"]

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    header_text = "\n".join(lines[:25])

    df = _text_line_parser(lines, header_text, warnings, source="image")
    if df.empty:
        warnings.append("OCR: no parseable transactions found")

    return df, header_text, "image", warnings


def _preprocess(file_path: str):
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
    hdr_idx = None
    for i, ln in enumerate(lines):
        if sum(1 for k in HDR if k in ln.lower()) >= 3:
            hdr_idx = i
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
        return bool(m and m.start() < 50)

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

    df = pd.DataFrame(rows)
    warnings.append(
        f"Line parser ({source}): {len(df)} transactions"
        + (f", bank={bank_code}" if bank_code else "")
    )
    return df