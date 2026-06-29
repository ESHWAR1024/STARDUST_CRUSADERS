"""
Phase 6 — Format-specific parsers  (v2 — final)

Each parser returns:
  (raw_df, header_text, source_format, warnings_list)

Bank coverage:
  PDF  : IDFC First Bank, Yes Bank, RBL/Ratnakar, Bank of Baroda,
         Bandhan, Federal, generic word-position, text-line fallback
  XLSX : BOB/Federal (BENEF cols), Paytm (COD_DRCR), Kerala Gramin,
         IndusInd XLS (single-amount + balance indicator)
  CSV  : Axis Bank (skip metadata rows), tab-separated (Yash Dubey)
  Image: Tesseract OCR with bbox alignment
  JSON / TSV: generic
"""

import io, os, re
import numpy as np
import pandas as pd
from collections import defaultdict


# ── shared helpers ─────────────────────────────────────────────────────────

def _read_text_lines(path, n=25):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                return [f.readline() for _ in range(n)]
        except Exception:
            continue
    return []

def _all_col_keywords():
    from ingestion_config import COLUMN_ROLE_KEYWORDS
    kws = set()
    for lst in COLUMN_ROLE_KEYWORDS.values():
        kws.update(k.lower() for k in lst)
    return kws


# ══════════════════════════════════════════════════════════════════════════════
# CSV
# ══════════════════════════════════════════════════════════════════════════════
def parse_csv(file_path):
    warnings = []
    raw_lines = _read_text_lines(file_path, 25)
    header_text = " ".join(raw_lines[:10])
    all_kw = _all_col_keywords()

    # detect separator
    sample = "".join(raw_lines[:5])
    sep = "\t" if sample.count("\t") > sample.count(",") else ","

    # find the header row (first row with ≥3 keyword hits)
    skip = 0
    for i, line in enumerate(raw_lines):
        hits = sum(1 for kw in all_kw if kw.lower() in line.lower())
        if hits >= 3:
            skip = i
            break

    df = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        for s in (sep, None):
            try:
                kw = {"sep": s, "engine": "python"} if s is None else {"sep": s}
                df = pd.read_csv(
                    file_path, encoding=encoding, skiprows=skip,
                    skip_blank_lines=True, dtype=str, on_bad_lines="skip", **kw
                )
                if len(df.columns) >= 4:
                    break
            except Exception as e:
                warnings.append(f"CSV enc={encoding}: {e}")
        if df is not None and len(df.columns) >= 4:
            break

    if df is None or df.empty:
        warnings.append("CSV parsing failed")
        return pd.DataFrame(), header_text, "csv", warnings
    return df, header_text, "csv", warnings


# ══════════════════════════════════════════════════════════════════════════════
# XLSX / XLS
# ══════════════════════════════════════════════════════════════════════════════
def parse_xlsx(file_path):
    warnings = []
    ext = os.path.splitext(file_path)[1].lower()

    # read first 20 rows raw for header detection
    try:
        engine = "xlrd" if ext == ".xls" else "openpyxl"
        raw = pd.read_excel(file_path, header=None, dtype=str,
                            nrows=20, engine=engine)
        header_text = " ".join(
            str(v) for row in raw.values
            for v in row if pd.notna(v) and str(v) != "nan"
        )
    except Exception as e:
        warnings.append(f"Excel header scan error: {e}")
        header_text = ""
        raw = pd.DataFrame()

    all_kw = _all_col_keywords()
    best_row, best_score = 0, 0
    for i in range(min(20, len(raw))):
        row_text = " ".join(str(v).lower() for v in raw.iloc[i].values if pd.notna(v))
        score = sum(1 for kw in all_kw if kw in row_text)
        if score > best_score:
            best_score, best_row = score, i

    df = None
    for hrow in list(dict.fromkeys([best_row, 0, 1, 2, 3])):
        for engine in (("xlrd" if ext == ".xls" else "openpyxl"), None):
            try:
                kw = {"engine": engine} if engine else {}
                df = pd.read_excel(
                    file_path, header=hrow, dtype=str,
                    skip_blank_lines=True, **kw
                )
                if (len(df.columns) >= 4 and
                        not all(str(c).startswith("Unnamed") for c in df.columns)):
                    break
            except Exception:
                continue
        if df is not None and len(df.columns) >= 4:
            break

    if df is None or df.empty:
        warnings.append("Excel parsing failed entirely")
        return pd.DataFrame(), header_text, "xlsx", warnings

    warnings.append(f"XLSX: {len(df)} rows, {len(df.columns)} cols, hdr@row {best_row}")
    return df, header_text, "xlsx", warnings


# ══════════════════════════════════════════════════════════════════════════════
# PDF  — cascading: table → bank-specific text → word-position → line-split
# ══════════════════════════════════════════════════════════════════════════════
def parse_pdf(file_path, password=None, password_candidates=None):
    try:
        import pdfplumber
    except ImportError:
        return pd.DataFrame(), "", "pdf", ["pdfplumber not installed"]

    warnings = []
    header_text = ""
    candidates = ([password] if password else []) + list(password_candidates or [])

    # ── open (handle encryption) ──────────────────────────────────────────
    def _open():
        try:
            return pdfplumber.open(file_path)
        except Exception as e:
            if "password" in str(e).lower() or "encrypt" in str(e).lower():
                for pw in candidates:
                    try:
                        return pdfplumber.open(file_path, password=pw)
                    except Exception:
                        continue
                raise Exception(
                    f"PASSWORD_PROTECTED: PDF requires password. "
                    f"Tried {len(candidates)} candidate(s)."
                ) from e
            raise

    all_rows, headers, n_cols = [], None, 0
    all_text_lines = []

    STRATEGIES = [
        {"vertical_strategy":"lines","horizontal_strategy":"lines",
         "snap_tolerance":3,"join_tolerance":3},
        {"vertical_strategy":"lines_strict","horizontal_strategy":"lines_strict",
         "snap_tolerance":3,"join_tolerance":3},
        {"vertical_strategy":"text","horizontal_strategy":"text",
         "snap_tolerance":3,"join_tolerance":3},
        {},
    ]

    try:
        with _open() as pdf:
            n_pages = len(pdf.pages)
            text_chunks = []

            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    text_chunks.append(t)
                    all_text_lines.extend(t.splitlines())

            header_text = "\n".join(text_chunks)
            bank_hint = _detect_bank_hint(header_text)

            for page in pdf.pages:
                tables = []
                for strat in STRATEGIES:
                    try:
                        tables = (page.extract_tables(strat) if strat
                                  else page.extract_tables())
                        if tables and any(
                            t and len(t) >= 1 and len(t[0]) >= 5
                            for t in tables
                        ):
                            break
                        tables = []
                    except Exception:
                        tables = []

                for table in tables:
                    if not table:
                        continue
                    if headers is None and len(table) < 2:
                        continue
                    if headers is not None and len(table[0]) != n_cols:
                        continue
                    if headers is None and not _is_txn_table(table):
                        continue

                    cleaned = [
                        [str(c).strip() if c is not None else ""
                         for c in row]
                        for row in table
                        if any(str(c).strip() for c in row if c is not None)
                    ]
                    if not cleaned:
                        continue

                    if headers is None:
                        hi = _find_hdr_row(cleaned)
                        raw_h = cleaned[hi]
                        headers = _merge_split_hdrs(raw_h)
                        n_cols = len(headers)
                        data_rows = cleaned[hi + 1:]
                    else:
                        data_rows = cleaned
                        test_h = _merge_split_hdrs(cleaned[0])
                        if cleaned[0] == headers or test_h == headers:
                            data_rows = cleaned[1:]

                    for row in data_rows:
                        if len(row) < n_cols:
                            row += [""] * (n_cols - len(row))
                        all_rows.append(row[:n_cols])

    except Exception as e:
        err = str(e)
        if "PASSWORD_PROTECTED" in err:
            return pd.DataFrame(), header_text, "pdf", [err]
        warnings.append(f"PDF error: {err}")

    if all_rows and headers:
        df = pd.DataFrame(all_rows, columns=headers).replace("", pd.NA)
        warnings.append(f"PDF table: {len(df)} rows, {len(headers)} cols "
                        f"(bank={bank_hint})")
        return df, header_text, "pdf", warnings

    # ── fallback cascade ──────────────────────────────────────────────────
    warnings.append("No tables — trying text fallbacks")
    return _pdf_text_fallback(
        file_path, header_text, bank_hint,
        all_text_lines, warnings
    )


def _detect_bank_hint(text):
    tl = text.lower()
    if "idfc" in tl or "pioneer" in tl:          return "idfc"
    if "yes bank" in tl or "yesb" in tl:          return "yesbank"
    if "ratnakar" in tl or "rbl bank" in tl:      return "rbl"
    if "bank of baroda" in tl or "barb" in tl:    return "bob"
    if "bandhan" in tl:                            return "bandhan"
    if "federal bank" in tl:                       return "federal"
    return "generic"

def _merge_split_hdrs(hdrs):
    merged, skip = [], False
    for i, cell in enumerate(hdrs):
        if skip: skip = False; continue
        c = str(cell).strip()
        if i + 1 < len(hdrs):
            nc = str(hdrs[i+1]).strip()
            if nc and nc[0] in (".", ",", "/", " ") and len(nc) < 10:
                merged.append((c + nc).strip()); skip = True; continue
        merged.append(c)
    return merged

def _find_hdr_row(rows):
    from ingestion_config import COLUMN_ROLE_KEYWORDS
    kws = set(k.lower() for lst in COLUMN_ROLE_KEYWORDS.values() for k in lst)
    best, score = 0, 0
    for i, row in enumerate(rows[:8]):
        s = sum(1 for c in row if any(k in str(c).lower() for k in kws))
        if s > score: score, best = s, i
    return best

def _is_txn_table(table):
    if not table or len(table) < 2 or len(table[0]) < 4:
        return False
    h = " ".join(str(c).lower() for c in table[0])
    return (any(k in h for k in ["date","txn","tran","post"]) and
            any(k in h for k in ["debit","credit","withdrawal","deposit","balance","amount"]))


# ── PDF fallbacks ─────────────────────────────────────────────────────────

def _pdf_text_fallback(file_path, header_text, bank_hint, all_text_lines, warnings):
    try:
        import pdfplumber

        # collect words per page
        page_words = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_words.append(
                    page.extract_words(x_tolerance=2, y_tolerance=2)
                )

        # Layer A: bank-specific
        if bank_hint == "idfc":
            df = _parse_idfc(page_words, warnings)
            if not df.empty:
                warnings.append("PDF: IDFC word-position parser")
                return df, header_text, "pdf", warnings

        if bank_hint == "yesbank":
            df = _parse_yesbank(all_text_lines, warnings)
            if not df.empty:
                warnings.append("PDF: Yes Bank text-line parser")
                return df, header_text, "pdf", warnings

        if bank_hint == "rbl":
            df = _parse_rbl(page_words, warnings)
            if not df.empty:
                warnings.append("PDF: RBL word-position parser")
                return df, header_text, "pdf", warnings

        if bank_hint == "bob":
            df = _parse_bob(all_text_lines, warnings)
            if not df.empty:
                warnings.append("PDF: Bank of Baroda text parser")
                return df, header_text, "pdf", warnings

        # Layer B: generic word-position
        with pdfplumber.open(file_path) as pdf:
            df = _word_position_parser(pdf, warnings)
        if not df.empty:
            warnings.append("PDF: generic word-position parser")
            return df, header_text, "pdf", warnings

        # Layer C: double-space text split
        df = _text_line_df(all_text_lines, double_space=True)
        if not df.empty:
            warnings.append("PDF: text-line double-space parser")
            return df, header_text, "pdf", warnings

        # Layer D: single-space split
        df = _text_line_df(all_text_lines, double_space=False)
        if not df.empty:
            warnings.append("PDF: text-line single-space parser")
            return df, header_text, "pdf", warnings

        warnings.append("PDF: all fallbacks failed")
        return pd.DataFrame(), header_text, "pdf", warnings

    except Exception as e:
        return pd.DataFrame(), header_text, "pdf", warnings + [f"Fallback crash: {e}"]


# ── IDFC word-position ────────────────────────────────────────────────────
_IDFC_DATE_RE   = re.compile(r'^\d{1,2}/\d{2}/\d{2,4}$')
_IDFC_AMOUNT_RE = re.compile(r'^[\d,]+\.\d{2}(?:Cr|Dr)?$', re.I)
IDFC_BANDS = [12, 83, 127, 382, 452, 519]

def _idfc_band(x):
    return min(range(len(IDFC_BANDS)), key=lambda i: abs(x - IDFC_BANDS[i]))

_IDFC_SKIP = {"statement","of","account","customer","id","no","for",
              "to","trans","date","and","value","time","transaction",
              "details","cheque","debit","credit","balance"}

def _parse_idfc(page_words, warnings):
    rows = []
    for words in page_words:
        if not words: continue
        yg = defaultdict(list)
        for w in words:
            y = round(float(w.get("top",0))/6)*6
            yg[y].append((float(w.get("x0",0)), str(w.get("text",""))))

        header_y = None
        for y in sorted(yg):
            txts = {t.lower() for _,t in yg[y]}
            if {"debit","credit","balance"}.issubset(txts):
                header_y = y; break
        if header_y is None: continue

        pending = ""
        for y in sorted(yg):
            if y <= header_y: continue
            items = sorted(yg[y], key=lambda t: t[0])
            row_text = " ".join(t for _,t in items)
            if any(k in row_text.lower() for k in [
                "statement of account","customer id","account no",
                "statement for","page","generated"
            ]): continue

            lx, lt = items[0]
            is_txn = bool(_IDFC_DATE_RE.match(lt)) and lx < 60

            if is_txn:
                cells = [""] * len(IDFC_BANDS)
                for x, t in items:
                    ci = _idfc_band(x)
                    cells[ci] = (cells[ci] + " " + t).strip()
                if pending:
                    cells[2] = (pending + " " + cells[2]).strip()
                    pending = ""
                dp = cells[0].split()
                rows.append({
                    "date":      dp[0] if dp else "",
                    "time":      dp[1] if len(dp)>1 else "00:00",
                    "narration": cells[2],
                    "debit":     cells[3],
                    "credit":    cells[4],
                    "balance":   cells[5],
                    "utr_ref":   "",
                })
            else:
                part = " ".join(
                    t for x,t in items
                    if x > 60 and t.lower() not in _IDFC_SKIP
                )
                if part.strip():
                    pending = (pending + " " + part).strip()

    if not rows: return pd.DataFrame()
    warnings.append(f"IDFC: {len(rows)} txns")
    return pd.DataFrame(rows)


# ── Yes Bank text-line ────────────────────────────────────────────────────
_YB_TXN_RE = re.compile(
    r'^(\d{2}-[A-Z]{3}-\d{4})\s+'
    r'(\d{2}-[A-Z]{3}-\d{4})\s+'
    r'(.+?)\s+'
    r'([\d,]+\.\d{2})\s+'
    r'([\d,]+\.\d{2})\s+'
    r'([\d,]+\.\d{2})$'
)

def _parse_yesbank(lines, warnings):
    rows, pending = [], ""
    SKIP = {"TXN DATE","VALUE DATE","DESCRIPTION","REFERENCE",
            "STATEMENT OF ACCOUNT","CUSTOMER ID","ACCOUNT NO",
            "PERIOD","OPENING BALANCE","CLOSING BALANCE","PAGE"}
    for line in lines:
        line = line.strip()
        if not line: continue
        if any(k in line.upper() for k in SKIP):
            pending = ""; continue
        m = _YB_TXN_RE.match(line)
        if m:
            td,vd,desc,dr,cr,bal = m.groups()
            rows.append({
                "date": td, "time":"00:00:00",
                "narration": (pending+" "+desc).strip(),
                "debit": dr, "credit": cr, "balance": bal, "utr_ref":"",
            })
            pending = ""
        else:
            if (len(line) > 5
                and not re.match(r'^[\d,]+\.\d{2}', line)
                and not re.match(r'^\d{2}-[A-Z]{3}', line)):
                pending = (pending + " " + line).strip()
            else:
                pending = ""
    if not rows: return pd.DataFrame()
    warnings.append(f"YesBank: {len(rows)} txns")
    return pd.DataFrame(rows)


# ── RBL / Ratnakar word-position ─────────────────────────────────────────
_RBL_BANDS = [30, 77, 222, 272, 371, 439, 492]
_RBL_COLS  = ["date","narration","cheque_no","value_date","debit","credit","balance"]
_RBL_DATE  = re.compile(r'^\d{2}-[A-Za-z]{3}-\d{4}$')

def _rbl_band(x):
    return min(range(len(_RBL_BANDS)), key=lambda i: abs(x-_RBL_BANDS[i]))

def _parse_rbl(page_words, warnings):
    rows, in_txns = [], False
    for words in page_words:
        if not words: continue
        yg = defaultdict(list)
        for w in words:
            y = round(float(w.get("top",0))/6)*6
            yg[y].append((float(w.get("x0",0)), str(w.get("text",""))))
        for y in sorted(yg):
            items = sorted(yg[y], key=lambda t: t[0])
            row_text = " ".join(t for _,t in items)
            if "DATE" in row_text and "WITHDRAWAL" in row_text:
                in_txns = True; continue
            if not in_txns: continue
            if any(k in row_text.upper() for k in [
                "STATEMENT OF","PERIOD :","CLOSING BALANCE",
                "OPENING BALANCE","PAGE","TOTAL"
            ]): continue
            if not items: continue
            lx, lt = items[0]
            if lx > 60 or not _RBL_DATE.match(lt): continue
            cells = [""] * len(_RBL_COLS)
            for x,t in items:
                cells[_rbl_band(x)] = (cells[_rbl_band(x)]+" "+t).strip()
            rows.append({
                "date":cells[0],"narration":cells[1],"utr_ref":cells[2],
                "debit":cells[4],"credit":cells[5],"balance":cells[6],
            })
    if not rows: return pd.DataFrame()
    warnings.append(f"RBL: {len(rows)} txns")
    return pd.DataFrame(rows)


# ── Bank of Baroda text-line ──────────────────────────────────────────────
_BOB_FULL = re.compile(
    r'^(\d{2}-\d{2}-\d{2,4})\s+(.+?)\s+'
    r'([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})(?:Cr|Dr)?$'
)
_BOB_3COL = re.compile(
    r'^(\d{2}-\d{2}-\d{2,4})\s+(.+?)\s+'
    r'([\d,]+\.\d{2})(?:Cr|Dr)?\s+([\d,]+\.\d{2})(?:Cr|Dr)?$'
)

def _parse_bob(lines, warnings):
    rows, in_txns, pending = [], False, ""
    for line in lines:
        line = line.strip()
        if not line or set(line) == {"-"}: continue
        if "DATE" in line and ("WITHDRAWAL" in line or "PARTICULARS" in line):
            in_txns = True; continue
        if not in_txns: continue
        if any(k in line.upper() for k in [
            "PAGE","TOTAL","CLOSING BALANCE","OPENING BALANCE","HELPLINE","BRANCH","MICR","IFSC"
        ]): continue
        m = _BOB_FULL.match(line)
        if m:
            rows.append({
                "date":m.group(1),
                "narration":(pending+" "+m.group(2)).strip(),
                "debit":m.group(3),"credit":m.group(4),"balance":m.group(5),
            })
            pending = ""
            continue
        m2 = _BOB_3COL.match(line)
        if m2:
            rows.append({
                "date":m2.group(1),
                "narration":(pending+" "+m2.group(2)).strip(),
                "debit":"","credit":"","balance":m2.group(4),
            })
            pending = ""
            continue
        if not re.match(r'^\d{2}-\d{2}', line) and len(line) > 3:
            pending = (pending+" "+line).strip()
        else:
            pending = ""
    if not rows: return pd.DataFrame()
    warnings.append(f"BOB: {len(rows)} txns")
    return pd.DataFrame(rows)


# ── Generic word-position ─────────────────────────────────────────────────
def _word_position_parser(pdf, warnings):
    from ingestion_config import COLUMN_ROLE_KEYWORDS
    kws = set(k.lower() for lst in COLUMN_ROLE_KEYWORDS.values() for k in lst)
    Y_TOL = 4
    all_rows_by_y = {}
    for page in pdf.pages:
        for w in page.extract_words(x_tolerance=2, y_tolerance=2):
            text = str(w.get("text","")).strip()
            if not text: continue
            y = round(float(w.get("top",0))/Y_TOL)*Y_TOL
            x = float(w.get("x0",0))
            all_rows_by_y.setdefault(y, []).append((x, text))
    if not all_rows_by_y: return pd.DataFrame()

    text_rows = [sorted(all_rows_by_y[y], key=lambda t:t[0])
                 for y in sorted(all_rows_by_y)]

    hdr_idx, best = 0, 0
    for i, row in enumerate(text_rows[:30]):
        row_text = " ".join(w for _,w in row).lower()
        s = sum(1 for kw in kws if kw in row_text)
        if s > best: best, hdr_idx = s, i
    if best < 2: return pd.DataFrame()

    hdr_words = text_rows[hdr_idx]
    col_xs = []
    for x,_ in sorted(hdr_words, key=lambda t:t[0]):
        snap = round(x/15)*15
        if not col_xs or snap - col_xs[-1] > 20:
            col_xs.append(snap)
    if len(col_xs) < 4: return pd.DataFrame()

    def _ac(x):
        return min(range(len(col_xs)), key=lambda i: abs(x-col_xs[i]))

    n = len(col_xs)
    hcells = [""]*n
    for x,w in hdr_words:
        hcells[_ac(x)] = (hcells[_ac(x)]+" "+w).strip()

    seen = {}
    deduped = []
    for nm in hcells:
        nm = nm or f"col_{len(deduped)}"
        if nm in seen: seen[nm]+=1; deduped.append(f"{nm}_{seen[nm]}")
        else: seen[nm]=0; deduped.append(nm)

    SKIP = ["account holder","ifsc","statement period","page ","generated"]
    rows = []
    for rw in text_rows[hdr_idx+1:]:
        if not rw: continue
        rt = " ".join(w for _,w in rw).lower()
        if any(p in rt for p in SKIP): continue
        cells = [""]*n
        for x,w in rw:
            cells[_ac(x)] = (cells[_ac(x)]+" "+w).strip()
        if sum(1 for c in cells if c.strip()) >= 3:
            rows.append(cells)
    if not rows: return pd.DataFrame()
    warnings.append(f"Word-pos: {len(rows)} rows, {n} cols")
    return pd.DataFrame(rows, columns=deduped)


# ── text-line split ───────────────────────────────────────────────────────
_HDR_KW = ["date","debit","credit","balance","narration","description",
           "withdrawal","deposit","particulars","remarks","amount","ref",
           "cheque","txn","post"]

def _text_line_df(lines, double_space=True):
    hdr_idx = None
    for i, line in enumerate(lines):
        hits = sum(1 for k in _HDR_KW if k in line.lower())
        if hits >= 3: hdr_idx = i; break
    if hdr_idx is None: return pd.DataFrame()

    splitter = (lambda l: [p.strip() for p in re.split(r'\s{2,}', l) if p.strip()]
                if double_space
                else (lambda l: l.strip().split()))
    if not double_space:
        splitter = lambda l: l.strip().split()

    col_names = splitter(lines[hdr_idx])
    if len(col_names) < 4: return pd.DataFrame()
    n = len(col_names)
    SKIP = ["account holder","ifsc","page","statement period","generated"]
    rows = []
    for line in lines[hdr_idx+1:]:
        if not line or len(line) < 8: continue
        if any(k in line.lower() for k in SKIP): continue
        parts = splitter(line)
        if abs(len(parts)-n) > 3: continue
        while len(parts) < n: parts.append("")
        rows.append(parts[:n])
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows, columns=col_names)


# ══════════════════════════════════════════════════════════════════════════════
# Image / OCR
# ══════════════════════════════════════════════════════════════════════════════
def parse_image(file_path):
    try:
        import pytesseract
        from PIL import Image
        import PIL.ImageEnhance as IE
    except ImportError:
        return pd.DataFrame(), "", "image", ["pytesseract/pillow not installed"]

    warnings = []
    try:
        img = Image.open(file_path).convert("L")
        w,h = img.size
        scale = max(1, 2400//max(w,1))
        if scale > 1:
            img = img.resize((w*scale, h*scale), Image.LANCZOS)
        img = IE.Contrast(img).enhance(1.5)
        img = IE.Sharpness(img).enhance(2.0)
        tsv = pytesseract.image_to_data(
            img, config="--psm 6 --oem 3",
            output_type=pytesseract.Output.DATAFRAME
        )
        raw_text = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        header_text = " ".join(lines[:5])
    except Exception as e:
        return pd.DataFrame(), "", "image", [f"OCR error: {e}"]

    df = _ocr_bbox_df(tsv, warnings)
    if df.empty and lines:
        warnings.append("OCR bbox failed → text-line fallback")
        df = _text_line_df(lines, double_space=True)
    if df.empty and lines:
        df = _text_line_df(lines, double_space=False)
    if df.empty:
        warnings.append("OCR: no parseable table found")
    return df, header_text, "image", warnings


def _ocr_bbox_df(tsv, warnings):
    tsv = tsv[tsv["conf"] > 30].copy()
    tsv = tsv[tsv["text"].fillna("").str.strip().ne("")]
    if tsv.empty:
        warnings.append("OCR: no confident words")
        return pd.DataFrame()

    logical = []
    for (blk,ln), grp in tsv.groupby(["block_num","line_num"]):
        grp = grp.sort_values("left")
        logical.append({
            "text":  " ".join(grp["text"].tolist()),
            "words": grp["text"].tolist(),
            "lefts": grp["left"].tolist(),
            "top":   grp["top"].min(),
        })
    logical.sort(key=lambda x: x["top"])

    HDR_KW = ["date","debit","credit","balance","narration","description",
              "withdrawal","deposit","particulars","amount","ref","txn"]
    hdr_idx = None
    for i,ll in enumerate(logical):
        hits = sum(1 for k in HDR_KW if k in ll["text"].lower())
        if hits >= 3: hdr_idx = i; break
    if hdr_idx is None:
        warnings.append("OCR: header row not found")
        return pd.DataFrame()

    hl = logical[hdr_idx]
    fw, fl = [], []
    VOCAB = {"date","post","txn","tran","narration","description","particulars",
             "debit","credit","withdrawal","deposit","balance","ref","cheque",
             "dr","cr","no","id","amt","amount"}
    for w,x in zip(hl["words"], hl["lefts"]):
        if w.lower().strip(".:(),/-") in VOCAB:
            fw.append(w); fl.append(x)
    if len(fw) < 3:
        fw, fl = hl["words"], hl["lefts"]

    col_xs = sorted(fl)
    bands = [col_xs[0]]
    for x in col_xs[1:]:
        if x - bands[-1] > 40: bands.append(x)
    n = len(bands)

    def _ac(x):
        return min(range(n), key=lambda i: abs(x-bands[i]))

    hcells = [""]*n
    for w,x in zip(fw,fl):
        hcells[_ac(x)] = (hcells[_ac(x)]+" "+w).strip()
    hcells = [c or f"col_{i}" for i,c in enumerate(hcells)]

    seen = {}; deduped = []
    for nm in hcells:
        if nm in seen: seen[nm]+=1; deduped.append(f"{nm}_{seen[nm]}")
        else: seen[nm]=0; deduped.append(nm)

    SKIP = ["account holder","ifsc","statement period","page"]
    rows = []
    for ll in logical[hdr_idx+1:]:
        if not ll["text"].strip() or len(ll["text"]) < 8: continue
        if any(k in ll["text"].lower() for k in SKIP): continue
        cells = [""]*n
        for w,x in zip(ll["words"],ll["lefts"]):
            cells[_ac(x)] = (cells[_ac(x)]+" "+w).strip()
        if sum(1 for c in cells if c.strip()) >= 3:
            rows.append(cells)
    if not rows:
        warnings.append("OCR: header found but no data rows")
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=deduped)


# ══════════════════════════════════════════════════════════════════════════════
# JSON
# ══════════════════════════════════════════════════════════════════════════════
def parse_json(file_path):
    import json
    warnings = []
    try:
        with open(file_path,"r",encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records, header_text = data, ""
        elif isinstance(data, dict):
            header_text = str(list(data.keys()))
            records = next(
                (v for v in data.values() if isinstance(v, list) and v), None
            )
            if records is None:
                return pd.DataFrame(), header_text, "json", ["No list in JSON"]
        else:
            return pd.DataFrame(), "", "json", ["Unsupported JSON structure"]
        return pd.json_normalize(records), header_text, "json", warnings
    except Exception as e:
        return pd.DataFrame(), "", "json", [f"JSON error: {e}"]


# ══════════════════════════════════════════════════════════════════════════════
# TSV / pipe
# ══════════════════════════════════════════════════════════════════════════════
def parse_tsv(file_path):
    warnings = []
    try:
        with open(file_path,"r",encoding="utf-8",errors="replace") as f:
            sample = f.read(500)
        delim = "\t" if "\t" in sample else "|" if "|" in sample else ","
        df = pd.read_csv(file_path, sep=delim, dtype=str,
                         encoding="utf-8", on_bad_lines="skip")
        return df, "", "tsv", warnings
    except Exception as e:
        return pd.DataFrame(), "", "tsv", [f"TSV error: {e}"]