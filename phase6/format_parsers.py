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

import io, os, re, shutil
import numpy as np
import pandas as pd

# Opts into pandas' upcoming replace() behavior so `.replace("", np.nan)`
# doesn't try (and warn about) an implicit downcast before our explicit
# `.infer_objects(copy=False)` runs. Purely silences a noisy FutureWarning
# that fires on every single parsed file — output is unaffected since we
# already call infer_objects() ourselves right after.
pd.set_option("future.no_silent_downcasting", True)
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
    try:
        from phase6.ingestion_config import COLUMN_ROLE_KEYWORDS
    except ImportError:
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
def _detect_excel_engine(file_path):
    """
    Some .xls files in this dataset are actually xlsx (zip-based) content
    that was just saved/exported with a .xls extension. Guessing the
    engine from the extension makes xlrd fail immediately on those (xlrd
    2.x deliberately dropped xlsx support) — logging a scary-looking but
    harmless "not supported" warning on every single one before falling
    back correctly. Sniffing the real file signature picks the right
    engine on the first try and removes the warning entirely.
    """
    try:
        with open(file_path, "rb") as fh:
            sig = fh.read(8)
        if sig[:4] == b"PK\x03\x04":
            return "openpyxl"       # real xlsx/zip container
        if sig[:8] == b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1":
            return "xlrd"           # real legacy .xls (OLE2/BIFF)
    except Exception:
        pass
    ext = os.path.splitext(file_path)[1].lower()
    return "xlrd" if ext == ".xls" else "openpyxl"


def parse_xlsx(file_path):
    warnings = []
    engine_guess = _detect_excel_engine(file_path)

    # read first 20 rows raw for header detection
    try:
        raw = pd.read_excel(file_path, header=None, dtype=str,
                            nrows=20, engine=engine_guess)
        header_text = " ".join(
            str(v) for row in raw.values
            for v in row if pd.notna(v) and str(v) != "nan"
        )
    except ImportError as e:
        warnings.append(f"Excel header scan import failed: {e}")
        header_text = ""
        raw = pd.DataFrame()
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
        for engine in (engine_guess, None):
            try:
                kw = {"engine": engine} if engine else {}
                try:
                    df = pd.read_excel(
                        file_path, header=hrow, dtype=str,
                        skip_blank_lines=True, **kw
                    )
                except TypeError:
                    # Older/newer pandas may not accept skip_blank_lines for read_excel
                    df = pd.read_excel(file_path, header=hrow, dtype=str, **kw)
                except ImportError as e:
                    warnings.append(f"Excel read engine import failed: {e}")
                    continue

                if (len(df.columns) >= 4 and
                        not all(str(c).startswith("Unnamed") for c in df.columns)):
                    break
            except Exception as e:
                warnings.append(f"Excel read error (hrow={hrow}, engine={engine}): {e}")
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
    except Exception as e:
        return pd.DataFrame(), "", "pdf", [f"Failed to import pdfplumber: {e}"]

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
    total_text_chars = 0

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
            winning_strategy_idx = None  # once a strategy works, try it first on later pages

            # Single pass per page: extract text AND attempt tables together,
            # then immediately flush pdfplumber's per-page cache (chars,
            # rects, edges, lines). Without this, a 500-1000 page statement
            # keeps every page's full layout graph alive in memory for the
            # whole file — that's the main driver of the RAM blowups on
            # large PDFs, not worker count.
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    text_chunks.append(t)
                    all_text_lines.extend(t.splitlines())
                    total_text_chars += len(t)

                tables = []
                strat_order = STRATEGIES
                if winning_strategy_idx is not None:
                    # try the strategy that already worked on this document
                    # first — most statements are consistent page to page,
                    # so this normally avoids the other 3 attempts entirely.
                    strat_order = [STRATEGIES[winning_strategy_idx]] + [
                        s for i, s in enumerate(STRATEGIES) if i != winning_strategy_idx
                    ]
                for strat in strat_order:
                    try:
                        candidate = (page.extract_tables(strat) if strat
                                     else page.extract_tables())
                        if candidate and any(
                            t and len(t) >= 1 and len(t[0]) >= 4
                            for t in candidate
                        ):
                            tables = candidate
                            winning_strategy_idx = STRATEGIES.index(strat) if strat else len(STRATEGIES) - 1
                            break
                    except Exception:
                        continue

                for table in tables:
                    if not table:
                        continue

                    cleaned = [
                        [
                            re.sub(r'\s+', ' ', re.sub(r'\s+-\s+', '-', re.sub(r'-\s+', '-', str(c).replace('\n', ' ')))).strip()
                            if c is not None else ""
                            for c in row
                        ]
                        for row in table
                        if any(str(c).strip() for c in row if c is not None)
                    ]
                    if not cleaned:
                        continue

                    if headers is None and len(cleaned) < 2:
                        continue
                    if headers is not None and len(cleaned[0]) != n_cols:
                        continue
                    if headers is None and not _is_txn_table(cleaned):
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

                # Release this page's cached geometry now that we're done
                # with it — critical for large multi-hundred-page PDFs.
                try:
                    page.flush_cache()
                except Exception:
                    pass

            header_text = "\n".join(text_chunks)
            # Bank identity should come from the statement's own header
            # (branch/IFSC/account-holder block on page 1), not the whole
            # document. On a 100+ page statement, some OTHER bank's name
            # is almost guaranteed to show up in a transaction narration
            # somewhere (e.g. "NEFT to XYZ Bank") — scanning the full text
            # let that override the correct detection.
            first_page_text = text_chunks[0] if text_chunks else header_text
            bank_hint = _detect_bank_hint(first_page_text)

    except Exception as e:
        err = str(e)
        if "PASSWORD_PROTECTED" in err:
            return pd.DataFrame(), header_text, "pdf", [err]
        warnings.append(f"PDF error: {err}")
        # text_chunks may hold partial progress if the exception hit
        # mid-document (e.g. a bad page later in a huge PDF) — use it.
        try:
            if not header_text and text_chunks:
                header_text = "\n".join(text_chunks)
        except NameError:
            pass
        try:
            first_page_text = text_chunks[0] if text_chunks else header_text
        except NameError:
            first_page_text = header_text
        bank_hint = _detect_bank_hint(first_page_text)

    if all_rows and headers:
        df = pd.DataFrame(all_rows, columns=headers).replace("", np.nan).infer_objects(copy=False)
        if _is_valid_transaction_table(df):
            warnings.append(f"PDF table: {len(df)} rows, {len(headers)} cols "
                            f"(bank={bank_hint})")
            return df, header_text, "pdf", warnings
        warnings.append(
            f"PDF table detected but invalid transaction table: {len(df)} rows, {len(headers)} cols "
            f"(bank={bank_hint})"
        )

    # ── fallback cascade ──────────────────────────────────────────────────
    warnings.append("No tables — trying text fallbacks")
    # A page is only worth OCR'ing if it doesn't already have a usable text
    # layer. Real digital statements average hundreds of chars/page here;
    # true scans average close to 0. This avoids burning 10-25+s per file
    # rendering-and-OCR'ing pages that already parsed fine as text and
    # were just an unusual table layout (this was silently discarding
    # files as "no rows found" purely because OCR fallback timed out or
    # produced garbage on a perfectly good digital PDF).
    avg_chars_per_page = (total_text_chars / n_pages) if n_pages else 0
    likely_scanned = avg_chars_per_page < 40
    return _pdf_text_fallback(
        file_path, header_text, bank_hint,
        all_text_lines, warnings, allow_ocr=likely_scanned
    )


def _detect_bank_hint(text):
    tl = text.lower()
    if "idfc" in tl or "pioneer" in tl:          return "idfc"
    if "yes bank" in tl or "yesb" in tl:          return "yesbank"
    if "ratnakar" in tl or "rbl bank" in tl:      return "rbl"
    if "bank of baroda" in tl or "barb" in tl:    return "bob"
    if "bandhan" in tl:                            return "bandhan"
    if "federal bank" in tl:                       return "federal"
    if "kerala gramin" in tl or "kerala grameena" in tl or "kerala gramin bank" in tl:
        return "kerala_gramin"
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
    try:
        from phase6.ingestion_config import COLUMN_ROLE_KEYWORDS
    except ImportError:
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

def _pdf_text_fallback(file_path, header_text, bank_hint, all_text_lines, warnings, allow_ocr=True):
    try:
        import pdfplumber

        # page_words (word-level bbox data for every page) is only needed
        # by the IDFC/RBL band-position parsers below. Collecting it
        # unconditionally for every PDF — including huge ones where it's
        # never used — was doing a second full-document layout pass for
        # nothing. Layer D (_word_position_parser) does its own pass later
        # if it's actually needed.
        page_words = []
        if bank_hint in ("idfc", "rbl"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_words.append(
                        page.extract_words(x_tolerance=2, y_tolerance=2)
                    )
                    try:
                        page.flush_cache()
                    except Exception:
                        pass

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

        if bank_hint == "kerala_gramin":
            df, kw = _parse_kerala_gramin_txt(all_text_lines)
            if not df.empty:
                warnings.extend(kw)
                warnings.append("PDF: Kerala Gramin fixed-width text parser")
                return df, header_text, "pdf", warnings

        df = _parse_date_amount_lines(all_text_lines, warnings)
        if not df.empty:
            warnings.append("PDF: date/amount line parser")
            return df, header_text, "pdf", warnings

        # Layer B: double-space text split
        df = _text_line_df(all_text_lines, double_space=True)
        if not df.empty:
            warnings.append("PDF: text-line double-space parser")
            return df, header_text, "pdf", warnings

        # Layer C: single-space split
        df = _text_line_df(all_text_lines, double_space=False)
        if not df.empty:
            warnings.append("PDF: text-line single-space parser")
            return df, header_text, "pdf", warnings

        # Layer D: generic word-position
        with pdfplumber.open(file_path) as pdf:
            df = _word_position_parser(pdf, warnings)
        if not df.empty:
            if _is_valid_transaction_table(df):
                warnings.append("PDF: generic word-position parser")
                return df, header_text, "pdf", warnings
            warnings.append("PDF: generic word-position parser produced invalid table")

        # Layer E: scanned PDF OCR fallback — only worth trying if this
        # PDF doesn't already have a real text layer. Running OCR on a
        # PDF that has extractable text (it just didn't match any of our
        # table/layout heuristics) burns 10-25+s per file rendering pages
        # to images and produces lower-quality output than the text we
        # already had — this was the main cause of files being silently
        # discarded as "no rows found" in the Secondary dataset.
        if allow_ocr:
            ocr_df, ocr_header, ocr_warnings = _pdf_ocr_fallback(file_path, warnings)
            if not ocr_df.empty:
                return ocr_df, header_text or ocr_header, "pdf", ocr_warnings
        else:
            warnings.append(
                "PDF: skipped OCR fallback (page already has a real text "
                "layer — OCR would not help; check table/layout heuristics instead)"
            )

        warnings.append("PDF: all fallbacks failed")
        return pd.DataFrame(), header_text, "pdf", warnings

    except Exception as e:
        return pd.DataFrame(), header_text, "pdf", warnings + [f"Fallback crash: {e}"]


def _pdf_ocr_fallback(file_path, warnings):
    try:
        import pytesseract
        from PIL import Image
    except Exception as e:
        return pd.DataFrame(), "", warnings + [f"PDF OCR fallback dependency import failed: {e}"]

    if shutil.which("tesseract") is None:
        return pd.DataFrame(), "", warnings + [
            "Tesseract executable not found on PATH. Install tesseract-ocr and ensure it is available in your PATH."
        ]

    try:
        import pdfplumber
    except Exception as e:
        return pd.DataFrame(), "", warnings + [f"PDF OCR fallback failed to import pdfplumber: {e}"]

    ocr_warnings = list(warnings)
    page_images = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:5]:
                try:
                    img = page.to_image(resolution=200).original.convert("L")
                    page_images.append(img)
                except Exception:
                    continue
                finally:
                    try:
                        page.flush_cache()
                    except Exception:
                        pass
    except Exception as e:
        return pd.DataFrame(), "", ocr_warnings + [f"PDF OCR fallback open error: {e}"]

    if not page_images:
        return pd.DataFrame(), "", ocr_warnings + ["PDF OCR fallback failed: no renderable pages"]

    for img in page_images:
        df, header_text, source_format, new_warnings = _ocr_image_df(img, ocr_warnings, source_format="pdf")
        if not df.empty:
            ocr_warnings = new_warnings
            ocr_warnings.append("PDF: scanned image OCR parser")
            return df, header_text, ocr_warnings

    return pd.DataFrame(), "", ocr_warnings + ["PDF OCR fallback failed"]


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


def _parse_date_amount_lines(lines, warnings):
    rows = []
    header_seen = False
    prev_balance = None
    # Accepts plain "20.00", suffixed "20.00CR"/"20.00 DR", AND
    # parenthesized "20.00(Cr)" — Kotak Mahindra statements use the
    # parenthesized form, which the plain-suffix pattern silently
    # rejected, causing every row to fail and fall through to the far
    # less reliable generic word-position parser.
    amount_re = re.compile(r'^[\d,]+\.\d{2}(?:\s*\(?(?:CR|DR)\)?)?$', re.IGNORECASE)
    # A line that's ONLY a bare amount (optionally with Cr/Dr) — used to
    # detect a balance that wrapped onto its own line (Central Bank of
    # India does this when the narration pushes the row past one line).
    bare_amount_re = re.compile(r'^[\d,]+\.\d{2}\s*(?:CR|DR)?$', re.IGNORECASE)
    # DD-MM-YYYY or DD/MM/YY(YY) — some banks (HDFC, Central Bank of
    # India) use slashes and/or 2-digit years instead of the dashed
    # 4-digit form this parser originally only accepted.
    date_re = r'(\d{2}[-/]\d{2}[-/]\d{2,4})'
    row_re = re.compile(rf'^{date_re}(?:\s+{date_re})?(\S*)\s+(.+)$')
    # Header keyword sets — broadened beyond one hardcoded string so this
    # generic parser catches any "date + narration + trailing amount(s) +
    # balance" statement layout (DCB Bank, Kotak, and others all match
    # this shape but each phrases the header line differently).
    _date_kw = ("date",)
    _amt_kw = ("withdraw", "deposit", "debit", "credit", "amount", "balance", "particular")

    n = len(lines)
    i = 0
    while i < n:
        line = str(lines[i]).strip()
        i += 1
        if not line:
            continue
        if not header_seen:
            ll = line.lower()
            # Some statements (e.g. Karnataka Bank) never print a repeated
            # column-header row at all — the first real signal is the
            # "OPENING BALANCE" / "BROUGHT FORWARD" line right before the
            # transactions start.
            if (any(k in ll for k in _date_kw) and any(k in ll for k in _amt_kw)) \
                    or "opening balance" in ll or "brought forward" in ll:
                header_seen = True
                if "opening balance" in ll or "brought forward" in ll:
                    tail = re.split(r'[:\s]+', line)[-1].strip()
                    if bare_amount_re.match(tail):
                        prev_balance, _ = _simple_amount_parse(tail)
                continue
            # No header line and no opening-balance line at all (seen on
            # some HDFC exports) — bootstrap from the data itself: if this
            # line already looks like a complete transaction row (date +
            # narration + 2+ trailing amounts), that's proof enough that
            # we're already past any header and can start right here.
            probe = row_re.match(line)
            if probe:
                probe_tail = re.split(r'\s+', probe.group(4))
                probe_amounts = sum(1 for t in probe_tail[-3:] if amount_re.match(t))
                if probe_amounts >= 2:
                    header_seen = True
                    i -= 1  # reprocess this line as a transaction below
                    continue
            continue
        if line.startswith("Id Date"):
            continue
        if line.startswith("Account Opening balance") or line.startswith("Opening Balance"):
            continue
        if line.startswith("Brought Forward") or line.upper().startswith("BROUGHT FORWARD"):
            bf_val, _ = _simple_amount_parse(line.split(":", 1)[-1].strip())
            if bare_amount_re.match(line.split(":", 1)[-1].strip()):
                prev_balance = bf_val
            continue
        if line.startswith("B/F"):
            # "B/F 0.00(Cr)" — Kotak's opening balance line. Capturing it
            # means even the FIRST transaction row gets a real balance to
            # compare against, instead of falling back to a guess.
            bf_val, _ = _simple_amount_parse(line[3:].strip())
            prev_balance = bf_val
            continue
        if line.startswith("Total(") or line.startswith("Total ") or line.startswith("Closing Balance"):
            continue
        if line.startswith("Manager/") or line.startswith("Date :"):
            continue
        if line.startswith("***"):
            continue
        if line.startswith("Report for the Period") or line.startswith("Service OutLet"):
            continue
        if line.startswith("Page") or ("Page" in line and not amount_re.search(line)):
            continue
        # Pure narration-continuation / placeholder lines (Central Bank of
        # India prints ". . SOME MORE TEXT ." for wrapped narration) —
        # not a new transaction, and not a bare balance either. Skip.
        if re.match(r'^[.\s]*$', line) or (line.startswith(".") and not bare_amount_re.match(line)):
            continue

        m = row_re.match(line)
        if not m:
            continue
        date, _post_date, txn_code, rest = m.groups()
        tokens = re.split(r'\s+', rest)
        trailing = []
        while tokens and amount_re.match(tokens[-1]):
            trailing.insert(0, tokens.pop())
        # drop a lone "." / "-" placeholder token that sits where an
        # empty Chq.No/Debit/Credit column would be
        while tokens and tokens[-1] in (".", "-"):
            tokens.pop()

        wrapped_balance = None
        if len(trailing) == 1:
            # Only one amount on this line — peek at the next non-blank
            # line: if it's a bare amount, the balance wrapped onto its
            # own line rather than staying on the transaction row.
            j = i
            while j < n and not str(lines[j]).strip():
                j += 1
            if j < n and bare_amount_re.match(str(lines[j]).strip()):
                wrapped_balance = str(lines[j]).strip()
                i = j + 1

        if len(trailing) < 2 and wrapped_balance is None:
            continue

        if wrapped_balance is not None:
            balance_token = wrapped_balance
            amount_tokens = trailing
        else:
            balance_token = trailing[-1]
            amount_tokens = trailing[:-1]
        narration = " ".join(tokens).strip()
        utr_ref = ""
        debit = credit = 0.0
        bal_val, bal_sign = _simple_amount_parse(balance_token)

        if len(amount_tokens) == 1:
            amt_val, amt_sign = _simple_amount_parse(amount_tokens[0])
            # The balance's own (Cr)/(Dr) suffix says whether the ACCOUNT
            # is in credit or overdrawn overall — it says nothing about
            # whether THIS transaction was a debit or credit (almost every
            # row has "(Cr)" on Kotak statements regardless of direction,
            # since the account rarely goes overdrawn). The only reliable
            # signal is whether the balance moved up or down versus the
            # previous row.
            if prev_balance is not None:
                if bal_val > prev_balance + 1e-6:
                    credit = amt_val
                elif bal_val < prev_balance - 1e-6:
                    debit = amt_val
                else:
                    credit = amt_val
            elif bal_sign == "-":
                # No prior balance to compare against (first row, and no
                # B/F opening balance was found) — an overdrawn balance
                # after a single transaction at least tells us it was a
                # debit; otherwise we can't know, so default to credit.
                debit = amt_val
            else:
                credit = amt_val
        else:
            amt1, sign1 = _simple_amount_parse(amount_tokens[-2])
            amt2, sign2 = _simple_amount_parse(amount_tokens[-1])
            if sign2 or sign1:
                credit = amt1 if sign2 in ("+", "") else 0.0
                debit = amt1 if sign2 == "-" else 0.0
            else:
                credit = amt2

        if bal_val is not None:
            prev_balance = bal_val

        if txn_code and txn_code.isalnum() and not txn_code.startswith("0"):
            utr_ref = txn_code
            narration = narration
        rows.append({
            "date": date,
            "time": "00:00:00",
            "narration": narration,
            "debit": debit,
            "credit": credit,
            "balance": balance_token,
            "utr_ref": utr_ref,
        })

    if not rows:
        return pd.DataFrame()
    warnings.append(f"Date/amount line parser: {len(rows)} txns")
    return pd.DataFrame(rows)


def _simple_amount_parse(val):
    s = str(val).strip()
    if not s:
        return 0.0, ""
    sign = ""
    # Strip a trailing parenthesized suffix first — "20.00(Cr)" — before
    # falling back to the plain "20.00CR" form.
    m = re.match(r'^(.*?)\s*\((CR|DR)\)\s*$', s, re.IGNORECASE)
    if m:
        s, tag = m.group(1).strip(), m.group(2).upper()
        sign = "+" if tag == "CR" else "-"
    elif s.upper().endswith("CR"):
        sign = "+"
        s = s[:-2].strip()
    elif s.upper().endswith("DR"):
        sign = "-"
        s = s[:-2].strip()
    s = s.replace(",", "")
    try:
        return float(s), sign
    except ValueError:
        return 0.0, sign


def _is_valid_transaction_table(df):
    if df.empty:
        return False
    try:
        try:
            from phase6.schema_detector import assign_column_roles, parse_date, parse_amount
        except ImportError:
            from schema_detector import assign_column_roles, parse_date, parse_amount
        roles = assign_column_roles(df.columns.tolist(), df=df)
    except Exception:
        return False

    if not roles.get("date"):
        return False
    if not (roles.get("debit") or roles.get("credit") or roles.get("amount")):
        return False

    valid_rows = 0
    for _, row in df.head(30).iterrows():
        date_val = parse_date(str(row[roles["date"]]))
        if not date_val:
            continue
        if roles.get("debit") and roles.get("credit"):
            d, _ = parse_amount(row[roles["debit"]])
            c, _ = parse_amount(row[roles["credit"]])
            if d != 0.0 or c != 0.0:
                valid_rows += 1
        elif roles.get("amount"):
            a, _ = parse_amount(row[roles["amount"]])
            if a != 0.0:
                valid_rows += 1
        if valid_rows >= 2:
            return True
    return False


# ── Generic word-position ─────────────────────────────────────────────────
def _word_position_parser(pdf, warnings):
    try:
        from phase6.ingestion_config import COLUMN_ROLE_KEYWORDS
    except ImportError:
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
        try:
            page.flush_cache()
        except Exception:
            pass
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
        if hits >= 3:
            hdr_idx = i
            break
    if hdr_idx is None:
        return pd.DataFrame()

    splitter = (lambda l: [p.strip() for p in re.split(r'\s{2,}', l) if p.strip()]
                if double_space
                else (lambda l: l.strip().split()))
    if not double_space:
        splitter = lambda l: l.strip().split()

    col_names = splitter(lines[hdr_idx])
    if len(col_names) < 4:
        return pd.DataFrame()
    n = len(col_names)

    SKIP = ["account holder", "ifsc", "page", "statement period", "generated"]
    merged_rows = []
    buffer = ""
    date_re = re.compile(r'^[0-3]?\d/[0-1]?\d/\d{2,4}\b')

    for line in lines[hdr_idx+1:]:
        if not line or len(line) < 8:
            continue
        lower = line.lower()
        if any(k in lower for k in SKIP):
            continue
        if date_re.match(line.strip()):
            if buffer:
                merged_rows.append(buffer)
            buffer = line.strip()
        else:
            if buffer:
                buffer = f"{buffer} {line.strip()}"
            else:
                buffer = line.strip()
    if buffer:
        merged_rows.append(buffer)

    rows = []
    for line in merged_rows:
        parts = splitter(line)
        if abs(len(parts) - n) > 3:
            continue
        while len(parts) < n:
            parts.append("")
        rows.append(parts[:n])

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=col_names)


# ══════════════════════════════════════════════════════════════════════════════
# Image / OCR
# ══════════════════════════════════════════════════════════════════════════════
def parse_image(file_path):
    try:
        import pytesseract
        from PIL import Image
        import PIL.ImageEnhance as IE
    except Exception as e:
        return pd.DataFrame(), "", "image", [f"Failed to import OCR dependencies: {e}"]

    if shutil.which("tesseract") is None:
        return pd.DataFrame(), "", "image", [
            "Tesseract executable not found on PATH. Install tesseract-ocr and ensure it is available in your PATH."
        ]

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

    return _ocr_image_df(img, warnings, source_format="image")


def _ocr_image_df(img, warnings, source_format="image"):
    try:
        import pytesseract
        from PIL import Image, ImageEnhance as IE
    except Exception as e:
        return pd.DataFrame(), "", source_format, warnings + [f"OCR helper import failed: {e}"]

    if shutil.which("tesseract") is None:
        return pd.DataFrame(), "", source_format, warnings + [
            "Tesseract executable not found on PATH. Install tesseract-ocr and ensure it is available in your PATH."
        ]

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

    df = _ocr_bbox_df(tsv, warnings)
    if df.empty and lines:
        warnings.append("OCR bbox failed → text-line fallback")
        df = _text_line_df(lines, double_space=True)
    if df.empty and lines:
        df = _text_line_df(lines, double_space=False)
    if df.empty:
        warnings.append("OCR: no parseable table found")
    return df, header_text, source_format, warnings


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


# ── Kerala Gramin / fixed-width TXT special-case parser
def _parse_kerala_gramin_txt(lines):
    """Attempt to parse Kerala Gramin / similar fixed-width bank text dumps.
    Looks for lines starting with a date and trailing amount tokens.
    Returns (df, warnings_list) where warnings_list may be empty.
    """
    warnings = []
    rows = []
    date_re = re.compile(r'^\s*\d{1,2}[-/ ]\d{1,2}[-/ ]\d{2,4}')
    amt_re = re.compile(r'[\d,]+\.\d{2}')
    for raw in lines:
        line = str(raw).strip()
        if not line:
            continue
        if not date_re.match(line):
            continue
        # split on 2+ spaces or tabs (common fixed-width separators)
        parts = re.split(r'\t|\s{2,}', line)
        # fallback to whitespace split if nothing else
        if len(parts) <= 1:
            parts = line.split()

        # heuristics: first part date, last part balance, second-last maybe amount
        date = parts[0]
        balance = ''
        debit = ''
        credit = ''
        narration = ''
        if len(parts) >= 3 and amt_re.search(parts[-1]):
            balance = parts[-1]
            # try detect amount before balance
            if len(parts) >= 4 and amt_re.search(parts[-2]):
                amt = parts[-2]
                narration = ' '.join(parts[1:-2]).strip()
            else:
                amt = parts[-2] if len(parts) >= 2 else ''
                narration = ' '.join(parts[1:-1]).strip()
        elif len(parts) >= 2 and amt_re.search(parts[-1]):
            amt = parts[-1]
            narration = ' '.join(parts[1:-1]).strip()
        else:
            # no obvious amounts; skip
            continue

        if isinstance(amt, str) and amt_re.search(amt):
            amt_clean = amt.replace(',', '')
            # determine dr/cr by presence of CR/DR suffix or by position
            if re.search(r'CR$|Cr$|CR\b', amt) or re.search(r'CR$|Cr$|CR\b', balance if balance else ''):
                credit = amt
            elif re.search(r'DR$|Dr$|DR\b', amt) or re.search(r'DR$|Dr$|DR\b', balance if balance else ''):
                debit = amt
            else:
                # ambiguous: treat as credit if balance increases, else debit
                credit = amt
        else:
            continue

        rows.append({
            "date": date,
            "time": "00:00:00",
            "narration": narration,
            "debit": debit,
            "credit": credit,
            "balance": balance,
            "utr_ref": "",
        })

    if not rows:
        return pd.DataFrame(), []
    warnings.append(f"KeralaGraminTXT: {len(rows)} rows parsed")
    return pd.DataFrame(rows), warnings


# ══════════════════════════════════════════════════════════════════════════════
# TXT
# ══════════════════════════════════════════════════════════════════════════════
def parse_txt(file_path):
    warnings = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\n") for line in f]
    except Exception as e:
        return pd.DataFrame(), "", "txt", [f"TXT read error: {e}"]

    if not lines:
        return pd.DataFrame(), "", "txt", ["TXT file is empty"]

    sample = "\n".join(lines[:20])
    if "\t" in sample or "|" in sample or ("," in sample and sample.count(",") > sample.count(" ")):
        df, header_text, source_format, parse_warnings = parse_tsv(file_path)
        warnings.extend(parse_warnings)
        if not df.empty:
            warnings.insert(0, "TXT delimiter parser")
            return df, header_text, "txt", warnings

    nonblank = [line.rstrip() for line in lines if line.strip()]
    if not nonblank:
        return pd.DataFrame(), "", "txt", ["TXT file contains only blank lines"]

    # Special-case: Kerala Gramin / fixed-width style statements
    try:
        sample_head = "\n".join(nonblank[:10]).lower()
    except Exception:
        sample_head = ""
    if "kerala gramin" in sample_head or re.search(r'^\s*\d{1,2}[-/ ]\d{1,2}[-/ ]\d{2,4}', nonblank[0]):
        df_k, kw = _parse_kerala_gramin_txt(nonblank)
        if not df_k.empty:
            warnings.extend(kw)
            warnings.insert(0, "TXT: Kerala Gramin fixed-width parser")
            return df_k, "", "txt", warnings

    df = _text_line_df(nonblank, double_space=True)
    if not df.empty:
        warnings.append("TXT: fixed-width text parser")
        return df, "", "txt", warnings

    df = _text_line_df(nonblank, double_space=False)
    if not df.empty:
        warnings.append("TXT: single-space text parser")
        return df, "", "txt", warnings

    df, header_text, source_format, parse_warnings = parse_tsv(file_path)
    warnings.extend(parse_warnings)
    warnings.append("TXT: all text fallbacks failed, attempted delimiter parse")
    return df, header_text, "txt", warnings


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