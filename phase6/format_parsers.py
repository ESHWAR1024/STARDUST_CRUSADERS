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
    from ingestion_config import BANK_FORMAT_REGISTRY
    all_col_keywords = set()
    for fmt in BANK_FORMAT_REGISTRY.values():
        all_col_keywords.update([
            fmt["date_col"], fmt["narration_col"],
            fmt["debit_col"], fmt["credit_col"], fmt["balance_col"]
        ])

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
def parse_pdf(file_path: str) -> tuple:
    try:
        import pdfplumber
    except ImportError:
        return pd.DataFrame(), "", "pdf", ["pdfplumber not installed"]

    warnings = []
    header_text = ""
    all_rows = []
    headers = None

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Capture header text from first page
                if page_num == 0:
                    text = page.extract_text() or ""
                    header_text = text[:600]

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
                    if not table or len(table) < 2:
                        continue

                    # Skip account info blocks (e.g. Account Holder / IFSC tables)
                    if not _is_transaction_table(table):
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
    from ingestion_config import BANK_FORMAT_REGISTRY
    all_col_keywords = set()
    for fmt in BANK_FORMAT_REGISTRY.values():
        all_col_keywords.update([
            fmt["date_col"].lower(), fmt["narration_col"].lower(),
            fmt["debit_col"].lower(), fmt["credit_col"].lower(),
            fmt["balance_col"].lower(),
        ])

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
        raw_text = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
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
    Reconstruct a table from Tesseract bounding-box output.

    Strategy:
    1. Filter confident words only (conf > 30)
    2. Cluster word left-edges into column bands using a gap threshold
    3. For each text line (same block_num/line_num), assign words to columns
       by which band their left-edge falls into
    4. Detect the header row, build column names, then parse data rows
    """
    import numpy as np

    # Filter low-confidence and empty detections
    tsv = tsv[tsv["conf"] > 30].copy()
    tsv = tsv[tsv["text"].str.strip().astype(bool)].copy()
    if tsv.empty:
        warnings.append("OCR: no confident words detected")
        return pd.DataFrame()

    # Build logical lines: group by (block_num, line_num)
    line_groups = tsv.groupby(["block_num", "line_num"])
    logical_lines = []
    for (blk, ln), grp in line_groups:
        grp = grp.sort_values("left")
        text = " ".join(grp["text"].tolist())
        left_positions = grp["left"].tolist()
        words = grp["text"].tolist()
        logical_lines.append({
            "block": blk, "line": ln,
            "text": text,
            "words": words,
            "lefts": left_positions,
            "top": grp["top"].min(),
        })

    logical_lines.sort(key=lambda x: (x["block"], x["top"]))

    # Find header line using keyword scoring
    header_keywords = [
        "date", "debit", "credit", "balance", "narration",
        "description", "withdrawal", "deposit", "particulars",
        "remarks", "amount", "ref", "cheque", "txn", "post",
    ]
    header_idx = None
    for i, ll in enumerate(logical_lines):
        hits = sum(1 for kw in header_keywords if kw in ll["text"].lower())
        if hits >= 3:
            header_idx = i
            break

    if header_idx is None:
        warnings.append("OCR bbox: could not identify header row")
        return pd.DataFrame()

    # Discover column x-boundaries from the header line's word positions
    header_ll = logical_lines[header_idx]
    col_lefts = sorted(header_ll["lefts"])

    # Merge word fragments within 40px into the same column slot
    GAP = 40
    col_bands = [col_lefts[0]]
    for x in col_lefts[1:]:
        if x - col_bands[-1] > GAP:
            col_bands.append(x)

    def assign_col(x_pos):
        """Find which column band this x position belongs to."""
        best = 0
        best_dist = abs(x_pos - col_bands[0])
        for i, band in enumerate(col_bands[1:], 1):
            d = abs(x_pos - band)
            if d < best_dist:
                best_dist = d
                best = i
        return best

    # Build column names from header word positions
    n_cols = len(col_bands)
    header_cells = [""] * n_cols
    for word, left in zip(header_ll["words"], header_ll["lefts"]):
        col_idx = assign_col(left)
        header_cells[col_idx] = (header_cells[col_idx] + " " + word).strip()
    header_cells = [c if c else f"col_{i}" for i, c in enumerate(header_cells)]

    if len(set(header_cells)) < 4:
        warnings.append(f"OCR bbox: only {n_cols} distinct column bands found")
        return pd.DataFrame()

    # Parse data rows
    rows = []
    for ll in logical_lines[header_idx + 1:]:
        text = ll["text"]
        if not text.strip() or len(text) < 8:
            continue
        # Skip meta lines
        if any(kw in text.lower() for kw in ["account holder", "ifsc", "statement period", "page"]):
            continue

        row_cells = [""] * n_cols
        for word, left in zip(ll["words"], ll["lefts"]):
            col_idx = assign_col(left)
            row_cells[col_idx] = (row_cells[col_idx] + " " + word).strip()

        # Only keep rows that have something in at least 3 columns
        non_empty = sum(1 for c in row_cells if c.strip())
        if non_empty >= 3:
            rows.append(row_cells)

    if not rows:
        warnings.append("OCR bbox: header found but no data rows could be reconstructed")
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=header_cells)


def _ocr_linefallback_to_dataframe(lines: list, warnings: list) -> pd.DataFrame:
    """Last-resort: try single-space splitting on the header line."""
    header_keywords = [
        "date", "debit", "credit", "balance", "narration",
        "description", "withdrawal", "deposit", "particulars",
        "remarks", "amount", "ref", "cheque", "txn",
    ]
    header_idx = None
    for i, line in enumerate(lines):
        hits = sum(1 for kw in header_keywords if kw in line.lower())
        if hits >= 3:
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame()

    # Try double-space first, then single
    header_line = lines[header_idx]
    col_names = re.split(r"\s{2,}", header_line.strip())
    if len(col_names) < 4:
        col_names = re.split(r"\s+", header_line.strip())

    col_names = [c.strip() for c in col_names if c.strip()]
    n_cols = len(col_names)
    if n_cols < 4:
        return pd.DataFrame()

    rows = []
    for line in lines[header_idx + 1:]:
        if not line or len(line) < 8:
            continue
        if any(kw in line.lower() for kw in ["account holder", "ifsc", "page"]):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        parts = [p.strip() for p in parts if p.strip()]
        if abs(len(parts) - n_cols) > 2:
            continue
        while len(parts) < n_cols:
            parts.append("")
        rows.append(parts[:n_cols])

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=col_names)
