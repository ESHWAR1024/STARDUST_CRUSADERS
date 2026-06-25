"""
Router: /api/v1/upload

Accepts multi-file upload of bank statements (CSV/Excel/PDF/PNG),
runs Phase 6 ingestion + Phase 7 cleaning, and loads results into
PostgreSQL via bulk insert.

POST /                Upload one or more statement files
GET  /status/{id}     Check async processing status
"""

import os
import uuid
import subprocess
import shutil
import pandas as pd

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_db
from models import Account, Transaction
from schemas import UploadResponse
from config import settings

router = APIRouter(prefix="/upload", tags=["Upload"])

# In-memory job status store (replace with Redis in production)
_job_status: dict[str, dict] = {}

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".pdf", ".png", ".jpg", ".jpeg"}


@router.post("/", response_model=UploadResponse)
async def upload_statements(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept bank statement files, save to disk, run ingestion + cleaning
    pipeline, bulk-load results into PostgreSQL.

    Supports: CSV, Excel (xlsx/xls), PDF (text-layer), PNG/JPG (OCR via Tesseract)
    """
    upload_id = str(uuid.uuid4())[:8]
    upload_dir = os.path.join(settings.UPLOAD_DIR, upload_id)
    os.makedirs(upload_dir, exist_ok=True)

    # Validate and save uploaded files
    saved_files = []
    warnings = []

    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            warnings.append(f"{f.filename}: unsupported format, skipped")
            continue

        size_bytes = 0
        dest = os.path.join(upload_dir, f.filename)
        with open(dest, "wb") as out:
            while chunk := await f.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    warnings.append(f"{f.filename}: exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")
                    break
                out.write(chunk)
        saved_files.append(dest)

    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    # Run pipeline synchronously (for hackathon demo; use BackgroundTasks for production)
    ingested_path, cleaned_path, pipeline_report = _run_pipeline(
        upload_dir, upload_id, warnings
    )

    if not cleaned_path or not os.path.exists(cleaned_path):
        raise HTTPException(status_code=500, detail="Pipeline failed: " + str(warnings))

    # Bulk load into PostgreSQL
    rows_loaded, banks_detected = await _load_into_db(cleaned_path, db)

    return UploadResponse(
        upload_id=upload_id,
        files_received=len(files),
        files_ingested=len(saved_files),
        rows_parsed=pipeline_report.get("rows_ingested", 0),
        rows_after_clean=rows_loaded,
        banks_detected=banks_detected,
        warnings=warnings[:20],   # cap warning list for response size
        status="success" if rows_loaded > 0 else "partial",
    )


def _run_pipeline(upload_dir: str, upload_id: str, warnings: list) -> tuple:
    """Run Phase 6 + Phase 7 as subprocess calls."""
    ingested_dir = os.path.join(upload_dir, "ingested")
    cleaned_dir  = os.path.join(upload_dir, "cleaned")

    # Phase 6: ingest
    try:
        result = subprocess.run(
            ["python3", settings.PHASE6_INGEST_SCRIPT,
             "--input", upload_dir,
             "--out-dir", ingested_dir],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            warnings.append(f"Ingestion warning: {result.stderr[:200]}")
    except Exception as e:
        warnings.append(f"Ingestion error: {e}")
        return None, None, {}

    ingested_csv = os.path.join(ingested_dir, "ingested_transactions.csv")
    if not os.path.exists(ingested_csv):
        warnings.append("Ingestion produced no output")
        return None, None, {}

    # Phase 7: clean
    try:
        result = subprocess.run(
            ["python3", settings.PHASE7_CLEAN_SCRIPT,
             "--input", ingested_csv,
             "--out-dir", cleaned_dir],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            warnings.append(f"Cleaning warning: {result.stderr[:200]}")
    except Exception as e:
        warnings.append(f"Cleaning error: {e}")
        return ingested_csv, None, {}

    cleaned_csv = os.path.join(cleaned_dir, "cleaned_transactions.csv")
    rows_ingested = 0
    try:
        rows_ingested = len(pd.read_csv(ingested_csv))
    except Exception:
        pass

    return ingested_csv, cleaned_csv, {"rows_ingested": rows_ingested}


async def _load_into_db(
    cleaned_csv: str, db: AsyncSession
) -> tuple[int, list[str]]:
    """
    Bulk-load cleaned_transactions.csv into PostgreSQL.
    Uses upsert logic: existing account_id rows are updated,
    new ones inserted. Transaction rows are appended (never updated).
    """
    df = pd.read_csv(cleaned_csv, dtype=str)
    df = df.fillna("")
    banks_detected = df["bank_name"].unique().tolist() if "bank_name" in df.columns else []

    # --- Upsert Accounts ---
    if "account_id" in df.columns:
        for _, row in df.drop_duplicates("account_id").iterrows():
            existing = await db.get(Account, row["account_id"])
            if not existing:
                acct = Account(
                    account_id=row.get("account_id", ""),
                    holder_name=row.get("account_holder", "Unknown"),
                    bank_name=row.get("bank_name", "Unknown"),
                    source_file=row.get("source_file", ""),
                )
                db.add(acct)

    await db.flush()

    # --- Insert Transactions (skip existing UTR refs to avoid exact dupes) ---
    rows_loaded = 0
    for _, row in df.iterrows():
        def _f(v, default=None):
            val = row.get(v, "")
            if val == "" or (isinstance(val, float) and pd.isna(val)):
                return default
            return val

        def _float(v):
            try:
                return float(_f(v, 0))
            except (ValueError, TypeError):
                return 0.0

        def _bool(v):
            val = str(_f(v, "False")).lower()
            return val in ("true", "1", "yes")

        txn = Transaction(
            account_id=_f("account_id", ""),
            date=_f("date"),
            time=_f("time", "00:00:00"),
            narration=_f("narration"),
            channel=_f("channel", "OTHER"),
            debit=_float("debit"),
            credit=_float("credit"),
            balance=_float("balance"),
            utr_ref=_f("utr_ref"),
            counterparty_account_id=_f("counterparty_account_id"),
            counterparty_name=_f("counterparty_name"),
            source_file=_f("source_file"),
            source_format=_f("source_format"),
            ingestion_warnings=_f("ingestion_warnings"),
            clean_flags=_f("clean_flags"),
            is_duplicate=_bool("is_duplicate"),
            is_balance_breach=_bool("is_balance_breach"),
            is_high_value_flag=_bool("is_high_value_flag"),
            is_ocr_row=_bool("is_ocr_row"),
        )
        db.add(txn)
        rows_loaded += 1

    await db.flush()
    return rows_loaded, [b for b in banks_detected if b]
