"""
Phase 6 - Ingestion Pipeline

Main entry point. Takes a single file OR a directory of mixed-format
bank statements, routes each file to the correct parser, normalizes
to the unified schema, and writes:
  - ingested_transactions.csv   (all statements merged, unified schema)
  - ingestion_report.csv        (per-file summary: rows parsed, warnings, bank detected)

Usage:
    python ingest.py --input output/sample_bank_statements/ --out-dir ingested/
    python ingest.py --input statement_ACC000042_SBI.csv   --out-dir ingested/
"""

import os
import argparse
import pandas as pd
from pathlib import Path

from ingestion_config import SUPPORTED_EXTENSIONS
from format_parsers import parse_csv, parse_xlsx, parse_pdf, parse_image
from normalizer import normalize


def ingest_file(file_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Ingest a single file. Returns (normalized_df, report_row).
    """
    ext = Path(file_path).suffix.lower()
    fname = os.path.basename(file_path)

    report = {
        "file": fname,
        "extension": ext,
        "bank_detected": "",
        "rows_parsed": 0,
        "rows_after_clean": 0,
        "parse_warnings": "",
        "status": "ok",
    }

    if ext not in SUPPORTED_EXTENSIONS:
        report["status"] = "skipped"
        report["parse_warnings"] = f"Unsupported extension: {ext}"
        return pd.DataFrame(), report

    # --- Route to correct parser ---
    try:
        if ext == ".csv":
            raw_df, header_text, source_format, parse_warnings = parse_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            raw_df, header_text, source_format, parse_warnings = parse_xlsx(file_path)
        elif ext == ".pdf":
            raw_df, header_text, source_format, parse_warnings = parse_pdf(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            raw_df, header_text, source_format, parse_warnings = parse_image(file_path)
        else:
            report["status"] = "skipped"
            return pd.DataFrame(), report
    except Exception as e:
        report["status"] = "error"
        report["parse_warnings"] = f"Parser crash: {e}"
        return pd.DataFrame(), report

    report["rows_parsed"] = len(raw_df)
    if parse_warnings:
        report["parse_warnings"] = " | ".join(parse_warnings)

    if raw_df.empty:
        report["status"] = "empty"
        return pd.DataFrame(), report

    # --- Normalize to unified schema ---
    try:
        normalized_df, norm_warnings = normalize(
            raw_df, header_text, file_path, source_format
        )
    except Exception as e:
        report["status"] = "normalization_error"
        report["parse_warnings"] += f" | Normalization crash: {e}"
        return pd.DataFrame(), report

    report["rows_after_clean"] = len(normalized_df)
    report["bank_detected"] = normalized_df["bank_name"].iloc[0] if not normalized_df.empty else ""
    if norm_warnings:
        report["parse_warnings"] += " | " + " | ".join(norm_warnings)

    return normalized_df, report


def ingest_directory(input_path: str, out_dir: str):
    """Ingest all supported files in a directory."""
    files = [
        os.path.join(input_path, f)
        for f in sorted(os.listdir(input_path))
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return _run_pipeline(files, out_dir)


def ingest_single(file_path: str, out_dir: str):
    """Ingest a single file."""
    return _run_pipeline([file_path], out_dir)


def _run_pipeline(files: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    all_dfs = []
    report_rows = []

    print(f"\n{'='*60}")
    print(f"Phase 6 — Ingestion Pipeline")
    print(f"Files to process: {len(files)}")
    print(f"{'='*60}")

    for file_path in files:
        fname = os.path.basename(file_path)
        print(f"\n  [{files.index(file_path)+1}/{len(files)}] {fname}")

        df, report = ingest_file(file_path)
        report_rows.append(report)

        if not df.empty:
            all_dfs.append(df)
            print(f"    ✓ Bank    : {report['bank_detected']}")
            print(f"    ✓ Rows    : {report['rows_parsed']} parsed → {report['rows_after_clean']} clean")
            if report["parse_warnings"]:
                print(f"    ⚠ Warnings: {report['parse_warnings'][:120]}")
        else:
            print(f"    ✗ Status  : {report['status']}")
            if report["parse_warnings"]:
                print(f"    ✗ Reason  : {report['parse_warnings'][:120]}")

    # --- Merge and write ---
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        # Deduplicate on (account_id, date, narration, debit, credit)
        before_dedup = len(merged)
        merged = merged.drop_duplicates(
            subset=["account_id", "date", "narration", "debit", "credit"]
        ).reset_index(drop=True)
        dedup_removed = before_dedup - len(merged)

        out_csv = os.path.join(out_dir, "ingested_transactions.csv")
        merged.to_csv(out_csv, index=False)
    else:
        merged = pd.DataFrame()
        dedup_removed = 0
        print("\n  ✗ No data was successfully ingested.")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(os.path.join(out_dir, "ingestion_report.csv"), index=False)

    _print_summary(report_rows, merged, dedup_removed, out_dir)
    return merged, report_df


def _print_summary(report_rows, merged, dedup_removed, out_dir):
    total = len(report_rows)
    ok = sum(1 for r in report_rows if r["status"] == "ok")
    errored = sum(1 for r in report_rows if r["status"] in ("error", "normalization_error", "empty"))
    skipped = sum(1 for r in report_rows if r["status"] == "skipped")

    print(f"\n{'='*60}")
    print("INGESTION SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed    : {total}")
    print(f"  Successfully      : {ok}")
    print(f"  Errors/empty      : {errored}")
    print(f"  Skipped           : {skipped}")
    if not merged.empty:
        print(f"Total rows ingested : {len(merged) + dedup_removed}")
        print(f"Duplicates removed  : {dedup_removed}")
        print(f"Final clean rows    : {len(merged)}")
        print(f"Banks detected      : {', '.join(merged['bank_name'].unique())}")
        print(f"Formats ingested    : {', '.join(merged['source_format'].unique())}")
    print(f"\nOutputs written to: {os.path.abspath(out_dir)}")
    print(f"  ingested_transactions.csv")
    print(f"  ingestion_report.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 — Bank Statement Ingestion Pipeline")
    parser.add_argument("--input", required=True, help="File or directory to ingest")
    parser.add_argument("--out-dir", default="ingested", help="Output directory")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        ingest_directory(args.input, args.out_dir)
    elif os.path.isfile(args.input):
        ingest_single(args.input, args.out_dir)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
