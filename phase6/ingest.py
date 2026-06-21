"""
Phase 6 — Ingestion Pipeline

Entry point. Processes a single file OR a directory of mixed-format bank
statements, normalises everything to the unified schema, and writes:
  • ingested_transactions.csv
  • ingestion_report.csv

Usage
-----
  python ingest.py --input samples/             --out-dir ingested/
  python ingest.py --input statement_SBI.csv    --out-dir ingested/
  python ingest.py --input scanned_page.png     --out-dir ingested/
"""

import os, argparse
import pandas as pd
from pathlib import Path

from ingestion_config import SUPPORTED_EXTENSIONS
from format_parsers   import parse_csv, parse_xlsx, parse_pdf, parse_image
from normalizer       import normalize


# ── Single-file ingestion ─────────────────────────────────────────────────────

def ingest_file(file_path: str) -> tuple[pd.DataFrame, dict]:
    ext   = Path(file_path).suffix.lower()
    fname = os.path.basename(file_path)

    report = {
        "file":             fname,
        "extension":        ext,
        "bank_detected":    "",
        "rows_parsed":      0,
        "rows_after_clean": 0,
        "parse_warnings":   "",
        "status":           "ok",
    }

    if ext not in SUPPORTED_EXTENSIONS:
        report["status"]         = "skipped"
        report["parse_warnings"] = f"Unsupported extension: {ext}"
        return pd.DataFrame(), report

    try:
        if   ext == ".csv":
            raw_df, header_text, fmt, pw = parse_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            raw_df, header_text, fmt, pw = parse_xlsx(file_path)
        elif ext == ".pdf":
            raw_df, header_text, fmt, pw = parse_pdf(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            raw_df, header_text, fmt, pw = parse_image(file_path)
        else:
            report["status"] = "skipped"
            return pd.DataFrame(), report
    except Exception as e:
        report["status"]         = "error"
        report["parse_warnings"] = f"Parser crash: {e}"
        return pd.DataFrame(), report

    report["rows_parsed"] = len(raw_df)
    if pw:
        report["parse_warnings"] = " | ".join(pw)

    if raw_df.empty:
        report["status"] = "empty"
        return pd.DataFrame(), report

    try:
        norm_df, nw = normalize(raw_df, header_text, file_path, fmt)
    except Exception as e:
        report["status"]          = "normalization_error"
        report["parse_warnings"] += f" | Normalization crash: {e}"
        return pd.DataFrame(), report

    report["rows_after_clean"] = len(norm_df)
    if not norm_df.empty:
        report["bank_detected"] = norm_df["bank_name"].iloc[0]
    if nw:
        report["parse_warnings"] += " | " + " | ".join(nw)

    return norm_df, report


# ── Pipeline ──────────────────────────────────────────────────────────────────

def ingest_directory(input_path: str, out_dir: str):
    files = sorted(
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return _run_pipeline(files, out_dir)


def ingest_single(file_path: str, out_dir: str):
    return _run_pipeline([file_path], out_dir)


def _run_pipeline(files: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    all_dfs, report_rows = [], []

    print(f"\n{'═' * 62}")
    print(f"  Phase 6 — Ingestion Pipeline   ({len(files)} file(s))")
    print(f"{'═' * 62}")

    for idx, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        print(f"\n  [{idx}/{len(files)}] {fname}")

        df, report = ingest_file(fp)
        report_rows.append(report)

        rows_in = report["rows_parsed"]
        rows_ok = report["rows_after_clean"]
        bank    = report["bank_detected"]
        warns   = report["parse_warnings"]
        status  = report["status"]

        if not df.empty:
            all_dfs.append(df)
            print(f"    ✓  Bank    : {bank}")
            print(f"    ✓  Rows    : {rows_in} parsed → {rows_ok} clean")
            if warns:
                print(f"    ⚠  Warnings: {warns[:160]}")
        else:
            label = {
                "ok":                  "All rows dropped during normalisation",
                "empty":               "Parser returned empty table",
                "error":               "Parser crashed",
                "normalization_error": "Normaliser crashed",
                "skipped":             "Skipped (unsupported extension)",
            }.get(status, status)
            print(f"    ✗  Status  : {label}")
            if warns:
                print(f"    ✗  Detail  : {warns[:160]}")

    # ── Merge & deduplicate ───────────────────────────────────────────────
    if all_dfs:
        merged       = pd.concat(all_dfs, ignore_index=True)
        before_dedup = len(merged)
        merged = merged.drop_duplicates(
            subset=["account_id", "date", "narration", "debit", "credit"]
        ).reset_index(drop=True)
        dedup_removed = before_dedup - len(merged)
        merged.to_csv(os.path.join(out_dir, "ingested_transactions.csv"), index=False)
    else:
        merged, dedup_removed = pd.DataFrame(), 0
        print("\n  ✗  No data was successfully ingested.")

    pd.DataFrame(report_rows).to_csv(
        os.path.join(out_dir, "ingestion_report.csv"), index=False
    )
    _print_summary(report_rows, merged, dedup_removed, out_dir)
    return merged, pd.DataFrame(report_rows)


def _print_summary(report_rows, merged, dedup_removed, out_dir):
    total   = len(report_rows)
    ok      = sum(1 for r in report_rows if r["rows_after_clean"] > 0)
    failed  = sum(1 for r in report_rows
                  if r["status"] in ("error", "normalization_error", "empty")
                  or (r["status"] == "ok" and r["rows_after_clean"] == 0))
    skipped = sum(1 for r in report_rows if r["status"] == "skipped")

    print(f"\n{'═' * 62}")
    print("  INGESTION SUMMARY")
    print(f"{'═' * 62}")
    print(f"  Files processed    : {total}")
    print(f"    Successfully      : {ok}")
    print(f"    Failed / empty    : {failed}")
    print(f"    Skipped           : {skipped}")
    if not merged.empty:
        print(f"  Total rows ingested : {len(merged) + dedup_removed}")
        print(f"  Duplicates removed  : {dedup_removed}")
        print(f"  Final clean rows    : {len(merged)}")
        print(f"  Banks detected      : {', '.join(merged['bank_name'].unique())}")
        print(f"  Formats ingested    : {', '.join(merged['source_format'].unique())}")
    print(f"\n  Output → {os.path.abspath(out_dir)}/")
    print(f"    • ingested_transactions.csv")
    print(f"    • ingestion_report.csv")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Phase 6 — Bank Statement Ingestion Pipeline"
    )
    ap.add_argument("--input",   required=True, help="File or directory to ingest")
    ap.add_argument("--out-dir", default="ingested", help="Output directory")
    args = ap.parse_args()

    if   os.path.isdir(args.input):  ingest_directory(args.input, args.out_dir)
    elif os.path.isfile(args.input): ingest_single(args.input, args.out_dir)
    else:
        print(f"Error: '{args.input}' is not a valid file or directory")