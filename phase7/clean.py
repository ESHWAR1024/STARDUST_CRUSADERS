"""
Phase 7 - Data Cleaning Engine

Orchestrates all cleaning passes on the ingested_transactions.csv
produced by Phase 6. Outputs:

  cleaned_transactions.csv  — clean data ready for Phase 8/9/10
  cleaning_report.json      — full audit trail of every change made
  suspect_accounts.csv      — accounts with balance integrity issues
                              (critical for investigators)

Design principle: NEVER silently drop data. Every removal is logged.
Every modification is traceable. The investigator must be able to
explain to a court exactly what the system did and why.

Usage:
    python clean.py --input ../phase6/ingested/ingested_transactions.csv
                    --out-dir cleaned/
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime

from cleaning_config import CLEANED_OUTPUT_COLS
from deduplicator import run_deduplication
from validator import (
    validate_dates,
    clean_amounts,
    validate_balance_continuity,
    flag_statistical_outliers,
    normalise_text_fields,
)


def run_cleaning_pipeline(input_path: str, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Phase 7 — Data Cleaning Engine")
    print(f"{'='*60}")

    # --- Load ---
    df = pd.read_csv(input_path, dtype=str)
    # Convert numeric columns
    for col in ("debit", "credit", "balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    print(f"Loaded : {len(df)} rows, {df['account_id'].nunique()} accounts")
    print(f"Sources: {df['source_format'].value_counts().to_dict()}")

    master_report = {
        "run_timestamp": datetime.now().isoformat(),
        "input_file": os.path.basename(input_path),
        "rows_input": len(df),
    }

    # Initialise audit columns
    df["clean_flags"]         = ""
    df["is_duplicate"]        = False
    df["is_balance_breach"]   = False
    df["is_high_value_flag"]  = False
    df["is_ocr_row"]          = df["source_format"].isin(["image"])

    # -----------------------------------------------------------------------
    # Pass 1: Deduplication
    # -----------------------------------------------------------------------
    print("\n[1/5] Deduplication ...")
    df, dedup_report = run_deduplication(df)
    master_report["deduplication"] = dedup_report
    print(f"      Exact dupes removed : {dedup_report['exact_duplicates_found']}")
    print(f"      Near dupes flagged  : {dedup_report['near_duplicates_flagged']}")

    # -----------------------------------------------------------------------
    # Pass 2: Date Validation
    # -----------------------------------------------------------------------
    print("\n[2/5] Date validation ...")
    df, date_report = validate_dates(df)
    master_report["date_validation"] = date_report

    # Merge date flags into clean_flags
    df["clean_flags"] = df.apply(
        lambda r: _merge_flags(r["clean_flags"], r.get("_date_flags", "")), axis=1
    )
    df = df.drop(columns=["_date_flags"], errors="ignore")

    print(f"      Null dates          : {date_report['null_dates']}")
    print(f"      Bad format dates    : {date_report['bad_format_dates']}")
    print(f"      Out-of-range dates  : {date_report['out_of_range_dates']}")

    # -----------------------------------------------------------------------
    # Pass 3: Amount Cleaning
    # -----------------------------------------------------------------------
    print("\n[3/5] Amount cleaning ...")
    df, amount_report = clean_amounts(df)
    master_report["amount_cleaning"] = amount_report

    # Flag zero-debit-credit rows
    zero_mask = (df["debit"] == 0.0) & (df["credit"] == 0.0)
    df.loc[zero_mask, "clean_flags"] = df.loc[zero_mask, "clean_flags"].apply(
        lambda f: _merge_flags(f, "ZERO_DEBIT_AND_CREDIT")
    )
    print(f"      Amount corrections  : {amount_report['amount_corrections']}")
    print(f"      Zero debit+credit   : {amount_report['zero_debit_credit_rows']}")

    # -----------------------------------------------------------------------
    # Pass 4: Balance Continuity
    # -----------------------------------------------------------------------
    print("\n[4/5] Balance continuity validation ...")
    df, balance_report = validate_balance_continuity(df)
    master_report["balance_validation"] = {
        k: v for k, v in balance_report.items() if k != "suspect_accounts"
    }
    master_report["balance_validation"]["n_suspect_accounts"] = len(
        balance_report["suspect_accounts"]
    )

    print(f"      Accounts checked    : {balance_report['accounts_checked']}")
    print(f"      Accounts w/ breaches: {balance_report['accounts_with_breaches']}")
    print(f"      Breach rows         : {balance_report['total_breach_rows']}")
    if balance_report["suspect_accounts"]:
        print(f"      ⚠ SUSPECT ACCOUNTS  : {len(balance_report['suspect_accounts'])}")
        for sa in balance_report["suspect_accounts"]:
            print(f"        {sa['account_id']} — {sa['breach_ratio']*100:.1f}% rows breach "
                  f"({sa['breach_rows']}/{sa['total_rows']})")

    # -----------------------------------------------------------------------
    # Pass 5: Statistical Outlier Flagging + Text Normalisation
    # -----------------------------------------------------------------------
    print("\n[5/5] Outlier flagging + text normalisation ...")
    df, outlier_report = flag_statistical_outliers(df)
    df, text_report    = normalise_text_fields(df)
    master_report["outlier_flagging"]   = outlier_report
    master_report["text_normalisation"] = text_report

    print(f"      Outlier rows flagged : {outlier_report['outlier_rows_flagged']}")
    print(f"      Narrations cleaned   : {text_report['narrations_cleaned']}")
    print(f"      Channels normalised  : {text_report['channels_normalised']}")

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    # Ensure all required columns exist
    for col in CLEANED_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""

    cleaned = df[CLEANED_OUTPUT_COLS].copy()
    master_report["rows_output"] = len(cleaned)
    master_report["rows_with_any_flag"] = int(
        (cleaned["clean_flags"].fillna("") != "").sum() |
        cleaned["is_duplicate"].sum() |
        cleaned["is_balance_breach"].sum() |
        cleaned["is_high_value_flag"].sum()
    )

    # Main cleaned output
    cleaned_path = os.path.join(out_dir, "cleaned_transactions.csv")
    cleaned.to_csv(cleaned_path, index=False)

    # Suspect accounts
    suspect_path = os.path.join(out_dir, "suspect_accounts.csv")
    if balance_report["suspect_accounts"]:
        pd.DataFrame(balance_report["suspect_accounts"]).to_csv(suspect_path, index=False)
    else:
        pd.DataFrame(columns=["account_id", "breach_ratio", "breach_rows", "total_rows"]
                     ).to_csv(suspect_path, index=False)

    # Full cleaning report
    report_path = os.path.join(out_dir, "cleaning_report.json")
    with open(report_path, "w") as f:
        json.dump(master_report, f, indent=2)

    _print_summary(master_report, out_dir)
    return cleaned


def _merge_flags(existing: str, new_flag: str) -> str:
    existing = str(existing).strip() if existing else ""
    new_flag = str(new_flag).strip() if new_flag else ""
    if not new_flag:
        return existing
    if not existing:
        return new_flag
    return existing + " | " + new_flag


def _print_summary(report: dict, out_dir: str):
    print(f"\n{'='*60}")
    print("CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"Rows input              : {report['rows_input']}")
    print(f"Rows output             : {report['rows_output']}")
    print(f"Exact dupes removed     : {report['deduplication']['exact_duplicates_found']}")
    print(f"Rows with any flag      : {report['rows_with_any_flag']}")
    print(f"Balance breach rows     : {report['balance_validation']['total_breach_rows']}")
    print(f"Suspect accounts        : {report['balance_validation']['n_suspect_accounts']}")
    print(f"High-value outlier rows : {report['outlier_flagging']['outlier_rows_flagged']}")
    print(f"\nOutputs → {os.path.abspath(out_dir)}")
    print(f"  cleaned_transactions.csv  ← feed to Phase 8/9/10")
    print(f"  suspect_accounts.csv      ← flag for investigators")
    print(f"  cleaning_report.json      ← full audit trail")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 — Data Cleaning Engine")
    parser.add_argument("--input",   required=True, help="ingested_transactions.csv from Phase 6")
    parser.add_argument("--out-dir", default="cleaned", help="Output directory")
    args = parser.parse_args()
    run_cleaning_pipeline(args.input, args.out_dir)
