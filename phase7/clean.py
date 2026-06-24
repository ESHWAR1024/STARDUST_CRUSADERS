"""
Phase 7 — Data Cleaning Engine

Input : ingested_transactions.csv  (from Phase 6)
Output directory contains:

  cleaned_transactions.csv       ← ready for Phase 8 / 9 / 10
  removed_data.csv               ← ALL rows removed (exact duplicates)
                                    with reason column — court-ready audit
  flagged_data.csv               ← ALL rows kept but flagged, with every
                                    flag reason listed per row
  all_actions.csv                ← row-level log of every single change
                                    made during cleaning (what, why, before/after)
  suspect_accounts.csv           ← accounts with balance integrity issues
  cleaning_report.json           ← machine-readable full audit trail
  cleaning_summary.txt           ← human-readable narrative summary

Design principle: NOTHING is silently dropped.
Every removal is in removed_data.csv with its reason.
Every flag is in flagged_data.csv with its reason.
Every value change is in all_actions.csv.
An investigator can reconstruct exactly what the system did.

Usage:
    python clean.py --input ../phase6/ingested/ingested_transactions.csv
                    --out-dir cleaned/
"""

import os, json, argparse
import pandas as pd
from datetime import datetime

from cleaning_config import CLEANED_OUTPUT_COLS
from deduplicator   import run_deduplication
from validator      import (
    validate_dates,
    clean_amounts,
    validate_balance_continuity,
    flag_statistical_outliers,
    flag_velocity_bursts,
    normalise_text_fields,
)


def run_cleaning_pipeline(input_path: str, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print("  Phase 7 — Data Cleaning Engine")
    print(f"{'='*62}")

    # ── Load ─────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path, dtype=str)
    for col in ("debit", "credit", "balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    print(f"  Loaded : {len(df):,} rows across {df['account_id'].nunique()} accounts")
    print(f"  Sources: {df['source_format'].value_counts().to_dict()}")

    report = {
        "run_timestamp": datetime.now().isoformat(),
        "input_file":    os.path.basename(input_path),
        "rows_input":    len(df),
    }

    # Audit log — every action taken on every row
    all_actions: list[dict] = []

    # Initialise audit flag columns
    df["clean_flags"]        = ""
    df["is_duplicate"]       = False
    df["is_balance_breach"]  = False
    df["is_high_value_flag"] = False
    df["is_velocity_flag"]   = False
    df["is_ocr_row"]         = df["source_format"].isin(["image"])

    # ── Pass 1: Deduplication ─────────────────────────────────────────────
    print("\n  [1/6] Deduplication ...")
    df, dedup_report, dedup_audit = run_deduplication(df)

    # Log removed rows as actions
    for _, row in dedup_audit["exact_duplicates_removed"].iterrows():
        all_actions.append({
            "row_index":   "REMOVED",
            "account_id":  row.get("account_id", ""),
            "date":        row.get("date", ""),
            "narration":   str(row.get("narration", ""))[:80],
            "debit":       row.get("debit", ""),
            "credit":      row.get("credit", ""),
            "balance":     row.get("balance", ""),
            "source_file": row.get("source_file", ""),
            "action_type": "EXACT_DUPLICATE_REMOVED",
            "detail":      "Row removed: identical account+date+narration+amounts already exists",
        })

    report["deduplication"] = dedup_report
    print(f"      Exact dupes removed  : {dedup_report['exact_duplicates_found']}")
    print(f"      Near dupes flagged   : {dedup_report['near_duplicates_flagged']}")

    # ── Pass 2: Date Validation ───────────────────────────────────────────
    print("\n  [2/6] Date validation ...")
    df, date_report, date_actions = validate_dates(df)
    df["clean_flags"] = df.apply(
        lambda r: _merge(r["clean_flags"], r.get("_date_flags", "")), axis=1
    )
    df = df.drop(columns=["_date_flags"], errors="ignore")
    all_actions.extend(date_actions)
    report["date_validation"] = date_report

    print(f"      Null dates           : {date_report['null_dates']}")
    print(f"      Bad format dates     : {date_report['bad_format_dates']}")
    print(f"      Out-of-range dates   : {date_report['out_of_range_dates']}")

    # ── Pass 3: Amount Cleaning ───────────────────────────────────────────
    print("\n  [3/6] Amount cleaning ...")
    df, amount_report, amount_actions = clean_amounts(df)
    zero_mask = (df["debit"] == 0.0) & (df["credit"] == 0.0)
    df.loc[zero_mask, "clean_flags"] = df.loc[zero_mask, "clean_flags"].apply(
        lambda f: _merge(f, "ZERO_DEBIT_AND_CREDIT")
    )
    all_actions.extend(amount_actions)
    report["amount_cleaning"] = amount_report

    print(f"      Amount corrections   : {amount_report['amount_corrections']}")
    print(f"      Zero debit+credit    : {amount_report['zero_debit_credit_rows']}")

    # ── Pass 4: Balance Continuity ────────────────────────────────────────
    print("\n  [4/6] Balance continuity ...")
    df, balance_report, balance_actions = validate_balance_continuity(df)
    all_actions.extend(balance_actions)
    report["balance_validation"] = {
        k: v for k, v in balance_report.items() if k != "suspect_accounts"
    }
    report["balance_validation"]["n_suspect_accounts"] = len(balance_report["suspect_accounts"])

    print(f"      Accounts checked     : {balance_report['accounts_checked']}")
    print(f"      Accounts w/ breaches : {balance_report['accounts_with_breaches']}")
    print(f"      Breach rows          : {balance_report['total_breach_rows']}")
    if balance_report["suspect_accounts"]:
        print(f"      ⚠  SUSPECT ACCOUNTS  : {len(balance_report['suspect_accounts'])}")
        for sa in balance_report["suspect_accounts"]:
            print(f"         {sa['account_id']} — {sa['breach_ratio']*100:.0f}% breach "
                  f"({sa['breach_rows']}/{sa['total_rows']}) [{sa['severity']}]")

    # ── Pass 5: Statistical Outliers ──────────────────────────────────────
    print("\n  [5/6] Outlier flagging + velocity check ...")
    df, outlier_report, outlier_actions = flag_statistical_outliers(df)
    df, velocity_report, velocity_actions = flag_velocity_bursts(df)
    all_actions.extend(outlier_actions)
    all_actions.extend(velocity_actions)
    report["outlier_flagging"] = outlier_report
    report["velocity_flagging"] = velocity_report

    print(f"      Outlier rows flagged  : {outlier_report['outlier_rows_flagged']}")
    print(f"      Velocity rows flagged : {velocity_report['velocity_rows_flagged']}")
    if velocity_report["velocity_accounts_flagged"] > 0:
        print(f"      ⚠  Velocity accounts : {velocity_report['velocity_accounts_flagged']}")

    # ── Pass 6: Text Normalisation ────────────────────────────────────────
    print("\n  [6/6] Text normalisation ...")
    df, text_report, text_actions = normalise_text_fields(df)
    all_actions.extend(text_actions)
    report["text_normalisation"] = text_report

    print(f"      Narrations cleaned   : {text_report['narrations_cleaned']}")
    print(f"      Channels normalised  : {text_report['channels_normalised']}")

    # ── Build flag columns ────────────────────────────────────────────────
    for col in CLEANED_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = "" if col == "clean_flags" else False

    cleaned = df[CLEANED_OUTPUT_COLS].copy()
    report["rows_output"] = len(cleaned)

    any_flag = (
        (cleaned["clean_flags"].fillna("") != "") |
        cleaned["is_duplicate"].astype(bool) |
        cleaned["is_balance_breach"].astype(bool) |
        cleaned["is_high_value_flag"].astype(bool) |
        cleaned["is_velocity_flag"].astype(bool)
    )
    report["rows_with_any_flag"] = int(any_flag.sum())
    report["rows_fully_clean"]   = int((~any_flag).sum())

    # ── Write all output files ────────────────────────────────────────────
    _write_outputs(cleaned, dedup_audit, df, all_actions,
                   balance_report, report, out_dir)

    _print_summary(report, out_dir)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
def _write_outputs(cleaned, dedup_audit, df_full, all_actions,
                   balance_report, report, out_dir):

    # 1. Main cleaned output
    cleaned.to_csv(os.path.join(out_dir, "cleaned_transactions.csv"), index=False)

    # 2. removed_data.csv — every row physically removed, with reason
    removed_parts = []

    exact_removed = dedup_audit.get("exact_duplicates_removed", pd.DataFrame())
    if not exact_removed.empty:
        exact_removed = exact_removed.copy()
        exact_removed["removal_reason"] = "EXACT_DUPLICATE"
        exact_removed["removal_detail"] = (
            "Identical account_id + date + narration + debit + credit already exists. "
            "Second occurrence removed."
        )
        removed_parts.append(exact_removed)

    if removed_parts:
        removed_df = pd.concat(removed_parts, ignore_index=True)
    else:
        removed_df = pd.DataFrame(columns=["account_id","date","narration",
                                            "debit","credit","removal_reason","removal_detail"])
    removed_df.to_csv(os.path.join(out_dir, "removed_data.csv"), index=False)

    # 3. flagged_data.csv — every row kept but flagged, with all reasons
    any_flag_mask = (
        (cleaned["clean_flags"].fillna("") != "") |
        cleaned["is_duplicate"].astype(bool) |
        cleaned["is_balance_breach"].astype(bool) |
        cleaned["is_high_value_flag"].astype(bool) |
        cleaned["is_velocity_flag"].astype(bool)
    )
    flagged_df = cleaned[any_flag_mask].copy()

    # Build a human-readable reason column
    def _describe_flags(row):
        reasons = []
        flags = str(row.get("clean_flags", "")).strip()
        if flags:
            reasons.append(flags)
        if row.get("is_duplicate"):
            reasons.append("NEAR_DUPLICATE: same date+amounts, different narration")
        if row.get("is_balance_breach"):
            reasons.append("BALANCE_BREACH: balance does not reconcile with prior row")
        if row.get("is_high_value_flag"):
            reasons.append("HIGH_VALUE_OUTLIER: amount exceeds 3×IQR for this account")
        if row.get("is_velocity_flag"):
            reasons.append("VELOCITY_BURST: rapid succession of debits in short window")
        if row.get("is_ocr_row"):
            reasons.append("OCR_SOURCE: lower data confidence (scanned image)")
        return " | ".join(reasons)

    flagged_df["all_flag_reasons"] = flagged_df.apply(_describe_flags, axis=1)
    flagged_df.to_csv(os.path.join(out_dir, "flagged_data.csv"), index=False)

    # 4. Near duplicates (flagged but kept — separate file for easy review)
    near_flagged = dedup_audit.get("near_duplicates_flagged", pd.DataFrame())
    if not near_flagged.empty:
        near_flagged.to_csv(os.path.join(out_dir, "near_duplicates_flagged.csv"), index=False)

    # 5. all_actions.csv — every single change made during cleaning
    if all_actions:
        actions_df = pd.DataFrame(all_actions)
        actions_df.to_csv(os.path.join(out_dir, "all_actions.csv"), index=False)
    else:
        pd.DataFrame(columns=["row_index","account_id","date","narration",
                               "debit","credit","balance","source_file",
                               "action_type","detail"]
                     ).to_csv(os.path.join(out_dir, "all_actions.csv"), index=False)

    # 6. suspect_accounts.csv
    suspects = balance_report.get("suspect_accounts", [])
    pd.DataFrame(suspects if suspects else [],
                 columns=["account_id","breach_ratio","breach_rows","total_rows","severity"]
                 ).to_csv(os.path.join(out_dir, "suspect_accounts.csv"), index=False)

    # 7. cleaning_report.json
    with open(os.path.join(out_dir, "cleaning_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # 8. cleaning_summary.txt — human-readable narrative
    _write_summary_txt(report, balance_report, len(removed_df),
                       len(flagged_df), all_actions, out_dir)


def _write_summary_txt(report, balance_report, n_removed, n_flagged,
                        all_actions, out_dir):
    lines = [
        "=" * 70,
        "PHASE 7 — DATA CLEANING SUMMARY",
        f"Run at: {report['run_timestamp']}",
        f"Input : {report['input_file']}",
        "=" * 70,
        "",
        "OVERVIEW",
        "-" * 40,
        f"  Rows loaded              : {report['rows_input']:,}",
        f"  Rows in cleaned output   : {report['rows_output']:,}",
        f"  Rows removed             : {report['rows_input'] - report['rows_output']:,}",
        f"  Rows flagged (kept)      : {n_flagged:,}",
        f"  Rows fully clean         : {report['rows_fully_clean']:,}",
        "",
        "PASS 1 — DEDUPLICATION",
        "-" * 40,
        f"  Exact duplicates removed : {report['deduplication']['exact_duplicates_found']}",
        f"  → Cause: same statement uploaded twice or overlapping date ranges",
        f"  → Action: second occurrence removed; first kept",
        f"  Near duplicates flagged  : {report['deduplication']['near_duplicates_flagged']}",
        f"  → Cause: same amounts, narration differs (OCR variation / channel diff)",
        f"  → Action: flagged is_duplicate=True; human review required",
        "",
        "PASS 2 — DATE VALIDATION",
        "-" * 40,
        f"  Null dates      : {report['date_validation']['null_dates']}",
        f"  → Rows kept but skipped in chronological checks",
        f"  Bad format      : {report['date_validation']['bad_format_dates']}",
        f"  → Flagged BAD_DATE_FORMAT; could not parse as YYYY-MM-DD",
        f"  Out of range    : {report['date_validation']['out_of_range_dates']}",
        f"  → Flagged DATE_OUT_OF_RANGE; year outside 2000–2030",
        "",
        "PASS 3 — AMOUNT CLEANING",
        "-" * 40,
        f"  Corrections made         : {report['amount_cleaning']['amount_corrections']}",
        f"  → Types: bracket negatives, CR/DR suffixes, lakh commas, ₹ symbol",
        f"  Zero debit+credit rows   : {report['amount_cleaning']['zero_debit_credit_rows']}",
        f"  → Flagged ZERO_DEBIT_AND_CREDIT; likely PDF summary/header rows",
        "",
        "PASS 4 — BALANCE CONTINUITY",
        "-" * 40,
        f"  Accounts checked         : {report['balance_validation']['accounts_checked']}",
        f"  Accounts with breaches   : {report['balance_validation']['accounts_with_breaches']}",
        f"  Breach rows total        : {report['balance_validation']['total_breach_rows']}",
        f"  → Formula: prev_balance + credit - debit ≈ current_balance",
        f"  → Tolerance: ₹1 (bank rounding)",
        f"  → Breach means: OCR corruption OR statement tampering",
    ]

    suspects = balance_report.get("suspect_accounts", [])
    if suspects:
        lines += [
            f"",
            f"  ⚠  SUSPECT ACCOUNTS ({len(suspects)}):",
        ]
        for sa in suspects:
            lines.append(
                f"     • {sa['account_id']} — {sa['breach_ratio']*100:.0f}% of rows breach "
                f"({sa['breach_rows']}/{sa['total_rows']}) [{sa['severity']}]"
            )
        lines.append(
            "     These accounts need investigator review — balances do not reconcile."
        )

    lines += [
        "",
        "PASS 5 — STATISTICAL OUTLIERS + VELOCITY",
        "-" * 40,
        f"  Outlier rows flagged     : {report['outlier_flagging']['outlier_rows_flagged']}",
        f"  → Method: per-account IQR (not global threshold)",
        f"  → Amount > Q3 + 3×IQR for that specific account",
        f"  Velocity rows flagged    : {report['velocity_flagging']['velocity_rows_flagged']}",
        f"  Velocity accounts        : {report['velocity_flagging']['velocity_accounts_flagged']}",
        f"  → Rapid successive debits within short window (mule signal)",
        "",
        "PASS 6 — TEXT NORMALISATION",
        "-" * 40,
        f"  Narrations cleaned       : {report['text_normalisation']['narrations_cleaned']}",
        f"  → OCR noise stripped, whitespace collapsed, uppercased",
        f"  Channels normalised      : {report['text_normalisation']['channels_normalised']}",
        f"  → Variants mapped: ATM WDL→ATM, NACH→ECS, CLG→CHEQUE, etc.",
        "",
        "ACTION LOG SUMMARY",
        "-" * 40,
    ]

    # Count by action type
    from collections import Counter
    action_counts = Counter(a["action_type"] for a in all_actions)
    for atype, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {atype:<40}: {count:,}")

    lines += [
        "",
        "OUTPUT FILES",
        "-" * 40,
        "  cleaned_transactions.csv      ← Feed to Phase 8/9/10",
        "  removed_data.csv              ← Every removed row with reason",
        "  flagged_data.csv              ← Every flagged row with reasons",
        "  near_duplicates_flagged.csv   ← Near-dupes for human review",
        "  all_actions.csv               ← Row-level audit log of every change",
        "  suspect_accounts.csv          ← Balance integrity failures",
        "  cleaning_report.json          ← Machine-readable full report",
        "  cleaning_summary.txt          ← This file",
        "=" * 70,
    ]

    with open(os.path.join(out_dir, "cleaning_summary.txt"), "w") as f:
        f.write("\n".join(lines))


def _merge(existing: str, new_flag: str) -> str:
    e = str(existing).strip() if existing else ""
    n = str(new_flag).strip()  if new_flag  else ""
    if not n: return e
    if not e: return n
    return e + " | " + n


def _print_summary(report, out_dir):
    print(f"\n{'='*62}")
    print("  CLEANING SUMMARY")
    print(f"{'='*62}")
    print(f"  Rows input              : {report['rows_input']:,}")
    print(f"  Rows output             : {report['rows_output']:,}")
    print(f"  Exact dupes removed     : {report['deduplication']['exact_duplicates_found']}")
    print(f"  Near dupes flagged      : {report['deduplication']['near_duplicates_flagged']}")
    print(f"  Balance breach rows     : {report['balance_validation']['total_breach_rows']}")
    print(f"  Suspect accounts        : {report['balance_validation']['n_suspect_accounts']}")
    print(f"  High-value outlier rows : {report['outlier_flagging']['outlier_rows_flagged']}")
    print(f"  Velocity burst rows     : {report['velocity_flagging']['velocity_rows_flagged']}")
    print(f"  Rows with any flag      : {report['rows_with_any_flag']:,}")
    print(f"  Rows fully clean        : {report['rows_fully_clean']:,}")
    print(f"\n  Outputs → {os.path.abspath(out_dir)}/")
    print(f"    • cleaned_transactions.csv     ← feed to Phase 8/9/10")
    print(f"    • removed_data.csv             ← all removed rows with reasons")
    print(f"    • flagged_data.csv             ← all flagged rows with reasons")
    print(f"    • all_actions.csv              ← every change made, row by row")
    print(f"    • suspect_accounts.csv         ← balance integrity failures")
    print(f"    • cleaning_report.json         ← machine-readable audit trail")
    print(f"    • cleaning_summary.txt         ← human-readable narrative")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 7 — Data Cleaning Engine")
    ap.add_argument("--input",   required=True)
    ap.add_argument("--out-dir", default="cleaned")
    args = ap.parse_args()
    run_cleaning_pipeline(args.input, args.out_dir)