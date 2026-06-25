"""
Phase 7 — Deduplicator

Two duplicate classes:

  EXACT  — same account_id + date + narration + debit + credit.
           Caused by re-uploaded statements or overlapping export windows.
           Safe to drop; the removed rows are saved to the audit file.

  NEAR   — same account + date + amounts, narration differs slightly.
           Caused by OCR variation or bank narration truncation across
           channels. FLAGGED but kept — a human must confirm before removal
           because the same amount on the same date could be two real txns.

Returns a 3-tuple: (cleaned_df, report_dict, audit_dict)
  audit_dict has two DataFrames: exact_removed and near_flagged — both
  written to separate CSV files by clean.py so investigators can review
  every single row that was touched.
"""

import pandas as pd
from cleaning_config import EXACT_DEDUP_KEYS, NEAR_DEDUP_KEYS


def run_deduplication(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    report = {
        "exact_duplicates_found":  0,
        "near_duplicates_flagged": 0,
        "rows_before":             len(df),
        "rows_after":              0,
    }

    df = df.copy()
    df["is_duplicate"] = False

    # ── 1. Exact deduplication ────────────────────────────────────────────
    exact_mask      = df.duplicated(subset=EXACT_DEDUP_KEYS, keep="first")
    exact_removed   = df.loc[exact_mask].copy().reset_index(drop=True)
    exact_removed["removal_reason"] = "EXACT_DUPLICATE"
    exact_removed["duplicate_of_row"] = (
        df[~exact_mask]
        .reset_index()
        .set_index(EXACT_DEDUP_KEYS)
        .reindex(exact_removed.set_index(EXACT_DEDUP_KEYS).index)["index"]
        .values
        if not exact_removed.empty else []
    )

    df.loc[exact_mask, "is_duplicate"] = True
    df = df[~exact_mask].reset_index(drop=True)
    report["exact_duplicates_found"] = int(exact_mask.sum())

    # ── 2. Near-duplicate flagging ────────────────────────────────────────
    near_flag_idx = []
    for _, group in df.groupby(NEAR_DEDUP_KEYS, sort=False):
        if len(group) >= 2:
            near_flag_idx.extend(group.index[1:].tolist())

    near_flagged = pd.DataFrame()
    if near_flag_idx:
        df.loc[near_flag_idx, "is_duplicate"] = True
        near_flagged = df.loc[near_flag_idx].copy().reset_index(drop=True)
        near_flagged["flag_reason"] = "NEAR_DUPLICATE_SAME_AMOUNT_DIFFERENT_NARRATION"
        report["near_duplicates_flagged"] = len(near_flag_idx)

    report["rows_after"] = len(df)

    audit = {
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_flagged":  near_flagged if not near_flagged.empty
                                    else pd.DataFrame(),
    }
    return df, report, audit