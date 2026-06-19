"""
Phase 7 - Deduplicator

Handles two classes of duplicates common in seized multi-source datasets:

1. EXACT duplicates — identical account_id + date + narration + amounts.
   Cause: same statement uploaded twice, overlapping date ranges across
   two export files from the same bank, or a statement re-exported after
   a period rollover. Safe to drop the second occurrence.

2. NEAR duplicates — same account + date + amounts but narration differs
   slightly. Cause: OCR variation (one source PDF, one scanned PNG of the
   same statement), bank truncating narration differently across channels,
   or NEFT/UPI reference numbers differing slightly between sender/receiver
   copies. These are FLAGGED but not auto-dropped — an investigator must
   confirm because near-dupe detection across different accounts could
   accidentally merge separate legitimate transactions.
"""

import pandas as pd
from cleaning_config import EXACT_DEDUP_KEYS, NEAR_DEDUP_KEYS


def run_deduplication(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns (cleaned_df, report).
    Adds 'is_duplicate' column — True for rows that are exact duplicates
    of an earlier row (the first occurrence is kept, later ones marked).
    """
    report = {
        "exact_duplicates_found": 0,
        "near_duplicates_flagged": 0,
        "rows_before": len(df),
        "rows_after": 0,
    }

    df = df.copy()
    df["is_duplicate"] = False

    # --- 1. Exact deduplication ---
    exact_mask = df.duplicated(subset=EXACT_DEDUP_KEYS, keep="first")
    n_exact = int(exact_mask.sum())
    df.loc[exact_mask, "is_duplicate"] = True
    report["exact_duplicates_found"] = n_exact

    # Drop exact dupes immediately — they add zero information
    df = df[~exact_mask].reset_index(drop=True)

    # --- 2. Near-duplicate flagging (same account + date + amounts) ---
    near_groups = df.groupby(NEAR_DEDUP_KEYS, sort=False)
    near_flag_indices = []

    for _, group in near_groups:
        if len(group) < 2:
            continue
        # Multiple rows with same account/date/amounts but different narrations
        # Flag all but the first as near-dupes
        near_flag_indices.extend(group.index[1:].tolist())

    n_near = len(near_flag_indices)
    if near_flag_indices:
        # Don't drop — flag for human review
        df.loc[near_flag_indices, "is_duplicate"] = True
        report["near_duplicates_flagged"] = n_near

    report["rows_after"] = len(df)
    return df, report
