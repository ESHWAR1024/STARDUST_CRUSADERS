"""
Phase 7 - Cleaning Engine Configuration

All thresholds here are STATISTICAL or STRUCTURAL — never hardcoded
rupee amounts or fixed date ranges. This is deliberate: the system must
generalise to the judges' unseen dataset which may have completely
different account sizes, date windows, or transaction volumes.

Rule of thumb used throughout:
  - Structural rules  : things that are always wrong (null date, both
                        debit AND credit == 0 with no balance change)
  - Statistical rules : things that are suspicious relative to the
                        account's own distribution (IQR-based outliers,
                        velocity z-scores) — these FLAG, never auto-drop
"""

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
# Exact dedup key: if all these fields match, it's the same transaction
# loaded twice (common when a statement is re-uploaded or overlaps with
# another export).
EXACT_DEDUP_KEYS = ["account_id", "date", "narration", "debit", "credit"]

# Near-dedup: same account + same date + amounts match but narration differs
# slightly (OCR variation, bank truncation). Flagged but NOT auto-removed —
# a human should confirm before dropping.
NEAR_DEDUP_KEYS = ["account_id", "date", "debit", "credit"]
NEAR_DEDUP_MAX_NARRATION_DISTANCE = 5   # Levenshtein distance threshold

# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------
DATE_FORMAT = "%Y-%m-%d"

# Rows with unparseable dates are KEPT but flagged — dropping them would
# silently remove transactions that might be critical fraud evidence.
# The graph layer and ML layer simply skip rows with null dates.
DATE_NULL_ACTION = "flag"      # "flag" | "drop"

# ---------------------------------------------------------------------------
# Amount cleaning
# ---------------------------------------------------------------------------
# OCR and PDF extraction can produce amounts with:
#   - Brackets for negatives: (1234.56) → -1234.56
#   - Indian lakh commas:     1,23,456  → 123456
#   - Currency prefix:        ₹1234     → 1234
#   - Trailing letters:       1234.56CR → 1234.56 (and flip sign)
AMOUNT_BRACKET_NEGATIVE = True     # (1234) → -1234
AMOUNT_STRIP_CURRENCY   = True     # ₹, INR, Rs.
AMOUNT_HANDLE_CR_DR     = True     # "1234CR" means credit, "1234DR" means debit

# Statistical outlier detection (per account, not global)
# Transactions beyond IQR_MULTIPLIER × IQR above Q3 are FLAGGED as high-value
# outliers — not dropped. IQR is computed separately for debit and credit.
OUTLIER_IQR_MULTIPLIER = 3.0
OUTLIER_MIN_TXN_COUNT  = 10   # need at least this many txns to compute IQR

# ---------------------------------------------------------------------------
# Balance continuity validation
# ---------------------------------------------------------------------------
# For accounts where we have sequential rows, check:
#   prior_balance + credit - debit ≈ current_balance
# Tolerance for floating point + bank rounding
BALANCE_TOLERANCE = 1.0   # ₹1 rounding tolerance

# If more than this fraction of rows fail the balance check, flag the
# entire account as "statement integrity suspect" — critical for
# investigators: a tampered statement won't reconcile.
BALANCE_BREACH_ACCOUNT_FLAG_THRESHOLD = 0.05   # 5% of rows

# ---------------------------------------------------------------------------
# Narration standardisation
# ---------------------------------------------------------------------------
# OCR introduces noise characters. Strip these from narration strings.
NARRATION_STRIP_CHARS = r"[|\\<>{}[\]~`]"

# Normalise common OCR character substitutions in narrations
# These are the most frequent Tesseract errors on printed bank statements
OCR_CHAR_FIXES = {
    "0": ["O", "o"],   # zero vs letter O
    "1": ["I", "l"],   # one vs I/l
    "5": ["S"],        # 5 vs S in amounts
}

# ---------------------------------------------------------------------------
# Channel normalisation
# ---------------------------------------------------------------------------
# OCR and narration parsing sometimes produces garbled channel values.
# Map known variants to canonical names.
CHANNEL_NORMALISE = {
    "UPI/IMPS": "UPI",
    "UPI-IMPS": "UPI",
    "IMPS/UPI": "IMPS",
    "NEFT/RTGS": "NEFT",
    "ATM WDL":  "ATM",
    "ATM-WDL":  "ATM",
    "NACH":     "ECS",
    "SI":       "ECS",    # Standing Instruction
    "CHQ":      "CHEQUE",
    "CLG":      "CHEQUE",
    "OTHER":    "OTHER",
}

# ---------------------------------------------------------------------------
# Output columns (ground truth cols never enter the cleaned output)
# ---------------------------------------------------------------------------
CLEANED_OUTPUT_COLS = [
    "account_id", "account_holder", "bank_name",
    "date", "time", "narration", "channel",
    "debit", "credit", "balance",
    "utr_ref", "counterparty_name",
    "source_file", "source_format",
    # cleaning audit trail
    "clean_flags",          # pipe-separated list of issues found on this row
    "is_duplicate",         # bool
    "is_balance_breach",    # bool — balance continuity failed for this row
    "is_high_value_flag",   # bool — statistical outlier within account
    "is_ocr_row",           # bool — came from image/OCR source
]
