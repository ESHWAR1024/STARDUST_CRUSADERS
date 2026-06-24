"""
Phase 7 — Cleaning Engine Configuration

All thresholds are STATISTICAL or STRUCTURAL — never hardcoded rupee
amounts or fixed date ranges, so the system generalises to the judges'
unseen dataset regardless of account sizes or date windows.

Rule of thumb:
  Structural rules  → always wrong (null date, zero debit+credit)
  Statistical rules → suspicious relative to each account's own
                      distribution — these FLAG, never auto-drop
"""

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
EXACT_DEDUP_KEYS = ["account_id", "date", "narration", "debit", "credit"]
NEAR_DEDUP_KEYS  = ["account_id", "date", "debit", "credit"]

# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------
DATE_FORMAT              = "%Y-%m-%d"
DATE_VALID_YEAR_MIN      = 2000
DATE_VALID_YEAR_MAX      = 2030

# ---------------------------------------------------------------------------
# Amount cleaning
# ---------------------------------------------------------------------------
AMOUNT_BRACKET_NEGATIVE  = True   # (1234.56) → -1234.56
AMOUNT_STRIP_CURRENCY    = True   # ₹, INR, Rs.
AMOUNT_HANDLE_CR_DR      = True   # "1234.56CR" / "1234.56DR"

# ---------------------------------------------------------------------------
# Balance continuity
# ---------------------------------------------------------------------------
BALANCE_TOLERANCE                      = 1.0    # ₹1 rounding allowance
BALANCE_BREACH_ACCOUNT_FLAG_THRESHOLD  = 0.05   # flag account if >5% rows breach

# ---------------------------------------------------------------------------
# Statistical outlier flagging (per-account IQR)
# ---------------------------------------------------------------------------
OUTLIER_IQR_MULTIPLIER   = 3.0
OUTLIER_MIN_TXN_COUNT    = 10     # minimum rows before IQR is computed

# ---------------------------------------------------------------------------
# Velocity check (rapid successive large transfers — fraud signal)
# ---------------------------------------------------------------------------
VELOCITY_WINDOW_MINUTES  = 30     # check for N+ debits within this window
VELOCITY_MIN_TXNS        = 3      # minimum txns in window to flag
VELOCITY_MIN_AMOUNT      = 0      # minimum amount per txn (0 = any amount)

# ---------------------------------------------------------------------------
# Narration normalisation
# ---------------------------------------------------------------------------
NARRATION_STRIP_CHARS    = r"[|\\<>{}[\]~`°]"

# ---------------------------------------------------------------------------
# Channel normalisation map
# ---------------------------------------------------------------------------
CHANNEL_NORMALISE = {
    "UPI/IMPS":  "UPI",
    "UPI-IMPS":  "UPI",
    "IMPS/UPI":  "IMPS",
    "NEFT/RTGS": "NEFT",
    "ATM WDL":   "ATM",
    "ATM-WDL":   "ATM",
    "NACH":      "ECS",
    "SI":        "ECS",
    "CHQ":       "CHEQUE",
    "CLG":       "CHEQUE",
}

# ---------------------------------------------------------------------------
# Final output columns (no ground-truth labels — cleaning audit only)
# ---------------------------------------------------------------------------
CLEANED_OUTPUT_COLS = [
    "account_id", "account_holder", "bank_name",
    "date", "time", "narration", "channel",
    "debit", "credit", "balance",
    "utr_ref", "counterparty_name",
    "source_file", "source_format",
    "clean_flags",
    "is_duplicate",
    "is_balance_breach",
    "is_high_value_flag",
    "is_ocr_row",
    "is_velocity_flag",
]