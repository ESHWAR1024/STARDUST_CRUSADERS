"""
Phase 6 - Ingestion Pipeline Configuration

Defines:
- UNIFIED_SCHEMA: the canonical column names every parser must output
- BANK_FORMAT_REGISTRY: per-bank column mappings + date formats
- File type detection helpers
"""

# ---------------------------------------------------------------------------
# Unified schema — every parser must produce exactly these columns
# ---------------------------------------------------------------------------
UNIFIED_SCHEMA = [
    "account_id",           # from filename or metadata
    "account_holder",       # from file header if present
    "bank_name",            # detected from file header / registry
    "date",                 # ISO format YYYY-MM-DD
    "time",                 # HH:MM:SS (00:00:00 if not present)
    "narration",            # transaction description
    "channel",              # inferred from narration where possible
    "debit",                # float, 0.0 if not applicable
    "credit",               # float, 0.0 if not applicable
    "balance",              # float
    "utr_ref",              # reference / cheque number
    "counterparty_name",    # from narration where extractable
    "source_file",          # original filename for traceability
    "source_format",        # csv | xlsx | pdf | image
    "ingestion_warnings",   # pipe-separated list of issues found
]

# ---------------------------------------------------------------------------
# Per-bank format registry — maps each bank's native column names to
# UNIFIED_SCHEMA fields. The ingestion pipeline auto-detects which bank
# a statement belongs to by matching column headers.
# ---------------------------------------------------------------------------
BANK_FORMAT_REGISTRY = {
    "SBI": {
        "bank_name": "State Bank of India",
        "date_col": "Txn Date",
        "narration_col": "Description",
        "ref_col": "Ref No./Cheque No.",
        "debit_col": "Debit",
        "credit_col": "Credit",
        "balance_col": "Balance",
        "date_formats": ["%d/%m/%y", "%d/%m/%Y", "%d-%m-%Y"],
    },
    "HDFC": {
        "bank_name": "HDFC Bank",
        "date_col": "Date",
        "narration_col": "Narration",
        "ref_col": "Chq./Ref.No.",
        "debit_col": "Withdrawal Amt.",
        "credit_col": "Deposit Amt.",
        "balance_col": "Closing Balance",
        "date_formats": ["%d/%m/%Y", "%d/%m/%y"],
    },
    "ICICI": {
        "bank_name": "ICICI Bank",
        "date_col": "Transaction Date",
        "narration_col": "Transaction Remarks",
        "ref_col": "Cheque Number",
        "debit_col": "Withdrawal Amount (INR)",
        "credit_col": "Deposit Amount (INR)",
        "balance_col": "Balance (INR)",
        "date_formats": ["%d-%b-%Y", "%d/%m/%Y"],
    },
    "AXIS": {
        "bank_name": "Axis Bank",
        "date_col": "Tran Date",
        "narration_col": "Particulars",
        "ref_col": "Cheque No",
        "debit_col": "Debit",
        "credit_col": "Credit",
        "balance_col": "Balance",
        "date_formats": ["%Y-%m-%d", "%d/%m/%Y"],
    },
    "CANARA": {
        "bank_name": "Canara Bank",
        "date_col": "Date",
        "narration_col": "Particulars",
        "ref_col": "Instrument Id",
        "debit_col": "Withdrawals",
        "credit_col": "Deposits",
        "balance_col": "Balance",
        "date_formats": ["%d-%m-%Y", "%d/%m/%Y"],
    },
    "PNB": {
        "bank_name": "Punjab National Bank",
        "date_col": "Post Date",
        "narration_col": "Remarks",
        "ref_col": "Cheque No/Ref No",
        "debit_col": "Debit",
        "credit_col": "Credit",
        "balance_col": "Balance",
        "date_formats": ["%d.%m.%Y", "%d/%m/%Y"],
    },
}

# Keyword → bank code mapping for header-line bank detection
BANK_NAME_KEYWORDS = {
    "state bank": "SBI",
    "sbi": "SBI",
    "hdfc": "HDFC",
    "icici": "ICICI",
    "axis": "AXIS",
    "canara": "CANARA",
    "punjab national": "PNB",
    "pnb": "PNB",
}

# Channel inference: if these strings appear in narration → assign channel
CHANNEL_KEYWORDS = [
    ("UPI", ["upi", "gpay", "phonepe", "paytm", "bhim"]),
    ("NEFT", ["neft"]),
    ("IMPS", ["imps"]),
    ("RTGS", ["rtgs"]),
    ("ATM", ["atm wdl", "atm withdrawal", "cash withdrawal"]),
    ("ECS", ["ecs", "nach"]),
    ("BILLPAY", ["billdesk", "billpay", "utility", "electricity", "water bill"]),
    ("CHEQUE", ["clg", "clearing", "chq", "cheque"]),
]

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
