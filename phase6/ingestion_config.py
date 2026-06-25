"""
Phase 6 - Ingestion Configuration

Designed for arbitrary financial datasets — not just standard bank statements.
The judges may provide any CSV/Excel/PDF with transaction data in any format.

Three-tier detection strategy:
  Tier 1: Keyword matching on column headers (handles known formats)
  Tier 2: Content-based column inference (handles unknown header names)
  Tier 3: Positional fallback (Col1/Col2 style — last resort)
"""

UNIFIED_SCHEMA = [
    "account_id", "account_holder", "bank_name",
    "date", "time", "narration", "channel",
    "debit", "credit", "balance",
    "utr_ref", "counterparty_name",
    "source_file", "source_format", "ingestion_warnings",
]

# ---------------------------------------------------------------------------
# Tier 1: Keyword fragments for column role detection
# Checked as case-insensitive substrings in column header names
# ---------------------------------------------------------------------------
COLUMN_ROLE_KEYWORDS = {
    "date": [
        "txn date", "tran date", "transaction date", "post date",
        "value date", "posting date", "trans date", "entry date",
        "date of transaction", "booking date", "process date",
        "effective date", "settlement date", "txn_date", "trn_date",
        "date",   # short match last so specific ones score higher
    ],
    "narration": [
        "description", "narration", "particulars", "remarks",
        "transaction details", "transaction particulars",
        "transaction remarks", "details of transaction",
        "beneficiary", "transaction type", "entry description",
        "memo", "note", "details", "narrative", "text",
        "txn_desc", "trn_desc", "trans_desc",
    ],
    "debit": [
        "debit amount", "withdrawal amount", "amount debited",
        "debit amt", "withdrawal amt", "dr amount", "dr amt",
        "amount (dr)", "paid out", "money out", "debit",
        "withdrawal", "withdrawals", "dr",
    ],
    "credit": [
        "credit amount", "deposit amount", "amount credited",
        "credit amt", "deposit amt", "cr amount", "cr amt",
        "amount (cr)", "paid in", "money in", "credit",
        "deposit", "deposits", "cr",
    ],
    "balance": [
        "closing balance", "running balance", "available balance",
        "balance amount", "closing bal", "bal amount",
        "current balance", "book balance", "ledger balance",
        "balance",
    ],
    "ref": [
        "reference number", "cheque number", "transaction id",
        "transaction ref", "ref no", "cheque no", "utr number",
        "chq/ref", "instrument id", "tran id", "txn id",
        "vch no", "voucher no", "journal no", "entry no",
        "txn_ref", "ref_num", "reference",
    ],
    "amount": [
        # Single-amount-column formats (+/- or CR/DR notation)
        "amount(+cr/-dr)", "amount (+cr/-dr)", "net amount",
        "transaction amount", "txn amount", "amount",
    ],
}

# ---------------------------------------------------------------------------
# Tier 2: Content-based detection patterns
# Used when column headers are unknown (Col1, A, B, etc.)
# Each pattern is applied to the first N data rows of each column
# ---------------------------------------------------------------------------
import re

CONTENT_PATTERNS = {
    "date": [
        # ISO date: 2026-01-15
        re.compile(r'^\d{4}-\d{2}-\d{2}$'),
        # Indian date: 15/01/2026 or 15-01-2026
        re.compile(r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$'),
        # Month-name date: 15-Jan-2026
        re.compile(r'^\d{1,2}[- ][A-Za-z]{3}[- ]\d{2,4}$'),
    ],
    "amount": [
        # Pure numeric with optional commas and decimal
        re.compile(r'^[\d,]+\.?\d*$'),
        # With CR/DR suffix
        re.compile(r'^[\d,]+\.?\d*\s*(CR|DR|Cr|Dr)$'),
        # Bracket negative
        re.compile(r'^\([\d,]+\.?\d*\)$'),
    ],
}

# ---------------------------------------------------------------------------
# Hindi / regional language column names (transliterated)
# ---------------------------------------------------------------------------
HINDI_COLUMN_MAP = {
    # Date equivalents
    "दिनांक": "date", "तारीख": "date", "दिन": "date",
    # Amount equivalents
    "राशि": "amount", "रकम": "amount", "धनराशि": "amount",
    # Debit equivalents
    "नामे": "debit", "डेबिट": "debit",
    # Credit equivalents
    "जमा": "credit", "क्रेडिट": "credit",
    # Balance equivalents
    "शेष": "balance", "बकाया": "balance", "शेष राशि": "balance",
    # Narration equivalents
    "विवरण": "narration", "ब्यौरा": "narration",
}

# ---------------------------------------------------------------------------
# Bank name detection
# ---------------------------------------------------------------------------
BANK_NAME_KEYWORDS = {
    "state bank of india": "State Bank of India",
    "sbi":                 "State Bank of India",
    "hdfc":                "HDFC Bank",
    "icici":               "ICICI Bank",
    "axis":                "Axis Bank",
    "canara":              "Canara Bank",
    "punjab national":     "Punjab National Bank",
    "pnb":                 "Punjab National Bank",
    "kotak":               "Kotak Mahindra Bank",
    "yes bank":            "Yes Bank",
    "indusind":            "IndusInd Bank",
    "bank of india":       "Bank of India",
    "bank of baroda":      "Bank of Baroda",
    "union bank":          "Union Bank of India",
    "central bank":        "Central Bank of India",
    "indian bank":         "Indian Bank",
    "idbi":                "IDBI Bank",
    "federal bank":        "Federal Bank",
    "south indian":        "South Indian Bank",
    "karnataka bank":      "Karnataka Bank",
    "bandhan":             "Bandhan Bank",
    "rbl":                 "RBL Bank",
    "idfc":                "IDFC First Bank",
    "au small":            "AU Small Finance Bank",
}

# ---------------------------------------------------------------------------
# Date formats to try in order
# ---------------------------------------------------------------------------
DATE_FORMATS = [
    "%d/%m/%Y", "%d/%m/%y",
    "%d-%m-%Y", "%d-%m-%y",
    "%d-%b-%Y", "%d-%b-%y",
    "%d.%m.%Y", "%d.%m.%y",
    "%Y-%m-%d",
    "%d %b %Y", "%d %b %y",
    "%d %B %Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%B-%Y",
]

# ---------------------------------------------------------------------------
# Channel inference keywords
# ---------------------------------------------------------------------------
CHANNEL_KEYWORDS = [
    ("UPI",     ["upi", "gpay", "phonepe", "paytm", "bhim", "googlepay"]),
    ("NEFT",    ["neft"]),
    ("IMPS",    ["imps"]),
    ("RTGS",    ["rtgs"]),
    ("ATM",     ["atm", "cash withdrawal", "atm wdl", "cwdl"]),
    ("ECS",     ["ecs", "nach", "si-", "standing instruction"]),
    ("CHEQUE",  ["clg", "clearing", "chq", "cheque", "chq dep"]),
    ("BILLPAY", ["billdesk", "billpay", "bbps", "electricity", "utility"]),
    ("CASH",    ["cash dep", "cash deposit", "counter deposit"]),
    ("WIRE",    ["swift", "wire", "foreign", "remittance", "international"]),
]

SUPPORTED_EXTENSIONS = {
    ".csv", ".xlsx", ".xls", ".pdf",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ".json", ".tsv", ".txt",
}