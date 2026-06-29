"""
Phase 6 — Ingestion Configuration  (v2)

Three-tier detection:
  Tier 1 : Keyword matching on column headers
  Tier 2 : Content-based column inference
  Tier 3 : Positional fallback

Now includes:
  • All banks found in the CIDECODE dataset (Yes Bank, IDFC, Bandhan,
    RBL/Ratnakar, Bank of Baroda, Federal, Paytm, Kerala Gramin …)
  • IDFC-specific "Cr" suffix balance parsing
  • Additional channel keywords (NACH, BBPS, SWIFT, etc.)
  • Hindi / regional language column headers
"""

import re

# ---------------------------------------------------------------------------
# Unified output schema  (every parser must produce these columns)
# ---------------------------------------------------------------------------
UNIFIED_SCHEMA = [
    "account_id", "account_holder", "bank_name",
    "date", "time", "narration", "channel",
    "debit", "credit", "balance",
    "utr_ref", "counterparty_name", "counterparty_account",
    "counterparty_ifsc",
    "source_file", "source_format", "ingestion_warnings",
]

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".csv", ".xlsx", ".xls", ".pdf",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ".json", ".tsv", ".txt",
}

# ---------------------------------------------------------------------------
# Tier 1 — keyword fragments for column role detection
# Checked as case-insensitive substrings in column header names
# ---------------------------------------------------------------------------
COLUMN_ROLE_KEYWORDS = {
    "date": [
        "txn date", "tran date", "transaction date", "post date",
        "value date", "posting date", "trans date", "entry date",
        "date of transaction", "booking date", "process date",
        "effective date", "settlement date", "txn_date", "trn_date",
        "dat_txn", "processing date",
        "date",
    ],
    "narration": [
        "description", "narration", "particulars", "remarks",
        "transaction details", "transaction particulars",
        "transaction remarks", "details of transaction",
        "memo", "note", "details", "narrative", "text",
        "txn_desc", "trn_desc", "trans_desc",
        "tran particular", "tran_rmks", "tran rmks",
        "txt_txn_desc", "tran_particular",
        "transaction description", "entry description",
        "beneficiary", "transaction type",
    ],
    "debit": [
        "debit amount", "withdrawal amount", "amount debited",
        "debit amt", "withdrawal amt", "dr amount", "dr amt",
        "amount (dr)", "paid out", "money out",
        "withdrawals", "withdrawal", "debit",
        "dr", "amt_txn_lcy_dr",
        # Paytm-style: COD_DRCR column value 'D' handled separately
    ],
    "credit": [
        "credit amount", "deposit amount", "amount credited",
        "credit amt", "deposit amt", "cr amount", "cr amt",
        "amount (cr)", "paid in", "money in",
        "deposits", "deposit", "credit",
        "cr", "amt_txn_lcy_cr",
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
        "instrument no", "chqno", "chq no", "ref_txn_no",
        "tran_id",
    ],
    "amount": [
        "amount(+cr/-dr)", "amount (+cr/-dr)", "net amount",
        "transaction amount", "txn amount", "amount",
        "amt_txn_lcy",
    ],
    "drcr": [
        # Single column that says 'D' or 'C' / 'DR' or 'CR'
        "cod_drcr", "dr/cr", "cr/dr", "drcr", "type",
        "part tran type", "part_tran_type",
        "balance indicator", "balance_indicator",
        "cr_dr",
    ],
}

# ---------------------------------------------------------------------------
# Tier 2 — content-based detection patterns
# ---------------------------------------------------------------------------
CONTENT_PATTERNS = {
    "date": [
        re.compile(r'^\d{4}-\d{2}-\d{2}$'),
        re.compile(r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$'),
        re.compile(r'^\d{1,2}[- ][A-Za-z]{3}[- ]\d{2,4}$'),
    ],
    "amount": [
        re.compile(r'^[\d,]+\.?\d*$'),
        re.compile(r'^[\d,]+\.?\d*\s*(CR|DR|Cr|Dr)$'),
        re.compile(r'^\([\d,]+\.?\d*\)$'),
    ],
}

# ---------------------------------------------------------------------------
# Hindi / regional language column names (transliterated)
# ---------------------------------------------------------------------------
HINDI_COLUMN_MAP = {
    "दिनांक": "date", "तारीख": "date", "दिन": "date",
    "राशि": "amount", "रकम": "amount", "धनराशि": "amount",
    "नामे": "debit", "डेबिट": "debit",
    "जमा": "credit", "क्रेडिट": "credit",
    "शेष": "balance", "बकाया": "balance", "शेष राशि": "balance",
    "विवरण": "narration", "ब्यौरा": "narration",
}

# ---------------------------------------------------------------------------
# Bank name detection keywords
# Extended with all banks found in CIDECODE dataset
# ---------------------------------------------------------------------------
BANK_NAME_KEYWORDS = {
    "state bank of india":      "State Bank of India",
    "sbi":                      "State Bank of India",
    "hdfc":                     "HDFC Bank",
    "icici":                    "ICICI Bank",
    "axis bank":                "Axis Bank",
    "axis":                     "Axis Bank",
    "canara":                   "Canara Bank",
    "punjab national":          "Punjab National Bank",
    "pnb":                      "Punjab National Bank",
    "kotak":                    "Kotak Mahindra Bank",
    "yes bank":                 "Yes Bank",
    "indusind":                 "IndusInd Bank",
    "bank of india":            "Bank of India",
    "bank of baroda":           "Bank of Baroda",
    "union bank":               "Union Bank of India",
    "central bank":             "Central Bank of India",
    "indian bank":              "Indian Bank",
    "idbi":                     "IDBI Bank",
    "federal bank":             "Federal Bank",
    "south indian":             "South Indian Bank",
    "karnataka bank":           "Karnataka Bank",
    "bandhan":                  "Bandhan Bank",
    "rbl bank":                 "RBL Bank",
    "ratnakar":                 "RBL Bank",
    "idfc":                     "IDFC First Bank",
    "au small":                 "AU Small Finance Bank",
    "paytm":                    "Paytm Payments Bank",
    "payments bank":            "Paytm Payments Bank",
    "kerala gramin":            "Kerala Gramin Bank",
    "gramin bank":              "Kerala Gramin Bank",
    "mahindra bank":            "Kotak Mahindra Bank",
    "bank of maharashtra":      "Bank of Maharashtra",
    "uco bank":                 "UCO Bank",
    "indian overseas":          "Indian Overseas Bank",
    "punjab & sind":            "Punjab & Sind Bank",
    "citi bank":                "Citibank",
    "citibank":                 "Citibank",
    "deutsche":                 "Deutsche Bank",
    "standard chartered":       "Standard Chartered Bank",
    "hsbc":                     "HSBC Bank",
    "baroda":                   "Bank of Baroda",
    "pioneer":                  "IDFC First Bank",   # "Pioneer Holdings" in IDFC statement
    "nsdl":                     "NSDL Payments Bank",
    "airtel":                   "Airtel Payments Bank",
    "fino":                     "Fino Payments Bank",
    "jio":                      "Jio Payments Bank",
    "saraswat":                 "Saraswat Bank",
    "nainital":                 "Nainital Bank",
    "lakshmi vilas":            "Lakshmi Vilas Bank",
    "tamilnad mercantile":      "Tamilnad Mercantile Bank",
}

# ---------------------------------------------------------------------------
# Date formats to try (in order)
# ---------------------------------------------------------------------------
DATE_FORMATS = [
    # Indian standard
    "%d/%m/%Y", "%d/%m/%y",
    "%d-%m-%Y", "%d-%m-%y",
    # Month-name
    "%d-%b-%Y", "%d-%b-%y",
    "%d %b %Y", "%d %b %y",
    "%d %B %Y", "%d-%B-%Y",
    # Dot-separated
    "%d.%m.%Y", "%d.%m.%y",
    # ISO
    "%Y-%m-%d",
    # American (less likely but possible)
    "%m/%d/%Y",
    "%Y/%m/%d",
    # Short year edge-cases
    "%d%m%Y",
]

# ---------------------------------------------------------------------------
# Channel inference keywords
# ---------------------------------------------------------------------------
CHANNEL_KEYWORDS = [
    ("UPI",     ["upi", "gpay", "phonepe", "paytm", "bhim", "googlepay",
                 "@okhdfcbank", "@oksbi", "@ybl", "@axl", "@okaxis",
                 "@ibl", "@kotak", "@okicici"]),
    ("NEFT",    ["neft", "nft/", "inft"]),
    ("IMPS",    ["imps"]),
    ("RTGS",    ["rtgs", "blkrtgs"]),
    ("ATM",     ["atm", "cash withdrawal", "atm wdl", "cwdl", "nfs cash"]),
    ("ECS",     ["ecs", "nach", "si-", "standing instruction", "auto debit"]),
    ("CHEQUE",  ["clg", "clearing", "chq", "cheque", "chq dep",
                 "bb/chq", "cheque deposit"]),
    ("BILLPAY", ["billdesk", "billpay", "bbps", "electricity",
                 "utility", "broadband", "recharge"]),
    ("CASH",    ["cash dep", "cash deposit", "counter deposit",
                 "cash credit", "by cash", "branch cash"]),
    ("WIRE",    ["swift", "wire transfer", "foreign", "remittance",
                 "international", "nostro"]),
    ("LOAN",    ["loan disbursement", "emi", "loan emi", "housing loan",
                 "vehicle loan", "personal loan"]),
    ("INTEREST",["interest credit", "int credit", "sbint", "interest",
                 "int on", "tds on int"]),
    ("CHARGES", ["charges", "service charge", "gst", "maintenance",
                 "fee", "penalty", "fine"]),
    ("REVERSAL",["reversal", "reverse", "refund", "cashback", "chargeback"]),
]

# ---------------------------------------------------------------------------
# IDFC-specific: balance column ends in 'Cr' or 'Dr'
# Detected during parsing so amounts are signed correctly
# ---------------------------------------------------------------------------
IDFC_BALANCE_CR_SUFFIX = True   # flag used by normalizer

# ---------------------------------------------------------------------------
# Paytm XLSX: has explicit COD_DRCR column ('C' = credit, 'D' = debit)
# and COD_ACCT_NO_FROM / COD_ACCT_NO_TO for graph edges
# ---------------------------------------------------------------------------
PAYTM_DRCR_COL   = "COD_DRCR"
PAYTM_FROM_COL   = "COD_ACCT_NO_FROM"
PAYTM_TO_COL     = "COD_ACCT_NO_TO"
PAYTM_AMOUNT_COL = "AMT_TXN_LCY"

# ---------------------------------------------------------------------------
# BOB / Federal XLSX: has explicit beneficiary account + IFSC columns
# ---------------------------------------------------------------------------
BOB_BENEF_ACCT_COL = "BENEF/REMIT ACCT NO"
BOB_BENEF_IFSC_COL = "BENEF/REMIT IFSC CODE"
BOB_BENEF_NAME_COL = "BENEF/REMIT ACCT NAME"