"""
Phase 6 - Ingestion Pipeline Configuration

Defines:
- UNIFIED_SCHEMA: the canonical column names every parser must output
- BANK_FORMAT_REGISTRY: per-bank column mappings + date formats
- File type detection helpers
"""

# ---------------------------------------------------------------------------
# Unified schema
# ---------------------------------------------------------------------------
UNIFIED_SCHEMA = [
    "account_id",           
    "account_holder",       
    "bank_name",            
    "date",                 
    "time",                 
    "narration",            
    "channel",              
    "debit",                
    "credit",               
    "balance",              
    "utr_ref",              
    "counterparty_name",    
    "source_file",          
    "source_format",        
    "ingestion_warnings",   
]

# ---------------------------------------------------------------------------
# Per-bank format registry
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
    "AIRTEL": {
        "bank_name": "Airtel Payments Bank",
        "date_col": "Date",
        "narration_col": "Particulars",
        "ref_col": "Transaction ID",
        "debit_col": "Withdrawal",
        "credit_col": "Deposit",
        "balance_col": "Balance",
        "date_formats": ["%d-%m-%Y", "%d/%m/%Y"],
    },
}

# Replace your current BANK_NAME_KEYWORDS with this expanded list
BANK_NAME_KEYWORDS = {
    # Top Public & Private
    "state bank": "SBI", "sbi": "SBI",
    "hdfc": "HDFC",
    "icici": "ICICI",
    "axis": "AXIS",
    "canara": "CANARA",
    "punjab national": "PNB", "pnb": "PNB",
    "bank of baroda": "BOB", "bob": "BOB",
    "bank of india": "BOI", "boi": "BOI",
    "union bank": "UBI",
    "kotak": "KOTAK",
    "indusind": "INDUSIND",
    "yes bank": "YES",
    "idfc": "IDFC",
    
    # Payments & Small Finance
    "airtel payments bank": "AIRTEL",
    "airtel": "AIRTEL",
    "paytm": "PAYTM",
    "jio payments": "JIO",
    "au small": "AU_SFB",
    "equitas": "EQUITAS",
}

# Expanded semantic column roles to catch far more variants and formatting quirks
GENERIC_COLUMN_ROLES = {
    "date_col": ["date", "txn date", "tran date", "transaction date", "value date", "post date"],
    "narration_col": ["narration", "description", "particulars", "remarks", "transaction remarks", "details", "chq / ref"],
    "ref_col": ["reference no", "ref no", "cheque", "chq", "transaction id", "txn id", "instrument", "utr", "ref."],
    "debit_col": ["withdrawal", "debit", "dr amount", "dr.", "dr", "amount", "paid out"],
    "credit_col": ["deposit", "credit", "cr amount", "cr.", "cr", "paid in"],
    "balance_col": ["balance", "closing balance", "bal.", "bal"],
}

GENERIC_MIN_REQUIRED_ROLES = 4

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