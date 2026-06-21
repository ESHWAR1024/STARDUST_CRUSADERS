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
    # Top Public & Private Banks
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
    "central bank": "CBI", "cbi": "CBI",
    "indian bank": "INDIAN",
    "indian overseas": "IOB", "iob": "IOB",
    "uco bank": "UCO",
    "bank of maharashtra": "BOM",
    "punjab & sind": "PSB",
    "federal bank": "FEDERAL",
    "south indian bank": "SIB",
    "rbl bank": "RBL",
    "bandhan": "BANDHAN",
    "karur vysya": "KVB",
    "city union": "CUB",
    "jammu & kashmir": "JKB",
    "karnataka bank": "KBL",
    "dhanlaxmi": "DHANLAXMI",
    "csb bank": "CSB",
    "nainital bank": "NAINITAL",

    # Foreign Banks (Major presence)
    "standard chartered": "SCB", "stanchart": "SCB",
    "citi": "CITI",
    "hsbc": "HSBC",
    "dbs": "DBS",
    "barclays": "BARCLAYS",
    "deutsche": "DEUTSCHE",

    # Payments & Small Finance Banks (SFBs)
    "airtel payments": "AIRTEL", "airtel": "AIRTEL",
    "paytm": "PAYTM",
    "jio payments": "JIO",
    "india post": "IPPB", "ippb": "IPPB",
    "fino payments": "FINO",
    "au small": "AU_SFB",
    "equitas": "EQUITAS",
    "ujjivan": "UJJIVAN",
    "esaf": "ESAF",
    "utkarsh": "UTKARSH",
    "suryoday": "SURYODAY",
    "jana small": "JANA",
    "capital small": "CAPITAL_SFB",
    "shivalik": "SHIVALIK",
    "unity small": "UNITY",

    # Major NBFCs (Lending, Muthoot, Bajaj, etc.)
    "bajaj finance": "BAJAJ_FIN", "bajaj finserv": "BAJAJ_FIN",
    "tata capital": "TATA_CAP",
    "muthoot": "MUTHOOT",
    "manappuram": "MANAPPURAM",
    "shriram finance": "SHRIRAM", "shriram transport": "SHRIRAM",
    "mahindra finance": "MAHINDRA_FIN", "m&m financial": "MAHINDRA_FIN",
    "cholamandalam": "CHOLA",
    "l&t finance": "LT_FIN",
    "aditya birla capital": "AB_CAPITAL",
    "hdb financial": "HDB_FIN",
    "pnb housing": "PNB_HOUSING",
    "lic housing": "LIC_HOUSING",
    "hdfc credila": "CREDILA",
    "poonawalla": "POONAWALLA", "magma": "POONAWALLA",
    "iifl": "IIFL",
    "edelweiss": "EDELWEISS",
    "capri global": "CAPRI",
    "piramal": "PIRAMAL",
    "five star business": "FIVE_STAR",
    "arohan": "AROHAN",
    "spandana": "SPANDANA",
    "satin creditcare": "SATIN",
    "creditaccess": "CREDITACCESS", "grameen koota": "CREDITACCESS",
}
# Expanded semantic column roles to catch far more variants and formatting quirks
GENERIC_COLUMN_ROLES = {
    "date_col": ["date", "txn date", "tran date", "transaction date", "value date", "post date", "posting date", "dt"],
    "narration_col": ["narration", "description", "particulars", "remarks", "transaction remarks", "details", "chq / ref", "narrative"],
    "ref_col": ["reference no", "ref no", "cheque", "chq", "transaction id", "txn id", "instrument", "utr", "ref.", "ref"],
    "debit_col": ["withdrawal", "debit", "dr amount", "dr.", "dr", "amount", "paid out", "out", "withdraw"],
    "credit_col": ["deposit", "credit", "cr amount", "cr.", "cr", "paid in", "in", "receipt"],
    "balance_col": ["balance", "closing balance", "bal.", "bal", "running balance"],
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