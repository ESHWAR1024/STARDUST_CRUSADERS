"""
Phase 5 - Synthetic Data Generation Engine
Global configuration: simulation window, population mix, behavioral
distributions, and fraud-injection parameters.

All randomness in the engine is seeded from RANDOM_SEED so a run is
reproducible end-to-end (important for hackathon demos - you want the
same "interesting" fraud ring to show up every time you run the demo).
"""

import numpy as np
import random
from datetime import date

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Simulation window
# ---------------------------------------------------------------------------
SIM_START_DATE = date(2026, 1, 1)
SIM_END_DATE = date(2026, 3, 31)   # 90-day window: enough hops for layering chains

# ---------------------------------------------------------------------------
# Population size & persona mix
# ---------------------------------------------------------------------------
N_ACCOUNTS = 500

# Persona distribution for *legitimate* accounts (must sum to 1.0)
PERSONA_MIX = {
    "salaried": 0.38,
    "student": 0.18,
    "business": 0.10,
    "shopkeeper": 0.12,
    "family": 0.22,
}

# Fraction of total accounts that belong to the hidden fraud ecosystem.
# Kept deliberately small (real-world STR hit rates are low single digits)
# so fraud has to be *found*, not stumbled into.
FRAUD_ACCOUNT_RATIO = 0.07          # ~7% of accounts touch fraud in some role
N_FRAUD_RINGS = 4                    # number of independent fraud networks
FRAUD_TYPOLOGY_WEIGHTS = {
    "layering_chain": 0.30,
    "fan_out_mule_network": 0.30,
    "fan_in_collector": 0.20,
    "smurfing_structuring": 0.20,
}

# ---------------------------------------------------------------------------
# Simulated banks (used to vary statement format per account -> stresses
# Phase 6 ingestion / schema-mapping logic)
# ---------------------------------------------------------------------------
BANKS = [
    {"name": "State Bank of India", "ifsc_prefix": "SBIN", "code": "SBI"},
    {"name": "HDFC Bank", "ifsc_prefix": "HDFC", "code": "HDFC"},
    {"name": "ICICI Bank", "ifsc_prefix": "ICIC", "code": "ICICI"},
    {"name": "Axis Bank", "ifsc_prefix": "UTIB", "code": "AXIS"},
    {"name": "Canara Bank", "ifsc_prefix": "CNRB", "code": "CANARA"},
    {"name": "Punjab National Bank", "ifsc_prefix": "PUNB", "code": "PNB"},
]

# ---------------------------------------------------------------------------
# Opening balance ranges by persona (₹) - lognormal-ish via uniform-on-log
# ---------------------------------------------------------------------------
OPENING_BALANCE_RANGE = {
    "salaried": (15_000, 250_000),
    "student": (500, 15_000),
    "business": (50_000, 800_000),
    "shopkeeper": (10_000, 150_000),
    "family": (20_000, 300_000),
    "mule": (200, 3_000),            # mule accounts sit near-empty until used
    "fresh_layering_node": (0, 500), # thin shell accounts opened to launder
}

# ---------------------------------------------------------------------------
# Salary / income parameters
# ---------------------------------------------------------------------------
SALARY_RANGE = {
    "salaried": (22_000, 180_000),
    "business": (None, None),        # businesses earn via customer credits, not salary
}
SALARY_CREDIT_DAY_RANGE = (1, 5)     # salary lands 1st-5th, right-skewed to 1st
SALARY_CREDIT_HOUR_RANGE = (0, 6)    # NEFT batch processing, early hours

STUDENT_ALLOWANCE_RANGE = (3_000, 15_000)

# Family accounts model a household's primary income (one or more earners
# pooling into a shared account) - without this, family accounts have only
# outflows (rent/utility/discretionary) and no inflow, which was driving
# them deeply negative.
FAMILY_INCOME_RANGE = (28_000, 220_000)

# ---------------------------------------------------------------------------
# Recurring obligation parameters
# ---------------------------------------------------------------------------
RENT_DAY_RANGE = (1, 3)
RENT_RATIO_OF_INCOME = (0.18, 0.30)   # kept below 1/3 of income so rent alone can't drain account
RENT_PROBABILITY = 0.55                # not everyone rents (some own / live with family)

UTILITY_BILL_DAY_RANGE = (8, 20)
UTILITY_BILL_RANGE = (800, 6_000)

EMI_PROBABILITY = 0.30
EMI_DAY_RANGE = (5, 10)
# EMI is sized as a fraction of monthly income (when known) rather than a
# flat rupee range, so a lower-income account can't get hit with an EMI
# bigger than its own salary. Flat fallback range only used when no income
# figure exists for the account (e.g. business/shopkeeper).
EMI_INCOME_RATIO_RANGE = (0.08, 0.22)
EMI_FLAT_FALLBACK_RANGE = (3_000, 18_000)

# ---------------------------------------------------------------------------
# Day-to-day spending (UPI / POS / ATM) - Poisson arrival process per persona
# ---------------------------------------------------------------------------
# NOTE: for "business" and "shopkeeper" this models ONLY their own personal/
# household discretionary spend - their business inflow is modeled
# separately via BUSINESS_COLLECTION_RATE / BUSINESS_COLLECTION_AMOUNT_RANGE
# below. Reusing one rate for both business income AND personal spend was
# the other big driver of runaway negative balances for those personas.
DAILY_TXN_RATE = {              # expected number of discretionary txns / day
    "salaried": 0.9,
    "student": 1.3,
    "business": 1.1,
    "shopkeeper": 1.3,
    "family": 1.2,              # household is many people but single account; moderated
    "mule": 0.05,
    "fresh_layering_node": 0.02,
}

DISCRETIONARY_AMOUNT_RANGE = {
    "salaried": (100, 4_000),
    "student": (50, 1_500),
    "business": (100, 3_000),
    "shopkeeper": (100, 2_500),
    "family": (100, 2_500),     # household daily spend, not per-person
    "mule": (50, 500),
    "fresh_layering_node": (0, 0),
}

# Business/shopkeeper customer-collection inflow (their actual "income").
BUSINESS_COLLECTION_RATE = {
    "business": 5.0,
    "shopkeeper": 6.5,
}
BUSINESS_COLLECTION_AMOUNT_RANGE = {
    "business": (200, 9_000),
    "shopkeeper": (100, 6_000),
}

# Time-of-day weighting for discretionary spend (24 buckets, hour 0-23).
# Peaks around lunch and evening - mirrors real UPI usage curves.
HOURLY_WEIGHTS = np.array([
    0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0,   # 0-7
    1.5, 1.8, 1.6, 1.4, 2.2, 1.8, 1.2, 1.1,   # 8-15
    1.4, 1.9, 2.4, 2.6, 2.2, 1.6, 1.0, 0.5    # 16-23
])
HOURLY_WEIGHTS = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()

ATM_WITHDRAWAL_PROBABILITY_PER_WEEK = 0.6
ATM_WITHDRAWAL_RANGE = (1_000, 15_000)
# Scales the base ATM range per persona so a student isn't statistically
# pulling out the same cash as a business account.
ATM_SCALE_BY_PERSONA = {
    "salaried": 1.0,
    "student": 0.25,
    "business": 1.3,
    "shopkeeper": 1.0,
    "family": 0.85,
}

# Safety floor enforced during simulation: discretionary spend / ATM
# withdrawals are scaled down (never skipped outright, to keep the income
# vs. expense story coherent) so an account can't be driven arbitrarily
# deep into the red purely by the random spending process. A small
# negative floor is still allowed - real accounts do dip into a brief
# overdraft - but not by lakhs.
BALANCE_FLOOR = {
    "salaried": -3_000,
    "student": -1_000,
    "business": -15_000,
    "shopkeeper": -8_000,
    "family": -3_000,
}

# ---------------------------------------------------------------------------
# Fraud behavioral parameters
# ---------------------------------------------------------------------------
# Layering: money decays slightly at each hop (the "commission" mules/launderers
# skim) and moves fast - hours, not days.
LAYERING_CHAIN_LENGTH_RANGE = (3, 6)
LAYERING_HOP_DELAY_HOURS_RANGE = (0.5, 18)
LAYERING_SKIM_RATIO_RANGE = (0.02, 0.08)     # 2-8% taken at each hop
LAYERING_SEED_AMOUNT_RANGE = (80_000, 600_000)

# Fan-out: one inbound lump sum immediately split to many mule legs
FAN_OUT_LEG_COUNT_RANGE = (4, 12)
FAN_OUT_DELAY_MINUTES_RANGE = (5, 240)
FAN_OUT_SEED_AMOUNT_RANGE = (100_000, 900_000)

# Fan-in: many small/medium inbound credits converge on one collector account
# in a tight time window before the collector cashes out / moves it onward
FAN_IN_SOURCE_COUNT_RANGE = (5, 15)
FAN_IN_WINDOW_HOURS_RANGE = (4, 72)
FAN_IN_LEG_AMOUNT_RANGE = (5_000, 45_000)

# Smurfing / structuring: break a large sum into multiple transfers each kept
# under common reporting/scrutiny thresholds, spread across several days and
# sometimes several source accounts, into one beneficiary.
CTR_LIKE_THRESHOLD = 50_000          # stay under this to "structure"
SMURF_LEG_COUNT_RANGE = (6, 20)
SMURF_LEG_AMOUNT_RANGE = (8_000, 48_000)
SMURF_SPREAD_DAYS_RANGE = (3, 14)

# Exit channels for laundered money at the end of a chain/ring
EXIT_CHANNELS = ["ATM_CASH_WITHDRAWAL", "INTERNATIONAL_REMITTANCE", "CRYPTO_EXCHANGE_TRANSFER"]
EXIT_CHANNEL_WEIGHTS = [0.55, 0.25, 0.20]

# Odd-hour bias for fraud-related movement (a real soft signal investigators
# use, but kept probabilistic, not absolute, so it doesn't trivially separate
# fraud from noise)
FRAUD_ODD_HOUR_BIAS = 0.35   # probability a fraud hop is forced into 00:00-05:00

# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------
def seed_all(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
