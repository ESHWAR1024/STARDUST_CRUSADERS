"""
Phase 5 - Behavioral simulation engine for legitimate accounts.

For every normal account (salaried / student / business / shopkeeper /
family) this module generates a stream of "raw" transaction events across
the simulation window: salary/income credits, rent, utility bills, EMIs,
discretionary UPI/POS spend, and occasional ATM withdrawals.

Output of this module is a flat list of raw event dicts (unordered,
un-balanced). config.SIM_START_DATE / SIM_END_DATE bound the window.
ledger.py is responsible for sorting per account and computing running
balances - this module only decides WHAT happens and WHEN.

A meaningful share of discretionary spend resolves to a transfer between
two accounts that both exist in our population (so the transaction graph
has real internal structure for the graph-analytics phases later), the
rest resolves to an external counterparty that only exists as narration
text (mirrors how most lines on a real bank statement reference parties
outside any seized account list).
"""

import random
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

import config

fake = Faker("en_IN")

MERCHANT_CATEGORIES = [
    ("Swiggy", "FOOD DELIVERY"), ("Zomato", "FOOD DELIVERY"),
    ("BigBasket", "GROCERY"), ("DMart", "GROCERY"),
    ("Amazon", "SHOPPING"), ("Flipkart", "SHOPPING"),
    ("BookMyShow", "ENTERTAINMENT"), ("Netflix", "SUBSCRIPTION"),
    ("Uber", "TRAVEL"), ("Ola", "TRAVEL"), ("IRCTC", "TRAVEL"),
    ("Jio Recharge", "UTILITY"), ("Airtel Recharge", "UTILITY"),
    ("Apollo Pharmacy", "MEDICAL"), ("PVR Cinemas", "ENTERTAINMENT"),
    ("Local Kirana Store", "GROCERY"), ("Petrol Pump", "FUEL"),
]

UTILITY_PROVIDERS = ["BESCOM Electricity", "BWSSB Water", "Indane Gas",
                      "Airtel Broadband", "ACT Fibernet", "BBMP Property Tax"]


def _random_hour():
    return int(np.random.choice(24, p=config.HOURLY_WEIGHTS))


def _ts(d, hour=None, minute=None):
    if hour is None:
        hour = _random_hour()
    if minute is None:
        minute = random.randint(0, 59)
    return datetime(d.year, d.month, d.day, hour, minute, random.randint(0, 59))


def _daterange(start, end):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _month_starts(start, end):
    months = []
    cur = start.replace(day=1)
    while cur <= end:
        months.append(cur)
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months


def _event(account_id, ts, narration, channel, debit=0.0, credit=0.0,
           counterparty_account_id=None, counterparty_name=None, utr_ref=None,
           category=None):
    return {
        "account_id": account_id,
        "timestamp": ts,
        "narration": narration,
        "channel": channel,
        "debit": round(debit, 2),
        "credit": round(credit, 2),
        "counterparty_account_id": counterparty_account_id,
        "counterparty_name": counterparty_name,
        "utr_ref": utr_ref,
        "category": category,
        "is_fraud": False,
        "fraud_pattern": None,
        "fraud_ring_id": None,
    }


def _linked_transfer(sender, receiver, ts, amount, purpose, channel="UPI"):
    """Two linked legs of an internal P2P transfer (debit + credit)."""
    utr = f"{channel}{random.randint(10**11, 10**12 - 1)}"
    debit_leg = _event(
        sender.account_id, ts,
        f"{channel}-{receiver.holder_name.upper()}-{purpose}",
        channel, debit=amount,
        counterparty_account_id=receiver.account_id,
        counterparty_name=receiver.holder_name, utr_ref=utr,
    )
    credit_leg = _event(
        receiver.account_id, ts,
        f"{channel}-{sender.holder_name.upper()}-{purpose}",
        channel, credit=amount,
        counterparty_account_id=sender.account_id,
        counterparty_name=sender.holder_name, utr_ref=utr,
    )
    return [debit_leg, credit_leg]


# ---------------------------------------------------------------------------
# Income
# ---------------------------------------------------------------------------
def gen_salary_events(acct):
    events = []
    if acct.persona not in ("salaried", "family") or not acct.monthly_income:
        return events
    employer = acct.meta.get("employer_name", "EMPLOYER PVT LTD")
    label = "SALARY" if acct.persona == "salaried" else "HOUSEHOLD INCOME"
    for month_start in _month_starts(config.SIM_START_DATE, config.SIM_END_DATE):
        day = int(np.clip(np.random.exponential(1.5) + config.SALARY_CREDIT_DAY_RANGE[0],
                           *config.SALARY_CREDIT_DAY_RANGE))
        try:
            d = month_start.replace(day=day)
        except ValueError:
            continue
        if d < config.SIM_START_DATE or d > config.SIM_END_DATE:
            continue
        hour = random.randint(*config.SALARY_CREDIT_HOUR_RANGE)
        amount = acct.monthly_income * random.uniform(0.97, 1.03)
        events.append(_event(
            acct.account_id, _ts(d, hour),
            f"NEFT CR-{employer.upper()}-{label} {d.strftime('%b%Y').upper()}",
            "NEFT", credit=amount, counterparty_name=employer,
        ))
    return events


def gen_student_allowance_events(acct):
    events = []
    if acct.persona != "student" or not acct.monthly_income:
        return events
    guardian = acct.meta.get("guardian_name", "PARENT")
    for month_start in _month_starts(config.SIM_START_DATE, config.SIM_END_DATE):
        day = random.randint(1, 7)
        try:
            d = month_start.replace(day=day)
        except ValueError:
            continue
        if d < config.SIM_START_DATE or d > config.SIM_END_DATE:
            continue
        amount = acct.monthly_income * random.uniform(0.9, 1.1)
        events.append(_event(
            acct.account_id, _ts(d),
            f"IMPS-{guardian.upper()}-MONTHLY ALLOWANCE",
            "IMPS", credit=amount, counterparty_name=guardian,
        ))
    return events


def gen_business_collection_events(acct, population_by_persona):
    """Frequent inbound customer payments for business/shopkeeper accounts.
    This is the persona's actual income stream - kept on its own
    rate/amount config so it isn't accidentally reused for personal
    discretionary spend (which used to double-count and drain balances)."""
    events = []
    if acct.persona not in ("business", "shopkeeper"):
        return events
    rate = config.BUSINESS_COLLECTION_RATE[acct.persona]
    lo, hi = config.BUSINESS_COLLECTION_AMOUNT_RANGE[acct.persona]
    candidates = population_by_persona.get("salaried", []) + population_by_persona.get("family", [])

    for d in _daterange(config.SIM_START_DATE, config.SIM_END_DATE):
        n_today = np.random.poisson(rate)
        for _ in range(n_today):
            amount = float(np.exp(np.random.uniform(np.log(max(lo, 1)), np.log(hi))))
            if candidates and random.random() < 0.25:
                customer = random.choice(candidates)
                events += _linked_transfer(customer, acct, _ts(d), amount,
                                            "PAYMENT RECEIVED")
            else:
                customer_name = fake.name()
                events.append(_event(
                    acct.account_id, _ts(d),
                    f"UPI CR-{customer_name.upper()}-PAYMENT RECEIVED",
                    "UPI", credit=amount, counterparty_name=customer_name,
                ))
    return events


def gen_supplier_payment_events(acct):
    """Periodic outbound supplier/wholesale payments for business/shopkeeper."""
    events = []
    if acct.persona not in ("business", "shopkeeper"):
        return events
    for month_start in _month_starts(config.SIM_START_DATE, config.SIM_END_DATE):
        for _ in range(random.randint(1, 3)):
            day = random.randint(1, 28)
            try:
                d = month_start.replace(day=day)
            except ValueError:
                continue
            if d < config.SIM_START_DATE or d > config.SIM_END_DATE:
                continue
            supplier = fake.company()
            amount = random.uniform(5_000, 60_000)
            events.append(_event(
                acct.account_id, _ts(d, hour=random.randint(9, 18)),
                f"NEFT DR-{supplier.upper()}-SUPPLIER PAYMENT",
                "NEFT", debit=amount, counterparty_name=supplier,
            ))
    return events


# ---------------------------------------------------------------------------
# Recurring obligations
# ---------------------------------------------------------------------------
def gen_rent_events(acct):
    events = []
    if acct.persona not in ("salaried", "family"):
        return events
    landlord = acct.meta.get("landlord_name")
    if not landlord:
        return events
    base_income = acct.monthly_income or random.uniform(25_000, 80_000)
    ratio = random.uniform(*config.RENT_RATIO_OF_INCOME)
    rent_amount = base_income * ratio
    for month_start in _month_starts(config.SIM_START_DATE, config.SIM_END_DATE):
        day = random.randint(*config.RENT_DAY_RANGE)
        try:
            d = month_start.replace(day=day)
        except ValueError:
            continue
        if d < config.SIM_START_DATE or d > config.SIM_END_DATE:
            continue
        events.append(_event(
            acct.account_id, _ts(d, hour=random.randint(8, 12)),
            f"UPI-{landlord.upper()}-HOUSE RENT {d.strftime('%b').upper()}",
            "UPI", debit=round(rent_amount, 2), counterparty_name=landlord,
        ))
    return events


def gen_utility_bill_events(acct):
    events = []
    if acct.persona not in ("salaried", "family", "business", "shopkeeper"):
        return events
    n_bills = random.randint(1, 3)
    providers = random.sample(UTILITY_PROVIDERS, k=min(n_bills, len(UTILITY_PROVIDERS)))
    for month_start in _month_starts(config.SIM_START_DATE, config.SIM_END_DATE):
        for provider in providers:
            day = random.randint(*config.UTILITY_BILL_DAY_RANGE)
            try:
                d = month_start.replace(day=day)
            except ValueError:
                continue
            if d < config.SIM_START_DATE or d > config.SIM_END_DATE:
                continue
            amount = random.uniform(*config.UTILITY_BILL_RANGE)
            events.append(_event(
                acct.account_id, _ts(d, hour=random.randint(7, 22)),
                f"BILLDESK-{provider.upper()}-UTILITY BILL",
                "BILLPAY", debit=amount, counterparty_name=provider,
            ))
    return events


def gen_emi_events(acct):
    events = []
    if acct.persona not in ("salaried", "business", "shopkeeper"):
        return events
    if random.random() >= config.EMI_PROBABILITY:
        return events
    if acct.monthly_income:
        emi_amount = acct.monthly_income * random.uniform(*config.EMI_INCOME_RATIO_RANGE)
    else:
        emi_amount = random.uniform(*config.EMI_FLAT_FALLBACK_RANGE)
    lender = random.choice(["Bajaj Finserv", "HDFC Loan Cell", "SBI Card",
                             "ICICI Personal Loan", "Tata Capital"])
    for month_start in _month_starts(config.SIM_START_DATE, config.SIM_END_DATE):
        day = random.randint(*config.EMI_DAY_RANGE)
        try:
            d = month_start.replace(day=day)
        except ValueError:
            continue
        if d < config.SIM_START_DATE or d > config.SIM_END_DATE:
            continue
        events.append(_event(
            acct.account_id, _ts(d, hour=random.randint(0, 6)),
            f"ECS DEBIT-{lender.upper()}-LOAN EMI",
            "ECS", debit=round(emi_amount, 2), counterparty_name=lender,
        ))
    return events


# ---------------------------------------------------------------------------
# Discretionary day-to-day spend + ATM
# ---------------------------------------------------------------------------
def gen_discretionary_events(acct, population_by_persona):
    events = []
    if acct.persona not in config.DAILY_TXN_RATE:
        return events
    rate = config.DAILY_TXN_RATE[acct.persona]
    lo, hi = config.DISCRETIONARY_AMOUNT_RANGE[acct.persona]
    if hi <= 0:
        return events

    internal_pool = (population_by_persona.get("student", []) +
                      population_by_persona.get("family", []) +
                      population_by_persona.get("salaried", []))
    atm_scale = config.ATM_SCALE_BY_PERSONA.get(acct.persona, 1.0)

    for d in _daterange(config.SIM_START_DATE, config.SIM_END_DATE):
        n_today = np.random.poisson(rate)
        for _ in range(n_today):
            amount = float(np.exp(np.random.uniform(np.log(max(lo, 1)), np.log(hi))))
            if internal_pool and random.random() < 0.15:
                other = random.choice(internal_pool)
                if other.account_id != acct.account_id:
                    purpose = random.choice(["SPLIT BILL", "SENT MONEY", "GIFT", "REIMBURSEMENT"])
                    events += _linked_transfer(acct, other, _ts(d), amount, purpose)
                    continue
            merchant, category = random.choice(MERCHANT_CATEGORIES)
            events.append(_event(
                acct.account_id, _ts(d),
                f"UPI-{merchant.upper()}-{category}",
                "UPI", debit=amount, counterparty_name=merchant,
                category="discretionary",
            ))

        # weekly ATM withdrawal chance, evaluated roughly once/week
        if d.weekday() == 5 and random.random() < config.ATM_WITHDRAWAL_PROBABILITY_PER_WEEK:
            amt = random.uniform(*config.ATM_WITHDRAWAL_RANGE) * atm_scale
            events.append(_event(
                acct.account_id, _ts(d, hour=random.randint(9, 21)),
                f"ATM WDL-{acct.bank_code} ATM {fake.city().upper()}",
                "ATM", debit=round(amt, 2), category="discretionary",
            ))
    return events


def _apply_spending_floor(acct, events):
    """
    Walk an account's full chronological event stream and scale down (never
    skip outright unless headroom is negligible) any event tagged
    category="discretionary" - i.e. genuinely optional UPI spend / ATM
    withdrawals - so it never drives the account past its persona's
    BALANCE_FLOOR. Fixed obligations (rent/utility/EMI/supplier payments)
    and all credits are left untouched, since those should be free to
    occasionally push a balance tight - that's realistic; uncapped
    discretionary spend compounding into lakhs of overdraft is not.
    """
    floor = config.BALANCE_FLOOR.get(acct.persona, -3_000)
    events_sorted = sorted(events, key=lambda e: e["timestamp"])
    balance = acct.opening_balance
    kept = []
    for ev in events_sorted:
        if ev["category"] == "discretionary" and ev["debit"] > 0:
            projected = balance - ev["debit"]
            if projected < floor:
                available = balance - floor
                if available < 50:
                    continue  # no headroom at all - drop this discretionary txn
                ev["debit"] = round(available, 2)
        balance = round(balance + ev["credit"] - ev["debit"], 2)
        kept.append(ev)
    return kept


def simulate_account(acct, population_by_persona):
    """Generate the full legitimate event stream for one account, with a
    spending-floor pass applied to keep balances from going deeply negative."""
    events = []
    events += gen_salary_events(acct)
    events += gen_student_allowance_events(acct)
    events += gen_business_collection_events(acct, population_by_persona)
    events += gen_supplier_payment_events(acct)
    events += gen_rent_events(acct)
    events += gen_utility_bill_events(acct)
    events += gen_emi_events(acct)
    events += gen_discretionary_events(acct, population_by_persona)
    # Apply floor-clamping only to events owned by THIS account (debit legs
    # of linked transfers to other accounts are included in the list, but
    # those get clamped when we process the sending account's own event set).
    own_events = [e for e in events if e["account_id"] == acct.account_id]
    other_events = [e for e in events if e["account_id"] != acct.account_id]
    return _apply_spending_floor(acct, own_events) + other_events


def simulate_population(legit_accounts):
    """
    Run behavior simulation for every legitimate account.
    Returns a flat list of raw events plus the resulting cross-account
    transfer legs (already included, since linked transfers append both
    legs into the same list keyed by their own account_id).
    """
    population_by_persona = {}
    for a in legit_accounts:
        population_by_persona.setdefault(a.persona, []).append(a)

    all_events = []
    for acct in legit_accounts:
        all_events += simulate_account(acct, population_by_persona)
    return all_events
