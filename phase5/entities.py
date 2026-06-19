"""
Phase 5 - Entity generation.

Builds the population of accounts: normal personas (salaried, student,
business, shopkeeper, family) plus the hidden fraud population (mules,
collectors, layering shell accounts, ring members). Every account gets a
realistic-looking account number, IFSC, holder name, and opening balance.

This module only creates the *static* account roster. Behavior over time
is simulated separately in behavior_engine.py and fraud_engine.py.
"""

import random
import numpy as np
from faker import Faker

import config

fake = Faker("en_IN")


class Account:
    """A single bank account in the synthetic ecosystem."""

    __slots__ = (
        "account_id", "holder_name", "persona", "bank_name", "bank_code",
        "ifsc", "account_number", "opening_balance", "balance",
        "is_fraud", "fraud_role", "fraud_ring_id", "monthly_income",
        "open_date", "meta",
    )

    def __init__(self, account_id, holder_name, persona, bank, opening_balance,
                 is_fraud=False, fraud_role=None, fraud_ring_id=None,
                 monthly_income=None, open_date=None, meta=None):
        self.account_id = account_id
        self.holder_name = holder_name
        self.persona = persona
        self.bank_name = bank["name"]
        self.bank_code = bank["code"]
        self.ifsc = f"{bank['ifsc_prefix']}0{random.randint(100000, 999999):06d}"[:11]
        self.account_number = str(random.randint(10**13, 10**14 - 1))  # 14-digit acct no
        self.opening_balance = round(opening_balance, 2)
        self.balance = self.opening_balance
        self.is_fraud = is_fraud
        self.fraud_role = fraud_role            # mule | collector | layering_node | ring_member | None
        self.fraud_ring_id = fraud_ring_id
        self.monthly_income = monthly_income
        self.open_date = open_date
        self.meta = meta or {}

    def to_dict(self):
        return {
            "account_id": self.account_id,
            "holder_name": self.holder_name,
            "persona": self.persona,
            "bank_name": self.bank_name,
            "bank_code": self.bank_code,
            "ifsc": self.ifsc,
            "account_number": self.account_number,
            "opening_balance": self.opening_balance,
            "is_fraud": self.is_fraud,
            "fraud_role": self.fraud_role,
            "fraud_ring_id": self.fraud_ring_id,
            "monthly_income": self.monthly_income,
            "open_date": self.open_date,
        }


def _random_bank():
    return random.choice(config.BANKS)


def _opening_balance(persona):
    lo, hi = config.OPENING_BALANCE_RANGE[persona]
    if hi <= lo:
        return lo
    # log-uniform: avoids everyone clustering near the midpoint
    return float(np.exp(np.random.uniform(np.log(max(lo, 1)), np.log(hi))))


def generate_legitimate_population(n_accounts: int) -> list:
    """Generate the normal financial ecosystem accounts."""
    accounts = []
    personas = list(config.PERSONA_MIX.keys())
    weights = list(config.PERSONA_MIX.values())

    for i in range(n_accounts):
        persona = random.choices(personas, weights=weights, k=1)[0]
        bank = _random_bank()
        opening = _opening_balance(persona)

        monthly_income = None
        if persona == "salaried":
            lo, hi = config.SALARY_RANGE["salaried"]
            monthly_income = round(float(np.exp(np.random.uniform(np.log(lo), np.log(hi)))), 2)
        elif persona == "student":
            lo, hi = config.STUDENT_ALLOWANCE_RANGE
            monthly_income = round(random.uniform(lo, hi), 2)
        elif persona == "family":
            lo, hi = config.FAMILY_INCOME_RANGE
            monthly_income = round(float(np.exp(np.random.uniform(np.log(lo), np.log(hi)))), 2)

        open_date = fake.date_between(start_date="-5y", end_date="-30d")

        meta = {}
        if persona in ("salaried", "family"):
            meta["employer_name"] = fake.company()
        if persona in ("salaried", "family"):
            if random.random() < config.RENT_PROBABILITY:
                meta["landlord_name"] = fake.name()
        if persona in ("business", "shopkeeper"):
            meta["business_category"] = random.choice(
                ["Kirana Store", "Electronics Retail", "Pharmacy", "Textiles",
                 "Hardware Supplies", "Restaurant", "Mobile Recharge & Accessories",
                 "Stationery & Printing"]
            )
        if persona == "student":
            meta["guardian_name"] = fake.name()

        acct = Account(
            account_id=f"ACC{i:06d}",
            holder_name=fake.name(),
            persona=persona,
            bank=bank,
            opening_balance=opening,
            is_fraud=False,
            monthly_income=monthly_income,
            open_date=open_date,
            meta=meta,
        )
        accounts.append(acct)
    return accounts


def generate_fraud_population(start_index: int, role_counts: dict) -> list:
    """
    Generate accounts that exist purely to serve fraud typologies, with
    EXACT counts per role (mule / collector / fresh_layering_node) as
    determined by fraud_engine.plan_fraud_rings(). Pre-planning the exact
    counts (rather than sampling roles independently of ring demand)
    guarantees every planned ring actually gets built instead of silently
    running out of accounts of the role it needed.
    """
    accounts = []
    i = start_index
    for role, count in role_counts.items():
        for _ in range(count):
            bank = _random_bank()
            balance_key = "mule" if role in ("mule", "collector") else "fresh_layering_node"
            opening = _opening_balance(balance_key)
            open_date = fake.date_between(start_date="-9M", end_date="-15d")

            acct = Account(
                account_id=f"ACC{i:06d}",
                holder_name=fake.name(),
                persona="mule_network",
                bank=bank,
                opening_balance=opening,
                is_fraud=True,
                fraud_role=role,
                open_date=open_date,
            )
            accounts.append(acct)
            i += 1
    return accounts


def build_population(role_counts: dict):
    """
    Returns (accounts, legit_accounts, fraud_accounts). The fraud
    population size/roles are dictated exactly by role_counts (computed
    from the fraud ring plan), and legitimate accounts fill the remainder
    of config.N_ACCOUNTS.
    """
    n_fraud = sum(role_counts.values())
    n_legit = max(config.N_ACCOUNTS - n_fraud, 1)

    legit = generate_legitimate_population(n_legit)
    fraud = generate_fraud_population(n_legit, role_counts)

    all_accounts = legit + fraud
    return all_accounts, legit, fraud
