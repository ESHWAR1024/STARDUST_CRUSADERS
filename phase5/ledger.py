"""
Phase 5 - Ledger assembly.

Combines the raw event streams from behavior_engine (legitimate activity)
and fraud_engine (injected fraud typologies), then per account:
  1. sorts events chronologically
  2. computes a running balance starting from the account's opening balance
  3. assigns a globally unique transaction_id and a UTR/reference number
     (reusing the shared utr_ref for linked transfer legs so both sides of
     a transfer carry the same reference - exactly like a real bank pair)

Output: one flat, balanced master transaction table covering the whole
population - this is the canonical ground-truth ledger that per-bank
"statement" exports (statement_formatter.py) are sliced from.
"""

import random
from collections import defaultdict


def build_ledger(accounts, events):
    by_account = defaultdict(list)
    for ev in events:
        by_account[ev["account_id"]].append(ev)

    acct_index = {a.account_id: a for a in accounts}
    txn_counter = 0
    rows = []

    for account_id, acct_events in by_account.items():
        acct = acct_index[account_id]
        acct_events.sort(key=lambda e: e["timestamp"])
        balance = acct.opening_balance

        for ev in acct_events:
            balance = round(balance + ev["credit"] - ev["debit"], 2)
            txn_counter += 1
            utr = ev["utr_ref"] or f"TXN{random.randint(10**11, 10**12 - 1)}"

            rows.append({
                "transaction_id": f"TXN{txn_counter:08d}",
                "utr_ref": utr,
                "account_id": account_id,
                "account_holder": acct.holder_name,
                "bank_name": acct.bank_name,
                "date": ev["timestamp"].date().isoformat(),
                "time": ev["timestamp"].strftime("%H:%M:%S"),
                "timestamp": ev["timestamp"].isoformat(),
                "narration": ev["narration"],
                "channel": ev["channel"],
                "debit": ev["debit"] if ev["debit"] else 0.0,
                "credit": ev["credit"] if ev["credit"] else 0.0,
                "balance": balance,
                "counterparty_account_id": ev.get("counterparty_account_id"),
                "counterparty_name": ev.get("counterparty_name"),
                # ground-truth-only columns (stripped before "investigation system" sees the data)
                "is_fraud": ev.get("is_fraud", False),
                "fraud_pattern": ev.get("fraud_pattern"),
                "fraud_ring_id": ev.get("fraud_ring_id"),
            })

        # update the account's stored balance to the final computed value
        acct.balance = balance

    rows.sort(key=lambda r: (r["account_id"], r["timestamp"]))
    return rows
