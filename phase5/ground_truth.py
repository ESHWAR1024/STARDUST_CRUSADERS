"""
Phase 5 - Ground truth export.

Fraud labels are written to files completely separate from the master
clean dataset / statement exports. The investigation system (Phases 7-11)
should never read these directly - they exist so your team can measure
precision/recall of the graph-analytics and ML detection layers against
a known answer key, and so the demo presenter can narrate "here's what
was hidden, and here's what the system found on its own."
"""

import pandas as pd


def export_account_labels(accounts, ring_summaries, out_path):
    rows = []
    ring_typology = {r["ring_id"]: r["typology"] for r in ring_summaries}
    for a in accounts:
        rows.append({
            "account_id": a.account_id,
            "holder_name": a.holder_name,
            "persona": a.persona,
            "is_fraud": a.is_fraud,
            "fraud_role": a.fraud_role,
            "fraud_ring_id": a.fraud_ring_id,
            "fraud_typology": ring_typology.get(a.fraud_ring_id) if a.fraud_ring_id else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def export_transaction_labels(ledger_rows, out_path):
    df = pd.DataFrame(ledger_rows)[
        ["transaction_id", "account_id", "is_fraud", "fraud_pattern", "fraud_ring_id"]
    ]
    df.to_csv(out_path, index=False)
    return df


def export_ring_summary(ring_summaries, out_path):
    pd.DataFrame(ring_summaries).to_csv(out_path, index=False)
