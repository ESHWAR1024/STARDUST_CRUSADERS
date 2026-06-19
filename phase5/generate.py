"""
Phase 5 - Main orchestrator.

Run with:
    python generate.py [--n-accounts 500] [--out-dir output]

Pipeline:
  1. seed RNGs
  2. build account population (legitimate + fraud-role accounts)
  3. simulate legitimate behavior -> raw events
  4. organize fraud accounts into rings, build fraud typology events
  5. merge events, build the balanced ledger (running balances, txn IDs)
  6. export: master clean dataset, sample "as received" multi-format
     statements, and ground-truth label files
  7. print a summary so you can sanity-check the run before using it
     downstream (Phase 6 ingestion, Phase 8/9 graph analytics, etc.)
"""

import argparse
import os
import random
import pandas as pd

import config
import entities
import behavior_engine
import fraud_engine
import ledger as ledger_mod
import statement_formatter
import ground_truth


def run(n_accounts=None, out_dir="output", n_sample_statements=12):
    config.seed_all()
    if n_accounts:
        config.N_ACCOUNTS = n_accounts

    os.makedirs(out_dir, exist_ok=True)

    print("[1/6] Planning fraud rings and building account population ...")
    ring_plans, role_totals = fraud_engine.plan_fraud_rings()
    all_accounts, legit_accounts, fraud_accounts = entities.build_population(role_totals)
    print(f"      -> {len(legit_accounts)} legitimate accounts, {len(fraud_accounts)} fraud-role accounts "
          f"({role_totals})")

    print("[2/6] Simulating legitimate behavior ...")
    legit_events = behavior_engine.simulate_population(legit_accounts)
    print(f"      -> {len(legit_events)} legitimate transaction events")

    print("[3/6] Building fraud rings and injecting fraud typologies ...")
    fraud_events, ring_summaries = fraud_engine.build_fraud_rings(fraud_accounts, ring_plans)
    print(f"      -> {len(ring_summaries)} fraud rings built, {len(fraud_events)} fraud transaction events")
    for r in ring_summaries:
        print(f"         {r['ring_id']}: {r['typology']} ({len(r['accounts'])} accounts)")

    print("[4/6] Assembling balanced ledger (running balances + transaction IDs) ...")
    all_events = legit_events + fraud_events
    rows = ledger_mod.build_ledger(all_accounts, all_events)
    print(f"      -> {len(rows)} total transactions across {len(all_accounts)} accounts")

    print("[5/6] Exporting master dataset, sample multi-format statements, and ground truth ...")
    rows_df = pd.DataFrame(rows)

    master_path = os.path.join(out_dir, "master_transactions_clean.csv")
    statement_formatter.export_master_clean(rows, master_path)

    accounts_path = os.path.join(out_dir, "accounts_master.csv")
    pd.DataFrame([a.to_dict() for a in all_accounts]).to_csv(accounts_path, index=False)

    # sample accounts for multi-format "as received" statement exports:
    # mix of normal accounts and a few fraud-ring accounts so the demo can
    # show ingestion working on both clean and adversarial inputs
    normal_sample = random.sample(legit_accounts, min(8, len(legit_accounts)))
    fraud_sample = random.sample(fraud_accounts, min(4, len(fraud_accounts)))
    sample_ids = [a.account_id for a in normal_sample + fraud_sample]

    statements_dir = os.path.join(out_dir, "sample_bank_statements")
    manifest = statement_formatter.export_sample_bank_statements(
        rows_df, all_accounts, sample_ids, statements_dir
    )

    gt_dir = os.path.join(out_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)
    ground_truth.export_account_labels(all_accounts, ring_summaries,
                                        os.path.join(gt_dir, "account_labels.csv"))
    ground_truth.export_transaction_labels(rows, os.path.join(gt_dir, "transaction_labels.csv"))
    ground_truth.export_ring_summary(ring_summaries, os.path.join(gt_dir, "ring_summary.csv"))

    print("[6/6] Done.\n")
    _print_summary(all_accounts, fraud_accounts, rows_df, ring_summaries, manifest, out_dir)
    return rows_df, all_accounts, ring_summaries


def _print_summary(all_accounts, fraud_accounts, rows_df, ring_summaries, manifest, out_dir):
    n_fraud_txns = int(rows_df["is_fraud"].sum())
    total_txns = len(rows_df)
    print("=" * 70)
    print("SYNTHETIC DATA GENERATION SUMMARY")
    print("=" * 70)
    print(f"Accounts total           : {len(all_accounts)}")
    print(f"  legitimate              : {len(all_accounts) - len(fraud_accounts)}")
    print(f"  fraud-role               : {len(fraud_accounts)}")
    print(f"Transactions total        : {total_txns}")
    print(f"  fraud-tagged (hidden)    : {n_fraud_txns} ({n_fraud_txns/total_txns:.2%})")
    print(f"Fraud rings                : {len(ring_summaries)}")
    print(f"Date range                 : {config.SIM_START_DATE} to {config.SIM_END_DATE}")
    print(f"Sample statement files     : {len(manifest)} (in {out_dir}/sample_bank_statements)")
    print("-" * 70)
    print(f"Outputs written to: {os.path.abspath(out_dir)}")
    print("  master_transactions_clean.csv   <- unified clean dataset (no fraud labels)")
    print("  accounts_master.csv             <- account roster")
    print("  sample_bank_statements/         <- multi-format 'as received' test fixtures")
    print("  ground_truth/                   <- account/transaction/ring labels (evaluation only)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 synthetic bank statement data generator")
    parser.add_argument("--n-accounts", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="output")
    args = parser.parse_args()
    run(n_accounts=args.n_accounts, out_dir=args.out_dir)
