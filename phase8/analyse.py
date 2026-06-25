"""
Phase 8 — Investigation Analytics Engine
Main orchestrator

Input : cleaned_transactions.csv  (Phase 7 output)
Output directory contains:

  analytics_transactions.csv     ← original rows + all Phase 8 flags
  round_trips.csv                ← every detected round-trip
  layering_chains.csv            ← every detected layering chain
  fan_in.csv                     ← collector account findings
  fan_out.csv                    ← distribution account findings
  smurfing.csv                   ← structuring findings
  odd_hours.csv                  ← odd-hour activity findings
  beneficiaries.csv              ← per-account beneficiary profile table
  risk_scores.csv                ← per-account risk score + tier + reasoning
  money_trails/                  ← forward/backward trails for top-risk accounts
      trail_<ACC>_forward.csv
      trail_<ACC>_backward.csv
  graph_summary.json             ← graph statistics
  analytics_report.json          ← full machine-readable findings summary
  analytics_summary.txt          ← human-readable narrative for investigators

Usage:
    python analyse.py --input ../phase7/cleaned/cleaned_transactions.csv
                      --out-dir analytics/
    python analyse.py --input cleaned_transactions.csv --out-dir analytics/ --top-trails 10
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime

from graph_builder    import build_graphs, graph_summary
from pattern_detectors import (
    detect_round_trips, detect_layering,
    detect_fan_in, detect_fan_out,
    detect_smurfing, detect_odd_hours,
)
from money_trail      import trace_forward, trace_backward
from risk_scorer      import analyse_beneficiaries, compute_risk_scores
from analytics_config import ANALYTICS_FLAG_COLS


def run_analytics(
    input_path:  str,
    out_dir:     str,
    top_trails:  int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full Phase 8 pipeline.
    Returns (analytics_df, risk_df).
    """
    os.makedirs(out_dir, exist_ok=True)
    trails_dir = os.path.join(out_dir, "money_trails")
    os.makedirs(trails_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print("  Phase 8 — Investigation Analytics Engine")
    print(f"{'='*65}")

    # ── Load Phase 7 output ─────────────────────────────────────────────
    df = pd.read_csv(input_path, dtype=str)
    for col in ("debit", "credit", "balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ("is_duplicate", "is_balance_breach",
                "is_high_value_flag", "is_ocr_row", "is_velocity_flag"):
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False,
                                    "true": True, "false": False}).fillna(False)
        else:
            df[col] = False

    print(f"\n  Loaded : {len(df):,} rows | {df['account_id'].nunique()} accounts")

    report = {
        "run_timestamp": datetime.now().isoformat(),
        "input_file":    os.path.basename(input_path),
        "rows_input":    len(df),
        "accounts":      int(df["account_id"].nunique()),
    }

    # Initialise Phase 8 flag columns
    df["is_round_trip"]    = False
    df["is_layering"]      = False
    df["is_fan_in"]        = False
    df["is_fan_out"]       = False
    df["is_smurfing"]      = False
    df["is_odd_hour"]      = False
    df["analytics_flags"]  = ""

    # ── Step 1: Build transaction graph ────────────────────────────────
    print("\n  [1/8] Building transaction graph ...")
    txn_graph, account_graph = build_graphs(df)
    g_summary = graph_summary(txn_graph, account_graph)
    report["graph"] = g_summary
    print(f"        Nodes : {g_summary['txn_graph_nodes']:,}  |  "
          f"Edges : {g_summary['txn_graph_edges']:,}  |  "
          f"Components : {g_summary['weakly_connected_comps']}")

    # ── Step 2: Round-trip detection ───────────────────────────────────
    print("\n  [2/8] Round-trip detection ...")
    rt_findings, rt_idx = detect_round_trips(df, txn_graph)
    df.loc[list(rt_idx), "is_round_trip"] = True
    _append_flag(df, list(rt_idx), "ROUND_TRIP")
    report["round_trips"] = len(rt_findings)
    print(f"        Found : {len(rt_findings)} round-trips "
          f"({len(rt_idx)} transactions flagged)")

    # ── Step 3: Layering detection ─────────────────────────────────────
    print("\n  [3/8] Layering chain detection ...")
    lay_findings, lay_idx = detect_layering(df, txn_graph)
    df.loc[list(lay_idx), "is_layering"] = True
    _append_flag(df, list(lay_idx), "LAYERING")
    report["layering_chains"] = len(lay_findings)
    print(f"        Found : {len(lay_findings)} layering chains "
          f"({len(lay_idx)} transactions flagged)")
    for f in lay_findings:
        print(f"          {f['chain']}  [{f['severity']}]")

    # ── Step 4: Fan-in detection ───────────────────────────────────────
    print("\n  [4/8] Fan-in (collector) detection ...")
    fi_findings, fi_idx = detect_fan_in(df)
    df.loc[list(fi_idx), "is_fan_in"] = True
    _append_flag(df, list(fi_idx), "FAN_IN")
    report["fan_in"] = len(fi_findings)
    print(f"        Found : {len(fi_findings)} collector accounts "
          f"({len(fi_idx)} transactions flagged)")

    # ── Step 5: Fan-out detection ──────────────────────────────────────
    print("\n  [5/8] Fan-out (distribution) detection ...")
    fo_findings, fo_idx = detect_fan_out(df)
    df.loc[list(fo_idx), "is_fan_out"] = True
    _append_flag(df, list(fo_idx), "FAN_OUT")
    report["fan_out"] = len(fo_findings)
    print(f"        Found : {len(fo_findings)} distribution accounts "
          f"({len(fo_idx)} transactions flagged)")

    # ── Step 6: Smurfing detection ─────────────────────────────────────
    print("\n  [6/8] Smurfing / structuring detection ...")
    sm_findings, sm_idx = detect_smurfing(df)
    df.loc[list(sm_idx), "is_smurfing"] = True
    _append_flag(df, list(sm_idx), "SMURFING")
    report["smurfing"] = len(sm_findings)
    print(f"        Found : {len(sm_findings)} smurfing patterns "
          f"({len(sm_idx)} transactions flagged)")

    # ── Step 7: Odd-hour detection ─────────────────────────────────────
    print("\n  [7/8] Odd-hour activity detection ...")
    oh_findings, oh_idx = detect_odd_hours(df)
    df.loc[list(oh_idx), "is_odd_hour"] = True
    _append_flag(df, list(oh_idx), "ODD_HOUR")
    report["odd_hours"] = len(oh_findings)
    print(f"        Found : {len(oh_findings)} accounts with odd-hour activity "
          f"({len(oh_idx)} transactions flagged)")

    # ── Step 8: Beneficiary analysis + Risk scoring ────────────────────
    print("\n  [8/8] Beneficiary analysis + Risk scoring ...")
    bene_df = analyse_beneficiaries(df)
    risk_df = compute_risk_scores(
        df, rt_findings, lay_findings, fi_findings,
        fo_findings, sm_findings, oh_findings, bene_df,
    )
    report["accounts_scored"] = len(risk_df)
    report["critical_accounts"] = int((risk_df["risk_tier"] == "CRITICAL").sum())
    report["high_accounts"]     = int((risk_df["risk_tier"] == "HIGH").sum())
    report["medium_accounts"]   = int((risk_df["risk_tier"] == "MEDIUM").sum())

    print(f"        Accounts scored : {len(risk_df)}")
    print(f"        CRITICAL        : {report['critical_accounts']}")
    print(f"        HIGH            : {report['high_accounts']}")
    print(f"        MEDIUM          : {report['medium_accounts']}")

    # ── Money trails for top-N highest-risk accounts ───────────────────
    print(f"\n  [+] Tracing money trails for top {top_trails} accounts ...")
    top_accounts = risk_df.head(top_trails)["account_id"].tolist()
    trail_manifest = []

    for acc_id in top_accounts:
        # Forward
        fwd_trails = trace_forward(acc_id, txn_graph, df)
        fwd_rows   = [r for t in fwd_trails for r in t.to_records()]
        fwd_path   = os.path.join(trails_dir, f"trail_{acc_id}_forward.csv")
        pd.DataFrame(fwd_rows).to_csv(fwd_path, index=False)   # empty CSV if no trails
        trail_manifest.append({
            "account":   acc_id,
            "direction": "forward",
            "hops":      len(fwd_rows),
            "trails":    len(fwd_trails),
            "status":    "ok" if fwd_rows else "no_internal_counterparties",
            "file":      os.path.basename(fwd_path),
        })

        # Backward
        bwd_trails = trace_backward(acc_id, txn_graph, df)
        bwd_rows   = [r for t in bwd_trails for r in t.to_records()]
        bwd_path   = os.path.join(trails_dir, f"trail_{acc_id}_backward.csv")
        pd.DataFrame(bwd_rows).to_csv(bwd_path, index=False)   # empty CSV if no trails
        trail_manifest.append({
            "account":   acc_id,
            "direction": "backward",
            "hops":      len(bwd_rows),
            "trails":    len(bwd_trails),
            "status":    "ok" if bwd_rows else "no_internal_counterparties",
            "file":      os.path.basename(bwd_path),
        })

    report["trail_manifest"] = trail_manifest

    # ── Export all outputs ─────────────────────────────────────────────
    print("\n  Exporting outputs ...")

    # 0. Export account_graph for Phase 9 (community detection, PageRank, centrality)
    import pickle
    graph_pkl_path = os.path.join(out_dir, "account_graph.pkl")
    with open(graph_pkl_path, "wb") as f:
        pickle.dump(account_graph, f)
    print(f"        account_graph.pkl saved → Phase 9 ready")

    # 1. analytics_transactions.csv — all rows + Phase 8 flags
    out_cols = [c for c in ANALYTICS_FLAG_COLS if c in df.columns]
    df[out_cols].to_csv(
        os.path.join(out_dir, "analytics_transactions.csv"), index=False
    )

    # 2. Per-pattern finding files
    _save(rt_findings,  os.path.join(out_dir, "round_trips.csv"))
    _save(lay_findings, os.path.join(out_dir, "layering_chains.csv"))
    _save(fi_findings,  os.path.join(out_dir, "fan_in.csv"))
    _save(fo_findings,  os.path.join(out_dir, "fan_out.csv"))
    _save(sm_findings,  os.path.join(out_dir, "smurfing.csv"))
    _save(oh_findings,  os.path.join(out_dir, "odd_hours.csv"))

    # 3. Beneficiary profiles
    bene_df.to_csv(os.path.join(out_dir, "beneficiaries.csv"), index=False)

    # 4. Risk scores
    risk_df.to_csv(os.path.join(out_dir, "risk_scores.csv"), index=False)

    # 5. Graph summary
    with open(os.path.join(out_dir, "graph_summary.json"), "w") as f:
        json.dump(g_summary, f, indent=2)

    # 6. Full analytics report
    report["rows_output"] = len(df)
    report["total_flagged_rows"] = int(
        df[["is_round_trip", "is_layering", "is_fan_in",
            "is_fan_out", "is_smurfing", "is_odd_hour"]].any(axis=1).sum()
    )
    with open(os.path.join(out_dir, "analytics_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # 7. Human-readable summary
    _write_summary(report, risk_df, rt_findings, lay_findings,
                   fi_findings, fo_findings, sm_findings, oh_findings, out_dir)

    _print_final_summary(report, out_dir)
    return df, risk_df, account_graph, txn_graph


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_flag(df: pd.DataFrame, idxs: list, flag: str):
    """Append flag text to the analytics_flags column for flagged rows."""
    if not idxs:
        return
    current = df.loc[idxs, "analytics_flags"]
    df.loc[idxs, "analytics_flags"] = current.apply(
        lambda x: (str(x) + " | " + flag).strip(" | ") if str(x) else flag
    )


def _save(findings: list, path: str):
    pd.DataFrame(findings).to_csv(path, index=False) if findings \
        else pd.DataFrame().to_csv(path, index=False)


def _write_summary(
    report, risk_df,
    rt, lay, fi, fo, sm, oh,
    out_dir,
):
    lines = [
        "=" * 70,
        "PHASE 8 — INVESTIGATION ANALYTICS SUMMARY",
        f"Run at : {report['run_timestamp']}",
        f"Input  : {report['input_file']}",
        "=" * 70,
        "",
        "OVERVIEW",
        "-" * 40,
        f"  Rows analysed           : {report['rows_input']:,}",
        f"  Accounts analysed       : {report['accounts']}",
        f"  Rows with analytics flag: {report.get('total_flagged_rows', 0):,}",
        "",
        "GRAPH STATISTICS",
        "-" * 40,
        f"  Graph nodes             : {report['graph']['txn_graph_nodes']}",
        f"  Graph edges (transfers) : {report['graph']['txn_graph_edges']}",
        f"  Connected components    : {report['graph']['weakly_connected_comps']}",
        "",
        "PATTERN DETECTION RESULTS",
        "-" * 40,
        f"  Round-trips             : {len(rt)}",
        f"  Layering chains         : {len(lay)}",
        f"  Fan-in collectors       : {len(fi)}",
        f"  Fan-out distributors    : {len(fo)}",
        f"  Smurfing / structuring  : {len(sm)}",
        f"  Odd-hour accounts       : {len(oh)}",
        "",
        "RISK SCORE DISTRIBUTION",
        "-" * 40,
        f"  CRITICAL (≥75)          : {report['critical_accounts']}",
        f"  HIGH     (≥50)          : {report['high_accounts']}",
        f"  MEDIUM   (≥25)          : {report['medium_accounts']}",
        f"  LOW      (<25)          : "
        f"{report['accounts_scored'] - report['critical_accounts'] - report['high_accounts'] - report['medium_accounts']}",
        "",
    ]

    # Top 10 highest-risk accounts
    if not risk_df.empty:
        lines += ["TOP 10 HIGHEST-RISK ACCOUNTS", "-" * 40]
        for _, row in risk_df.head(10).iterrows():
            lines.append(
                f"  [{row['risk_tier']:8s}] {row['account_id']:12s} "
                f"Score:{row['risk_score']:5.1f}  {row['account_holder']}"
            )
            lines.append(f"             Patterns : {row['active_patterns']}")
            lines.append(f"             Reasoning: {row['risk_reasoning'][:100]}")
            lines.append("")

    # Layering chain detail
    if lay:
        lines += ["LAYERING CHAINS DETAIL", "-" * 40]
        for f in lay:
            lines.append(f"  [{f['severity']:8s}] {f['chain']}")
            lines.append(
                f"             {f['chain_length']} hops | "
                f"₹{f['start_amount']:,.0f} → ₹{f['end_amount']:,.0f} "
                f"({f['skim_ratio']*100:.1f}% skimmed)"
            )
            lines.append("")

    # Round-trips
    if rt:
        lines += ["ROUND-TRIP DETAIL", "-" * 40]
        for f in rt[:10]:
            lines.append(f"  {f['description']}")

    lines += [
        "",
        "OUTPUT FILES",
        "-" * 40,
        "  account_graph.pkl            ← NetworkX DiGraph → feed directly to Phase 9",
        "  analytics_transactions.csv   ← all rows + Phase 8 flags → Feed to Phase 9/10",
        "  round_trips.csv              ← round-trip findings",
        "  layering_chains.csv          ← layering chain findings",
        "  fan_in.csv                   ← collector account findings",
        "  fan_out.csv                  ← distribution account findings",
        "  smurfing.csv                 ← structuring findings",
        "  odd_hours.csv                ← odd-hour findings",
        "  beneficiaries.csv            ← beneficiary profiles per account",
        "  risk_scores.csv              ← per-account risk score, tier, reasoning",
        "  money_trails/                ← forward/backward fund traces for top accounts",
        "  graph_summary.json           ← graph statistics",
        "  analytics_report.json        ← machine-readable full report",
        "  analytics_summary.txt        ← this file",
        "=" * 70,
    ]

    with open(os.path.join(out_dir, "analytics_summary.txt"), "w") as f:
        f.write("\n".join(lines))


def _print_final_summary(report, out_dir):
    print(f"\n{'='*65}")
    print("  PHASE 8 COMPLETE")
    print(f"{'='*65}")
    print(f"  Round-trips detected     : {report['round_trips']}")
    print(f"  Layering chains detected : {report['layering_chains']}")
    print(f"  Fan-in collectors        : {report['fan_in']}")
    print(f"  Fan-out distributors     : {report['fan_out']}")
    print(f"  Smurfing patterns        : {report['smurfing']}")
    print(f"  Odd-hour accounts        : {report['odd_hours']}")
    print(f"  CRITICAL risk accounts   : {report['critical_accounts']}")
    print(f"  HIGH risk accounts       : {report['high_accounts']}")
    print(f"\n  Outputs → {os.path.abspath(out_dir)}/")
    print(f"    • analytics_transactions.csv  ← feed to Phase 9/10")
    print(f"    • risk_scores.csv             ← ranked account list")
    print(f"    • layering_chains.csv         ← money trail chains")
    print(f"    • money_trails/               ← per-account hop traces")
    print(f"    • analytics_summary.txt       ← human-readable report")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Phase 8 — Investigation Analytics Engine"
    )
    ap.add_argument("--input",      required=True,
                    help="cleaned_transactions.csv from Phase 7")
    ap.add_argument("--out-dir",    default="analytics",
                    help="Output directory")
    ap.add_argument("--top-trails", type=int, default=5,
                    help="Number of top-risk accounts to trace money trails for")
    args = ap.parse_args()
    run_analytics(args.input, args.out_dir, args.top_trails)
