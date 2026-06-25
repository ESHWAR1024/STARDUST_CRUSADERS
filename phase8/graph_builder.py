"""
Phase 8 — Graph Builder

Builds two NetworkX graphs from the cleaned transaction dataframe:

  TXN_GRAPH   — directed multigraph where every edge is one transaction
                node = account_id
                edge attrs: amount, date, timestamp, utr_ref, narration, channel

  ACCOUNT_GRAPH — simple weighted digraph where edge weight = total flow
                  between two accounts (collapsed from TXN_GRAPH)
                  Used for community detection and risk propagation

Both graphs are returned so downstream modules (round-trip, layering,
fan-in/out, graph analytics) can share the same object rather than
rebuilding it each time.
"""

import pandas as pd
import networkx as nx
from datetime import datetime


def build_graphs(df: pd.DataFrame) -> tuple[nx.MultiDiGraph, nx.DiGraph]:
    """
    Input : cleaned_transactions DataFrame (Phase 7 output)
    Output: (txn_graph, account_graph)

    Node attribute  is_internal=True  → account_id from the dataset
                    is_internal=False → external counterparty name (merchant, unknown)
    money_trail.py respects this flag: BFS only follows is_internal nodes,
    so traces don't dead-end into "Swiggy" or "IRCTC".
    """
    txn_graph     = nx.MultiDiGraph()
    account_graph = nx.DiGraph()

    # All known internal account IDs
    internal_ids = set(df["account_id"].unique())

    # Add all internal account nodes first
    for acc_id in internal_ids:
        meta = df[df["account_id"] == acc_id].iloc[0]
        txn_graph.add_node(acc_id,
            holder=meta.get("account_holder", ""),
            bank=meta.get("bank_name", ""),
            is_internal=True,
        )
        account_graph.add_node(acc_id,
            holder=meta.get("account_holder", ""),
            bank=meta.get("bank_name", ""),
            is_internal=True,
        )

    # Process transfer edges — only rows that have a counterparty
    transfer_df = df[
        df["counterparty_name"].notna() &
        (df["counterparty_name"].str.strip() != "") &
        (df["debit"] > 0)
    ].copy()

    for _, row in transfer_df.iterrows():
        src  = row["account_id"]
        dst  = str(row["counterparty_name"]).strip()
        amt  = float(row["debit"])
        ts   = _parse_ts(row)
        utr  = str(row.get("utr_ref", "")).strip()
        nar  = str(row.get("narration", "")).strip()
        chan = str(row.get("channel", "")).strip()
        date = str(row.get("date", "")).strip()

        # Add counterparty node if not already present
        # Tag as internal only if it matches a known account_id
        dst_is_internal = dst in internal_ids
        if dst not in txn_graph:
            txn_graph.add_node(dst,
                holder=dst,
                bank="UNKNOWN",
                is_internal=dst_is_internal,
            )
        if dst not in account_graph:
            account_graph.add_node(dst,
                holder=dst,
                bank="UNKNOWN",
                is_internal=dst_is_internal,
            )

        # TXN_GRAPH: one edge per transaction
        txn_graph.add_edge(
            src, dst,
            amount=amt,
            timestamp=ts,
            date=date,
            utr_ref=utr,
            narration=nar,
            channel=chan,
            account_id=src,
        )

        # ACCOUNT_GRAPH: accumulate total flow
        if account_graph.has_edge(src, dst):
            account_graph[src][dst]["total_amount"] += amt
            account_graph[src][dst]["txn_count"]    += 1
        else:
            account_graph.add_edge(
                src, dst,
                total_amount=amt,
                txn_count=1,
            )

    return txn_graph, account_graph


def _parse_ts(row) -> datetime:
    """Parse a timestamp from date + time columns, fall back gracefully."""
    date_str = str(row.get("date", "")).strip()
    time_str = str(row.get("time", "00:00:00")).strip()
    if not time_str or time_str == "nan":
        time_str = "00:00:00"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return datetime(2000, 1, 1)


def graph_summary(txn_graph: nx.MultiDiGraph, account_graph: nx.DiGraph) -> dict:
    return {
        "txn_graph_nodes":        txn_graph.number_of_nodes(),
        "txn_graph_edges":        txn_graph.number_of_edges(),
        "account_graph_nodes":    account_graph.number_of_nodes(),
        "account_graph_edges":    account_graph.number_of_edges(),
        "weakly_connected_comps": nx.number_weakly_connected_components(account_graph),
    }
