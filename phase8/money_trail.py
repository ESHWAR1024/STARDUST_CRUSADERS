"""
Phase 8 — Money Trail & Fund Tracing Engine

Two complementary tracers:

  trace_forward(account_id)
    → Follows money OUT of an account hop by hop.
      Answers: "Where did this account's money ultimately go?"

  trace_backward(account_id)
    → Follows money INTO an account hop by hop.
      Answers: "Where did this account's money come from?"

Both use a time-ordered BFS on the TXN_GRAPH, matching credit legs
within TRAIL_MAX_HOP_HOURS of the preceding debit — the same
temporal-linkage logic real AML analysts use to chain NEFT/UPI hops.

Output: a list of TrailHop dicts that the reporting engine can render
as a human-readable money trail table and a graph path.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

import networkx as nx
import pandas as pd

from analytics_config import (
    TRAIL_MAX_HOPS, TRAIL_MIN_MATCH_RATIO, TRAIL_MAX_HOP_HOURS,
)


@dataclass
class TrailHop:
    hop_number:      int
    from_account:    str
    to_account:      str
    amount:          float
    date:            str
    timestamp:       datetime
    utr_ref:         str
    narration:       str
    channel:         str
    match_ratio:     float         # credit / previous debit
    cumulative_loss: float         # total % skimmed off the original amount


@dataclass
class MoneyTrail:
    root_account:  str
    direction:     str             # "forward" | "backward"
    seed_amount:   float
    hops:          list[TrailHop] = field(default_factory=list)
    terminal_node: Optional[str]  = None
    terminal_type: str            = ""   # e.g. "ATM", "CRYPTO", "INTERNATIONAL", "dead_end"
    total_hops:    int            = 0
    amount_recovered: float      = 0.0   # amount at end of trail

    def to_records(self) -> list[dict]:
        base = {
            "root_account": self.root_account,
            "direction":    self.direction,
            "seed_amount":  self.seed_amount,
            "terminal":     self.terminal_node,
            "terminal_type":self.terminal_type,
        }
        return [{**base, **asdict(h)} for h in self.hops]


def trace_forward(
    account_id: str,
    txn_graph:  nx.MultiDiGraph,
    df:         pd.DataFrame,
    seed_txn:   Optional[dict] = None,
) -> list[MoneyTrail]:
    """
    Follow money forward from account_id.
    Returns one MoneyTrail per distinct path found (BFS, max TRAIL_MAX_HOPS).
    """
    return _bfs_trace(account_id, txn_graph, df, direction="forward",
                      seed_txn=seed_txn)


def trace_backward(
    account_id: str,
    txn_graph:  nx.MultiDiGraph,
    df:         pd.DataFrame,
) -> list[MoneyTrail]:
    """
    Follow money backward into account_id.
    Returns one MoneyTrail per distinct path found (reverse BFS).
    """
    return _bfs_trace(account_id, txn_graph, df, direction="backward")


def _bfs_trace(
    root:      str,
    txn_graph: nx.MultiDiGraph,
    df:        pd.DataFrame,
    direction: str,
    seed_txn:  Optional[dict] = None,
) -> list[MoneyTrail]:
    """
    BFS on the transaction graph following temporal ordering.

    State in queue: (current_node, current_amount, current_time,
                     hops_so_far, trail_so_far, visited_set)
    """
    # Build per-account transaction index for fast lookup
    # {account_id -> sorted list of tx dicts}
    tx_index = _build_tx_index(df)

    # Seed amount: use the seed transaction if provided, else use the
    # largest single outflow from the root as the starting amount
    if seed_txn:
        seed_amount = seed_txn["amount"]
        seed_time   = seed_txn["timestamp"]
    else:
        root_txns   = tx_index.get(root, [])
        if not root_txns:
            return []
        outflows    = [t for t in root_txns if t["direction"] == "debit"] if direction == "forward" \
                      else [t for t in root_txns if t["direction"] == "credit"]
        if not outflows:
            return []
        outflows.sort(key=lambda t: t["amount"], reverse=True)
        seed_amount = outflows[0]["amount"]
        seed_time   = outflows[0]["timestamp"]

    completed_trails: list[MoneyTrail] = []

    # BFS queue entry: (node, amount, time, hop_count, hops_list, visited)
    queue = deque()
    queue.append((root, seed_amount, seed_time, 0, [], {root}))

    while queue:
        node, amount, last_time, hop_count, hops, visited = queue.popleft()

        if hop_count >= TRAIL_MAX_HOPS:
            _finalise_trail(root, direction, seed_amount, hops,
                            node, "max_hops_reached", completed_trails)
            continue

        # Get outgoing edges in the correct direction
        if direction == "forward":
            neighbors = list(txn_graph.successors(node))
        else:
            neighbors = list(txn_graph.predecessors(node))

        found_next = False
        for neighbor in neighbors:
            if neighbor in visited:
                continue   # avoid cycles

            # Fix: skip external nodes (merchants, unknown counterparties)
            # They are dead ends by design — not traceable internal accounts
            node_data = txn_graph.nodes.get(neighbor, {})
            if not node_data.get("is_internal", False):
                # Still record as terminal with type EXTERNAL
                if hops:
                    _finalise_trail(root, direction, seed_amount, hops,
                                    neighbor, "EXTERNAL_COUNTERPARTY", completed_trails)
                continue

            # Get all edges between node and neighbor in direction
            if direction == "forward":
                edges = txn_graph.get_edge_data(node, neighbor)
            else:
                edges = txn_graph.get_edge_data(neighbor, node)

            if not edges:
                continue

            # Find the temporally valid edge closest in time
            best_edge = _find_matching_edge(
                edges, last_time, amount, direction
            )
            if best_edge is None:
                continue

            edge_amount = best_edge["amount"]
            match_ratio = edge_amount / amount if amount > 0 else 0

            if match_ratio < TRAIL_MIN_MATCH_RATIO:
                continue   # too much skimmed — chain broken

            hop = TrailHop(
                hop_number      = hop_count + 1,
                from_account    = node if direction == "forward" else neighbor,
                to_account      = neighbor if direction == "forward" else node,
                amount          = edge_amount,
                date            = best_edge.get("date", ""),
                timestamp       = best_edge["timestamp"],
                utr_ref         = best_edge.get("utr_ref", ""),
                narration       = best_edge.get("narration", ""),
                channel         = best_edge.get("channel", ""),
                match_ratio     = round(match_ratio, 4),
                cumulative_loss = round(1 - (edge_amount / seed_amount), 4),
            )

            new_hops = hops + [hop]

            # Check if terminal
            terminal_type = _classify_terminal(neighbor, txn_graph, direction)
            if terminal_type:
                _finalise_trail(root, direction, seed_amount, new_hops,
                                neighbor, terminal_type, completed_trails)
                found_next = True
            else:
                queue.append((
                    neighbor, edge_amount, best_edge["timestamp"],
                    hop_count + 1, new_hops, visited | {neighbor}
                ))
                found_next = True

        if not found_next and hops:
            _finalise_trail(root, direction, seed_amount, hops,
                            node, "dead_end", completed_trails)

    return completed_trails


def _build_tx_index(df: pd.DataFrame) -> dict:
    """Build a {account_id: [tx_dict]} lookup for fast access."""
    index = {}
    for _, row in df.iterrows():
        acc = row["account_id"]
        ts  = _parse_ts(row)
        tx  = {
            "account_id":      acc,
            "amount":          float(row.get("debit", 0) or 0),
            "direction":       "debit" if float(row.get("debit", 0) or 0) > 0 else "credit",
            "timestamp":       ts,
            "date":            str(row.get("date", "")),
            "utr_ref":         str(row.get("utr_ref", "")),
            "narration":       str(row.get("narration", "")),
            "channel":         str(row.get("channel", "")),
            "counterparty":    str(row.get("counterparty_name", "")),
        }
        index.setdefault(acc, []).append(tx)

    for acc in index:
        index[acc].sort(key=lambda t: t["timestamp"])

    return index


def _find_matching_edge(
    edges: dict,
    last_time: datetime,
    amount: float,
    direction: str,
) -> Optional[dict]:
    """
    From all edges between two nodes, find the best temporal match:
    - must occur AFTER last_time
    - must occur within TRAIL_MAX_HOP_HOURS of last_time
    - amount closest to (but within ratio of) the seed amount
    """
    window_end = last_time + timedelta(hours=TRAIL_MAX_HOP_HOURS)
    candidates = []

    for edge_key, edge_data in edges.items():
        ts = edge_data.get("timestamp")
        if not isinstance(ts, datetime):
            continue
        if direction == "forward" and not (last_time <= ts <= window_end):
            continue
        if direction == "backward" and not (ts <= last_time):
            continue
        candidates.append(edge_data)

    if not candidates:
        return None

    # Pick the edge whose amount is closest to the source amount
    candidates.sort(key=lambda e: abs(e["amount"] - amount))
    return candidates[0]


def _classify_terminal(
    node: str,
    txn_graph: nx.MultiDiGraph,
    direction: str,
) -> str:
    """
    Determine if a node is a terminal (exit) node.
    Terminals: no further outgoing edges (dead end), or exit channels.
    """
    if direction == "forward":
        out_degree = txn_graph.out_degree(node)
    else:
        out_degree = txn_graph.in_degree(node)

    if out_degree == 0:
        return "dead_end"

    # Check if all outgoing narrations reference exit channels
    exit_keywords = ["ATM", "CASH WITHDRAWAL", "CRYPTO", "INTERNATIONAL",
                     "REMITTANCE", "WIRE", "FOREX"]
    if direction == "forward":
        for _, _, data in txn_graph.out_edges(node, data=True):
            narration = str(data.get("narration", "")).upper()
            for kw in exit_keywords:
                if kw in narration:
                    return kw.replace(" ", "_")

    return ""


def _finalise_trail(
    root, direction, seed_amount, hops, terminal, terminal_type, results
):
    if not hops:
        return
    trail = MoneyTrail(
        root_account     = root,
        direction        = direction,
        seed_amount      = seed_amount,
        hops             = hops,
        terminal_node    = terminal,
        terminal_type    = terminal_type,
        total_hops       = len(hops),
        amount_recovered = hops[-1].amount if hops else 0.0,
    )
    results.append(trail)


def _parse_ts(row) -> datetime:
    date_str = str(row.get("date", "")).strip()
    time_str = str(row.get("time", "00:00:00")).strip()
    if not time_str or time_str == "nan":
        time_str = "00:00:00"
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return datetime(2000, 1, 1)
