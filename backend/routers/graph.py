"""
Router: /api/v1/graph

Produces Cytoscape.js-ready node/edge payloads for:
  GET /fund-trace/{account_id}   Ego graph N hops from a seed account
  GET /ring/{ring_id}            Full ring subgraph
  GET /full                      Entire transaction graph (paginated,
                                 for overview visualization)

Node = Account.  Edge = internal Transaction (counterparty_account_id != null).

Edge weight = total amount transferred between the pair.
Risk score encoded as node colour gradient in the frontend.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from collections import defaultdict

from dependencies import get_db
from models import Account, Transaction, FraudRing, FraudRingMember
from schemas import GraphData, GraphNode, GraphEdge

router = APIRouter(prefix="/graph", tags=["Graph"])


async def _build_graph_from_account_ids(
    account_ids: list[str],
    db: AsyncSession,
    seed_account_id: Optional[str] = None,
) -> GraphData:
    """
    Core graph builder. Given a set of account IDs:
    1. Load account metadata (nodes)
    2. Load all internal transactions between those accounts (edges)
    3. Return Cytoscape.js-compatible payload
    """
    if not account_ids:
        return GraphData(nodes=[], edges=[], total_nodes=0,
                         total_edges=0, seed_account_id=seed_account_id)

    # --- Nodes ---
    acct_q = select(Account).where(Account.account_id.in_(account_ids))
    accounts = (await db.execute(acct_q)).scalars().all()

    # Get per-account transaction stats
    stats_q = (
        select(
            Transaction.account_id,
            func.count().label("txn_count"),
            func.sum(Transaction.debit).label("total_debit"),
            func.sum(Transaction.credit).label("total_credit"),
        )
        .where(Transaction.account_id.in_(account_ids))
        .group_by(Transaction.account_id)
    )
    stats_rows = (await db.execute(stats_q)).all()
    stats = {r.account_id: r for r in stats_rows}

    nodes = []
    for acct in accounts:
        s = stats.get(acct.account_id)
        nodes.append(GraphNode(
            id=acct.account_id,
            label=acct.holder_name,
            bank=acct.bank_name,
            risk_score=acct.risk_score,
            is_suspect=acct.is_suspect,
            fraud_role=acct.fraud_role,
            total_debit=float(s.total_debit or 0) if s else 0.0,
            total_credit=float(s.total_credit or 0) if s else 0.0,
            txn_count=int(s.txn_count or 0) if s else 0,
        ))

    # --- Edges: aggregate transfers between account pairs ---
    edge_q = (
        select(
            Transaction.account_id,
            Transaction.counterparty_account_id,
            Transaction.channel,
            Transaction.date,
            Transaction.utr_ref,
            func.sum(Transaction.debit + Transaction.credit).label("total_amount"),
            func.count().label("txn_count"),
            func.max(func.cast(Transaction.is_high_value_flag, func.Integer())).label("any_flag"),
        )
        .where(
            Transaction.account_id.in_(account_ids),
            Transaction.counterparty_account_id.in_(account_ids),
            Transaction.counterparty_account_id.isnot(None),
        )
        .group_by(
            Transaction.account_id,
            Transaction.counterparty_account_id,
            Transaction.channel,
            Transaction.date,
            Transaction.utr_ref,
        )
    )
    edge_rows = (await db.execute(edge_q)).all()

    # Deduplicate bidirectional pairs — keep one edge per pair
    seen_pairs = set()
    edges = []
    for i, row in enumerate(edge_rows):
        pair = tuple(sorted([row.account_id, row.counterparty_account_id]))
        edge_key = (pair, row.channel, row.date or "")
        if edge_key in seen_pairs:
            continue
        seen_pairs.add(edge_key)

        edges.append(GraphEdge(
            id=row.utr_ref or f"edge_{i}",
            source=row.account_id,
            target=row.counterparty_account_id,
            amount=float(row.total_amount or 0),
            channel=row.channel or "OTHER",
            date=row.date,
            is_fraud_flagged=bool(row.any_flag),
        ))

    return GraphData(
        nodes=nodes,
        edges=edges,
        total_nodes=len(nodes),
        total_edges=len(edges),
        seed_account_id=seed_account_id,
    )


# ---------------------------------------------------------------------------
# Fund trace — N-hop ego graph from a seed account
# ---------------------------------------------------------------------------
@router.get("/fund-trace/{account_id}", response_model=GraphData)
async def fund_trace(
    account_id: str,
    hops: int = Query(default=2, ge=1, le=4),
    db: AsyncSession = Depends(get_db),
):
    """
    Starting from seed account, expand N hops via internal counterparty
    links. This is the primary investigator tool: "starting from this
    victim/mule account, show me the full money trail."

    hops=1 → direct counterparties only
    hops=2 → counterparties of counterparties (typical for layering)
    hops=3 → full ring exposure (fan-in/fan-out)
    """
    account = await db.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    # BFS expansion
    visited = {account_id}
    frontier = {account_id}

    for _ in range(hops):
        if not frontier:
            break
        cp_q = (
            select(Transaction.counterparty_account_id)
            .where(
                Transaction.account_id.in_(frontier),
                Transaction.counterparty_account_id.isnot(None),
            )
            .distinct()
        )
        next_ids = set((await db.execute(cp_q)).scalars().all()) - visited
        visited |= next_ids
        frontier = next_ids

    return await _build_graph_from_account_ids(
        list(visited), db, seed_account_id=account_id
    )


# ---------------------------------------------------------------------------
# Ring subgraph
# ---------------------------------------------------------------------------
@router.get("/ring/{ring_id}", response_model=GraphData)
async def ring_graph(ring_id: str, db: AsyncSession = Depends(get_db)):
    """
    Full graph for a specific fraud ring.
    Used in the investigation report and ring detail panel.
    """
    ring = await db.get(FraudRing, ring_id)
    if not ring:
        raise HTTPException(status_code=404, detail="Fraud ring not found")

    member_q = select(FraudRingMember.account_id).where(
        FraudRingMember.ring_id == ring_id
    )
    account_ids = (await db.execute(member_q)).scalars().all()

    return await _build_graph_from_account_ids(
        list(account_ids), db, seed_account_id=None
    )


# ---------------------------------------------------------------------------
# Full graph (paginated by account count)
# ---------------------------------------------------------------------------
@router.get("/full", response_model=GraphData)
async def full_graph(
    limit_accounts: int = Query(default=100, ge=10, le=500),
    suspect_only: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """
    Overview graph — all accounts and their internal connections.
    limit_accounts caps node count for frontend performance.
    suspect_only=True shows only high-risk accounts and their neighbours.
    """
    q = select(Account.account_id).order_by(Account.risk_score.desc())
    if suspect_only:
        q = q.where(Account.is_suspect == True)
    q = q.limit(limit_accounts)

    account_ids = (await db.execute(q)).scalars().all()
    return await _build_graph_from_account_ids(list(account_ids), db)
