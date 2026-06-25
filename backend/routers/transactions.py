"""
Router: /api/v1/transactions

Endpoints:
  GET  /          Search/filter transactions across ALL accounts
  GET  /{id}      Single transaction detail
  GET  /stats     Aggregate stats (channel breakdown, daily volume, etc.)
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_db
from models import Transaction
from schemas import TransactionOut, PaginatedResponse

router = APIRouter(prefix="/transactions", tags=["Transactions"])


@router.get("/", response_model=PaginatedResponse)
async def search_transactions(
    account_id: Optional[str] = None,
    counterparty_account_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    channel: Optional[str] = None,
    narration_contains: Optional[str] = None,
    flagged_only: bool = False,
    suspect_only: bool = False,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    Flexible transaction search. The investigator uses this to answer
    questions like:
      - "Show me all ATM withdrawals above ₹50K in March"
      - "Find all transfers between ACC000017 and ACC000042"
      - "Show me every transaction flagged by the ML layer"
    """
    q = select(Transaction)
    filters = []

    if account_id:
        filters.append(Transaction.account_id == account_id)
    if counterparty_account_id:
        filters.append(
            Transaction.counterparty_account_id == counterparty_account_id
        )
    if date_from:
        filters.append(Transaction.date >= date_from)
    if date_to:
        filters.append(Transaction.date <= date_to)
    if channel:
        filters.append(Transaction.channel == channel.upper())
    if narration_contains:
        filters.append(Transaction.narration.ilike(f"%{narration_contains}%"))
    if min_amount is not None:
        filters.append(
            or_(Transaction.debit >= min_amount, Transaction.credit >= min_amount)
        )
    if max_amount is not None:
        filters.append(
            or_(Transaction.debit <= max_amount, Transaction.credit <= max_amount)
        )
    if flagged_only:
        filters.append(
            or_(
                Transaction.is_high_value_flag == True,
                Transaction.is_balance_breach == True,
                Transaction.is_duplicate == True,
            )
        )
    if suspect_only:
        filters.append(Transaction.final_risk_score >= 0.7)

    if filters:
        q = q.where(and_(*filters))

    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar_one()

    q = (
        q.order_by(Transaction.date.desc(), Transaction.time.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    txns = (await db.execute(q)).scalars().all()

    return PaginatedResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[TransactionOut.model_validate(t) for t in txns],
    )


@router.get("/stats/summary")
async def transaction_stats(db: AsyncSession = Depends(get_db)):
    """
    Aggregate stats for the dashboard overview panel.
    Returns channel breakdown, daily volume totals, flag counts.
    """
    channel_q = select(
        Transaction.channel,
        func.count().label("count"),
        func.sum(Transaction.debit + Transaction.credit).label("total_volume"),
    ).group_by(Transaction.channel).order_by(func.count().desc())

    channel_rows = (await db.execute(channel_q)).all()

    flag_q = select(
        func.count().label("total"),
        func.sum(func.cast(Transaction.is_high_value_flag, func.Integer())).label("high_value"),
        func.sum(func.cast(Transaction.is_balance_breach, func.Integer())).label("balance_breach"),
        func.sum(func.cast(Transaction.is_duplicate, func.Integer())).label("duplicates"),
        func.sum(func.cast(Transaction.is_ocr_row, func.Integer())).label("ocr_rows"),
    )
    flag_row = (await db.execute(flag_q)).one()

    return {
        "channels": [
            {
                "channel": r.channel,
                "count": r.count,
                "total_volume": float(r.total_volume or 0),
            }
            for r in channel_rows
        ],
        "flags": {
            "total_transactions": flag_row.total,
            "high_value_flagged": flag_row.high_value or 0,
            "balance_breach": flag_row.balance_breach or 0,
            "duplicates": flag_row.duplicates or 0,
            "ocr_rows": flag_row.ocr_rows or 0,
        },
    }


@router.get("/{txn_id}", response_model=TransactionOut)
async def get_transaction(txn_id: str, db: AsyncSession = Depends(get_db)):
    import uuid as _uuid
    try:
        uid = _uuid.UUID(txn_id)
    except ValueError:
        # Try by transaction_id string
        q = select(Transaction).where(Transaction.transaction_id == txn_id)
        txn = (await db.execute(q)).scalar_one_or_none()
    else:
        txn = await db.get(Transaction, uid)

    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return TransactionOut.model_validate(txn)
