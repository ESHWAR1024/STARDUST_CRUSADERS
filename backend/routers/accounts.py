"""
Router: /api/v1/accounts

Endpoints:
  GET  /                    List all accounts with risk scores + pagination
  GET  /{account_id}        Full account detail
  GET  /{account_id}/transactions   Transactions for one account
  GET  /{account_id}/risk-history   ML score history
  GET  /{account_id}/counterparties All accounts this one transacted with
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_db
from models import Account, Transaction, RiskScoreHistory
from schemas import (
    AccountOut, AccountSummary, TransactionOut,
    RiskScoreOut, PaginatedResponse,
)

router = APIRouter(prefix="/accounts", tags=["Accounts"])


# ---------------------------------------------------------------------------
# List accounts
# ---------------------------------------------------------------------------
@router.get("/", response_model=PaginatedResponse)
async def list_accounts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    bank_name: Optional[str] = None,
    is_suspect: Optional[bool] = None,
    min_risk_score: Optional[float] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List accounts sorted by risk_score descending.
    Investigators start here — highest-risk accounts surface first.
    """
    q = select(Account)

    if bank_name:
        q = q.where(Account.bank_name.ilike(f"%{bank_name}%"))
    if is_suspect is not None:
        q = q.where(Account.is_suspect == is_suspect)
    if min_risk_score is not None:
        q = q.where(Account.risk_score >= min_risk_score)
    if search:
        q = q.where(
            or_(
                Account.holder_name.ilike(f"%{search}%"),
                Account.account_id.ilike(f"%{search}%"),
                Account.account_number.ilike(f"%{search}%"),
            )
        )

    # Count total
    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar_one()

    # Paginate, sort by risk desc
    q = (
        q.order_by(Account.risk_score.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    accounts = (await db.execute(q)).scalars().all()

    return PaginatedResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[AccountOut.model_validate(a) for a in accounts],
    )


# ---------------------------------------------------------------------------
# Account detail
# ---------------------------------------------------------------------------
@router.get("/{account_id}", response_model=AccountOut)
async def get_account(account_id: str, db: AsyncSession = Depends(get_db)):
    account = await db.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    return AccountOut.model_validate(account)


# ---------------------------------------------------------------------------
# Transactions for one account
# ---------------------------------------------------------------------------
@router.get("/{account_id}/transactions", response_model=PaginatedResponse)
async def get_account_transactions(
    account_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    flagged_only: bool = False,
    db: AsyncSession = Depends(get_db),
):
    account = await db.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    q = select(Transaction).where(Transaction.account_id == account_id)

    if date_from:
        q = q.where(Transaction.date >= date_from)
    if date_to:
        q = q.where(Transaction.date <= date_to)
    if flagged_only:
        q = q.where(
            or_(
                Transaction.is_high_value_flag == True,
                Transaction.is_balance_breach == True,
                Transaction.final_risk_score >= 0.7,
            )
        )

    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar_one()

    q = q.order_by(Transaction.date, Transaction.time).offset(
        (page - 1) * page_size
    ).limit(page_size)

    txns = (await db.execute(q)).scalars().all()

    return PaginatedResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[TransactionOut.model_validate(t) for t in txns],
    )


# ---------------------------------------------------------------------------
# Risk score history
# ---------------------------------------------------------------------------
@router.get("/{account_id}/risk-history", response_model=list[RiskScoreOut])
async def get_risk_history(
    account_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    account = await db.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    q = (
        select(RiskScoreHistory)
        .where(RiskScoreHistory.account_id == account_id)
        .order_by(RiskScoreHistory.computed_at.desc())
        .limit(limit)
    )
    scores = (await db.execute(q)).scalars().all()
    return [RiskScoreOut.model_validate(s) for s in scores]


# ---------------------------------------------------------------------------
# Counterparties — all accounts this one sent/received money with
# (critical for fund-tracing: "who did this account deal with?")
# ---------------------------------------------------------------------------
@router.get("/{account_id}/counterparties", response_model=list[AccountSummary])
async def get_counterparties(
    account_id: str,
    db: AsyncSession = Depends(get_db),
):
    account = await db.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    # Find all internal counterparty account IDs
    q = (
        select(Transaction.counterparty_account_id)
        .where(
            Transaction.account_id == account_id,
            Transaction.counterparty_account_id.isnot(None),
        )
        .distinct()
    )
    cp_ids = (await db.execute(q)).scalars().all()

    if not cp_ids:
        return []

    cp_q = select(Account).where(Account.account_id.in_(cp_ids))
    counterparties = (await db.execute(cp_q)).scalars().all()
    return [AccountSummary.model_validate(a) for a in counterparties]
