"""
Router: /api/v1/dashboard + /api/v1/rings

Dashboard:
  GET /dashboard/stats    Single-call summary for the overview page

Fraud Rings:
  GET /rings/             List all detected rings
  GET /rings/{ring_id}    Ring detail with member list
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_db
from models import Account, Transaction, FraudRing, FraudRingMember, Investigation
from schemas import FraudRingOut, FraudRingMemberOut, DashboardStats

dashboard_router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
rings_router = APIRouter(prefix="/rings", tags=["Fraud Rings"])


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@dashboard_router.get("/stats", response_model=DashboardStats)
async def dashboard_stats(db: AsyncSession = Depends(get_db)):
    """
    Single API call that powers the entire dashboard overview panel.
    Computed from live DB — always reflects the latest ingested data.
    """
    total_accounts    = (await db.execute(select(func.count(Account.account_id)))).scalar_one()
    suspect_accounts  = (await db.execute(
        select(func.count(Account.account_id)).where(Account.is_suspect == True)
    )).scalar_one()
    total_txns        = (await db.execute(select(func.count(Transaction.id)))).scalar_one()
    flagged_txns      = (await db.execute(
        select(func.count(Transaction.id)).where(Transaction.is_high_value_flag == True)
    )).scalar_one()
    fraud_rings       = (await db.execute(select(func.count(FraudRing.ring_id)))).scalar_one()
    open_inv          = (await db.execute(
        select(func.count(Investigation.id)).where(Investigation.status == "open")
    )).scalar_one()

    amount_at_risk = (await db.execute(
        select(func.sum(Transaction.debit + Transaction.credit)).where(
            Transaction.is_high_value_flag == True
        )
    )).scalar_one() or 0.0

    banks = (await db.execute(
        select(Account.bank_name).distinct()
    )).scalars().all()

    return DashboardStats(
        total_accounts=total_accounts,
        suspect_accounts=suspect_accounts,
        total_transactions=total_txns,
        flagged_transactions=flagged_txns,
        fraud_rings_detected=fraud_rings,
        open_investigations=open_inv,
        total_amount_at_risk=float(amount_at_risk),
        banks_covered=[b for b in banks if b],
    )


# ---------------------------------------------------------------------------
# Fraud Rings
# ---------------------------------------------------------------------------
@rings_router.get("/", response_model=list[FraudRingOut])
async def list_rings(db: AsyncSession = Depends(get_db)):
    rings = (await db.execute(
        select(FraudRing).order_by(FraudRing.total_amount_moved.desc())
    )).scalars().all()
    return [FraudRingOut.model_validate(r) for r in rings]


@rings_router.get("/{ring_id}", response_model=FraudRingOut)
async def get_ring(ring_id: str, db: AsyncSession = Depends(get_db)):
    ring = await db.get(FraudRing, ring_id)
    if not ring:
        raise HTTPException(status_code=404, detail="Ring not found")
    return FraudRingOut.model_validate(ring)


@rings_router.get("/{ring_id}/members", response_model=list[FraudRingMemberOut])
async def get_ring_members(ring_id: str, db: AsyncSession = Depends(get_db)):
    ring = await db.get(FraudRing, ring_id)
    if not ring:
        raise HTTPException(status_code=404, detail="Ring not found")

    q = (
        select(FraudRingMember, Account)
        .join(Account, FraudRingMember.account_id == Account.account_id)
        .where(FraudRingMember.ring_id == ring_id)
    )
    rows = (await db.execute(q)).all()

    return [
        FraudRingMemberOut(
            account_id=member.account_id,
            holder_name=account.holder_name,
            bank_name=account.bank_name,
            role_in_ring=member.role_in_ring,
            amount_handled=member.amount_handled,
            risk_score=account.risk_score,
        )
        for member, account in rows
    ]
