"""
SQLAlchemy 2.0 ORM Models

Tables:
  accounts            - Bank accounts (legitimate + fraud suspects)
  transactions        - All financial transactions post-cleaning
  fraud_rings         - Detected fraud network metadata
  fraud_ring_members  - Account <-> Ring mapping (many-to-many)
  investigations      - Investigator-created case files
  risk_score_history  - ML risk score log (every scoring run recorded)
  reports             - Generated investigation reports / evidence docs
  evidence_items      - Individual evidence entries within an investigation

Design notes:
  - All PKs are UUIDs (uuid4) — avoids sequential ID guessing in the API
  - account_id uses the synthetic ACC000XXX string as PK (matches Phase 5)
  - Timestamps always stored as UTC
  - is_suspect / risk_score are denormalised onto accounts for fast querying
    and updated by the ML scoring pipeline (Phase 10)
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint, Index,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def utcnow():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 1. Accounts
# ---------------------------------------------------------------------------
class Account(Base):
    __tablename__ = "accounts"

    account_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    holder_name: Mapped[str] = mapped_column(String(200), nullable=False)
    bank_name: Mapped[str] = mapped_column(String(100), nullable=False)
    bank_code: Mapped[Optional[str]] = mapped_column(String(10))
    ifsc: Mapped[Optional[str]] = mapped_column(String(15))
    account_number: Mapped[Optional[str]] = mapped_column(String(20))
    persona: Mapped[Optional[str]] = mapped_column(String(30))
    opening_balance: Mapped[float] = mapped_column(Float, default=0.0)

    # ML / investigation fields (updated by Phase 10 scoring pipeline)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_suspect: Mapped[bool] = mapped_column(Boolean, default=False)
    fraud_role: Mapped[Optional[str]] = mapped_column(String(30))
    fraud_ring_id: Mapped[Optional[str]] = mapped_column(String(20))
    last_scored_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Source tracking
    source_file: Mapped[Optional[str]] = mapped_column(String(255))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Relationships
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="account", lazy="select"
    )
    risk_scores: Mapped[list["RiskScoreHistory"]] = relationship(
        back_populates="account", lazy="select"
    )
    ring_memberships: Mapped[list["FraudRingMember"]] = relationship(
        back_populates="account", lazy="select"
    )

    __table_args__ = (
        Index("ix_accounts_risk_score", "risk_score"),
        Index("ix_accounts_is_suspect", "is_suspect"),
        Index("ix_accounts_bank_name", "bank_name"),
    )

    def __repr__(self):
        return f"<Account {self.account_id} {self.holder_name}>"


# ---------------------------------------------------------------------------
# 2. Transactions
# ---------------------------------------------------------------------------
class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[Optional[str]] = mapped_column(String(30), index=True)
    account_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("accounts.account_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # Core financial fields
    date: Mapped[Optional[str]] = mapped_column(String(10))        # YYYY-MM-DD
    time: Mapped[Optional[str]] = mapped_column(String(8))         # HH:MM:SS
    narration: Mapped[Optional[str]] = mapped_column(Text)
    channel: Mapped[Optional[str]] = mapped_column(String(20))
    debit: Mapped[float] = mapped_column(Float, default=0.0)
    credit: Mapped[float] = mapped_column(Float, default=0.0)
    balance: Mapped[float] = mapped_column(Float, default=0.0)
    utr_ref: Mapped[Optional[str]] = mapped_column(String(50), index=True)

    # Counterparty (populated where available — critical for graph edges)
    counterparty_account_id: Mapped[Optional[str]] = mapped_column(
        String(20), index=True
    )
    counterparty_name: Mapped[Optional[str]] = mapped_column(String(200))

    # Source / ingestion metadata
    source_file: Mapped[Optional[str]] = mapped_column(String(255))
    source_format: Mapped[Optional[str]] = mapped_column(String(10))
    ingestion_warnings: Mapped[Optional[str]] = mapped_column(Text)

    # Phase 7 cleaning audit flags
    clean_flags: Mapped[Optional[str]] = mapped_column(Text)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    is_balance_breach: Mapped[bool] = mapped_column(Boolean, default=False)
    is_high_value_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    is_ocr_row: Mapped[bool] = mapped_column(Boolean, default=False)

    # Phase 10 ML risk output
    anomaly_score: Mapped[Optional[float]] = mapped_column(Float)
    xgb_fraud_prob: Mapped[Optional[float]] = mapped_column(Float)
    final_risk_score: Mapped[Optional[float]] = mapped_column(Float)
    fraud_pattern_predicted: Mapped[Optional[str]] = mapped_column(String(50))

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Relationships
    account: Mapped["Account"] = relationship(back_populates="transactions")

    __table_args__ = (
        Index("ix_transactions_date", "date"),
        Index("ix_transactions_channel", "channel"),
        Index("ix_transactions_final_risk", "final_risk_score"),
        Index("ix_transactions_account_date", "account_id", "date"),
    )

    def __repr__(self):
        return f"<Transaction {self.transaction_id} {self.date} ₹{self.debit or self.credit}>"


# ---------------------------------------------------------------------------
# 3. Fraud Rings
# ---------------------------------------------------------------------------
class FraudRing(Base):
    __tablename__ = "fraud_rings"

    ring_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    typology: Mapped[str] = mapped_column(String(50), nullable=False)
    # detected | confirmed | dismissed
    status: Mapped[str] = mapped_column(String(20), default="detected")
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)

    # Summary stats (denormalised for dashboard display)
    total_accounts: Mapped[int] = mapped_column(Integer, default=0)
    total_amount_moved: Mapped[float] = mapped_column(Float, default=0.0)
    date_first_txn: Mapped[Optional[str]] = mapped_column(String(10))
    date_last_txn: Mapped[Optional[str]] = mapped_column(String(10))

    # Narrative generated by LLM (Phase 11)
    llm_summary: Mapped[Optional[str]] = mapped_column(Text)

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    # Relationships
    members: Mapped[list["FraudRingMember"]] = relationship(
        back_populates="ring", lazy="select"
    )

    def __repr__(self):
        return f"<FraudRing {self.ring_id} {self.typology}>"


# ---------------------------------------------------------------------------
# 4. Fraud Ring Members (Account <-> Ring)
# ---------------------------------------------------------------------------
class FraudRingMember(Base):
    __tablename__ = "fraud_ring_members"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ring_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("fraud_rings.ring_id", ondelete="CASCADE"),
        nullable=False,
    )
    account_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("accounts.account_id", ondelete="CASCADE"),
        nullable=False,
    )
    role_in_ring: Mapped[Optional[str]] = mapped_column(String(30))
    # hub | mule | collector | layering_node | beneficiary
    amount_handled: Mapped[float] = mapped_column(Float, default=0.0)

    # Relationships
    ring: Mapped["FraudRing"] = relationship(back_populates="members")
    account: Mapped["Account"] = relationship(back_populates="ring_memberships")

    __table_args__ = (
        UniqueConstraint("ring_id", "account_id", name="uq_ring_account"),
    )


# ---------------------------------------------------------------------------
# 5. Investigations
# ---------------------------------------------------------------------------
class Investigation(Base):
    __tablename__ = "investigations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Seed account the investigator started from
    seed_account_id: Mapped[Optional[str]] = mapped_column(
        String(20), ForeignKey("accounts.account_id", ondelete="SET NULL")
    )

    # open | active | closed | escalated
    status: Mapped[str] = mapped_column(String(20), default="open")
    created_by: Mapped[str] = mapped_column(String(100), default="investigator")

    # Linked rings (stored as JSON array of ring_ids)
    linked_ring_ids: Mapped[Optional[list]] = mapped_column(JSONB, default=list)

    # LLM-generated case narrative (Phase 11)
    case_narrative: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    # Relationships
    reports: Mapped[list["Report"]] = relationship(
        back_populates="investigation", lazy="select"
    )
    evidence_items: Mapped[list["EvidenceItem"]] = relationship(
        back_populates="investigation", lazy="select"
    )

    __table_args__ = (
        Index("ix_investigations_status", "status"),
        Index("ix_investigations_created_at", "created_at"),
    )


# ---------------------------------------------------------------------------
# 6. Risk Score History
# ---------------------------------------------------------------------------
class RiskScoreHistory(Base):
    __tablename__ = "risk_score_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    account_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("accounts.account_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # Score breakdown from each ML layer
    isolation_forest_score: Mapped[Optional[float]] = mapped_column(Float)
    xgb_fraud_probability: Mapped[Optional[float]] = mapped_column(Float)
    graph_centrality_score: Mapped[Optional[float]] = mapped_column(Float)
    meta_learner_score: Mapped[Optional[float]] = mapped_column(Float)
    final_risk_score: Mapped[float] = mapped_column(Float, nullable=False)

    model_version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    # Features snapshot for explainability
    feature_snapshot: Mapped[Optional[dict]] = mapped_column(JSONB)

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Relationships
    account: Mapped["Account"] = relationship(back_populates="risk_scores")

    __table_args__ = (
        Index("ix_risk_score_history_computed_at", "computed_at"),
    )


# ---------------------------------------------------------------------------
# 7. Reports
# ---------------------------------------------------------------------------
class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    # pdf | excel | json
    format: Mapped[str] = mapped_column(String(10), default="pdf")
    file_path: Mapped[Optional[str]] = mapped_column(String(500))
    # Full report content as JSON for API delivery
    content: Mapped[Optional[dict]] = mapped_column(JSONB)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Relationships
    investigation: Mapped["Investigation"] = relationship(back_populates="reports")


# ---------------------------------------------------------------------------
# 8. Evidence Items
# ---------------------------------------------------------------------------
class EvidenceItem(Base):
    __tablename__ = "evidence_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # transaction | account | ring | external_doc
    evidence_type: Mapped[str] = mapped_column(String(30), nullable=False)
    reference_id: Mapped[Optional[str]] = mapped_column(String(100))
    description: Mapped[Optional[str]] = mapped_column(Text)
    amount_involved: Mapped[Optional[float]] = mapped_column(Float)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Relationships
    investigation: Mapped["Investigation"] = relationship(
        back_populates="evidence_items"
    )
