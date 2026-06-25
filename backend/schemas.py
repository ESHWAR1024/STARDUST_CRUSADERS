"""
Pydantic v2 Schemas — Request / Response models for all API endpoints.

Kept deliberately flat (no deep nesting) so the React frontend can
consume them without complex unwrapping. Every list endpoint returns
a paginated envelope: { total, page, page_size, items: [...] }
"""

import uuid
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
class PaginatedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[Any]


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------
class AccountBase(BaseModel):
    account_id: str
    holder_name: str
    bank_name: str
    bank_code: Optional[str] = None
    ifsc: Optional[str] = None
    account_number: Optional[str] = None
    persona: Optional[str] = None
    opening_balance: float = 0.0


class AccountOut(AccountBase):
    risk_score: float = 0.0
    is_suspect: bool = False
    fraud_role: Optional[str] = None
    fraud_ring_id: Optional[str] = None
    last_scored_at: Optional[datetime] = None
    ingested_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AccountSummary(BaseModel):
    """Lightweight version for list endpoints."""
    account_id: str
    holder_name: str
    bank_name: str
    risk_score: float
    is_suspect: bool
    fraud_role: Optional[str] = None
    total_transactions: Optional[int] = None
    total_debit: Optional[float] = None
    total_credit: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------
class TransactionOut(BaseModel):
    id: uuid.UUID
    transaction_id: Optional[str]
    account_id: str
    date: Optional[str]
    time: Optional[str]
    narration: Optional[str]
    channel: Optional[str]
    debit: float
    credit: float
    balance: float
    utr_ref: Optional[str]
    counterparty_account_id: Optional[str]
    counterparty_name: Optional[str]
    source_format: Optional[str]
    # Flags
    is_duplicate: bool
    is_balance_breach: bool
    is_high_value_flag: bool
    is_ocr_row: bool
    clean_flags: Optional[str]
    # ML scores (None until Phase 10 runs)
    anomaly_score: Optional[float]
    xgb_fraud_prob: Optional[float]
    final_risk_score: Optional[float]
    fraud_pattern_predicted: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class TransactionFilter(BaseModel):
    """Query params for transaction search."""
    account_id: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    channel: Optional[str] = None
    is_suspect_only: bool = False
    counterparty_account_id: Optional[str] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)


# ---------------------------------------------------------------------------
# Fraud Rings
# ---------------------------------------------------------------------------
class FraudRingOut(BaseModel):
    ring_id: str
    typology: str
    status: str
    confidence_score: Optional[float]
    total_accounts: int
    total_amount_moved: float
    date_first_txn: Optional[str]
    date_last_txn: Optional[str]
    llm_summary: Optional[str]
    detected_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FraudRingMemberOut(BaseModel):
    account_id: str
    holder_name: str
    bank_name: str
    role_in_ring: Optional[str]
    amount_handled: float
    risk_score: float

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Graph (Cytoscape.js format)
# ---------------------------------------------------------------------------
class GraphNode(BaseModel):
    id: str                          # account_id
    label: str                       # holder_name
    bank: str
    risk_score: float
    is_suspect: bool
    fraud_role: Optional[str]
    total_debit: float
    total_credit: float
    txn_count: int


class GraphEdge(BaseModel):
    id: str                          # utr_ref or generated
    source: str                      # account_id
    target: str                      # counterparty_account_id
    amount: float
    channel: str
    date: Optional[str]
    is_fraud_flagged: bool


class GraphData(BaseModel):
    """Cytoscape.js-ready graph payload."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    total_nodes: int
    total_edges: int
    seed_account_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Investigations
# ---------------------------------------------------------------------------
class InvestigationCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = None
    seed_account_id: Optional[str] = None


class InvestigationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    case_narrative: Optional[str] = None
    linked_ring_ids: Optional[list[str]] = None


class InvestigationOut(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    seed_account_id: Optional[str]
    status: str
    created_by: str
    linked_ring_ids: Optional[list]
    case_narrative: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Risk Scores
# ---------------------------------------------------------------------------
class RiskScoreOut(BaseModel):
    id: uuid.UUID
    account_id: str
    isolation_forest_score: Optional[float]
    xgb_fraud_probability: Optional[float]
    graph_centrality_score: Optional[float]
    meta_learner_score: Optional[float]
    final_risk_score: float
    model_version: str
    feature_snapshot: Optional[dict]
    computed_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# File Upload
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    upload_id: str
    files_received: int
    files_ingested: int
    rows_parsed: int
    rows_after_clean: int
    banks_detected: list[str]
    warnings: list[str]
    status: str   # "success" | "partial" | "failed"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
class ReportCreate(BaseModel):
    investigation_id: uuid.UUID
    title: str
    format: str = "pdf"   # pdf | excel | json


class ReportOut(BaseModel):
    id: uuid.UUID
    investigation_id: uuid.UUID
    title: str
    format: str
    file_path: Optional[str]
    generated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Dashboard Stats
# ---------------------------------------------------------------------------
class DashboardStats(BaseModel):
    total_accounts: int
    suspect_accounts: int
    total_transactions: int
    flagged_transactions: int
    fraud_rings_detected: int
    open_investigations: int
    total_amount_at_risk: float
    banks_covered: list[str]
