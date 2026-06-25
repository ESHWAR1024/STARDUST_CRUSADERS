"""
Phase 8 — FastAPI Backend Entry Point

/api/v1/ route tree:
  /upload           File ingestion (Phase 6+7 pipeline trigger)
  /accounts         Account management + fund tracing
  /transactions     Transaction search + stats
  /graph            Cytoscape.js graph payloads
  /investigations   Case file management
  /dashboard        Overview stats
  /rings            Fraud ring registry
  /health           Health check
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from database import create_tables

from routers.upload import router as upload_router
from routers.accounts import router as accounts_router
from routers.transactions import router as transactions_router
from routers.graph import router as graph_router
from routers.investigations import router as investigations_router
from routers.dashboard import dashboard_router, rings_router


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    await create_tables()   # no-op if tables already exist
    print(f"✓ {settings.APP_NAME} started")
    print(f"✓ Docs: http://localhost:8000/docs")
    yield
    # Shutdown (nothing to teardown for demo)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Financial Intelligence Platform for Law Enforcement — CIDECODE 2026",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow React dev server
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
PREFIX = "/api/v1"

app.include_router(upload_router,         prefix=PREFIX)
app.include_router(accounts_router,       prefix=PREFIX)
app.include_router(transactions_router,   prefix=PREFIX)
app.include_router(graph_router,          prefix=PREFIX)
app.include_router(investigations_router, prefix=PREFIX)
app.include_router(dashboard_router,      prefix=PREFIX)
app.include_router(rings_router,          prefix=PREFIX)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
