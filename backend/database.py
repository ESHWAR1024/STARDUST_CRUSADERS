"""
Database Setup — SQLAlchemy 2.0 async engine

Uses asyncpg driver for PostgreSQL.
All route handlers get an AsyncSession via the get_db dependency.

Connection pooling:
  pool_size=10, max_overflow=20 — handles concurrent investigator sessions
  pool_pre_ping=True            — drops stale connections silently
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from config import settings


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=settings.DEBUG,       # log all SQL in dev
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,    # avoids lazy-load errors after commit
    autocommit=False,
    autoflush=False,
)

# ---------------------------------------------------------------------------
# Declarative base — all models inherit from this
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Utility: create all tables (used at startup in dev/demo mode)
# ---------------------------------------------------------------------------
async def create_tables():
    """Drop-and-recreate all tables. Only call in dev/demo mode."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
