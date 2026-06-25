"""
Shared FastAPI Dependencies

get_db() is injected into every route handler that needs a database
session via FastAPI's Depends() mechanism.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from database import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session per request.
    Session is always closed after the request, even on exceptions.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
