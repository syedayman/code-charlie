"""
PostgreSQL LangGraph checkpointer for the Code Charlie agent.

Singleton per process. Connection pool sized via core.config.settings.
Reuses the same `checkpoints` / `checkpoint_writes` tables that KARR-AI's
PostgresSaver creates — thread_id UUIDs prevent collisions.
"""

from __future__ import annotations

import atexit
import logging
from typing import Optional

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection, errors
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from core.config import settings

logger = logging.getLogger(__name__)

_checkpointer: Optional[PostgresSaver] = None
_db_pool: Optional[ConnectionPool] = None


def _close_pool_on_exit() -> None:
    """Close the psycopg pool cleanly on process exit so background worker
    threads stop without the 5-second timeout warning."""
    global _db_pool
    if _db_pool is None:
        return
    try:
        _db_pool.close(timeout=2.0)
    except Exception:
        pass


atexit.register(_close_pool_on_exit)


def _ensure_sslmode(db_url: str) -> str:
    """Ensure sslmode=require is set for Supabase connections."""
    if "sslmode=" in db_url:
        return db_url
    separator = "&" if "?" in db_url else "?"
    return f"{db_url}{separator}sslmode=require"


def _pool_check(conn: Connection) -> None:
    """Health check for pooled connections (prevents stale pooler sockets)."""
    if conn.closed:
        raise errors.OperationalError("pooled connection is closed")
    conn.execute("SELECT 1")


def get_checkpointer() -> PostgresSaver:
    """Get or create the PostgreSQL checkpointer singleton."""
    global _checkpointer, _db_pool

    if _checkpointer is None or (_db_pool is not None and _db_pool.closed):
        db_url = _ensure_sslmode(settings.DATABASE_URL)

        logger.info("Initializing PostgreSQL checkpointer...")

        try:
            _db_pool = ConnectionPool(
                conninfo=db_url,
                kwargs={
                    "autocommit": True,
                    "row_factory": dict_row,
                    "prepare_threshold": 0,
                },
                min_size=settings.CHECKPOINT_POOL_MIN_SIZE,
                max_size=settings.CHECKPOINT_POOL_MAX_SIZE,
                timeout=settings.CHECKPOINT_POOL_TIMEOUT_SECONDS,
                max_idle=settings.CHECKPOINT_POOL_MAX_IDLE_SECONDS,
                max_lifetime=settings.CHECKPOINT_POOL_MAX_LIFETIME_SECONDS,
                check=_pool_check,
                open=True,
            )
            _db_pool.wait(timeout=settings.CHECKPOINT_POOL_TIMEOUT_SECONDS)

            _checkpointer = PostgresSaver(_db_pool)
            _checkpointer.setup()
        except Exception:
            logger.exception("PostgreSQL checkpointer initialization failed")
            if _db_pool is not None:
                try:
                    _db_pool.close(timeout=2.0)
                except Exception:
                    pass
            _db_pool = None
            _checkpointer = None
            raise

        logger.info("PostgreSQL checkpointer initialized successfully")

    return _checkpointer
