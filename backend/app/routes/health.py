"""
Health check routes.

Provides a simple endpoint to verify the API is running and that the
SQLite database is reachable.
"""

import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import get_settings
from app.database import get_connection
from app.logger import get_logger

router = APIRouter(prefix="/api/health", tags=["Health"])
settings = get_settings()
logger = get_logger(__name__)


@router.get("")
def health_check():
    """Return API status and a basic DB connectivity check."""
    db_status = "ok"
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
    except sqlite3.Error as exc:
        logger.error("Database health check failed: %s", exc)
        db_status = "error"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "database": db_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
