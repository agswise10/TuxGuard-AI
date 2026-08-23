"""
Conversation history storage.

Thin data-access layer over the `conversation_messages` SQLite table.
Kept separate from the AI assistant orchestration logic so storage concerns
don't leak into prompt-building / LLM-calling code.
"""

from app.config import get_settings
from app.database import get_connection
from app.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


def add_message(
    session_id: str,
    role: str,
    content: str,
    intent: str | None = None,
    confidence_score: float | None = None,
) -> None:
    """Persist a single conversation turn (user or assistant)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO conversation_messages (session_id, role, content, intent, confidence_score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, intent, confidence_score),
        )
        conn.commit()


def get_recent_messages(session_id: str, limit: int) -> list[dict]:
    """Return the most recent `limit` messages for a session, oldest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, intent, confidence_score, created_at
            FROM conversation_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def get_all_messages(session_id: str) -> list[dict]:
    """Return the full conversation history for a session, oldest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, intent, confidence_score, created_at
            FROM conversation_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def clear_session(session_id: str) -> int:
    """Delete all stored messages for a session. Returns rows deleted."""
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM conversation_messages WHERE session_id = ?", (session_id,)
        )
        conn.commit()
        return cursor.rowcount


def prune_session(session_id: str, keep_last: int) -> None:
    """Keep only the most recent `keep_last` messages for a session."""
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM conversation_messages
            WHERE session_id = ? AND id NOT IN (
                SELECT id FROM conversation_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (session_id, session_id, keep_last),
        )
        conn.commit()
