"""
SQLite database connection handling.

Kept intentionally simple for the PoC: a single helper that returns a
sqlite3 connection, plus a startup routine that creates the DB file / a
sample table if it doesn't exist yet.
"""

import sqlite3
from contextlib import contextmanager

from app.config import get_settings
from app.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


def get_connection() -> sqlite3.Connection:
    """Create a new SQLite connection with row access by column name."""
    conn = sqlite3.connect(settings.sqlite_db_path)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """FastAPI dependency-friendly context manager for a DB connection."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the SQLite database. Creates a minimal table so the
    health endpoint can prove connectivity end to end."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checked_at TEXT NOT NULL DEFAULT (datetime('now')),
                status TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                intent TEXT,
                confidence_score REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_session "
            "ON conversation_messages (session_id, created_at)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS command_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                description TEXT,
                command TEXT NOT NULL,
                source TEXT NOT NULL CHECK (source IN ('ai_generated', 'user_provided')),
                explanation TEXT,
                risk_level TEXT NOT NULL DEFAULT 'safe',
                blocked INTEGER NOT NULL DEFAULT 0,
                matched_rules TEXT,
                confidence_score REAL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'blocked', 'rejected', 'executed')),
                stdout TEXT,
                stderr TEXT,
                exit_code INTEGER,
                duration_seconds REAL,
                timed_out INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                executed_at TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_command_executions_session "
            "ON command_executions (session_id, created_at)"
        )

        # --- Sprint 7: Voice Assistant ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                transcript TEXT NOT NULL,
                ai_response_text TEXT,
                intent TEXT,
                confidence_score REAL,
                stt_engine TEXT,
                tts_engine TEXT,
                audio_duration_seconds REAL,
                status TEXT NOT NULL DEFAULT 'completed'
                    CHECK (status IN ('completed', 'transcribe_failed', 'assistant_failed', 'speak_failed')),
                error TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_voice_history_session "
            "ON voice_history (session_id, created_at)"
        )

        conn.commit()
    logger.info("Database initialized at %s", settings.sqlite_db_path)
