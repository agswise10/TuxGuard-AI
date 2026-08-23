"""
Command Execution storage - the audit trail for Safe Command Execution.

Every command that passes through the pipeline is recorded here at every
stage: when it's first generated/previewed (status="pending"), and again
when it's confirmed+executed, rejected, or blocked. Nothing in this module
executes anything - it is pure persistence, kept separate from
`command_console.py` (orchestration) and `command_executor.py` (the only
place that actually runs a command).
"""

import json

from app.database import get_connection
from app.logger import get_logger

logger = get_logger(__name__)


def save_pending(
    session_id: str,
    description: str | None,
    command: str,
    source: str,
    explanation: str,
    risk_level: str,
    blocked: bool,
    matched_rules: list[dict],
    confidence_score: float | None,
) -> int:
    """Persist a newly generated/previewed command awaiting confirmation.

    `status` starts as "blocked" if the safety analyzer refused it
    outright, otherwise "pending" (awaiting user confirmation).
    """
    status = "blocked" if blocked else "pending"
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO command_executions
                (session_id, description, command, source, explanation, risk_level,
                 blocked, matched_rules, confidence_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                description,
                command,
                source,
                explanation,
                risk_level,
                1 if blocked else 0,
                json.dumps(matched_rules),
                confidence_score,
                status,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def mark_executed(
    execution_id: int,
    command: str,
    risk_level: str,
    blocked: bool,
    matched_rules: list[dict],
    stdout: str,
    stderr: str,
    exit_code: int | None,
    duration_seconds: float,
    timed_out: bool,
) -> None:
    """Record the final result of actually running a command.

    Also re-persists `command`/`risk_level`/`matched_rules` in case the
    user edited the command between preview and confirmation - the audit
    trail should reflect what was ACTUALLY executed, not just what was
    originally previewed.
    """
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE command_executions
            SET command = ?, risk_level = ?, blocked = ?, matched_rules = ?,
                status = 'executed', stdout = ?, stderr = ?, exit_code = ?,
                duration_seconds = ?, timed_out = ?, executed_at = datetime('now')
            WHERE id = ?
            """,
            (
                command,
                risk_level,
                1 if blocked else 0,
                json.dumps(matched_rules),
                stdout,
                stderr,
                exit_code,
                duration_seconds,
                1 if timed_out else 0,
                execution_id,
            ),
        )
        conn.commit()


def mark_status(execution_id: int, status: str, command: str | None = None) -> None:
    """Update just the status (e.g. to 'rejected' or 'blocked'), optionally
    also updating the command text if the user edited it before rejecting."""
    with get_connection() as conn:
        if command is not None:
            conn.execute(
                "UPDATE command_executions SET status = ?, command = ? WHERE id = ?",
                (status, command, execution_id),
            )
        else:
            conn.execute(
                "UPDATE command_executions SET status = ? WHERE id = ?",
                (status, execution_id),
            )
        conn.commit()


def _row_to_record(row) -> dict:
    record = dict(row)
    try:
        record["matched_rules"] = json.loads(record["matched_rules"]) if record.get("matched_rules") else []
    except (json.JSONDecodeError, TypeError):
        record["matched_rules"] = []
    record["blocked"] = bool(record.get("blocked"))
    record["timed_out"] = bool(record.get("timed_out"))
    return record


def get_by_id(execution_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM command_executions WHERE id = ?", (execution_id,)
        ).fetchone()
    return _row_to_record(row) if row else None


def get_history(session_id: str, limit: int = 100) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM command_executions
            WHERE session_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def clear_session(session_id: str) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM command_executions WHERE session_id = ?", (session_id,)
        )
        conn.commit()
        return cursor.rowcount


def prune_session(session_id: str, keep_last: int) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM command_executions
            WHERE session_id = ? AND id NOT IN (
                SELECT id FROM command_executions
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (session_id, session_id, keep_last),
        )
        conn.commit()
