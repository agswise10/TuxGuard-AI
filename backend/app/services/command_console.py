"""
Safe Command Execution - orchestration layer.

Implements the full requested workflow:

    User -> AI generates command -> Preview -> Explain -> User confirms
    -> Execute -> Return output

Two entry points build a "pending" command awaiting confirmation:
  - `generate_command()`   : natural language -> AI proposes a command
  - `explain_command()`    : user's own command -> AI explains it

Both immediately run every candidate command through the deterministic
`command_safety` analyzer (never trusting the LLM's own risk judgment) and
persist the result via `execution_store`.

A single entry point actually runs anything:
  - `confirm_and_execute()`: the ONLY function in this project that can
    cause a command to run. It re-validates safety on the exact command
    text about to execute (in case it was edited after preview), refuses
    outright if blocked, refuses if not explicitly confirmed, and logs
    the full result - stdout, stderr, exit code, duration - regardless of
    outcome.
"""

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone

from app.config import get_settings
from app.logger import get_logger
from app.services import command_executor, execution_store
from app.services.command_prompts import (
    COMMAND_EXPLAIN_SYSTEM_PROMPT,
    COMMAND_SYSTEM_PROMPT,
    build_explain_prompt,
    build_generate_prompt,
)
from app.services.command_context import gather_command_context
from app.services.command_safety import analyze_command
from app.services.ollama_client import chat

settings = get_settings()
logger = get_logger(__name__)

# systemctl start|restart|stop|reload|try-restart legitimately take longer
# than the default execution budget: e.g. `systemctl restart
# NetworkManager-wait-online.service` waits for that unit's ExecStart
# (nm-online) to finish, which can take up to ~60s. `settings
# .service_restart_timeout_seconds` already exists for exactly this case but
# was never applied to an execute() call - every command silently got the
# flat default timeout regardless of this setting. This regex detects that
# one class of command so it can be given the longer budget instead.
_SERVICE_RESTART_RE = re.compile(
    r"\bsystemctl\b(?:\s+--\S+)*\s+(?:start|restart|stop|reload|try-restart)\b"
)


def _execution_timeout_for(command: str) -> int:
    """Pick the timeout budget for `command`.

    Defaults to `settings.execution_timeout_seconds`; uses the longer
    `settings.service_restart_timeout_seconds` for systemctl
    start/restart/stop/reload/try-restart commands, which can legitimately
    run longer than a normal command.
    """
    if _SERVICE_RESTART_RE.search(command):
        return settings.service_restart_timeout_seconds
    return settings.execution_timeout_seconds

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_code_fences(text: str) -> str:
    return _JSON_FENCE_RE.sub("", text).strip()


def _parse_llm_command_response(raw: str, fallback_command: str = "") -> tuple[dict, list[str]]:
    """Parse the LLM's JSON reply for command generation/explanation.

    Never raises - a malformed reply degrades to a low-confidence response
    (falling back to the raw text as the explanation, and to
    `fallback_command` - e.g. the user's original input when explaining -
    for the command field) instead of a 500.
    """
    warnings: list[str] = []
    cleaned = _strip_code_fences(raw)

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON was not an object")
    except (json.JSONDecodeError, ValueError) as exc:
        warnings.append(f"Model response was not valid JSON ({exc}); using fallback values.")
        return (
            {
                "command": fallback_command,
                "explanation": raw.strip() or "The generator did not return a usable response.",
                "confidence_score": 0.1,
            },
            warnings,
        )

    command = str(data.get("command") or fallback_command).strip()
    command = _strip_code_fences(command)
    if not command:
        warnings.append("Missing 'command' field in model response.")

    explanation = str(data.get("explanation") or "").strip()
    if not explanation:
        explanation = "The generator did not provide an explanation."
        warnings.append("Missing 'explanation' field in model response.")

    risk_notes = str(data.get("risk_notes") or "").strip()
    if risk_notes:
        explanation = f"{explanation}\n\nModel's own risk note: {risk_notes}"

    try:
        confidence = float(data.get("confidence_score", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
        warnings.append("Invalid 'confidence_score'; defaulted to 0.5.")
    confidence = max(0.0, min(1.0, confidence))

    return {"command": command, "explanation": explanation, "confidence_score": confidence}, warnings


def _summarize_context(context: dict) -> dict:
    summary = {}
    for key, value in context.items():
        if isinstance(value, list):
            summary[key] = f"{len(value)} item(s)"
        elif isinstance(value, dict):
            summary[key] = f"{len(value)} field(s)"
        else:
            summary[key] = value
    return summary


def _build_preview(
    session_id: str,
    description: str | None,
    source: str,
    command: str,
    explanation: str,
    confidence_score: float,
    context: dict,
    llm_warnings: list[str],
) -> dict:
    """Run the safety analyzer, persist as pending/blocked, and return the
    full preview payload shared by both generate_command() and
    explain_command()."""
    analysis = analyze_command(command)

    execution_id = execution_store.save_pending(
        session_id=session_id,
        description=description,
        command=command,
        source=source,
        explanation=explanation,
        risk_level=analysis.risk_level,
        blocked=analysis.blocked,
        matched_rules=[
            {"rule": m.rule, "description": m.description, "severity": m.severity}
            for m in analysis.matched_rules
        ],
        confidence_score=confidence_score,
    )
    execution_store.prune_session(session_id, keep_last=settings.execution_max_history_stored)

    return {
        "execution_id": execution_id,
        "session_id": session_id,
        "description": description,
        "source": source,
        "command": command,
        "explanation": explanation,
        "risk_level": analysis.risk_level,
        "blocked": analysis.blocked,
        "matched_rules": analysis.to_dict()["matched_rules"],
        "warnings": llm_warnings + analysis.warnings,
        "confidence_score": confidence_score,
        "status": "blocked" if analysis.blocked else "pending",
        "context_summary": _summarize_context(context),
        "timestamp": _now_iso(),
    }


async def generate_command(description: str, session_id: str | None = None) -> dict:
    """Step 1-2 of the workflow: AI generates a command from a description, then Preview.

    Raises OllamaUnavailableError if the LLM cannot be reached - callers
    (routes) should translate that into an HTTP 503.
    """
    session_id = session_id or str(uuid.uuid4())
    context = gather_command_context(description)

    messages = [
        {"role": "system", "content": COMMAND_SYSTEM_PROMPT},
        {"role": "user", "content": build_generate_prompt(description, context)},
    ]

    logger.info("Session %s: generating command for description=%r", session_id, description)
    raw_reply = await chat(messages)
    parsed, llm_warnings = _parse_llm_command_response(raw_reply)

    return _build_preview(
        session_id=session_id,
        description=description,
        source="ai_generated",
        command=parsed["command"],
        explanation=parsed["explanation"],
        confidence_score=parsed["confidence_score"],
        context=context,
        llm_warnings=llm_warnings,
    )


async def explain_command(command: str, session_id: str | None = None) -> dict:
    """Step 2-3 of the workflow for a user-provided command: Preview + Explain.

    Raises OllamaUnavailableError if the LLM cannot be reached.
    """
    session_id = session_id or str(uuid.uuid4())
    context = gather_command_context(command)

    messages = [
        {"role": "system", "content": COMMAND_EXPLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": build_explain_prompt(command, context)},
    ]

    logger.info("Session %s: explaining user-provided command=%r", session_id, command)
    raw_reply = await chat(messages)
    parsed, llm_warnings = _parse_llm_command_response(raw_reply, fallback_command=command)

    # Defense in depth: always trust the user's actual typed command over
    # anything the model may have altered when echoing it back.
    parsed["command"] = command

    return _build_preview(
        session_id=session_id,
        description=None,
        source="user_provided",
        command=parsed["command"],
        explanation=parsed["explanation"],
        confidence_score=parsed["confidence_score"],
        context=context,
        llm_warnings=llm_warnings,
    )


async def confirm_and_execute(execution_id: int, confirm: bool, edited_command: str | None = None) -> dict:
    """Steps 4-6: User confirms -> Execute -> Return output.

    This is the ONLY function in the entire project that can cause a
    command to actually run. Safety is re-checked here, immediately
    before execution, on the exact final command text - never trusting
    whatever was analyzed at preview time, in case the user edited it.
    """
    record = execution_store.get_by_id(execution_id)
    if not record:
        raise LookupError(f"No pending command found with id {execution_id}")

    if record["status"] not in ("pending", "blocked"):
        raise ValueError(
            f"Command {execution_id} has already been {record['status']} and cannot be run again. "
            "Generate or preview a new command instead."
        )

    final_command = (edited_command if edited_command is not None else record["command"]).strip()

    # Always re-analyze the exact command about to run - this is the real
    # gate, independent of whatever was shown at preview time.
    analysis = analyze_command(final_command)

    if analysis.blocked:
        execution_store.mark_status(execution_id, "blocked", command=final_command)
        logger.warning(
            "Blocked command execution attempt (id=%s): %s | reasons=%s",
            execution_id,
            final_command,
            [m.rule for m in analysis.matched_rules],
        )
        return {
            "execution_id": execution_id,
            "session_id": record["session_id"],
            "command": final_command,
            "status": "blocked",
            "risk_level": "blocked",
            "matched_rules": analysis.to_dict()["matched_rules"],
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "duration_seconds": 0.0,
            "timed_out": False,
            "executed_at": None,
            "message": "This command was blocked by the safety analyzer and was NOT executed.",
        }

    if not confirm:
        execution_store.mark_status(execution_id, "rejected", command=final_command)
        logger.info("Execution %s rejected by user (not confirmed): %s", execution_id, final_command)
        return {
            "execution_id": execution_id,
            "session_id": record["session_id"],
            "command": final_command,
            "status": "rejected",
            "risk_level": analysis.risk_level,
            "matched_rules": analysis.to_dict()["matched_rules"],
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "duration_seconds": 0.0,
            "timed_out": False,
            "executed_at": None,
            "message": "Execution was not confirmed, so nothing was run.",
        }

    logger.info(
        "Executing confirmed command (id=%s, risk=%s): %s",
        execution_id, analysis.risk_level, final_command,
    )
    # command_executor.execute() runs a blocking subprocess.run() with up to
    # execution_timeout_seconds (default 30s) of wall-clock time - longer
    # (service_restart_timeout_seconds) for systemctl start/restart/stop/
    # reload/try-restart, via _execution_timeout_for(). Calling it directly
    # here would freeze the asyncio event loop - and therefore the entire
    # API, not just this request - for that whole duration. Running it in a
    # worker thread keeps the event loop free for every other request while
    # this command executes.
    result = await asyncio.to_thread(
        command_executor.execute, final_command, _execution_timeout_for(final_command)
    )

    execution_store.mark_executed(
        execution_id=execution_id,
        command=final_command,
        risk_level=analysis.risk_level,
        blocked=False,
        matched_rules=analysis.to_dict()["matched_rules"],
        stdout=result["stdout"],
        stderr=result["stderr"],
        exit_code=result["exit_code"],
        duration_seconds=result["duration_seconds"],
        timed_out=result["timed_out"],
    )

    updated = execution_store.get_by_id(execution_id)
    logger.info(
        "Command executed (id=%s): exit_code=%s duration=%ss timed_out=%s",
        execution_id, result["exit_code"], result["duration_seconds"], result["timed_out"],
    )

    return {
        "execution_id": execution_id,
        "session_id": record["session_id"],
        "command": final_command,
        "status": "executed",
        "risk_level": analysis.risk_level,
        "matched_rules": analysis.to_dict()["matched_rules"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "exit_code": result["exit_code"],
        "duration_seconds": result["duration_seconds"],
        "timed_out": result["timed_out"],
        "executed_at": updated["executed_at"] if updated else _now_iso(),
        "message": None,
    }