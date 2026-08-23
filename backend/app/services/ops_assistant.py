"""
Real-time Linux Operations Copilot - tool-grounded orchestration layer.

Pipeline for every ops query (see `ops_intent_classifier.classify_ops_intent`):
  1. Detect intent              -> already done by the caller (ops_intent_classifier)
  2. Execute a real, whitelisted, read-only command on the host -> tool_executor.execute_tool()
  3. Collect the structured result                                -> tool_executor.ToolResult
  4. Build a prompt grounded in that data                         -> ops_prompts.build_ops_prompt()
  5. Ask the LLM to answer using ONLY that data                   -> ollama_client.chat()
  6. Return summary + detailed structured results + confidence + suggested actions

This module never invents system data itself and never lets the LLM
override what the command actually returned - `detailed_results` in the
final response is always the tool's own structured output, independent of
whatever the model says in its summary.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.config import get_settings
from app.logger import get_logger
from app.services import conversation_store
from app.services.llm_json import normalize_fields, parse_json_object
from app.services.ollama_client import chat
from app.services.ops_intent_classifier import OpsIntentResult
from app.services.ops_prompts import OPS_SYSTEM_PROMPT, build_ops_prompt
from app.services.tool_executor import ToolExecutionError, ToolName, execute_tool

settings = get_settings()
logger = get_logger(__name__)

# Safe, read-only fallback follow-up actions per tool, used only when the
# LLM itself returns no suggested_actions. Every entry is diagnostic/
# informational so it's always safe to surface even though the model didn't
# get a chance to tailor it.
_FALLBACK_ACTIONS: dict[str, list[dict]] = {
    ToolName.CPU_TOP.value: [
        {"command": "top -bn1 | head -20", "description": "Live snapshot of CPU usage across all processes.", "risk_level": "low"},
        {"command": "ps -p <pid> -o pid,ppid,cmd,%cpu,%mem", "description": "Inspect a specific process in more detail (replace <pid>).", "risk_level": "low"},
    ],
    ToolName.MEMORY_TOP.value: [
        {"command": "free -h", "description": "Overall RAM and swap usage.", "risk_level": "low"},
        {"command": "ps -p <pid> -o pid,ppid,cmd,%cpu,%mem", "description": "Inspect a specific process in more detail (replace <pid>).", "risk_level": "low"},
    ],
    ToolName.DISK_USAGE.value: [
        {"command": "du -ah / 2>/dev/null | sort -rh | head -20", "description": "Find the largest files/directories on the fullest filesystem.", "risk_level": "low"},
        {"command": "du -sh /var/log/* 2>/dev/null | sort -rh | head -10", "description": "Check whether logs are consuming the space.", "risk_level": "low"},
    ],
    ToolName.SERVICES_RUNNING.value: [
        {"command": "systemctl status <service-name>", "description": "Detailed status for a specific running service (replace <service-name>).", "risk_level": "low"},
    ],
    ToolName.SERVICES_FAILED.value: [
        {"command": "systemctl status <service-name>", "description": "See why a specific failed service failed (replace <service-name>).", "risk_level": "low"},
        {"command": "journalctl -u <service-name> --since '30 min ago'", "description": "Recent logs for a specific failed service (replace <service-name>).", "risk_level": "low"},
        {"command": "sudo systemctl restart <service-name>", "description": "Restart a specific failed service (replace <service-name>); briefly interrupts it.", "risk_level": "medium"},
    ],
    ToolName.LOGS_RECENT.value: [
        {"command": "journalctl -p err -n 50 --no-pager", "description": "Narrow the journal down to error-level entries only.", "risk_level": "low"},
    ],
    ToolName.LOGS_ERROR.value: [
        {"command": "journalctl -u <service-name> --since '1 hour ago'", "description": "Follow up on a specific service named in the errors (replace <service-name>).", "risk_level": "low"},
    ],
    ToolName.NETWORK_PORTS.value: [
        {"command": "ss -tulpn", "description": "Same listening sockets, plus which process owns each one.", "risk_level": "low"},
        {"command": "sudo ufw status verbose", "description": "Check whether the firewall is exposing these ports intentionally.", "risk_level": "low"},
    ],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fallback_actions(tool_name: str) -> list[dict]:
    bank = _FALLBACK_ACTIONS.get(tool_name, [])
    return [dict(item) for item in bank]


def _build_history_messages(session_id: str) -> list[dict]:
    recent = conversation_store.get_recent_messages(
        session_id, limit=settings.assistant_history_turns * 2
    )
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def _parse_ops_llm_response(raw: str, tool_name: str) -> tuple[dict, list[str]]:
    """Parse and validate the LLM's JSON reply for an ops/tool query.

    Mirrors `ai_assistant._parse_llm_response` but for the ops contract
    (`summary`/`suggested_actions` instead of `explanation`/
    `recommended_commands`). Never raises - a malformed reply degrades to a
    low-confidence summary instead of a 500.
    """
    warnings: list[str] = []

    try:
        data = parse_json_object(raw)
    except (ValueError,) as exc:
        warnings.append(f"Model response was not valid JSON ({exc}); showing raw text instead.")
        return (
            {
                "summary": raw.strip() or "The copilot did not return a usable response.",
                "suggested_actions": _fallback_actions(tool_name),
                "confidence_score": 0.2,
                "reasoning": "Response could not be parsed as structured JSON.",
            },
            warnings,
        )
    except Exception as exc:  # noqa: BLE001 - json.JSONDecodeError etc. from parse_json_object
        warnings.append(f"Model response was not valid JSON ({exc}); showing raw text instead.")
        return (
            {
                "summary": raw.strip() or "The copilot did not return a usable response.",
                "suggested_actions": _fallback_actions(tool_name),
                "confidence_score": 0.2,
                "reasoning": "Response could not be parsed as structured JSON.",
            },
            warnings,
        )

    data = normalize_fields(
        data, ("summary", "reasoning", "confidence_score", "suggested_actions")
    )

    summary = str(data.get("summary") or "").strip()
    if not summary:
        summary = "The copilot did not provide a summary."
        warnings.append("Missing 'summary' field in model response.")

    reasoning = str(data.get("reasoning") or "").strip()
    if not reasoning:
        reasoning = "No reasoning was provided by the model."
        warnings.append("Missing 'reasoning' field in model response.")

    try:
        confidence = float(data.get("confidence_score", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
        warnings.append("Invalid 'confidence_score'; defaulted to 0.5.")
    confidence = max(0.0, min(1.0, confidence))

    raw_actions = data.get("suggested_actions", [])
    actions: list[dict] = []
    if isinstance(raw_actions, list):
        for item in raw_actions:
            if not isinstance(item, dict) or not item.get("command"):
                continue
            risk = str(item.get("risk_level", "low")).lower()
            if risk not in ("low", "medium", "high"):
                risk = "low"
            actions.append(
                {
                    "command": str(item["command"]),
                    "description": str(item.get("description", "")).strip() or "No description provided.",
                    "risk_level": risk,
                }
            )
    else:
        warnings.append("'suggested_actions' was not a list; ignored.")

    if not actions:
        actions = _fallback_actions(tool_name)
        if actions:
            warnings.append("Model returned no suggested actions; substituted safe defaults.")

    return (
        {
            "summary": summary,
            "suggested_actions": actions,
            "confidence_score": confidence,
            "reasoning": reasoning,
        },
        warnings,
    )


async def process_ops_query(message: str, session_id: str | None, ops_result: OpsIntentResult) -> dict:
    """Run the full tool-grounded ops pipeline for a single user query.

    Returns a dict matching the extended ChatResponse schema (adds
    `tool_used` and `detailed_results` on top of the original fields).
    Raises `OllamaUnavailableError` if the LLM cannot be reached - callers
    (routes) should translate that into an HTTP 503, same as the original
    assistant pipeline.
    """
    session_id = session_id or str(uuid.uuid4())
    tool_name = ops_result.tool

    logger.info(
        "Session %s: ops intent -> tool=%s (matched=%s, confidence=%.2f)",
        session_id,
        tool_name.value,
        ops_result.matched_patterns,
        ops_result.confidence,
    )

    # 2 & 3. Execute the whitelisted tool and collect its structured result.
    try:
        tool_result = execute_tool(tool_name)
    except ToolExecutionError as exc:
        # Only raised for an unrecognized tool name - a programmer error in
        # the classifier/registry, not something a live host can trigger.
        logger.error("Tool execution rejected: %s", exc)
        raise

    tool_dict = tool_result.to_dict()

    # 4. Build the ops prompt grounded in that real data.
    history_messages = _build_history_messages(session_id)
    user_prompt = build_ops_prompt(message, tool_name.value, tool_result.display_command, tool_dict)
    llm_messages = (
        [{"role": "system", "content": OPS_SYSTEM_PROMPT}]
        + history_messages
        + [{"role": "user", "content": user_prompt}]
    )

    # 5. Ask the LLM to answer using only that data (raises OllamaUnavailableError on failure).
    raw_reply = await chat(llm_messages)

    # 6. Parse into summary / suggested actions / confidence / reasoning.
    parsed, llm_warnings = _parse_ops_llm_response(raw_reply, tool_name.value)

    warnings = list(llm_warnings)
    if not tool_result.success:
        warnings.append(f"Tool execution failed: {tool_result.error}")
    warnings.extend(tool_result.warnings)

    # Persist conversation turns exactly like the original assistant pipeline.
    conversation_store.add_message(session_id, role="user", content=message)
    conversation_store.add_message(
        session_id,
        role="assistant",
        content=parsed["summary"],
        intent=tool_name.value,
        confidence_score=parsed["confidence_score"],
    )
    conversation_store.prune_session(session_id, keep_last=settings.assistant_max_history_stored)

    return {
        "session_id": session_id,
        "intent": tool_name.value,
        "explanation": parsed["summary"],
        "recommended_commands": parsed["suggested_actions"],
        "confidence_score": parsed["confidence_score"],
        "reasoning": parsed["reasoning"],
        "context_summary": {
            "tool": tool_name.value,
            "command": tool_result.display_command,
            "row_count": tool_result.row_count,
            "success": tool_result.success,
        },
        "warnings": warnings,
        "timestamp": _now_iso(),
        "tool_used": tool_name.value,
        "detailed_results": tool_dict,
    }
