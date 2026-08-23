"""
AI Linux Operations Assistant - orchestration layer.

Pipeline for every user query:
  1. Understand user intent       -> intent_classifier.classify_intent()
  2. Gather relevant system info  -> context_builder.build_context()
  3. Build context                -> prompts.build_user_prompt()
  4. Send context to an LLM       -> ollama_client.chat()
  5. Return explanation, recommended commands, confidence score, reasoning

For everything except the Sprint 9 tool-grounded fast path described
below, the assistant NEVER executes commands - it only reads system state
(via psutil, in context_builder.py) and asks the LLM to recommend commands
as text for the user to review and run themselves.

Conversation history is persisted per session_id in SQLite
(conversation_store.py) and replayed back into the prompt on subsequent
turns so the assistant has short-term memory of the conversation.

Sprint 9 adds a tool-grounded fast path: for specific, high-precision ops
questions this copilot can answer from a real, whitelisted, read-only
command (top CPU/memory processes, disk usage, running/failed services,
recent/error logs, listening ports), `ops_assistant.process_ops_query()`
runs instead: it executes the command for real (`tool_executor.py`),
grounds the LLM prompt in that live data, and returns a response carrying
the same fields as before plus `tool_used` and `detailed_results`.

Routing-audit fix (pipeline ordering)
--------------------------------------
Previously, `ops_intent_classifier.classify_ops_intent()` ran FIRST, before
`intent_classifier.classify_intent()` even had a chance to run. Because its
raw-tool patterns include broad single words ("cpu", "memory", "disk",
"logs"), it was hijacking detector-backed questions - "check CPU and GPU
temperatures", "are there any predicted memory failures?", "what are the
active system faults?" - into a plain `ps`/`df`/`journalctl` command that
has no knowledge of driver anomalies, hardware/thermal health, SMART, or
predicted failures at all.

The fix is ordering: `classify_intent()` (the detector-backed classifier)
now runs FIRST. If it identifies a detector-backed intent (driver health,
hardware health, or failure prediction), the tool-grounded fast path is
skipped entirely and the question goes straight to the original,
detector-grounded pipeline (`context_builder.build_context()`, which reads
`issue_alert_store` - see that module for the routing audit's other half
of this fix). Only when the question is NOT detector-backed does
`classify_ops_intent()` get a chance to claim it for the raw-tool fast
path - and `ops_intent_classifier.py` itself now also declines (returns
`None`) for detector-backed phrasing that `classify_intent()` doesn't
specifically categorize (e.g. "active faults", "system health"), as a
second line of defense. Everything else about both pipelines - the actual
tools, prompts, parsing, and response shape - is unchanged.
"""

import json
import uuid
from datetime import datetime, timezone

from app.config import get_settings
from app.logger import get_logger
from app.services import conversation_store, deterministic_system_chat
from app.services.context_builder import build_context
from app.services.intent_classifier import Intent, classify_intent
from app.services.llm_json import normalize_fields, parse_json_object
from app.services.ollama_client import chat
from app.services.ops_assistant import process_ops_query
from app.services.ops_intent_classifier import classify_ops_intent
from app.services.prompts import SYSTEM_PROMPT, build_user_prompt

# Detector-backed intents must never be handed off to the raw-tool fast
# path (see module docstring): these are exactly the questions that need
# real driver/hardware/prediction data from `context_builder.py`, which a
# generic `ps`/`df`/`journalctl` command cannot provide.
_DETECTOR_BACKED_INTENTS = frozenset(
    {Intent.DRIVER_HEALTH, Intent.HARDWARE_HEALTH, Intent.FAILURE_PREDICTION}
)

settings = get_settings()
logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_object(raw: str) -> dict:
    """Thin wrapper kept for call-site compatibility; see llm_json.py."""
    return parse_json_object(raw)


# Safe, read-only-by-default commands to fall back on per detected issue
# category, used only when the LLM itself returns no recommended_commands.
# Every entry here is diagnostic/informational (or, where a change is
# implied, clearly labeled "medium" risk) so it's always safe to show even
# though the model didn't have a chance to tailor it to the exact question.
_FALLBACK_COMMANDS: dict[str, list[dict]] = {
    "performance": [
        {"command": "top -bn1 | head -20", "description": "Snapshot of the most CPU/memory-hungry processes right now.", "risk_level": "low"},
        {"command": "free -h", "description": "Shows how much RAM and swap are used vs. available.", "risk_level": "low"},
        {"command": "uptime", "description": "Shows load averages and how long the system has been running.", "risk_level": "low"},
        {"command": "ps aux --sort=-%cpu | head -10", "description": "Lists the top 10 processes by CPU usage.", "risk_level": "low"},
        {"command": "vmstat 1 5", "description": "Samples CPU, memory, and I/O activity over a few seconds.", "risk_level": "low"},
    ],
    "service_management": [
        {"command": "systemctl --failed", "description": "Lists any services that failed to start.", "risk_level": "low"},
        {"command": "systemctl list-units --type=service --state=running", "description": "Lists all services currently running.", "risk_level": "low"},
        {"command": "systemctl status <service-name>", "description": "Shows detailed status/recent logs for a specific service (replace <service-name>).", "risk_level": "low"},
        {"command": "journalctl -u <service-name> --since '10 min ago'", "description": "Shows recent log output for a specific service (replace <service-name>).", "risk_level": "low"},
        {"command": "sudo systemctl restart <service-name>", "description": "Restarts a specific service (replace <service-name>); briefly interrupts it.", "risk_level": "medium"},
    ],
    "docker": [
        {"command": "docker ps -a", "description": "Lists all containers, running and stopped, with their status.", "risk_level": "low"},
        {"command": "docker stats --no-stream", "description": "Shows a one-off snapshot of CPU/memory usage per container.", "risk_level": "low"},
        {"command": "docker logs --tail 100 <container-name>", "description": "Shows the last 100 log lines for a container (replace <container-name>).", "risk_level": "low"},
        {"command": "docker system df", "description": "Shows disk space used by images, containers, and volumes.", "risk_level": "low"},
    ],
    "file_search": [
        {"command": "df -h", "description": "Shows free/used disk space per mounted filesystem.", "risk_level": "low"},
        {"command": "du -ah / 2>/dev/null | sort -rh | head -20", "description": "Lists the 20 largest files/directories on the system.", "risk_level": "low"},
        {"command": "find / -xdev -type f -size +100M 2>/dev/null", "description": "Finds files larger than 100MB on the root filesystem.", "risk_level": "low"},
        {"command": "du -sh /var/log/* 2>/dev/null | sort -rh | head -10", "description": "Shows which log files are taking up the most space.", "risk_level": "low"},
    ],
    "log_analysis": [
        {"command": "journalctl -xe --no-pager | tail -50", "description": "Shows the most recent system journal entries with extra context.", "risk_level": "low"},
        {"command": "journalctl -p err -b", "description": "Shows error-level messages from the current boot.", "risk_level": "low"},
        {"command": "dmesg | tail -50", "description": "Shows the most recent kernel ring-buffer messages.", "risk_level": "low"},
        {"command": "tail -n 100 /var/log/syslog", "description": "Shows the last 100 lines of the main system log.", "risk_level": "low"},
        {"command": "grep -i error /var/log/syslog | tail -50", "description": "Searches the system log for recent error entries.", "risk_level": "low"},
    ],
    "network": [
        {"command": "ip a", "description": "Shows all network interfaces and their IP addresses.", "risk_level": "low"},
        {"command": "ss -tulpn", "description": "Lists listening TCP/UDP ports and the processes using them.", "risk_level": "low"},
        {"command": "ping -c 4 8.8.8.8", "description": "Tests basic internet connectivity with 4 pings.", "risk_level": "low"},
        {"command": "sudo ufw status verbose", "description": "Shows current firewall rules and status.", "risk_level": "low"},
        {"command": "ip route", "description": "Shows the system's routing table.", "risk_level": "low"},
    ],
    "users_sessions": [
        {"command": "who", "description": "Lists users currently logged in.", "risk_level": "low"},
        {"command": "w", "description": "Shows logged-in users and what they're currently running.", "risk_level": "low"},
        {"command": "last -n 20", "description": "Shows the 20 most recent login sessions.", "risk_level": "low"},
        {"command": "loginctl list-sessions", "description": "Lists active login sessions managed by systemd.", "risk_level": "low"},
    ],
    # Previously missing: any driver_health/hardware_health query with no
    # LLM-suggested commands (or a parse failure) silently fell through to
    # the "general" bank below via _fallback_commands()'s .get(..., general)
    # default - which is exactly why hardware-fault responses sometimes
    # showed irrelevant commands like `uname -a`/`uptime`/`df -h` that have
    # nothing to do with drivers or hardware health.
    "driver_health": [
        {"command": "lsmod", "description": "Lists currently loaded kernel modules/drivers.", "risk_level": "low"},
        {"command": "dmesg | tail -50", "description": "Shows recent kernel messages, including driver load/error events.", "risk_level": "low"},
        {"command": "journalctl -k -p err -b", "description": "Shows kernel-level error messages from the current boot.", "risk_level": "low"},
        {"command": "modinfo <module-name>", "description": "Shows details for a specific kernel module (replace <module-name>).", "risk_level": "low"},
    ],
    "hardware_health": [
        {"command": "sensors", "description": "Shows current temperatures/fan speeds from hardware sensors (requires lm-sensors).", "risk_level": "low"},
        {"command": "smartctl -a /dev/sda", "description": "Shows SMART health data for a disk (replace /dev/sda as needed; requires smartmontools).", "risk_level": "low"},
        {"command": "dmesg | grep -i -E 'error|fail|fault'", "description": "Searches kernel messages for hardware error/fault indications.", "risk_level": "low"},
        {"command": "journalctl -p err -b", "description": "Shows error-level messages from the current boot.", "risk_level": "low"},
    ],
    "failure_prediction": [
        {"command": "smartctl -a /dev/sda", "description": "Shows SMART health/wear data that can indicate an impending disk failure (replace /dev/sda as needed).", "risk_level": "low"},
        {"command": "df -h", "description": "Shows free/used disk space per mounted filesystem, useful for predicting when a disk will fill up.", "risk_level": "low"},
        {"command": "dmesg | tail -50", "description": "Shows recent kernel messages that may show early warning signs.", "risk_level": "low"},
    ],
    "general": [
        {"command": "uname -a", "description": "Shows kernel version and basic system information.", "risk_level": "low"},
        {"command": "uptime", "description": "Shows load averages and how long the system has been running.", "risk_level": "low"},
        {"command": "df -h", "description": "Shows free/used disk space per mounted filesystem.", "risk_level": "low"},
        {"command": "free -h", "description": "Shows how much RAM and swap are used vs. available.", "risk_level": "low"},
        {"command": "journalctl -xe --no-pager | tail -30", "description": "Shows the most recent system journal entries for context.", "risk_level": "low"},
    ],
}


def _fallback_commands(intent: str) -> list[dict]:
    """Return a fresh copy of the safe fallback commands for an intent.

    Falls back to the GENERAL bank for any intent without a dedicated list,
    so callers never get an empty result for a detected issue.
    """
    bank = _FALLBACK_COMMANDS.get(intent, _FALLBACK_COMMANDS["general"])
    return [dict(item) for item in bank]


def _parse_llm_response(raw: str, intent: str) -> tuple[dict, list[str]]:
    """Parse and validate the LLM's JSON reply, filling in safe defaults.

    Returns (parsed_dict, warnings). Never raises - a malformed reply
    degrades to a low-confidence explanation-only response instead of a
    500 error, since the failure mode here is "LLM formatting", not a
    system error.

    `intent` is the already-detected issue category (see intent_classifier)
    and is used solely to pick a safe fallback command set when the model
    itself doesn't return any recommended_commands - a detected issue
    should never come back with an empty command list.
    """
    warnings: list[str] = []

    try:
        data = _load_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        # TEMPORARY DIAGNOSTIC LOGGING - see ollama_client.py for the
        # matching raw-response log this pairs with.
        logger.warning("OLLAMA DIAG: JSON parse FAILED (%s) for intent=%s raw=%r", exc, intent, raw[:1500])
        warnings.append(f"Model response was not valid JSON ({exc}); showing raw text instead.")
        return (
            {
                "explanation": raw.strip() or "The assistant did not return a usable response.",
                "recommended_commands": _fallback_commands(intent),
                "confidence_score": 0.2,
                "reasoning": "Response could not be parsed as structured JSON.",
            },
            warnings,
        )

    # Normalize before reading any field: a model that returns valid JSON
    # with a slightly different field name (e.g. "answer" instead of
    # "explanation") should not be treated the same as one that omitted
    # the field entirely.
    data = normalize_fields(
        data, ("explanation", "reasoning", "confidence_score", "recommended_commands")
    )

    explanation = str(data.get("explanation") or "").strip()
    if not explanation:
        # TEMPORARY DIAGNOSTIC LOGGING - see ollama_client.py for the
        # matching raw-response log this pairs with. This is the "valid
        # JSON, missing our keys" case - logging the parsed dict's actual
        # keys/shape is what tells us what the model returned instead.
        logger.warning(
            "OLLAMA DIAG: JSON parsed OK but 'explanation' missing/empty for intent=%s. "
            "parsed_keys=%s parsed_dict=%r",
            intent, list(data.keys()), data,
        )
        explanation = "The assistant did not provide an explanation."
        warnings.append("Missing 'explanation' field in model response.")

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

    raw_commands = data.get("recommended_commands", [])
    commands: list[dict] = []
    if isinstance(raw_commands, list):
        for item in raw_commands:
            if not isinstance(item, dict) or not item.get("command"):
                continue
            risk = str(item.get("risk_level", "low")).lower()
            if risk not in ("low", "medium", "high"):
                risk = "low"
            commands.append(
                {
                    "command": str(item["command"]),
                    "description": str(item.get("description", "")).strip()
                    or "No description provided.",
                    "risk_level": risk,
                }
            )
    else:
        warnings.append("'recommended_commands' was not a list; ignored.")

    if not commands:
        # The model detected an issue (that's why we're here) but didn't
        # suggest anything actionable - fall back to a safe, intent-matched
        # command set rather than leaving the user with nothing to try.
        commands = _fallback_commands(intent)
        warnings.append("Model returned no recommended commands; substituted safe defaults.")

    return (
        {
            "explanation": explanation,
            "recommended_commands": commands,
            "confidence_score": confidence,
            "reasoning": reasoning,
        },
        warnings,
    )


def _build_history_messages(session_id: str) -> list[dict]:
    """Replay recent conversation turns as chat messages for context."""
    recent = conversation_store.get_recent_messages(
        session_id, limit=settings.assistant_history_turns * 2
    )
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def _summarize_context(context: dict) -> dict:
    """Produce a compact, human-friendly summary of what data was used.

    Full context already goes to the LLM; this trimmed-down version is for
    the API response so clients (including a future AI layer) can see what
    grounded the answer without re-parsing a large payload.
    """
    summary = {}
    for key, value in context.items():
        if isinstance(value, list):
            summary[key] = f"{len(value)} item(s)"
        elif isinstance(value, dict):
            summary[key] = f"{len(value)} field(s)"
        else:
            summary[key] = value
    return summary


async def process_query(message: str, session_id: str | None = None) -> dict:
    """Run the full assistant pipeline for a single user query.

    Returns a dict matching the ChatResponse schema. Raises
    OllamaUnavailableError if the LLM cannot be reached - callers (routes)
    should translate that into an HTTP 503.
    """
    session_id = session_id or str(uuid.uuid4())

    # 0. Deterministic system-chat fast path: for a fixed set of common
    # CPU/RAM/disk/temperature/process/driver/kernel/hardware questions,
    # bypass LLM reasoning entirely and answer directly from real system
    # collectors (see deterministic_system_chat.py). Returns None for
    # anything outside that known set, in which case every existing
    # pipeline below runs completely unchanged.
    deterministic_result = await deterministic_system_chat.try_handle(message)
    if deterministic_result is not None:
        conversation_store.add_message(session_id, role="user", content=message)
        conversation_store.add_message(
            session_id,
            role="assistant",
            content=deterministic_result["explanation"],
            intent=deterministic_result["intent"],
            confidence_score=deterministic_result["confidence_score"],
        )
        conversation_store.prune_session(session_id, keep_last=settings.assistant_max_history_stored)
        return {
            "session_id": session_id,
            **deterministic_result,
            "timestamp": _now_iso(),
        }

    # 1. Understand user intent FIRST, using the detector-backed classifier.
    # This must run before the raw-tool fast path below so that
    # driver/hardware/prediction questions are never hijacked away from the
    # real detector context (see module docstring for the routing-audit
    # fix).
    intent_result = classify_intent(message)
    logger.info(
        "Session %s: classified intent=%s (matched=%s)",
        session_id,
        intent_result.intent.value,
        intent_result.matched_keywords,
    )

    # 0. Tool-grounded fast path: only offered a chance when the question
    # is NOT one of the detector-backed intents. If this is one of the
    # specific ops questions we can answer from a real whitelisted command,
    # run that pipeline instead (see module docstring). Falls through to
    # the original context-builder pipeline for everything else, including
    # every detector-backed question.
    if intent_result.intent not in _DETECTOR_BACKED_INTENTS:
        ops_intent = classify_ops_intent(message)
        if ops_intent is not None:
            return await process_ops_query(message, session_id, ops_intent)

    # 2. Gather relevant system information
    context = build_context(intent_result.intent, message)

    # 3. Build context / prompt
    history_messages = _build_history_messages(session_id)
    user_prompt = build_user_prompt(message, intent_result.intent.value, context)
    llm_messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history_messages
        + [{"role": "user", "content": user_prompt}]
    )

    # 4. Send context to the LLM (raises OllamaUnavailableError on failure)
    raw_reply = await chat(llm_messages)

    # 5. Parse into explanation / commands / confidence / reasoning
    parsed, warnings = _parse_llm_response(raw_reply, intent_result.intent.value)

    # Persist conversation turns (store the user's raw message, and the
    # assistant's explanation as its "content" for future context replay).
    conversation_store.add_message(session_id, role="user", content=message)
    conversation_store.add_message(
        session_id,
        role="assistant",
        content=parsed["explanation"],
        intent=intent_result.intent.value,
        confidence_score=parsed["confidence_score"],
    )
    conversation_store.prune_session(session_id, keep_last=settings.assistant_max_history_stored)

    return {
        "session_id": session_id,
        "intent": intent_result.intent.value,
        "explanation": parsed["explanation"],
        "recommended_commands": parsed["recommended_commands"],
        "confidence_score": parsed["confidence_score"],
        "reasoning": parsed["reasoning"],
        "context_summary": _summarize_context(context),
        "warnings": warnings,
        "timestamp": _now_iso(),
    }