"""
Prompt templates for the real-time Linux Operations Copilot tool pipeline.

Distinct from `prompts.py` (the original "explain + recommend commands,
no data" assistant): every prompt built here is grounded in the output of
a command that was *just executed* on the live host by `tool_executor.py`.
The system prompt below tells the model exactly that, and forbids it from
answering with anything not present in that data.
"""

from __future__ import annotations

import json

# Sent on every ops-tool request. Kept short for the same latency reasons as
# prompts.SYSTEM_PROMPT, but the contract is different: "summary" instead of
# "explanation", and the rules make explicit that the data is live and real.
OPS_SYSTEM_PROMPT = """You are Linux Copilot XAI, a real-time Linux Operations Copilot.

A read-only Linux command was just executed on the live host and its parsed output is given to \
you below as CURRENT SYSTEM DATA. Rules: answer ONLY using that data - never invent processes, \
services, numbers, or log lines that are not present in it. If the data is empty or an error is \
noted, say so plainly and lower confidence_score. Reference real names/numbers from the data in \
your summary. Suggested follow-up commands must be safe; if a command would change system state \
(restart/stop/kill/delete/edit), mark it risk_level "medium" or "high" and say what it would do.

Respond with ONLY this JSON object, no markdown fences, no other text:
{"summary": "<2-3 plain-language sentences answering the user question, citing real data>", \
"suggested_actions": [{"command": "<shell command>", "description": "<why>", \
"risk_level": "low|medium|high"}], "confidence_score": <0.0-1.0>, \
"reasoning": "<brief basis for the answer, referencing the data>"}

Use an empty list for suggested_actions if none apply. All four fields are required."""


def build_tool_data_block(tool_name: str, display_command: str, tool_result: dict) -> str:
    """Render one tool's structured result into a compact text block."""
    payload = {
        "tool": tool_name,
        "command_executed": display_command,
        "success": tool_result.get("success"),
        "row_count": tool_result.get("row_count"),
        "data": tool_result.get("parsed"),
        "error": tool_result.get("error"),
        "warnings": tool_result.get("warnings") or [],
    }
    try:
        rendered = json.dumps(payload, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        rendered = str(payload)
    return rendered


def build_ops_prompt(message: str, tool_name: str, display_command: str, tool_result: dict) -> str:
    """Compose the final user-turn prompt for an ops/tool-grounded query.

    Follows the required shape: current system data, then the user
    question, then an explicit instruction to answer only from that data.
    """
    data_block = build_tool_data_block(tool_name, display_command, tool_result)
    return (
        f"Current System Data:\n{data_block}\n\n"
        f"User Question: {message}\n\n"
        "Answer only using the provided system data above. "
        "Respond with the JSON object described in your instructions."
    )
