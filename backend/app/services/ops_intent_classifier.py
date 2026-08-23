"""
Ops intent classifier for the real-time Linux Operations Copilot.

This is a second, narrower classifier that sits in front of the existing
`intent_classifier.py`. Where that one decides a broad *topic* (performance,
network, logs, ...) so the old psutil-based context builder can gather
loosely-related data, this one decides whether the question is one of the
specific, high-precision "run this exact tool and answer from live data"
questions this upgrade targets - e.g. "show top 5 CPU consuming processes"
-> run the CPU tool, not just "this is a performance question".

Deliberately rule-based and fully offline for the same reasons as the
existing classifier: instant, free, and this routing step doesn't need an
LLM call to be reliable for a fixed, well-known set of ops questions.

`classify_ops_intent()` returns `None` when the question doesn't clearly
match one of the whitelisted tools, so callers (see `ai_assistant.py`) can
fall back to the existing general-purpose assistant pipeline unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.tool_executor import ToolName


@dataclass
class OpsIntentResult:
    tool: ToolName
    matched_patterns: list[str]
    confidence: float


# Ordered so more specific patterns are checked before more generic ones,
# e.g. "failed services" must win over the generic "services" pattern, and
# "recent errors" must win over the generic "recent logs" pattern.
_OPS_PATTERNS: list[tuple[ToolName, list[str]]] = [
    (
        ToolName.SERVICES_FAILED,
        [
            r"\bfailed\s+services?\b",
            r"\bservices?\s+(that\s+)?(failed|have\s+failed|are\s+down)\b",
            r"\bservice\s+failures?\b",
            r"\bwhich\s+services?\s+failed\b",
            r"\bany\s+failed\s+units?\b",
        ],
    ),
    (
        ToolName.LOGS_ERROR,
        [
            r"\brecent\s+errors?\b",
            r"\bany\s+errors?\b",
            r"\berror\s+logs?\b",
            r"\bshow\b.*\berrors?\b",
            r"\bwhat\s+errors?\b",
            r"\berror\s+messages?\b",
        ],
    ),
    (
        ToolName.LOGS_RECENT,
        [
            r"\brecent\s+logs?\b",
            r"\blatest\s+logs?\b",
            r"\bshow\b.*\blogs?\b",
            r"\bjournal\s*ctl\b",
            r"\bsystem\s+journal\b",
        ],
    ),
    (
        ToolName.DISK_USAGE,
        [
            r"\bwhich\s+partition\b",
            r"\bpartition(s)?\b.*\b(full|almost\s+full|space)\b",
            r"\bdisk\s+(space|usage)\b",
            r"\bdisk\s+almost\s+full\b",
            r"\bfree\s+disk\s+space\b",
            r"\bhow\s+full\b.*\bdisk\b",
        ],
    ),
    (
        ToolName.MEMORY_TOP,
        [
            # Deliberately scoped to PROCESS-level memory questions only,
            # mirroring CPU_TOP below for the same reason - "what
            # percentage of RAM is currently being used?" must NOT match
            # here: it has no process in it and needs to fall through to
            # the general PERFORMANCE pipeline (context_builder.py),
            # which already reports the real system-wide
            # system_monitor.get_memory_info()["usage_percent"] instead
            # of a per-process %MEM figure from a handful of processes.
            r"\bmost\s+memory\b",
            r"\bhighest\s+memory\b",
            r"\bmemory\s+consuming\b",
            r"\btop\b.*\bmemory\b",
            r"\bwhich\s+(process|application|app)\b.*\bmemory\b",
            r"\bconsuming\s+.*\bmemory\b",
            # NOTE: deliberately no bare `\bram\b`, bare `\bmemory\b`, and
            # no generic `\bmemory\s+usage\b` here anymore - those
            # over-matched plain "what % of RAM is being used?" questions
            # and incorrectly routed them to this per-process tool (the
            # exact same class of bug already fixed for CPU_TOP below).
        ],
    ),
    (
        ToolName.CPU_TOP,
        [
            # Deliberately scoped to PROCESS-level CPU questions only -
            # "which process(es)/app(s) is/are eating CPU", "top CPU
            # consumers", etc. This must NOT match overall CPU utilization
            # questions like "what is my current CPU usage?" - those have
            # no process in them at all and need to fall through to the
            # general PERFORMANCE pipeline (context_builder.py), which
            # already reports the real system-wide
            # system_monitor.get_cpu_info()["usage_percent"] instead of a
            # per-process %CPU figure (which is per-core and can add up to
            # well over 100% across a handful of processes on a multi-core
            # box - not a meaningful "current CPU usage" answer).
            r"\bmost\s+cpu\b",
            r"\bhighest\s+cpu\b",
            r"\bcpu[- ]consuming\b",
            r"\bcpu[- ](intensive|heavy|hungry)\b",
            r"\btop\b.*\bcpu\b",
            r"\bcpu\b.*\btop\b",
            r"\bwhich\s+(process(es)?|application(s)?|app(s)?|program(s)?)\b.*\bcpu\b",
            r"\bconsuming\s+.*\bcpu\b",
            r"\b(process(es)?|application(s)?|app(s)?|program(s)?)\b.*\busing\b.*\bcpu\b",
            # NOTE: deliberately no bare `\bcpu\b` and no generic
            # `\bcpu\s+usage\b` / `\bcpu\s+utilization\b` here anymore -
            # those over-matched plain "what is my CPU usage?" questions
            # and incorrectly routed them to this per-process tool.
        ],
    ),
    (
        ToolName.SERVICES_RUNNING,
        [
            r"\brunning\s+services?\b",
            r"\bshow\b.*\bservices?\b",
            r"\blist\b.*\bservices?\b",
            r"\bwhich\s+services?\s+are\s+running\b",
        ],
    ),
    (
        ToolName.NETWORK_PORTS,
        [
            r"\bopen\s+ports?\b",
            r"\blistening\s+ports?\b",
            r"\bnetwork\s+ports?\b",
            r"\bnetwork\s+connections?\b",
            r"\bwhich\s+ports?\b",
            r"\bsockets?\b",
        ],
    ),
]


# Bare-word markers for each of the three system-resource domains this
# module has a single-purpose tool for (CPU_TOP, MEMORY_TOP, DISK_USAGE).
# Used only by `_is_multi_domain_overview` below, never for routing on
# their own.
_DOMAIN_MARKERS: dict[str, str] = {
    "cpu": r"\bcpu\b",
    "memory": r"\b(memory|ram)\b",
    "disk": r"\bdisk\b",
}


def _is_multi_domain_overview(text: str) -> bool:
    """True when a question spans 2+ of {cpu, memory, disk} at once.

    A compound question like "what is my current CPU, memory, and disk
    usage?" is a single-shot system overview, not one of the narrow
    per-tool questions this fast path targets. context_builder's
    PERFORMANCE/GENERAL context already returns all three real,
    system-wide figures together in one response (see
    context_builder.build_context). Without this guard, whichever single-
    metric tool pattern happens to match first (e.g. "disk usage") would
    grab the whole question and silently drop the other metrics from the
    answer - so an overview question like this is deliberately left
    unclassified here and falls through to the general pipeline instead.
    """
    hits = sum(1 for pattern in _DOMAIN_MARKERS.values() if re.search(pattern, text))
    return hits >= 2


def classify_ops_intent(message: str) -> OpsIntentResult | None:
    """Classify a message into a specific whitelisted tool, or None.

    Returns None (rather than a low-confidence guess) when nothing matches,
    so the caller knows to fall back to the general assistant pipeline
    instead of running a tool the question didn't actually ask for.
    """
    if not message or not message.strip():
        return None

    text = message.lower()

    if _is_multi_domain_overview(text):
        return None

    for tool, patterns in _OPS_PATTERNS:
        matches = [p for p in patterns if re.search(p, text)]
        if matches:
            confidence = min(0.65 + 0.1 * len(matches), 0.97)
            return OpsIntentResult(tool=tool, matched_patterns=matches, confidence=confidence)

    return None