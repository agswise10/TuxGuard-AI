"""
Intent classification for the AI Ops Assistant.

A lightweight, fast, fully offline keyword/pattern classifier. It decides
*what kind* of system context needs to be gathered before we ever talk to
the LLM - e.g. "why is my system slow?" needs CPU/memory/process data,
"restart nginx" needs service status.

Deliberately rule-based (not a second LLM call): it's instant, free, and
"good enough" for routing context. The LLM still does all the actual
reasoning and natural-language understanding for the final answer - this
step only decides which system facts are worth fetching first.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    DRIVER_HEALTH = "driver_health"             # "which driver anomalies were detected?"
    HARDWARE_HEALTH = "hardware_health"         # "is any hardware overheating / failing?"
    FAILURE_PREDICTION = "failure_prediction"   # "what's about to fail / when will disk fill up?"
    PERFORMANCE = "performance"              # "why is my system slow?"
    SERVICE_MANAGEMENT = "service_management"  # "restart nginx", "is ssh running?"
    DOCKER = "docker"                          # "show running docker containers"
    FILE_SEARCH = "file_search"                # "find large files"
    LOG_ANALYSIS = "log_analysis"               # "explain this error", "check logs"
    NETWORK = "network"                         # "why is my network slow", "check open ports"
    USERS_SESSIONS = "users_sessions"           # "who is logged in"
    GENERAL = "general"                         # fallback: general Linux Q&A


@dataclass
class IntentResult:
    intent: Intent
    matched_keywords: list[str]
    confidence: float  # heuristic confidence in the classification itself


# Ordered so more specific intents are checked before generic ones.
# NOTE: DRIVER_HEALTH / HARDWARE_HEALTH / FAILURE_PREDICTION are checked
# FIRST, ahead of LOG_ANALYSIS, NETWORK, and PERFORMANCE. Without this,
# a question like "which driver anomalies were detected in the kernel
# logs" matches LOG_ANALYSIS purely because it contains the word "logs",
# routes to generic system_monitor log context instead of the real
# driver_monitor findings, and leaves the LLM to fill the gap with
# invented/irrelevant detail. Putting the specific detector-backed
# intents first ensures they win that race.
_INTENT_PATTERNS: list[tuple[Intent, list[str]]] = [
    (
        Intent.DRIVER_HEALTH,
        [
            r"\bdriver(s)?\b",
            r"\bkernel\s+module(s)?\b",
            r"\bmodprobe\b",
            r"\bamdgpu\b",
            r"\bnvidia\b",
            r"\bkernel\s+log(s)?\b",
            r"\bdmesg\b",
            # NOTE: deliberately no bare `\bgpu\b` here - it over-matched
            # thermal/health questions like "what are the current CPU and
            # GPU temperatures?", stealing them from Intent.HARDWARE_HEALTH
            # below (which is what actually has the real temperature/
            # thermal detector data). A GPU *driver* question still
            # matches via `\bdriver(s)?\b`, `\bamdgpu\b`, or `\bnvidia\b`
            # above, so no legitimate driver-anomaly question is lost.
        ],
    ),
    (
        Intent.HARDWARE_HEALTH,
        [
            r"\bhardware\b",
            r"\boverheat(ing)?\b",
            r"\btemperature(s)?\b",
            r"\bthermal\b",
            r"\bfan\s+(failure|fault|speed)\b",
            r"\bsmart\b.*\bdisk\b",
            r"\bdisk\s+health\b",
            r"\bcpu\s+temp(erature)?\b",
        ],
    ),
    (
        Intent.FAILURE_PREDICTION,
        [
            r"\bpredict(ed|ion)?\b",
            r"\bupcoming\s+failure(s)?\b",
            r"\babout\s+to\s+fail\b",
            r"\bgoing\s+to\s+(run\s+out|fail)\b",
            r"\bwill\s+.*\bfail\b",
            r"\b(memory|disk|cpu)\s+exhaustion\b",
            r"\beta\b.*\b(fail|exhaust|full)\b",
            r"\bforecast\b",
            r"\btrend(ing)?\b.*\b(fail|exhaust|full)\b",
        ],
    ),
    (
        Intent.DOCKER,
        [
            r"\bdocker\b",
            r"\bcontainer(s)?\b",
            r"\bdocker[- ]compose\b",
            r"\bimage(s)?\b.*\bdocker\b",
        ],
    ),
    (
        Intent.SERVICE_MANAGEMENT,
        [
            r"\brestart\b",
            r"\bstop\b.*\bservice\b",
            r"\bstart\b.*\bservice\b",
            r"\bsystemctl\b",
            r"\bnginx\b",
            r"\bapache\b",
            r"\bservice\s+status\b",
            r"\bis\s+\w+\s+running\b",
            r"\benable\b.*\bservice\b",
        ],
    ),
    (
        Intent.FILE_SEARCH,
        [
            r"\blarge\s+files?\b",
            r"\bdisk\s+space\b",
            r"\bwhat.?s\s+using\s+(my\s+)?disk\b",
            r"\bfind\s+files?\b",
            r"\bfree\s+up\s+space\b",
        ],
    ),
    (
        Intent.LOG_ANALYSIS,
        [
            r"\berror\b",
            r"\bexception\b",
            r"\bstack\s?trace\b",
            r"\blogs?\b",
            r"\bjournal\b",
            r"\bwhy\s+did\s+.*\bfail\b",
            r"\bcrash(ed)?\b",
            r"\bexplain\s+this\b",
        ],
    ),
    (
        Intent.NETWORK,
        [
            r"\bnetwork\b",
            r"\bbandwidth\b",
            r"\bport(s)?\b",
            r"\bconnection(s)?\b",
            r"\bping\b",
            r"\bfirewall\b",
            r"\binternet\b",
        ],
    ),
    (
        Intent.USERS_SESSIONS,
        [
            r"\bwho\s+is\s+logged\s+in\b",
            r"\blogged.?in\s+users?\b",
            r"\bactive\s+sessions?\b",
            r"\bwho\s+is\s+on\s+(this|the)\s+system\b",
        ],
    ),
    (
        Intent.PERFORMANCE,
        [
            r"\bslow\b",
            r"\bhigh\s+cpu\b",
            r"\bcpu\s+usage\b",
            r"\bmemory\s+usage\b",
            r"\bram\b",
            r"\bload\s+average\b",
            r"\bperformance\b",
            r"\blagg(y|ing)\b",
            r"\bfreez(e|ing)\b",
            r"\bhang(ing|s)?\b",
        ],
    ),
]


def classify_intent(message: str) -> IntentResult:
    """Classify a natural-language message into an operational intent.

    Runs each intent's patterns against the message (case-insensitive) and
    returns the first intent with at least one match, ordered by
    specificity. Falls back to GENERAL if nothing matches.
    """
    if not message or not message.strip():
        return IntentResult(intent=Intent.GENERAL, matched_keywords=[], confidence=0.0)

    text = message.lower()

    for intent, patterns in _INTENT_PATTERNS:
        matches = [p for p in patterns if re.search(p, text)]
        if matches:
            # Confidence scales gently with number of distinct pattern hits,
            # capped so it never claims false certainty.
            confidence = min(0.6 + 0.1 * len(matches), 0.95)
            return IntentResult(intent=intent, matched_keywords=matches, confidence=confidence)

    return IntentResult(intent=Intent.GENERAL, matched_keywords=[], confidence=0.4)