"""
Deterministic System Chat - a small, rule-based fast path for the AI Chat
that answers common CPU/RAM/disk/temperature/process/driver/kernel/hardware
questions directly from real system collectors, bypassing LLM reasoning
entirely for that fixed set of questions.

Why this exists
----------------
`ai_assistant.process_query()` already has two LLM-backed paths (the
original context-builder pipeline and the Sprint 9 tool-grounded ops
pipeline) - both still call the LLM to phrase the final answer, so both
are still subject to occasional unreliable/inconsistent phrasing from the
model. For a small, well-known set of "what is my CPU usage" style
questions, that unreliability is unnecessary: the answer is a couple of
real numbers from `system_monitor.py` / `hardware_monitor.py` /
`driver_monitor.py`, formatted into a sentence. This module recognizes
those specific questions and answers them directly from live data, with
no LLM call in the loop at all.

Flow (see `try_handle`)
------------------------
    user message
        -> normalize
        -> keyword/regex rule matcher (`_CATEGORIES`, most specific first)
        -> matched?  -> run the matching handler against real collectors
                         -> deterministic natural-language answer
        -> no match? -> return None, caller falls through to the existing
                         LLM pipeline (tool-grounded ops path or the
                         original context-builder + LLM path) unchanged.

This module never talks to the LLM, never executes anything beyond the
existing read-only collectors it calls, and never touches the frontend,
the response schema, remediation/fix engine, or any other route. It only
changes what `POST /api/assistant/chat` answers for the specific
questions it recognizes - see `ai_assistant.process_query` for the single
call site that wires it in ahead of the existing pipelines.

Safety
------
Every handler below only *reads* system state via the existing collector
modules (`system_monitor`, `hardware_monitor`, `driver_monitor`). Nothing
here executes remediation, restarts anything, kills processes, loads or
unloads kernel modules, writes files, or touches
`fix_engine`/`issue_alert_store`'s mutation paths. This module is
read-only by construction: it has no import of, or call into, anything
that can change host state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from app.logger import get_logger
from app.services import driver_monitor, hardware_monitor, system_monitor

logger = get_logger(__name__)

_BYTES_PER_GB = 1024**3


def _gb(num_bytes: float | int | None) -> float:
    if not num_bytes:
        return 0.0
    return round(num_bytes / _BYTES_PER_GB, 1)


def _pct(value: float | None) -> str:
    return f"{value:.1f}%" if isinstance(value, (int, float)) else "unknown"


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass
class DeterministicResult:
    intent: str
    explanation: str
    context_summary: dict
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "explanation": self.explanation,
            "recommended_commands": [],
            "confidence_score": 1.0,
            "reasoning": (
                "Answered deterministically from live system data "
                f"(source: {self.context_summary.get('source', 'system collectors')}); "
                "no LLM reasoning was used for this response."
            ),
            "context_summary": self.context_summary,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Handlers - each one reads real data via the existing collectors and
# returns a DeterministicResult. No handler ever invents a value.
# ---------------------------------------------------------------------------


def _handle_temperature(message: str) -> DeterministicResult:
    reading = hardware_monitor.get_current_sensor_readings()
    sensors = reading.get("sensors") or []

    if not sensors:
        explanation = (
            "Temperature sensor data is currently unavailable on this system, so I can't "
            "report an actual CPU or GPU temperature reading. This is common on VMs, "
            "containers, and some cloud hosts where the kernel doesn't expose hardware "
            "sensors. I won't guess a value - if you have `lm-sensors` installed, `sensors` "
            "may expose more detail directly on the host."
        )
        return DeterministicResult(
            intent="cpu_temperature",
            explanation=explanation,
            context_summary={"source": "hardware_monitor.get_current_sensor_readings", "sensor_count": 0},
            warnings=["No temperature sensors were available on this host."],
        )

    cpu_keywords = ("cpu", "core", "package", "k10temp", "coretemp", "zenpower")
    cpu_sensor = next(
        (s for s in sensors if any(k in (s.get("chip", "") + s.get("label", "")).lower() for k in cpu_keywords)),
        None,
    )
    primary = cpu_sensor or sensors[0]

    lines = [
        f"Your current {primary.get('label') or primary.get('chip')} temperature is "
        f"{primary.get('current_celsius')}\u00b0C."
    ]

    other_sensors = [s for s in sensors if s is not primary]
    if other_sensors:
        extra = ", ".join(
            f"{s.get('label') or s.get('chip')}: {s.get('current_celsius')}\u00b0C" for s in other_sensors[:4]
        )
        lines.append(f"Other sensor readings: {extra}.")

    flagged = [
        s
        for s in sensors
        if isinstance(s.get("high_celsius"), (int, float)) and s.get("current_celsius", 0) >= s["high_celsius"]
    ]
    if flagged:
        lines.append(
            "One or more sensors are at or above their configured warning threshold, "
            "which suggests the system is running hotter than normal right now."
        )
    else:
        lines.append("These readings are within the normal range for the sensors reporting them.")

    return DeterministicResult(
        intent="cpu_temperature",
        explanation=" ".join(lines),
        context_summary={
            "source": "hardware_monitor.get_current_sensor_readings",
            "sensor_count": len(sensors),
            "primary_sensor": primary.get("label") or primary.get("chip"),
        },
        warnings=[],
    )


async def _handle_driver_problem(message: str) -> DeterministicResult:
    candidates = await driver_monitor.detect_driver_anomalies()

    if not candidates:
        explanation = (
            "No driver anomalies were found in the recent kernel logs. Your drivers appear "
            "to be functioning normally right now, based on the most recent kernel log scan."
        )
        return DeterministicResult(
            intent="driver_health",
            explanation=explanation,
            context_summary={"source": "driver_monitor.detect_driver_anomalies", "anomaly_count": 0},
            warnings=[],
        )

    parts = [f"I found {len(candidates)} driver anomaly pattern(s) in the recent kernel logs:"]
    for candidate in candidates[:5]:
        evidence = candidate.get("evidence", {})
        occurrences = evidence.get("occurrence_count", "an unknown number of")
        parts.append(
            f"- {candidate['title']} (severity: {candidate['severity']}): {candidate['problem']} "
            f"Seen {occurrences} time(s) in the scanned log window."
        )
    parts.append(
        "This indicates a driver-level anomaly in the kernel logs; it does not by itself "
        "confirm a hardware failure - see the hardware health check for sensor/SMART evidence."
    )

    return DeterministicResult(
        intent="driver_health",
        explanation="\n".join(parts),
        context_summary={
            "source": "driver_monitor.detect_driver_anomalies",
            "anomaly_count": len(candidates),
            "categories": [c.get("evidence", {}).get("category") for c in candidates],
        },
        warnings=[],
    )


def _handle_driver_inventory(message: str) -> DeterministicResult:
    data = driver_monitor.get_loaded_kernel_modules(limit=50)
    total = data.get("total_modules", 0)
    modules = data.get("modules", [])

    if data.get("errors"):
        explanation = (
            "I couldn't read the loaded kernel modules on this host "
            f"({'; '.join(data['errors'])})."
        )
        return DeterministicResult(
            intent="driver_inventory",
            explanation=explanation,
            context_summary={"source": "driver_monitor.get_loaded_kernel_modules", "total_modules": 0},
            warnings=list(data["errors"]),
        )

    example_names = ", ".join(m["name"] for m in modules[:4])
    explanation = (
        f"Your system currently has {total} loaded kernel module(s). "
        f"Examples include {example_names}, and other Linux kernel modules."
        if example_names
        else f"Your system currently has {total} loaded kernel module(s)."
    )

    return DeterministicResult(
        intent="driver_inventory",
        explanation=explanation,
        context_summary={"source": "driver_monitor.get_loaded_kernel_modules", "total_modules": total},
        warnings=[],
    )


async def _handle_hardware_health(message: str) -> DeterministicResult:
    hardware_candidates = await hardware_monitor.detect_hardware_issues()
    driver_candidates = await driver_monitor.detect_driver_anomalies()
    disk = system_monitor.get_disk_info()
    disk_problems = [p for p in disk.get("partitions", []) if (p.get("usage_percent") or 0) >= 90]

    parts: list[str] = []

    if hardware_candidates:
        titles = "; ".join(f"{c['title']} ({c['severity']})" for c in hardware_candidates[:5])
        parts.append(f"Hardware faults detected: {titles}.")
    else:
        parts.append("No hardware faults (temperature, fan, or disk SMART) were detected.")

    if driver_candidates:
        titles = "; ".join(f"{c['title']} ({c['severity']})" for c in driver_candidates[:5])
        parts.append(f"Driver/kernel anomalies detected in kernel logs: {titles}.")
    else:
        parts.append("No driver or kernel anomalies were found in the recent kernel logs.")

    if disk_problems:
        mounts = "; ".join(f"{p['mountpoint']} at {_pct(p.get('usage_percent'))}" for p in disk_problems)
        parts.append(f"Disk/storage usage is critically high on: {mounts}.")
    else:
        parts.append("No disk/storage partitions are critically low on space.")

    any_issue = bool(hardware_candidates or driver_candidates or disk_problems)
    if any_issue:
        parts.append(
            "Overall: at least one category above shows an active issue - review the specific "
            "item(s) flagged rather than treating the whole system as healthy or unhealthy."
        )
    else:
        parts.append(
            "Overall: no active hardware faults, driver anomalies, or critical disk usage were "
            "found in the categories checked (hardware sensors, kernel logs, disk usage). This "
            "reflects only those checks, not an exhaustive guarantee of full system health."
        )

    return DeterministicResult(
        intent="hardware_health",
        explanation=" ".join(parts),
        context_summary={
            "source": "hardware_monitor.detect_hardware_issues + driver_monitor.detect_driver_anomalies + system_monitor.get_disk_info",
            "hardware_fault_count": len(hardware_candidates),
            "driver_anomaly_count": len(driver_candidates),
            "disk_problem_count": len(disk_problems),
        },
        warnings=[],
    )


def _handle_cpu_performance(message: str) -> DeterministicResult:
    cpu = system_monitor.get_cpu_info()
    usage = cpu.get("usage_percent") or 0.0
    is_high = usage >= 80.0
    status = "high" if is_high else "within a normal range"

    parts = [f"Your current CPU usage is {_pct(usage)}. The system is currently operating {status}."]

    if "why" in message or is_high:
        processes = system_monitor.get_processes(limit=3, sort_by="cpu_percent")
        top = [p for p in processes.get("processes", []) if (p.get("cpu_percent") or 0) > 0]
        if top:
            listing = ", ".join(f"{p['name']} ({_pct(p.get('cpu_percent'))})" for p in top)
            parts.append(f"The top CPU-consuming process(es) right now: {listing}.")
        load = cpu.get("load_avg_1m")
        if load is not None:
            parts.append(f"1-minute load average is {load} across {cpu.get('logical_cores')} logical core(s).")

    return DeterministicResult(
        intent="cpu_performance",
        explanation=" ".join(parts),
        context_summary={"source": "system_monitor.get_cpu_info", "usage_percent": usage},
        warnings=[],
    )


def _handle_cpu_identity(message: str) -> DeterministicResult:
    cpu = system_monitor.get_cpu_info()
    model_name = cpu.get("model_name")
    if not model_name:
        explanation = (
            "I couldn't determine the exact CPU model on this host (no readable /proc/cpuinfo "
            "or lscpu output), but it has "
            f"{cpu.get('physical_cores') or 'an unknown number of'} physical core(s) and "
            f"{cpu.get('logical_cores') or 'an unknown number of'} logical thread(s)."
        )
    else:
        explanation = (
            f"Your system is using a {model_name} CPU with {cpu.get('physical_cores')} core(s) "
            f"and {cpu.get('logical_cores')} thread(s)."
        )
    return DeterministicResult(
        intent="cpu_identity",
        explanation=explanation,
        context_summary={"source": "system_monitor.get_cpu_info", "model_name": model_name},
        warnings=[],
    )


def _handle_cpu_cores(message: str) -> DeterministicResult:
    cpu = system_monitor.get_cpu_info()
    wants_threads = bool(re.search(r"\bthreads?\b", message))
    if wants_threads and not re.search(r"\bcores?\b", message):
        explanation = f"Your CPU has {cpu.get('logical_cores')} logical thread(s)."
    else:
        explanation = (
            f"Your CPU has {cpu.get('physical_cores')} physical core(s) and "
            f"{cpu.get('logical_cores')} logical thread(s)."
        )
    return DeterministicResult(
        intent="cpu_cores",
        explanation=explanation,
        context_summary={
            "source": "system_monitor.get_cpu_info",
            "physical_cores": cpu.get("physical_cores"),
            "logical_cores": cpu.get("logical_cores"),
        },
        warnings=[],
    )


def _handle_cpu_frequency(message: str) -> DeterministicResult:
    cpu = system_monitor.get_cpu_info()
    freq = cpu.get("frequency_mhz")
    if freq is None:
        explanation = "CPU frequency data is not available from this host."
    else:
        explanation = f"Your CPU is currently running at {freq} MHz."
    return DeterministicResult(
        intent="cpu_frequency",
        explanation=explanation,
        context_summary={"source": "system_monitor.get_cpu_info", "frequency_mhz": freq},
        warnings=[],
    )


def _handle_cpu_usage(message: str) -> DeterministicResult:
    cpu = system_monitor.get_cpu_info()
    usage = cpu.get("usage_percent") or 0.0
    status = "high" if usage >= 80.0 else "within a normal range"
    explanation = f"Your current CPU usage is {_pct(usage)}. The system is currently operating {status}."
    return DeterministicResult(
        intent="cpu_usage",
        explanation=explanation,
        context_summary={"source": "system_monitor.get_cpu_info", "usage_percent": usage},
        warnings=[],
    )


def _handle_memory(message: str) -> DeterministicResult:
    mem = system_monitor.get_memory_info()
    available_gb = _gb(mem.get("available_bytes"))
    total_gb = _gb(mem.get("total_bytes"))
    used_percent = mem.get("usage_percent") or 0.0

    if "available" in message or "free" in message:
        explanation = f"Your system currently has {available_gb} GB of RAM available out of {total_gb} GB total."
    elif "high" in message:
        status = "high" if used_percent >= 80.0 else "not unusually high"
        explanation = (
            f"Your current RAM usage is {_pct(used_percent)} "
            f"({_gb(mem.get('used_bytes'))} GB used of {total_gb} GB). That is {status} right now."
        )
    else:
        explanation = (
            f"Your current RAM usage is {_pct(used_percent)} "
            f"({_gb(mem.get('used_bytes'))} GB used, {available_gb} GB available, {total_gb} GB total)."
        )

    return DeterministicResult(
        intent="memory_usage",
        explanation=explanation,
        context_summary={
            "source": "system_monitor.get_memory_info",
            "usage_percent": used_percent,
            "available_gb": available_gb,
            "total_gb": total_gb,
        },
        warnings=[],
    )


def _handle_disk(message: str) -> DeterministicResult:
    disk = system_monitor.get_disk_info()
    partitions = disk.get("partitions", [])

    if not partitions:
        return DeterministicResult(
            intent="disk_usage",
            explanation="I couldn't read any disk partition usage on this host.",
            context_summary={"source": "system_monitor.get_disk_info", "partition_count": 0},
            warnings=["No disk partitions were reported."],
        )

    # The busiest real (non-error) partition is what "which disk" / "is my
    # disk full" questions care about.
    valid = [p for p in partitions if p.get("error") is None]
    busiest = max(valid, key=lambda p: p.get("usage_percent") or 0, default=partitions[0])

    if "which disk" in message:
        explanation = (
            f"Your busiest disk is mounted at {busiest['mountpoint']} ({busiest['device']}), "
            f"currently at {_pct(busiest.get('usage_percent'))} usage."
        )
    elif "available" in message or "free" in message:
        free_gb = _gb(busiest.get("free_bytes"))
        explanation = (
            f"You have {free_gb} GB of free disk space available on {busiest['mountpoint']} "
            f"({_pct(busiest.get('usage_percent'))} used)."
        )
    elif "full" in message:
        is_full = (busiest.get("usage_percent") or 0) >= 90
        explanation = (
            f"{'Yes' if is_full else 'No'} - {busiest['mountpoint']} is at "
            f"{_pct(busiest.get('usage_percent'))} usage, "
            f"{'which is critically full.' if is_full else 'which is not critically full.'}"
        )
    else:
        explanation = (
            f"Disk usage on {busiest['mountpoint']} ({busiest['device']}) is "
            f"{_pct(busiest.get('usage_percent'))}, with {_gb(busiest.get('free_bytes'))} GB free "
            f"out of {_gb(busiest.get('total_bytes'))} GB total."
        )

    return DeterministicResult(
        intent="disk_usage",
        explanation=explanation,
        context_summary={
            "source": "system_monitor.get_disk_info",
            "partition_count": len(partitions),
            "busiest_mountpoint": busiest.get("mountpoint"),
            "usage_percent": busiest.get("usage_percent"),
        },
        warnings=[],
    )


def _handle_processes(message: str) -> DeterministicResult:
    sort_by = "memory_percent" if "memory" in message else "cpu_percent"
    data = system_monitor.get_processes(limit=5, sort_by=sort_by)
    processes = data.get("processes", [])

    if not processes:
        explanation = "I couldn't read any running processes on this host."
        return DeterministicResult(
            intent="process_list",
            explanation=explanation,
            context_summary={"source": "system_monitor.get_processes", "process_count": 0},
            warnings=data.get("errors", []),
        )

    metric_key = sort_by
    metric_label = "memory" if sort_by == "memory_percent" else "CPU"

    if re.search(r"\bwhich process(es)?\b", message) or "most" in message or "consuming" in message or "using" in message:
        top = processes[0]
        explanation = (
            f"The process using the most {metric_label} right now is '{top['name']}' "
            f"(PID {top['pid']}) at {_pct(top.get(metric_key))} {metric_label} usage."
        )
    else:
        listing = ", ".join(f"{p['name']} ({_pct(p.get(metric_key))})" for p in processes[:5])
        explanation = (
            f"Top {len(processes[:5])} processes by {metric_label} usage right now: {listing}. "
            f"({data.get('total_processes')} processes total.)"
        )

    return DeterministicResult(
        intent="process_list",
        explanation=explanation,
        context_summary={
            "source": "system_monitor.get_processes",
            "sorted_by": sort_by,
            "process_count": data.get("total_processes"),
        },
        warnings=[],
    )


def _handle_kernel_system(message: str) -> DeterministicResult:
    version = system_monitor.get_system_version()
    uptime_seconds = system_monitor.get_uptime_seconds()

    days, remainder = divmod(int(uptime_seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    uptime_str = (f"{days}d " if days else "") + f"{hours}h {minutes}m"

    if "kernel" in message:
        explanation = f"You're running Linux kernel version {version.get('kernel_version')}."
    elif "hostname" in message:
        explanation = f"Your hostname is {version.get('hostname')}."
    elif "long" in message or "uptime" in message:
        explanation = f"Your system has been running for {uptime_str} (since last boot)."
    elif "operating system" in message or re.search(r"\bwhat\s+system\b", message):
        explanation = (
            f"You're running {version.get('ubuntu_version') or 'Linux'} "
            f"(kernel {version.get('kernel_version')}, {version.get('architecture')} architecture)."
        )
    else:
        explanation = (
            f"You're running {version.get('ubuntu_version') or 'Linux'} with kernel "
            f"{version.get('kernel_version')} on host '{version.get('hostname')}', up for {uptime_str}."
        )

    return DeterministicResult(
        intent="kernel_system_info",
        explanation=explanation,
        context_summary={
            "source": "system_monitor.get_system_version + system_monitor.get_uptime_seconds",
            "kernel_version": version.get("kernel_version"),
            "uptime_seconds": round(uptime_seconds),
        },
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Category registry - ordered MOST SPECIFIC FIRST. The first category whose
# any pattern matches wins; everything else is never checked for that
# message. See module docstring / README notes for why this ordering
# matters (e.g. "why is my CPU usage high" must not fall into a generic
# CPU-identity or CPU-usage bucket; "is there a driver problem" must not
# fall into the driver-inventory bucket).
# ---------------------------------------------------------------------------

_SyncHandler = Callable[[str], DeterministicResult]
_AsyncHandler = Callable[[str], Awaitable[DeterministicResult]]


@dataclass
class _Category:
    name: str
    patterns: list[str]
    handler: _SyncHandler | _AsyncHandler
    is_async: bool = False


_CATEGORIES: list[_Category] = [
    _Category(
        "temperature",
        [
            r"\btemperature",
            r"\boverheat",
            r"\btoo hot\b",
            r"\brunning hot\b",
            r"\btemperatures?\s+normal\b",
        ],
        _handle_temperature,
    ),
    _Category(
        "driver_problem",
        [
            r"\bdriver(s)?\b.*\b(problem|error|issue|fault|fail|healthy|health)\b",
            r"\b(problem|error|issue|fault|fail)\b.*\bdriver(s)?\b",
        ],
        _handle_driver_problem,
        is_async=True,
    ),
    _Category(
        "driver_inventory",
        [
            r"\b(which|what)\s+drivers?\b.*\b(loaded|installed)\b",
            r"\bloaded\s+(kernel\s+)?(driver|module)s?\b",
            r"\bkernel\s+modules?\b.*\bloaded\b",
            r"\bshow\b.*\b(loaded\s+)?(driver|module)s?\b",
            r"\blist\b.*\bdrivers?\b",
        ],
        _handle_driver_inventory,
    ),
    _Category(
        "hardware_health",
        [
            r"\bhardware\b.*\b(problem|fault|issue|failure|anomal)",
            r"\b(problem|fault|issue|failure|anomal)\w*\b.*\bhardware\b",
            r"\bis\s+my\s+system\s+healthy\b",
            r"\bsystem\s+healthy\b",
            r"\bdetect\b.*\bhardware\b",
            r"\bhardware\s+health\b",
        ],
        _handle_hardware_health,
        is_async=True,
    ),
    _Category(
        "processes",
        [
            r"\brunning\s+processes\b",
            r"\bshow\s+me\b.*\bprocesses\b",
            r"\bwhat\s+processes\s+are\s+running\b",
            r"\bwhich\s+process(es)?\b",
            r"\bwhat('?s| is)\s+using\s+my\s+cpu\b",
        ],
        _handle_processes,
    ),
    _Category(
        "cpu_performance",
        [
            r"\bwhy\b.*\bcpu\b.*\bhigh\b",
            r"\bcpu\b.*\b(usage|utilization)\b.*\bhigh\b",
            r"\bis\s+my\s+cpu\s+usage\s+high\b",
            r"\bhigh\s+cpu\b",
        ],
        _handle_cpu_performance,
    ),
    _Category(
        "cpu_identity",
        [
            r"\bwhich\s+cpu\b",
            r"\bwhat\s+cpu\s+do\s+i\s+have\b",
            r"\bwhat\s+cpu\s+is\b",
            r"\bwhat\s+processor\b",
        ],
        _handle_cpu_identity,
    ),
    _Category(
        "cpu_cores",
        [
            r"\bhow\s+many\s+.*\bcores?\b",
            r"\bhow\s+many\s+.*\bthreads?\b",
            r"\bcpu\s+cores?\b",
            r"\bcpu\s+threads?\b",
        ],
        _handle_cpu_cores,
    ),
    _Category(
        "cpu_frequency",
        [r"\bcpu\s+frequency\b", r"\bclock\s+speed\b"],
        _handle_cpu_frequency,
    ),
    _Category(
        "cpu_usage",
        [r"\bcpu\s+usage\b", r"\bcurrent\s+cpu\b", r"\bcpu\s+utilization\b"],
        _handle_cpu_usage,
    ),
    _Category(
        "memory",
        [
            r"\bram\b",
            r"\bmemory\s+usage\b",
            r"\bmemory\s+available\b",
            r"\bfree\s+memory\b",
            r"\bmemory\s+high\b",
        ],
        _handle_memory,
    ),
    _Category(
        "disk",
        [
            r"\bdisk\s+usage\b",
            r"\bdisk\s+space\b",
            r"\bdisk\s+full\b",
            r"\bwhich\s+disk\b",
            r"\bfree\s+disk\s+space\b",
        ],
        _handle_disk,
    ),
    _Category(
        "kernel_system",
        [
            r"\bkernel\s+version\b",
            r"\blinux\s+version\b",
            r"\bhostname\b",
            r"\bhow\s+long\b.*\brunning\b",
            r"\buptime\b",
            r"\boperating\s+system\b",
            r"\bwhat\s+system\b.*\brunning\b",
        ],
        _handle_kernel_system,
    ),
]


def _normalize(message: str) -> str:
    return re.sub(r"\s+", " ", (message or "").strip().lower())


def _match_category(text: str) -> _Category | None:
    for category in _CATEGORIES:
        for pattern in category.patterns:
            if re.search(pattern, text):
                return category
    return None


async def try_handle(message: str) -> dict | None:
    """Return a deterministic ChatResponse-shaped dict, or None if the
    message doesn't match any known system/hardware question category.

    Callers (see `ai_assistant.process_query`) should fall through to the
    existing LLM pipeline(s) unchanged whenever this returns None.
    """
    text = _normalize(message)
    if not text:
        return None

    category = _match_category(text)
    if category is None:
        return None

    try:
        if category.is_async:
            result = await category.handler(text)  # type: ignore[misc]
        else:
            result = category.handler(text)  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001 - a collector hiccup must never crash chat
        logger.error("Deterministic handler for category '%s' failed: %s", category.name, exc)
        return None

    logger.info("Deterministic system chat matched category=%s intent=%s", category.name, result.intent)
    return result.to_dict()
