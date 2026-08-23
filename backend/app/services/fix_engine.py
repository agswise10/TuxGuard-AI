"""
AI One-Click Fix Engine.

Detects a fixed set of common Ubuntu operational issues:

    - High CPU
    - High Memory
    - Disk Almost Full
    - Apache Down
    - Docker Container Stopped
    - Failed Service

For every issue found it produces the full explainability chain requested
by the product spec:

    Problem -> Reason -> Evidence -> Confidence Score -> Recommended Command

using the local LLM for the "Reason"/confidence/command triple, with a
deterministic fallback so the engine still works with Ollama offline.

Alert deduplication: each detector produces a cheap, AI-free "candidate"
(issue_id/type/title/problem/evidence/severity/fallback_command). Every
scan hands its full set of candidates to `issue_alert_store.reconcile()`,
which is the single place that decides, per `issue_id`, whether this is a
brand new issue (diagnose + create), the same issue with materially
different evidence (diagnose + update in place), the same issue with
unchanged evidence (a duplicate scan - ignored, no re-diagnosis, no new
alert), or an issue that's no longer observed (resolved - dropped). This
guarantees at most one active alert per issue_id, and that AI diagnosis
only ever runs for genuinely new or changed issues. See
`issue_alert_store.py` for the dedup logic itself.

Nothing in this module executes a command. Turning a recommendation into
a "One Click Fix" only ever *prepares* it: it runs the exact same
deterministic safety analyzer and persistence used everywhere else in the
project (`command_safety.analyze_command` + `execution_store.save_pending`,
via `command_console._build_preview`) so the resulting entry can only ever
be run through the existing, unmodified

    POST /api/commands/{execution_id}/confirm

endpoint, after an explicit user confirmation - exactly like AI-generated
and user-provided commands elsewhere in the app.
"""

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone

from app.config import get_settings
from app.logger import get_logger
from app.services import (
    command_console,
    docker_monitor,
    driver_monitor,
    failure_predictor,
    hardware_monitor,
    issue_alert_store,
    system_monitor,
)
from app.services.ollama_client import OllamaUnavailableError, chat

settings = get_settings()
logger = get_logger(__name__)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _severity_for(value: float, threshold: float) -> str:
    return "critical" if value >= threshold + 5 else "warning"


# ---------------------------------------------------------------------------
# Interactive / long-running / blocking command guard.
#
# `command_executor.execute()` runs every command non-interactively (no
# TTY, no stdin) and kills it after a fixed timeout. A command that is
# perfectly "safe" from a destructive standpoint - e.g. `systemctl
# status <service>` - can still sit there until it hits that timeout,
# because it (or its pager) was written for a human watching a terminal,
# not a one-shot automated fix. That reads to the user as "the fix
# failed" even though nothing actually went wrong.
#
# This guard runs on every AI-diagnosed `recommended_command` before it
# is ever handed back to the frontend: if it looks interactive,
# long-running, or otherwise likely to hang, it is swapped out for the
# issue's own deterministic, actionable `fallback_command` (e.g. a
# `systemctl restart` in place of a `systemctl status`) instead.
# ---------------------------------------------------------------------------
_BLOCKING_COMMAND_PATTERNS: list["re.Pattern[str]"] = [
    re.compile(r"\bsystemctl\b(?:\s+--\S+)*\s+status\b", re.IGNORECASE),
    re.compile(r"\bjournalctl\b[^\n]*(?:-f\b|--follow\b)", re.IGNORECASE),
    re.compile(r"\btail\b[^\n]*(?:-f\b|-F\b|--follow\b)", re.IGNORECASE),
    re.compile(r"\bwatch\b", re.IGNORECASE),
    re.compile(r"\b(?:less|more|most)\b", re.IGNORECASE),
    re.compile(r"\b(?:vi|vim|nvim|nano|emacs)\b", re.IGNORECASE),
    re.compile(r"\b(?:top|htop|iotop|iftop)\b", re.IGNORECASE),
    re.compile(r"\b(?:screen|tmux)\b", re.IGNORECASE),
    re.compile(r"\bman\b", re.IGNORECASE),
    re.compile(r"^\s*read\b", re.IGNORECASE),
    re.compile(r"\bsudo\s+-i\b|\bsu\s+-\b|\bsu\s*$", re.IGNORECASE),
    re.compile(r"\byes\b", re.IGNORECASE),
    re.compile(r"\bping\b(?![^\n]*-c\s*\d)", re.IGNORECASE),
]


def _is_blocking_command(command: str) -> bool:
    """True if `command` looks interactive, long-running, or otherwise

    likely to hang forever under the non-interactive One-Click Fix
    executor, which has no TTY and no way to send input.
    """
    command = (command or "").strip()
    if not command:
        return False
    return any(pattern.search(command) for pattern in _BLOCKING_COMMAND_PATTERNS)


def _ensure_non_blocking(command: str, fallback_command: str) -> str:
    """Guarantee the command handed back to the frontend never hangs.

    If `command` matches a known interactive/long-running/blocking
    pattern, swap it for the issue's own deterministic, actionable
    `fallback_command` (e.g. `sudo systemctl restart <service>`) instead
    of returning something written for an interactive terminal session.
    Otherwise return it unchanged.
    """
    if _is_blocking_command(command):
        logger.warning(
            "Recommended command looked interactive/long-running, replacing with fallback: %r -> %r",
            command,
            fallback_command,
        )
        return fallback_command
    return command


# ---------------------------------------------------------------------------
# Predicted-failure safety guard.
#
# `failure_predictor.py` itself never builds a `kill`/`pkill`/`killall`
# fallback_command for a predicted (not-yet-real) CPU/memory issue - see
# that module's docstring - but `_ai_diagnose()` below still lets the LLM
# freely suggest a `recommended_command`. Nothing stops a model from
# looking at a "top resource-consuming process" in the evidence and
# proposing to kill it, even though that process may be a perfectly
# legitimate workload (Ollama, a browser, an IDE, a build, ...) and the
# metric hasn't actually breached anything yet. This guard is the
# integration-layer backstop for that: regardless of what the LLM (or any
# future diagnosis path) proposes, a `predicted_failure` issue can never
# have a process-termination command handed back to the frontend - it
# always falls back to the detector's own safe, read-only diagnostic
# command instead. Real remediation for an *actual* high-CPU/high-memory
# breach still goes through `_detect_high_cpu`/`_detect_high_memory` and
# the normal reactive flow, exactly as before.
# ---------------------------------------------------------------------------
_PROCESS_KILL_PATTERN = re.compile(r"\b(?:kill|pkill|killall)\b", re.IGNORECASE)


def _is_process_kill_command(command: str) -> bool:
    return bool(_PROCESS_KILL_PATTERN.search((command or "").strip()))


def _ensure_no_prediction_kill(issue_type: str, command: str, fallback_command: str) -> str:
    """For `predicted_failure` issues only: never hand back a process-kill

    command, no matter where it came from (LLM suggestion or a future
    change elsewhere). Every other issue type is returned unchanged - a
    genuinely active `high_cpu`/`high_memory` issue is free to recommend
    whatever the existing behaviour already recommends for it.
    """
    if issue_type == "predicted_failure" and _is_process_kill_command(command):
        logger.warning(
            "Predicted-failure diagnosis recommended a process-termination command, "
            "replacing with the detector's safe diagnostic fallback: %r -> %r",
            command,
            fallback_command,
        )
        return fallback_command
    return command


# ---------------------------------------------------------------------------
# AI diagnosis: Reason + Confidence Score + Recommended Command
# (deterministic fallback if Ollama is unreachable)
# ---------------------------------------------------------------------------

def _fallback_reason(issue_type: str, evidence: dict) -> str:
    top = evidence.get("top_process") or {}
    reasons = {
        "high_cpu": (
            f"CPU usage is at {evidence.get('cpu_percent')}%, above the "
            f"{evidence.get('threshold')}% threshold. The top consumer is "
            f"'{top.get('name', 'an unknown process')}' (PID {top.get('pid', '?')})."
        ),
        "high_memory": (
            f"Memory usage is at {evidence.get('memory_percent')}%, above the "
            f"{evidence.get('threshold')}% threshold. The top consumer is "
            f"'{top.get('name', 'an unknown process')}' (PID {top.get('pid', '?')})."
        ),
        "disk_full": (
            (
                f"{evidence.get('mount_count')} writable filesystems are above the "
                f"{evidence.get('threshold')}% usage threshold: {evidence.get('affected_mounts')}."
            )
            if "affected_mounts" in evidence
            else (
                f"Disk usage on {evidence.get('mountpoint')} is at {evidence.get('disk_percent')}%, "
                f"above the {evidence.get('threshold')}% threshold."
            )
        ),
        "apache_down": (
            f"The Apache service '{evidence.get('service')}' is {evidence.get('active_state')}, "
            "expected 'active'."
        ),
        "docker_stopped": (
            f"Docker container '{evidence.get('container')}' is not running "
            f"(state: {evidence.get('state')})."
        ),
        "failed_service": (
            f"Systemd service '{evidence.get('service')}' has failed "
            f"(state: {evidence.get('active_state')})."
        ),
        "predicted_failure": (
            f"Trend analysis of recent samples projects this metric will reach its "
            f"{evidence.get('threshold')}% threshold in about {evidence.get('eta_minutes')} "
            "minute(s) if the current trend continues "
            f"(current: {evidence.get('current_percent')}%, "
            f"trend: {evidence.get('trend_percent_per_minute')}%/min, "
            f"confidence: {evidence.get('confidence')}, "
            f"fit quality R^2={evidence.get('fit_r_squared')} over "
            f"{evidence.get('samples_analyzed')} samples)."
        ),
        "hardware_fault": _hardware_fault_reason(evidence),
        "driver_anomaly": (
            f"Kernel log shows {evidence.get('occurrence_count')} occurrence(s) of a "
            f"'{evidence.get('category')}' anomaly pattern across the last "
            f"{evidence.get('log_lines_scanned')} scanned kernel log lines."
            + (
                f" Classification: {evidence.get('classification_note')}"
                if evidence.get("classification_note")
                else ""
            )
        ),
    }
    return reasons.get(issue_type, "A system issue was detected.")


def _hardware_fault_reason(evidence: dict) -> str:
    """Build a `hardware_fault` reason from whichever real evidence fields

    `hardware_monitor.py` actually populated for this candidate (a
    temperature reading, a stalled fan, or SMART disk health data are
    each shaped differently), instead of a generic evidence dump. Never
    invents a value that isn't present - a field simply isn't mentioned
    if the evidence doesn't have it, and a candidate whose shape isn't
    recognized at all still falls back to the raw evidence.
    """
    if evidence.get("current_celsius") is not None:
        return (
            f"Sensor '{evidence.get('label', evidence.get('sensor', 'unknown'))}' reads "
            f"{evidence.get('current_celsius')}\u00b0C, at or above the "
            f"{evidence.get('warning_threshold_celsius')}\u00b0C warning threshold "
            f"(critical threshold: {evidence.get('critical_threshold_celsius')}\u00b0C)."
        )
    if evidence.get("current_rpm") is not None:
        return (
            f"Fan '{evidence.get('label', evidence.get('sensor', 'unknown'))}' reports "
            f"{evidence.get('current_rpm')} RPM while system temperature is elevated "
            f"({evidence.get('max_temperature_celsius')}\u00b0C, at or above the "
            f"{evidence.get('temperature_warning_threshold_celsius')}\u00b0C warning threshold) - "
            "a stalled fan under heat load is an early hardware-failure signal."
        )
    if "smart_overall_verdict" in evidence:
        return (
            f"SMART health data for disk '{evidence.get('device', 'unknown')}' shows real evidence "
            f"of a drive problem: overall verdict={evidence.get('smart_overall_verdict')}, "
            f"critical warning={evidence.get('smart_critical_warning')}, "
            f"media/data integrity errors={evidence.get('smart_media_and_data_integrity_errors')}."
        )
    return (
        f"Hardware health check '{evidence.get('label', evidence.get('sensor', evidence.get('device', 'unknown')))}' "
        "reported values outside safe operating range. "
        f"Evidence: {json.dumps(evidence)}"
    )


def _fallback_confidence(issue_type: str, evidence: dict) -> float:
    """Deterministic confidence score used when the LLM is unavailable.

    `predicted_failure` candidates already carry their own explainable
    confidence (see `failure_predictor._confidence_score` - derived from
    the trend fit's R^2 and sample-window fullness), so reuse that value
    here instead of a generic constant: it keeps the AI-diagnosis and
    offline-fallback paths telling the same story for the same evidence,
    rather than silently discarding a number the detector already
    computed. Every other issue type is unaffected and keeps the existing
    fixed fallback confidence.
    """
    if issue_type == "predicted_failure":
        try:
            confidence = float(evidence.get("confidence"))
            if 0.0 <= confidence <= 1.0:
                return confidence
        except (TypeError, ValueError):
            pass
    return 0.55


def _fallback_diagnosis(issue_type: str, evidence: dict, fallback_command: str) -> dict:
    # Defensive: fallback_command is already curated to be safe and
    # non-interactive (and, for predicted_failure, never a process-kill -
    # see failure_predictor.py) for every current issue type, but this
    # keeps both guarantees explicit and future-proof if that ever
    # changes.
    command = _ensure_non_blocking(fallback_command, fallback_command)
    command = _ensure_no_prediction_kill(issue_type, command, fallback_command)
    return {
        "reason": _fallback_reason(issue_type, evidence),
        "confidence_score": _fallback_confidence(issue_type, evidence),
        "recommended_command": command,
        "recommended_action": "Review the evidence below, then run the recommended command if it looks correct.",
    }


async def _ai_diagnose(issue_type: str, problem: str, evidence: dict, fallback_command: str) -> dict:
    """Ask the local LLM to explain *why* the issue is happening and to

    confirm/refine the recommended fix command. Degrades gracefully (never
    raises) to a deterministic diagnosis if Ollama is unreachable or replies
    with something unusable - the fix engine must keep working either way.
    """
    prompt = (
        "You are an Ubuntu system reliability assistant inside a One-Click Fix Engine. "
        "Diagnose the issue below for a sysadmin. Respond with JSON ONLY (no markdown "
        "fences) with exactly these keys: reason (string explaining WHY this is "
        "happening, referencing the evidence), confidence_score (0.0-1.0 number), "
        "recommended_command (a single safe Linux command that would fix it), "
        "recommended_action (short string describing the fix).\n\n"
        f"Issue type: {issue_type}\nProblem: {problem}\nEvidence: {json.dumps(evidence)}\n"
        f"Default command if you are unsure: {fallback_command}"
    )
    try:
        raw = await chat(
            [
                {"role": "system", "content": "Respond with strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            # Short, dedicated timeout for this *background* call - see
            # settings.ollama_background_timeout_seconds. This is polled
            # from the frontend every ~30s and shares the one local
            # Ollama backend with live chat/ops requests; on timeout this
            # already falls through to `_fallback_diagnosis` below, so
            # nothing is lost, and a live chat request never has to wait
            # behind a stuck background call for anywhere near the full
            # interactive `ollama_timeout_seconds`.
            timeout_seconds=settings.ollama_background_timeout_seconds,
        )
        cleaned = _JSON_FENCE_RE.sub("", raw).strip()
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("LLM did not return a JSON object")

        confidence = data.get("confidence_score", 0.7)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.7

        fallback = _fallback_diagnosis(issue_type, evidence, fallback_command)
        recommended_command = (
            str(data.get("recommended_command") or "").strip() or fallback_command
        )
        # Safeguard: the LLM is free to suggest anything, including
        # inspection-only or interactive commands (e.g. `systemctl
        # status ...`) that were never meant to run unattended. Never
        # hand one of those back as the One-Click Fix - swap it for the
        # issue's own actionable, non-interactive fallback instead.
        recommended_command = _ensure_non_blocking(recommended_command, fallback_command)
        # Safeguard: for a *predicted* (not-yet-real) CPU/memory issue,
        # never hand back a process-termination command even if the LLM
        # proposed one - see `_ensure_no_prediction_kill` docstring.
        recommended_command = _ensure_no_prediction_kill(
            issue_type, recommended_command, fallback_command
        )
        return {
            "reason": str(data.get("reason") or "").strip() or fallback["reason"],
            "confidence_score": confidence,
            "recommended_command": recommended_command,
            "recommended_action": str(data.get("recommended_action") or "").strip()
            or fallback["recommended_action"],
        }
    except (OllamaUnavailableError, json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("AI diagnosis unavailable for issue '%s', using fallback: %s", issue_type, exc)
        return _fallback_diagnosis(issue_type, evidence, fallback_command)


def _make_candidate(
    issue_id: str,
    issue_type: str,
    title: str,
    problem: str,
    evidence: dict,
    severity: str,
    fallback_command: str,
) -> dict:
    """Build the cheap, AI-free representation of a currently-observed issue.

    No diagnosis happens here - `issue_alert_store.reconcile()` decides
    whether this candidate needs a fresh AI diagnosis at all (new or
    changed) or can be treated as a duplicate scan of an already-active
    alert (ignored).
    """
    return {
        "issue_id": issue_id,
        "issue_type": issue_type,
        "title": title,
        "severity": severity,
        "problem": problem,
        "evidence": evidence,
        "fallback_command": fallback_command,
    }


async def _diagnose_candidate(candidate: dict) -> dict:
    """Adapter passed to `issue_alert_store.reconcile()` as its `diagnose`

    callback - runs the AI diagnosis for one candidate. Only invoked for
    candidates the store has determined are new or materially changed.
    """
    return await _ai_diagnose(
        candidate["issue_type"],
        candidate["problem"],
        candidate["evidence"],
        candidate["fallback_command"],
    )


# ---------------------------------------------------------------------------
# Detectors - one per required issue type. Each is independent and
# defensive: a failure in one must never prevent the others from running.
# Detectors only gather evidence and build candidates - no AI calls here.
# ---------------------------------------------------------------------------

async def _detect_high_cpu(info: dict) -> dict | None:
    cpu_percent = (info.get("cpu") or {}).get("usage_percent")
    if cpu_percent is None or cpu_percent < settings.alert_cpu_threshold_percent:
        return None

    top_process: dict = {}
    try:
        procs = system_monitor.get_processes(limit=1, sort_by="cpu_percent")
        if procs["processes"]:
            top_process = procs["processes"][0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read top CPU process: %s", exc)

    evidence = {
        "cpu_percent": cpu_percent,
        "threshold": settings.alert_cpu_threshold_percent,
        "top_process": {
            "name": top_process.get("name"),
            "pid": top_process.get("pid"),
            "cpu_percent": top_process.get("cpu_percent"),
        },
    }
    fallback_command = (
        f"kill -15 {top_process['pid']}" if top_process.get("pid") else "top -b -o %CPU -n 1 | head -n 15"
    )
    return _make_candidate(
        issue_id="high_cpu",
        issue_type="high_cpu",
        title="High CPU Usage",
        problem=f"CPU usage reached {cpu_percent}% (threshold {settings.alert_cpu_threshold_percent}%).",
        evidence=evidence,
        severity=_severity_for(cpu_percent, settings.alert_cpu_threshold_percent),
        fallback_command=fallback_command,
    )


async def _detect_high_memory(info: dict) -> dict | None:
    memory_percent = (info.get("memory") or {}).get("usage_percent")
    if memory_percent is None or memory_percent < settings.alert_memory_threshold_percent:
        return None

    top_process: dict = {}
    try:
        procs = system_monitor.get_processes(limit=1, sort_by="memory_percent")
        if procs["processes"]:
            top_process = procs["processes"][0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read top memory process: %s", exc)

    evidence = {
        "memory_percent": memory_percent,
        "threshold": settings.alert_memory_threshold_percent,
        "top_process": {
            "name": top_process.get("name"),
            "pid": top_process.get("pid"),
            "memory_percent": top_process.get("memory_percent"),
        },
    }
    fallback_command = (
        f"kill -15 {top_process['pid']}"
        if top_process.get("pid")
        else "free -h && ps aux --sort=-%mem | head -n 10"
    )
    return _make_candidate(
        issue_id="high_memory",
        issue_type="high_memory",
        title="High Memory Usage",
        problem=f"Memory usage reached {memory_percent}% (threshold {settings.alert_memory_threshold_percent}%).",
        evidence=evidence,
        severity=_severity_for(memory_percent, settings.alert_memory_threshold_percent),
        fallback_command=fallback_command,
    )


# Snap/squashfs loop mounts are read-only, fixed-size package images that
# routinely report ~100% usage_percent - that is normal for them and is
# NOT a sign of a real storage problem. They're filtered out here, in the
# Fix Engine's detector, rather than in system_monitor.get_disk_info(),
# so the raw system-info API (used elsewhere, e.g. the dashboard's disk
# widget) is left completely untouched.
_IGNORED_MOUNT_PREFIXES = ("/snap/", "/var/lib/snapd/")
_IGNORED_DEVICE_PREFIXES = ("/dev/loop",)
_IGNORED_FS_TYPES = {"squashfs"}


def _is_ignorable_mount(part: dict) -> bool:
    """True for a snap/squashfs loop-mounted pseudo-filesystem.

    These are not real writable storage - flagging them as "Disk Full"
    is a false positive regardless of how full they report themselves.
    """
    mountpoint = part.get("mountpoint") or ""
    device = part.get("device") or ""
    fstype = (part.get("filesystem_type") or "").lower()

    if mountpoint == "/snap" or mountpoint.startswith(_IGNORED_MOUNT_PREFIXES):
        return True
    if device.startswith(_IGNORED_DEVICE_PREFIXES):
        return True
    if fstype in _IGNORED_FS_TYPES:
        return True
    return False


async def _detect_disk_almost_full(info: dict) -> list[dict]:
    """Detect real writable filesystems (/, /home, /var, /usr, /opt, /data,
    ...) that are almost full.

    Snap/squashfs loop mounts are ignored entirely (see
    `_is_ignorable_mount`) so they never generate a Disk Full / Disk
    Almost Full / critical storage alert.

    When more than one real partition breaches the threshold at once,
    they're reported as a single consolidated "Storage Incident"
    candidate (one issue_id, one card, one recommended fix) instead of a
    near-identical "Disk Almost Full" card per mount - this keeps
    `issue_alert_store.reconcile()`'s "one active alert per issue_id"
    guarantee meaningful for storage the same way it already is for
    every other detector.
    """
    partitions = (info.get("disk") or {}).get("partitions", [])
    threshold = settings.alert_disk_threshold_percent
    fallback_command = "sudo apt-get clean && sudo journalctl --vacuum-size=200M"

    breaching: list[dict] = []
    for part in partitions:
        if _is_ignorable_mount(part):
            continue
        usage = part.get("usage_percent") or 0
        if usage < threshold:
            continue
        breaching.append(
            {
                "mountpoint": part.get("mountpoint", "/"),
                "device": part.get("device"),
                "usage_percent": usage,
            }
        )

    if not breaching:
        return []

    if len(breaching) == 1:
        mount = breaching[0]
        mountpoint = mount["mountpoint"]
        usage = mount["usage_percent"]
        evidence = {
            "disk_percent": usage,
            "mountpoint": mountpoint,
            "device": mount["device"],
            "threshold": threshold,
        }
        return [
            _make_candidate(
                issue_id=f"disk_full:{mountpoint}",
                issue_type="disk_full",
                title=f"Disk Almost Full ({mountpoint})",
                problem=f"Disk usage on {mountpoint} reached {usage}% (threshold {threshold}%).",
                evidence=evidence,
                severity=_severity_for(usage, threshold),
                fallback_command=fallback_command,
            )
        ]

    # Multiple real partitions are low on space at the same time - group
    # them into one Storage Incident instead of one card each.
    breaching.sort(key=lambda m: m["usage_percent"], reverse=True)
    worst = breaching[0]
    mounts_summary = ", ".join(f"{m['mountpoint']} ({m['usage_percent']}%)" for m in breaching)
    evidence = {
        "affected_mounts": mounts_summary,
        "mount_count": len(breaching),
        "worst_mountpoint": worst["mountpoint"],
        "worst_usage_percent": worst["usage_percent"],
        "threshold": threshold,
    }
    return [
        _make_candidate(
            issue_id="disk_full:multiple",
            issue_type="disk_full",
            title=f"Storage Incident ({len(breaching)} mounts affected)",
            problem=(
                f"{len(breaching)} writable filesystems are above the {threshold}% "
                f"usage threshold: {mounts_summary}."
            ),
            evidence=evidence,
            severity=_severity_for(worst["usage_percent"], threshold),
            fallback_command=fallback_command,
        )
    ]


async def _detect_apache_down() -> dict | None:
    service_basename = (settings.fix_apache_service_name or "").strip()
    if not service_basename:
        return None

    try:
        services = system_monitor.get_services(limit=500)["services"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read services for Apache check: %s", exc)
        return None

    svc = next((s for s in services if s["name"].split(".")[0] == service_basename), None)
    if not svc or svc.get("active_state") == "active":
        return None  # not installed on this host, or already running - nothing to fix

    evidence = {
        "service": svc["name"],
        "active_state": svc.get("active_state"),
        "sub_state": svc.get("sub_state"),
    }
    fallback_command = f"sudo systemctl restart {svc['name']}"
    return _make_candidate(
        issue_id=f"apache_down:{svc['name']}",
        issue_type="apache_down",
        title="Apache Web Server Down",
        problem=f"Service '{svc['name']}' is {svc.get('active_state')}, expected 'active'.",
        evidence=evidence,
        severity="critical",
        fallback_command=fallback_command,
    )


async def _detect_docker_stopped() -> list[dict]:
    candidates: list[dict] = []
    try:
        result = docker_monitor.get_docker_containers(all_containers=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not query Docker for stopped-container check: %s", exc)
        return candidates

    if not result.get("available"):
        return candidates  # Docker not installed/reachable on this host - nothing to report

    for container in result.get("containers", []):
        state = (container.get("state") or "").lower()
        if state in ("running", ""):
            continue

        name = container.get("name") or container.get("id") or "unknown"
        evidence = {
            "container": name,
            "image": container.get("image"),
            "state": container.get("state"),
            "status": container.get("status"),
        }
        fallback_command = f"docker start {name}"
        candidates.append(
            _make_candidate(
                issue_id=f"docker_stopped:{name}",
                issue_type="docker_stopped",
                title=f"Docker Container Stopped ({name})",
                problem=f"Container '{name}' is {container.get('status') or state}.",
                evidence=evidence,
                severity="warning",
                fallback_command=fallback_command,
            )
        )
    return candidates


async def _detect_failed_services() -> list[dict]:
    candidates: list[dict] = []
    try:
        services = system_monitor.get_services(limit=500)["services"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read services for failed-service check: %s", exc)
        return candidates

    apache_basename = (settings.fix_apache_service_name or "").strip()

    for svc in services:
        if svc.get("active_state") != "failed":
            continue
        base_name = svc["name"].split(".")[0]
        if base_name == apache_basename:
            continue  # already reported as "Apache Down" - avoid duplicate cards

        evidence = {
            "service": svc["name"],
            "active_state": svc.get("active_state"),
            "sub_state": svc.get("sub_state"),
            "load_state": svc.get("load_state"),
        }
        fallback_command = f"sudo systemctl restart {svc['name']}"
        candidates.append(
            _make_candidate(
                issue_id=f"failed_service:{svc['name']}",
                issue_type="failed_service",
                title=f"Failed Service ({svc['name']})",
                problem=f"Service '{svc['name']}' has failed.",
                evidence=evidence,
                severity="critical",
                fallback_command=fallback_command,
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Aggregate detection pass
# ---------------------------------------------------------------------------

async def detect_all_issues() -> dict:
    """Run every detector once, then reconcile the result against the

    active-issue store so the same problem never produces more than one
    active alert. A previously-active issue that isn't observed this pass
    is treated as resolved and dropped.

    Every detector is isolated with its own try/except so one failing check
    (e.g. `docker` not installed) never prevents the others from reporting.
    """
    errors: list[str] = []

    try:
        info = system_monitor.get_system_info()
    except Exception as exc:  # noqa: BLE001
        logger.error("Fix engine could not read system info: %s", exc)
        info = {}
        errors.append(f"Could not read system info: {exc}")

    async def _safe(label: str, coro):
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001
            logger.error("Fix engine detector '%s' failed: %s", label, exc)
            errors.append(f"{label} detection failed: {exc}")
            return None

    (
        cpu_candidate,
        memory_candidate,
        disk_candidates,
        apache_candidate,
        docker_candidates,
        failed_service_candidates,
        predicted_failure_candidates,
        hardware_fault_candidates,
        driver_anomaly_candidates,
    ) = await asyncio.gather(
        _safe("high_cpu", _detect_high_cpu(info)),
        _safe("high_memory", _detect_high_memory(info)),
        _safe("disk_full", _detect_disk_almost_full(info)),
        _safe("apache_down", _detect_apache_down()),
        _safe("docker_stopped", _detect_docker_stopped()),
        _safe("failed_service", _detect_failed_services()),
        _safe("predicted_failure", failure_predictor.detect_failure_predictions(info)),
        _safe("hardware_fault", hardware_monitor.detect_hardware_issues()),
        _safe("driver_anomaly", driver_monitor.detect_driver_anomalies()),
    )

    candidates: list[dict] = []
    if cpu_candidate:
        candidates.append(cpu_candidate)
    if memory_candidate:
        candidates.append(memory_candidate)
    candidates.extend(disk_candidates or [])
    if apache_candidate:
        candidates.append(apache_candidate)
    candidates.extend(docker_candidates or [])
    candidates.extend(failed_service_candidates or [])
    candidates.extend(predicted_failure_candidates or [])
    candidates.extend(hardware_fault_candidates or [])
    candidates.extend(driver_anomaly_candidates or [])

    # Single reconciliation point: dedupes by issue_id, diagnoses only what's
    # new/changed, ignores unchanged duplicate scans, drops resolved issues.
    active_issues = await issue_alert_store.reconcile(candidates, _diagnose_candidate)

    # Strip internal-only bookkeeping fields before returning to the API.
    issues = [
        {k: v for k, v in issue.items() if k not in ("fallback_command", "evidence_signature")}
        for issue in active_issues
    ]
    issues.sort(key=lambda i: (i["severity"] != "critical", i["first_detected_at"]))

    return {
        "checked_at": _now_iso(),
        "total_issues": len(issues),
        "issues": issues,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# One-Click Fix: prepare (never execute) via the existing Safe Command
# Execution pipeline
# ---------------------------------------------------------------------------

def prepare_fix(
    session_id: str | None,
    issue_title: str,
    command: str,
    explanation: str,
    confidence_score: float,
) -> dict:
    """Turn a detected issue's recommended command into a pending entry in

    the existing Safe Command Execution pipeline.

    This performs NO execution whatsoever. It reuses
    `command_console._build_preview()` - the exact same deterministic
    safety analysis (`command_safety.analyze_command`) and persistence
    (`execution_store.save_pending`) used by the AI Assistant's command
    generation/explanation flow - so the returned preview can be run
    through the existing, unmodified `POST /api/commands/{id}/confirm`
    endpoint, and only after the user explicitly confirms.
    """
    session_id = session_id or str(uuid.uuid4())
    return command_console._build_preview(  # noqa: SLF001 - intentional reuse, see docstring
        session_id=session_id,
        description=f"One-Click Fix: {issue_title}",
        source="ai_generated",
        command=command,
        explanation=explanation,
        confidence_score=confidence_score,
        context={},
        llm_warnings=[],
    )


# ---------------------------------------------------------------------------
# Recovery verification: detect -> diagnose -> safe proposal -> confirmation
# -> execute -> verify.
#
# Everything up through "execute" already exists and is untouched: a fix
# is proposed by `prepare_fix()` above, safety-analyzed and persisted by
# the existing `command_safety`/`execution_store` pipeline, and only ever
# actually run after an explicit user confirmation via the existing
# `POST /api/commands/{execution_id}/confirm` endpoint - none of that is
# changed here.
#
# `verify_fix()` closes the loop for the last step using only functions
# that already exist: it re-runs the exact same detection pass
# `detect_all_issues()` already performs (which is how every issue was
# found in the first place) and reports whether `issue_id` is still part
# of the resulting active-issue set. If the underlying condition really
# was fixed, the relevant detector simply won't produce that candidate
# again on the next pass, `issue_alert_store.reconcile()` will have
# already dropped it as resolved, and this reports `resolved: True`. If
# the issue is still observed, it reports `resolved: False` along with
# the still-active issue's latest evidence, so the same explainability
# chain (Problem/Reason/Evidence/Confidence/Command) is available for a
# retry. This performs NO execution and creates no new execution
# framework - it is a read-only check built entirely out of the existing
# detect/reconcile pipeline, safe to call any time after a fix has been
# confirmed and executed (e.g. from a background poller or a future route
# added elsewhere, without any further change needed in this file).
# ---------------------------------------------------------------------------

async def verify_fix(issue_id: str) -> dict:
    """Re-run detection and report whether `issue_id` is still active.

    Reuses `detect_all_issues()` end to end (same detectors, same
    reconciliation against `issue_alert_store`) - this is intentionally
    not a separate/lighter-weight check, so "resolved" here means exactly
    what "no longer an active alert" means everywhere else in the app.
    """
    result = await detect_all_issues()
    still_active = next((issue for issue in result["issues"] if issue["issue_id"] == issue_id), None)
    return {
        "issue_id": issue_id,
        "checked_at": result["checked_at"],
        "resolved": still_active is None,
        "issue": still_active,
    }


# ---------------------------------------------------------------------------
# Server-side background detection loop.
#
# Root-cause fix (routing/freshness audit): before this, `detect_all_issues()`
# only ever ran when something called GET /api/fixes/detect - in practice,
# only the frontend dashboard, polling every ~30s while a browser tab is
# open (see frontend/script.js POLL_INTERVAL_MS). The AI Assistant's
# driver-health/hardware-health/failure-prediction chat answers are
# grounded entirely in `issue_alert_store` (see context_builder.py) and
# never trigger a scan themselves (deliberately - see context_builder's
# module docstring on why not). That meant chat-only or API-only usage,
# with no dashboard tab ever opened, always answered those questions from
# an empty store - not because there was no evidence of a problem, but
# because no scan had ever run.
#
# This loop runs the exact same `detect_all_issues()` on a fixed
# server-side interval, independent of any frontend, so the store is
# fresh no matter how the assistant is being used. It changes nothing
# about detection, dedup, or diagnosis logic - it only changes *when*
# the existing scan runs. Wired up from `main.py`'s startup/shutdown
# events; every iteration is isolated so one failed scan (e.g. Ollama
# briefly unavailable for a new-issue diagnosis) never stops the loop.
# ---------------------------------------------------------------------------

async def run_background_detection_loop(interval_seconds: float) -> None:
    """Run `detect_all_issues()` forever, sleeping `interval_seconds` between passes.

    Intended to be launched once via `asyncio.create_task()` at app
    startup and cancelled at shutdown. Never raises out of the loop - a
    failed pass is logged and the loop continues on the same schedule.
    """
    logger.info("Background detection loop starting (interval=%ss)", interval_seconds)
    try:
        while True:
            try:
                result = await detect_all_issues()
                logger.info(
                    "Background detection scan complete: %d active issue(s), %d error(s)",
                    result.get("total_issues", 0),
                    len(result.get("errors", [])),
                )
            except Exception as exc:  # noqa: BLE001 - one bad pass must never kill the loop
                logger.error("Background detection scan failed: %s", exc)
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("Background detection loop stopped")
        raise