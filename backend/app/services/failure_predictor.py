"""
Failure Predictor - trend-based failure prediction for the AI One-Click
Fix Engine.

This module adds *predictive* fault detection on top of the existing
*reactive* detectors in `fix_engine.py`. Where `fix_engine._detect_high_cpu`
etc. only fire once a threshold has already been breached, this module
watches the same CPU/memory/disk metrics over time and, using a simple
linear trend fit over a rolling window of samples, estimates *when* a
metric will cross its threshold if the current trend continues. If that
projected breach falls inside a configurable lookahead window, it is
surfaced as a "predicted_failure" candidate - so operators (or the
self-healing pipeline) can act before the problem actually occurs, not
just after.

Integration point: `fix_engine.detect_all_issues()` calls
`detect_failure_predictions(info)` once per scan (the dashboard already
polls `/api/fixes/detect` on a fixed interval, which is what feeds this
module's rolling history) alongside its other detectors via
`asyncio.gather`, and folds the returned candidates into the same
candidate list handed to `issue_alert_store.reconcile()`. From there
everything is identical to every other issue type already in this
project: `issue_alert_store` dedupes/tracks it, `fix_engine._ai_diagnose`
(or its deterministic fallback) produces the Reason/Confidence/Command
explanation, and `fix_engine.prepare_fix()` / the existing Safe Command
Execution pipeline is what turns it into an actual (user-confirmed) fix.
This module itself never diagnoses, never executes, and never talks to
the LLM - it only detects a trend and builds a candidate, exactly like
every other detector in `fix_engine.py`.

Candidates are emitted in the exact same shape as
`fix_engine._make_candidate()` produces (issue_id/issue_type/title/
problem/evidence/severity/fallback_command) so `fix_engine.py` can treat
them identically to its own detectors' output.

State (the rolling per-metric sample history) is in-memory and
process-local, matching the existing pattern in `issue_alert_store.py` -
no database table is needed for a short trend window.

--------------------------------------------------------------------------
Safety note (Problem Statement 2 - Autonomous Fault Detection & Self-
Healing OS)
--------------------------------------------------------------------------
This module ONLY predicts. It never kills a process, restarts a service,
reboots anything, touches kernel settings, or runs a shell command itself.
A *predicted* CPU/memory problem is, by definition, not a problem yet -
the "top resource-consuming process" reported in the evidence may well be
a completely legitimate, intentionally heavy workload (e.g. Ollama, a
browser, VS Code, a build). Terminating an arbitrary PID on nothing more
than a trend projection is unsafe and is deliberately NOT something this
module does: every `fallback_command` produced here is a **read-only
diagnostic command** (e.g. `ps`/`top`/`free`/`df`), never a `kill`. Any
actual remediation still has to go through the existing, unmodified
`command_safety` + user-confirmation + execution pipeline in
`fix_engine.py`, exactly like every other issue type.
"""

import asyncio
import time
from collections import deque
from datetime import datetime, timezone

from app.config import get_settings
from app.logger import get_logger
from app.services import system_monitor

settings = get_settings()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
#
# The window/lookahead/reliability knobs aren't all in Settings yet, so
# they're read via getattr() with sane defaults - this module works
# standalone today and will automatically pick up real config.py values
# once those settings are added there, with no further change needed
# here. The thresholds themselves intentionally reuse the same values the
# reactive detectors already use, so "predicted" and "actual" alerts
# agree on what "too high" means.
# ---------------------------------------------------------------------------
_WINDOW_SIZE = getattr(settings, "predictor_window_size", 20)  # samples kept per metric
_MIN_SAMPLES = getattr(settings, "predictor_min_samples", 5)  # samples needed before predicting
_LOOKAHEAD_MINUTES = float(getattr(settings, "predictor_lookahead_minutes", 30.0))

# Reliability gates - these are what keep the predictor from firing on
# noise. Both a minimum "fit quality" (R^2 of the linear regression) and
# a minimum slope are required before a trend is considered "meaningful"
# rather than random jitter around a flat line.
_MIN_R_SQUARED = float(getattr(settings, "predictor_min_r_squared", 0.5))
_MIN_SLOPE_PERCENT_PER_MINUTE = float(
    getattr(settings, "predictor_min_slope_percent_per_minute", 0.05)
)
# Guards against a degenerate/near-instant fit (e.g. several scans firing
# back-to-back within the same second): without a real spread of wall-clock
# time across the sample window, a tiny timing jitter can produce a wildly
# unrealistic slope/ETA even with a "clean" R^2. Samples must span at least
# this many seconds before a projection is trusted.
_MIN_SAMPLE_SPAN_SECONDS = float(getattr(settings, "predictor_min_sample_span_seconds", 10.0))

# Re-alert cooldown - once a prediction has been surfaced for a given
# metric key, don't surface a materially-unchanged prediction again for
# this many seconds. (`issue_alert_store` already collapses scans whose
# evidence is byte-for-byte identical, but live percentages drift by
# fractions of a point every poll, which would otherwise re-diagnose and
# re-alert an unchanged situation on almost every scan.)
_REALERT_COOLDOWN_SECONDS = float(getattr(settings, "predictor_realert_cooldown_seconds", 90.0))
# A cooled-down prediction still re-alerts immediately if the estimated
# time-to-breach has moved by more than this fraction (the situation is
# meaningfully better/worse than what was last reported).
_REALERT_ETA_CHANGE_FRACTION = 0.25

_CPU_THRESHOLD = settings.alert_cpu_threshold_percent
_MEMORY_THRESHOLD = settings.alert_memory_threshold_percent
_DISK_THRESHOLD = settings.alert_disk_threshold_percent

# Mirrors fix_engine._is_ignorable_mount: snap/squashfs loop mounts are
# read-only, fixed-size package images that routinely sit near 100% usage -
# that's normal for them and must never generate a predicted-disk-full
# alert. Duplicated here (rather than imported) to keep this module
# self-contained and avoid reaching into fix_engine's private helpers.
_IGNORED_MOUNT_PREFIXES = ("/snap/", "/var/lib/snapd/")
_IGNORED_DEVICE_PREFIXES = ("/dev/loop",)
_IGNORED_FS_TYPES = {"squashfs"}

# Read-only diagnostic commands only. Deliberately NEVER a `kill`/`pkill`/
# service-restart/reboot command: a predicted (not-yet-real) CPU/memory
# problem must never auto-suggest terminating an arbitrary process, since
# the top consumer may be a perfectly legitimate workload (Ollama, a
# browser, an IDE, a build, ...). If real remediation is ever warranted,
# a human decides that via the normal reactive `high_cpu`/`high_memory`
# detectors and the existing command_safety + confirmation pipeline - not
# from a trend projection alone.
_CPU_DIAGNOSTIC_COMMAND = "top -b -o %CPU -n 1 | head -n 15"
_MEMORY_DIAGNOSTIC_COMMAND = "free -h && ps aux --sort=-%mem | head -n 10"
_DISK_DIAGNOSTIC_COMMAND = "df -h && du -h --max-depth=1 / 2>/dev/null | sort -rh | head -n 15"

# Rolling history: metric key -> deque of (monotonic_seconds, value).
# time.monotonic() is used for the trend math (immune to wall-clock jumps);
# wall-clock time is only used for human-readable evidence fields.
_history: dict[str, deque[tuple[float, float]]] = {}
# Last prediction actually surfaced per metric key, used only for the
# re-alert cooldown described above. Value: (monotonic_time, eta_minutes).
_last_alerted: dict[str, tuple[float, float]] = {}
_lock = asyncio.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_ignorable_mount(part: dict) -> bool:
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


# ---------------------------------------------------------------------------
# Trend math - a plain ordinary-least-squares fit over (time, value) pairs.
# Deliberately dependency-free (no numpy/ML) to keep this lightweight.
# ---------------------------------------------------------------------------

def _fit_linear_trend(samples: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Fit value = intercept + slope * time_seconds over `samples`.

    Returns (slope, intercept, r_squared), where slope is in metric-units
    per second and r_squared (0..1) is how well the straight line
    explains the observed samples - the basis for both the "is this
    trend meaningful, or just noise?" gate and the confidence score.
    Returns None if there isn't enough spread in the samples to fit a
    line (e.g. all samples arrived at the same instant, or every sample
    has the exact same value).
    """
    n = len(samples)
    if n < 2:
        return None

    mean_t = sum(t for t, _ in samples) / n
    mean_v = sum(v for _, v in samples) / n
    denominator = sum((t - mean_t) ** 2 for t, _ in samples)
    if denominator == 0:
        return None

    numerator = sum((t - mean_t) * (v - mean_v) for t, v in samples)
    slope = numerator / denominator
    intercept = mean_v - slope * mean_t

    ss_tot = sum((v - mean_v) ** 2 for _, v in samples)
    if ss_tot == 0:
        # Every sample has the identical value - a perfectly flat line,
        # not a trend. Treat as "no meaningful fit" rather than dividing
        # by zero.
        return None
    ss_res = sum((v - (intercept + slope * t)) ** 2 for t, v in samples)
    r_squared = max(0.0, 1.0 - (ss_res / ss_tot))

    return slope, intercept, r_squared


def _eta_seconds_to_threshold(
    samples: list[tuple[float, float]], threshold: float
) -> tuple[float, float, float] | None:
    """Seconds from the most recent sample until the fitted trend line

    crosses `threshold`, together with the slope (per second) and
    r_squared that produced it, or None if there's no meaningful upward
    trend, the metric has already breached the threshold (that's the
    reactive detectors' job, not prediction), or the fit is too weak/
    noisy to trust (see _MIN_R_SQUARED / _MIN_SLOPE_PERCENT_PER_MINUTE).
    """
    time_span = samples[-1][0] - samples[0][0]
    if time_span < _MIN_SAMPLE_SPAN_SECONDS:
        return None  # samples too close together in time to trust a rate-of-change

    fit = _fit_linear_trend(samples)
    if fit is None:
        return None
    slope, intercept, r_squared = fit

    slope_per_minute = slope * 60.0
    if slope_per_minute < _MIN_SLOPE_PERCENT_PER_MINUTE:
        return None  # flat, improving, or too shallow to call a "trend"

    if r_squared < _MIN_R_SQUARED:
        return None  # too noisy/scattered to trust a straight-line projection

    last_t, last_v = samples[-1]
    if last_v >= threshold:
        return None  # already over threshold - not a "prediction" anymore

    t_breach = (threshold - intercept) / slope
    eta_seconds = t_breach - last_t
    if eta_seconds <= 0:
        return None

    return eta_seconds, slope, r_squared


def _confidence_score(r_squared: float, sample_count: int) -> float:
    """Explainable confidence: mostly "how well do the samples fit a

    straight line" (r_squared), with a smaller boost for having a fuller
    window of samples to fit against. Deliberately simple arithmetic
    (no hidden model) so the number in the UI can always be traced back
    to these two inputs.
    """
    sample_fraction = min(1.0, sample_count / _WINDOW_SIZE)
    confidence = 0.5 + (0.4 * r_squared) + (0.1 * sample_fraction)
    return round(min(0.95, max(0.3, confidence)), 2)


async def _record_and_predict(key: str, current_value: float, threshold: float) -> dict | None:
    """Append `current_value` to `key`'s rolling history, then check

    whether the resulting trend predicts a threshold breach inside the
    lookahead window, applying the reliability gates (minimum samples,
    minimum slope, minimum fit quality) and the re-alert cooldown.

    Returns a dict of prediction fields if a breach is predicted soon
    enough - and reliably enough - to act on, else None. A healthy metric
    with no upward trend never reaches the point of building this dict,
    so "current value is fine" alone can never produce a prediction.
    """
    async with _lock:
        history = _history.setdefault(key, deque(maxlen=_WINDOW_SIZE))
        history.append((time.monotonic(), current_value))
        if len(history) < _MIN_SAMPLES:
            return None
        samples = list(history)

    result = _eta_seconds_to_threshold(samples, threshold)
    if result is None:
        return None
    eta_seconds, slope_per_second, r_squared = result

    eta_minutes = eta_seconds / 60.0
    if eta_minutes > _LOOKAHEAD_MINUTES:
        return None  # trending toward the threshold, but too far out to act on yet

    sample_count = len(samples)
    confidence = _confidence_score(r_squared, sample_count)

    async with _lock:
        last = _last_alerted.get(key)
        now = time.monotonic()
        if last is not None:
            last_time, last_eta = last
            eta_delta_fraction = abs(eta_minutes - last_eta) / last_eta if last_eta > 0 else 1.0
            still_cooling_down = (now - last_time) < _REALERT_COOLDOWN_SECONDS
            barely_changed = eta_delta_fraction < _REALERT_ETA_CHANGE_FRACTION
            if still_cooling_down and barely_changed:
                return None  # same prediction as recently reported - don't spam a new alert
        _last_alerted[key] = (now, eta_minutes)

    return {
        "eta_minutes": eta_minutes,
        "slope_percent_per_minute": slope_per_second * 60.0,
        "sample_count": sample_count,
        "r_squared": round(r_squared, 3),
        "confidence": confidence,
    }


def _severity_for_eta(eta_minutes: float) -> str:
    """Escalate to critical once the predicted breach is imminent (inside

    the first quarter of the lookahead window) rather than merely
    "somewhere in the window" - mirrors fix_engine._severity_for's intent
    of reserving "critical" for the more urgent cases.
    """
    return "critical" if eta_minutes <= (_LOOKAHEAD_MINUTES * 0.25) else "warning"


def _eta_display(eta_minutes: float) -> str:
    """Human-readable ETA for the `problem` string, rounded to a sensible

    granularity instead of implying false precision (e.g. "~5 minutes"
    rather than "~4.83827 minutes").
    """
    if eta_minutes < 1:
        return "under a minute"
    if eta_minutes < 2:
        return "about 1 minute"
    return f"about {round(eta_minutes)} minutes"


def _make_candidate(
    issue_id: str,
    title: str,
    problem: str,
    evidence: dict,
    severity: str,
    fallback_command: str,
) -> dict:
    """Same shape as fix_engine._make_candidate() - issue_type is always

    "predicted_failure" here; the affected resource lives in `issue_id`
    and `evidence`, exactly like every other detector's candidates.
    """
    return {
        "issue_id": issue_id,
        "issue_type": "predicted_failure",
        "title": title,
        "severity": severity,
        "problem": problem,
        "evidence": evidence,
        "fallback_command": fallback_command,
    }


# ---------------------------------------------------------------------------
# Per-metric predictors. Each is independent and defensive: a failure in
# one must never prevent the others from running or crash the scan.
#
# NOTE ON SAFETY: none of these ever build a `kill`/`pkill`/restart/reboot
# command. `fallback_command` is always a read-only diagnostic (top/ps/
# free/df) - see the module docstring and the constants above. The "top
# resource-consuming process" is reported purely as *explanatory evidence*
# for a human to look at, never as a target to terminate.
# ---------------------------------------------------------------------------

async def _detect_cpu_trend(info: dict) -> dict | None:
    cpu_percent = (info.get("cpu") or {}).get("usage_percent")
    if cpu_percent is None:
        return None

    prediction = await _record_and_predict("cpu", cpu_percent, _CPU_THRESHOLD)
    if prediction is None:
        return None
    eta_minutes = prediction["eta_minutes"]

    top_process: dict = {}
    try:
        procs = system_monitor.get_processes(limit=1, sort_by="cpu_percent")
        if procs["processes"]:
            top_process = procs["processes"][0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failure predictor could not read top CPU process: %s", exc)

    evidence = {
        "current_percent": round(cpu_percent, 1),
        "threshold": _CPU_THRESHOLD,
        "trend_percent_per_minute": round(prediction["slope_percent_per_minute"], 2),
        "eta_minutes": round(eta_minutes, 1),
        "samples_analyzed": prediction["sample_count"],
        "fit_r_squared": prediction["r_squared"],
        "confidence": prediction["confidence"],
        "top_process": {
            "name": top_process.get("name"),
            "pid": top_process.get("pid"),
            "cpu_percent": top_process.get("cpu_percent"),
            "note": (
                "Reported for context only - may be a legitimate workload. "
                "This predictor never recommends terminating it automatically."
            ),
        },
        "reason": (
            f"CPU utilization has shown a sustained upward trend over the last "
            f"{prediction['sample_count']} samples (fit quality R^2="
            f"{prediction['r_squared']}) and is projected to cross the "
            f"{_CPU_THRESHOLD}% threshold if it continues."
        ),
    }
    return _make_candidate(
        issue_id="predicted_failure:cpu",
        title="Predicted CPU Saturation",
        problem=(
            f"CPU usage ({round(cpu_percent, 1)}%) is trending upward "
            f"(+{round(prediction['slope_percent_per_minute'], 1)}%/min) and is projected to reach the "
            f"{_CPU_THRESHOLD}% threshold in {_eta_display(eta_minutes)} if the trend continues."
        ),
        evidence=evidence,
        severity=_severity_for_eta(eta_minutes),
        fallback_command=_CPU_DIAGNOSTIC_COMMAND,
    )


async def _detect_memory_trend(info: dict) -> dict | None:
    memory_percent = (info.get("memory") or {}).get("usage_percent")
    if memory_percent is None:
        return None

    prediction = await _record_and_predict("memory", memory_percent, _MEMORY_THRESHOLD)
    if prediction is None:
        return None
    eta_minutes = prediction["eta_minutes"]

    top_process: dict = {}
    try:
        procs = system_monitor.get_processes(limit=1, sort_by="memory_percent")
        if procs["processes"]:
            top_process = procs["processes"][0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failure predictor could not read top memory process: %s", exc)

    evidence = {
        "current_percent": round(memory_percent, 1),
        "threshold": _MEMORY_THRESHOLD,
        "trend_percent_per_minute": round(prediction["slope_percent_per_minute"], 2),
        "eta_minutes": round(eta_minutes, 1),
        "samples_analyzed": prediction["sample_count"],
        "fit_r_squared": prediction["r_squared"],
        "confidence": prediction["confidence"],
        "top_process": {
            "name": top_process.get("name"),
            "pid": top_process.get("pid"),
            "memory_percent": top_process.get("memory_percent"),
            "note": (
                "Reported for context only - may be a legitimate workload. "
                "This predictor never recommends terminating it automatically."
            ),
        },
        "reason": (
            f"Memory utilization has shown a sustained upward trend over the last "
            f"{prediction['sample_count']} samples (fit quality R^2="
            f"{prediction['r_squared']}) and is projected to cross the "
            f"{_MEMORY_THRESHOLD}% threshold if it continues."
        ),
    }
    return _make_candidate(
        issue_id="predicted_failure:memory",
        title="Predicted Memory Exhaustion",
        problem=(
            f"Memory usage ({round(memory_percent, 1)}%) is trending upward "
            f"(+{round(prediction['slope_percent_per_minute'], 1)}%/min) and is projected to reach the "
            f"{_MEMORY_THRESHOLD}% threshold in {_eta_display(eta_minutes)} if the trend continues."
        ),
        evidence=evidence,
        severity=_severity_for_eta(eta_minutes),
        fallback_command=_MEMORY_DIAGNOSTIC_COMMAND,
    )


async def _detect_disk_trend(info: dict) -> list[dict]:
    candidates: list[dict] = []
    partitions = (info.get("disk") or {}).get("partitions", [])

    for part in partitions:
        if _is_ignorable_mount(part):
            continue

        usage_percent = part.get("usage_percent")
        if usage_percent is None:
            continue

        mountpoint = part.get("mountpoint", "/")
        key = f"disk:{mountpoint}"

        prediction = await _record_and_predict(key, usage_percent, _DISK_THRESHOLD)
        if prediction is None:
            continue
        eta_minutes = prediction["eta_minutes"]

        evidence = {
            "current_percent": round(usage_percent, 1),
            "threshold": _DISK_THRESHOLD,
            "mountpoint": mountpoint,
            "device": part.get("device"),
            "trend_percent_per_minute": round(prediction["slope_percent_per_minute"], 2),
            "eta_minutes": round(eta_minutes, 1),
            "samples_analyzed": prediction["sample_count"],
            "fit_r_squared": prediction["r_squared"],
            "confidence": prediction["confidence"],
            "reason": (
                f"Disk usage on {mountpoint} has shown a sustained upward trend over the last "
                f"{prediction['sample_count']} samples (fit quality R^2="
                f"{prediction['r_squared']}) and is projected to cross the "
                f"{_DISK_THRESHOLD}% threshold if it continues."
            ),
        }
        candidates.append(
            _make_candidate(
                issue_id=f"predicted_failure:disk:{mountpoint}",
                title=f"Predicted Disk Full ({mountpoint})",
                problem=(
                    f"Disk usage on {mountpoint} ({round(usage_percent, 1)}%) is trending upward "
                    f"(+{round(prediction['slope_percent_per_minute'], 1)}%/min) and is projected "
                    f"to reach the {_DISK_THRESHOLD}% threshold in {_eta_display(eta_minutes)} "
                    "if the trend continues."
                ),
                evidence=evidence,
                severity=_severity_for_eta(eta_minutes),
                fallback_command=_DISK_DIAGNOSTIC_COMMAND,
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Aggregate entry point - called from fix_engine.detect_all_issues()
# ---------------------------------------------------------------------------

async def detect_failure_predictions(info: dict) -> list[dict]:
    """Record this scan's CPU/memory/disk metrics and return any candidates

    for a threshold breach predicted within the lookahead window.

    `info` is the same `system_monitor.get_system_info()` payload the
    reactive detectors in `fix_engine.py` already use, so this can be
    dropped into `detect_all_issues()`'s existing `asyncio.gather` call
    with no additional system calls of its own beyond the same
    `get_processes()` lookups the reactive CPU/memory detectors already
    perform.

    Each metric's predictor is isolated with its own try/except so a
    failure reading one metric never prevents the others from reporting -
    matching every other detector in this project. This function only
    ever returns analysis/explanation - it never executes anything.
    """
    candidates: list[dict] = []

    try:
        cpu_candidate = await _detect_cpu_trend(info)
        if cpu_candidate:
            candidates.append(cpu_candidate)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failure predictor CPU trend detection failed: %s", exc)

    try:
        memory_candidate = await _detect_memory_trend(info)
        if memory_candidate:
            candidates.append(memory_candidate)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failure predictor memory trend detection failed: %s", exc)

    try:
        candidates.extend(await _detect_disk_trend(info))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failure predictor disk trend detection failed: %s", exc)

    return candidates


def get_history_snapshot() -> dict:
    """Read-only snapshot of the current rolling metric history, keyed by

    metric key, as (iso_timestamp, value) pairs. Mainly useful for
    diagnostics/tests - not required by the detection pipeline itself.
    """
    snapshot: dict[str, list[tuple[str, float]]] = {}
    now_monotonic = time.monotonic()
    now_wall = datetime.now(timezone.utc)
    for key, samples in _history.items():
        snapshot[key] = [
            ((now_wall - __import__("datetime").timedelta(seconds=now_monotonic - t)).isoformat(), v)
            for t, v in samples
        ]
    return snapshot


def clear() -> None:
    """Drop all rolling metric history and re-alert cooldown state.

    Mainly useful for tests.
    """
    _history.clear()
    _last_alerted.clear()