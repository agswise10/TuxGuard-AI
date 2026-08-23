"""
Hardware Health Monitor - fault detection for CPU/board temperature,
system fans, and disk SMART health.

Follows the exact same "detector" shape as everything else feeding the
AI One-Click Fix Engine (see `fix_engine.py`): each check here is a
read-only, defensive collector that turns raw sensor data into candidate
dicts shaped like `fix_engine._make_candidate()` produces
(issue_id/issue_type/title/problem/evidence/severity/fallback_command).
Nothing in this module diagnoses (that's `fix_engine._ai_diagnose`),
dedupes (that's `issue_alert_store.reconcile()`), or executes anything
(that's the existing Safe Command Execution pipeline, reached only after
an explicit user confirmation) - this module only observes hardware
state and reports it, exactly like `system_monitor.py` and
`docker_monitor.py` already do for their respective areas.

Integration point: `fix_engine.detect_all_issues()` calls
`detect_hardware_issues()` once per scan alongside its other detectors
via `asyncio.gather`, and folds the returned candidates into the same
candidate list handed to `issue_alert_store.reconcile()`.

Sensor availability varies wildly by host (bare metal vs VM vs
container, desktop vs server, vendor-specific kernel modules, `smartmontools`
installed or not, whether the running user has permission to query
`/dev/sdX`), so every check here degrades gracefully: a missing tool,
missing sensor, or permission error simply means "nothing to report from
that check" rather than a crash or a false alert. This mirrors
`system_monitor.get_system_info()`'s "one failing collector must never
break the rest of the response" design.

SMART health verdict, independent of process exit code.

`smartctl`'s exit status is a bitmask of many independent conditions
(see `man smartctl`, RETURN VALUES) - things like an unsupported
optional log page (e.g. NVMe drives commonly printing "Read Self-test
Log failed: Invalid Field in Command") set a bit and produce a non-zero
exit even when the drive's actual health is fine. The previous version
of this check used the shared `run_command()` helper, which correctly
treats a non-zero exit as failure for every other caller in this
project - but for `smartctl` specifically that meant a real "PASSED,
Critical Warning 0x00, Media and Data Integrity Errors 0" health report
could be discarded and reported as a hardware fault purely because of
an unrelated non-fatal exit bit.

`_detect_disk_smart_failures()` now invokes `smartctl` directly (still
read-only, still non-interactive, still `smartmontools` - no new
dependency) so it always sees the full stdout, and decides health from
the actual text: an explicit PASSED/OK self-assessment verdict, plus
(when present) a zero NVMe Critical Warning bitmask and zero Media and
Data Integrity Errors count, is healthy regardless of exit code. Only an
explicit FAILED/FAILING verdict, a non-zero Critical Warning, or a
non-zero Media/Data Integrity Errors count is treated as real fault
evidence. Anything else (no parseable verdict at all, e.g. smartctl
couldn't open the device) is inconclusive and reported as nothing, same
as before.
"""

import asyncio
import re
import subprocess
from datetime import datetime, timezone

import psutil

from app.config import get_settings
from app.logger import get_logger
from app.services import system_monitor

settings = get_settings()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
#
# Not in Settings yet, so read via getattr() with sane defaults - this
# module works standalone today and will automatically pick up real
# config.py values once those settings are added there, with no further
# change needed here (same approach as failure_predictor.py).
# ---------------------------------------------------------------------------
_TEMP_WARNING_CELSIUS = float(getattr(settings, "hardware_temp_warning_celsius", 80.0))
_TEMP_CRITICAL_CELSIUS = float(getattr(settings, "hardware_temp_critical_celsius", 95.0))
_SMARTCTL_TIMEOUT_SECONDS = int(getattr(settings, "hardware_smartctl_timeout_seconds", 10))

# Same filtering as fix_engine._is_ignorable_mount / failure_predictor -
# snap/squashfs loop mounts aren't real physical storage and never have
# meaningful SMART data of their own.
_IGNORED_MOUNT_PREFIXES = ("/snap/", "/var/lib/snapd/")
_IGNORED_DEVICE_PREFIXES = ("/dev/loop", "/dev/ram", "tmpfs", "overlay", "none")

# Partition device names (e.g. /dev/sda1) need to be reduced to their
# underlying physical disk (e.g. /dev/sda) before calling smartctl -
# SMART health lives on the disk, not the partition.
_PARTITION_SUFFIX_RE = re.compile(r"^(/dev/[a-z]+)\d+$")  # /dev/sda1 -> /dev/sda
_NVME_PARTITION_SUFFIX_RE = re.compile(r"^(/dev/nvme\d+n\d+)p\d+$")  # /dev/nvme0n1p1 -> /dev/nvme0n1


def _make_candidate(
    issue_id: str,
    title: str,
    problem: str,
    evidence: dict,
    severity: str,
    fallback_command: str,
) -> dict:
    """Same shape as fix_engine._make_candidate() - issue_type is always

    "hardware_fault" here; the specific component (sensor/fan/disk) lives
    in `issue_id` and `evidence`, exactly like every other detector's
    candidates.
    """
    return {
        "issue_id": issue_id,
        "issue_type": "hardware_fault",
        "title": title,
        "severity": severity,
        "problem": problem,
        "evidence": evidence,
        "fallback_command": fallback_command,
    }


def _severity_for_temp(current: float) -> str:
    return "critical" if current >= _TEMP_CRITICAL_CELSIUS else "warning"


# ---------------------------------------------------------------------------
# Temperature sensors
# ---------------------------------------------------------------------------

async def _detect_overheating() -> list[dict]:
    """Flag any temperature sensor reading at or above the warning

    threshold. `psutil.sensors_temperatures()` is Linux-only and, even
    there, is only populated when the host exposes `/sys/class/hwmon`
    entries (bare metal / most VMs with the right kernel modules;
    frequently empty in containers) - both cases degrade to "no data",
    not an error.

    A single elevated reading is reported at "warning" severity, never
    escalated to "critical" (and therefore never treated as a confirmed
    hardware fault) unless it is at or above the sensor's own critical
    threshold - CPU/GPU/NVMe chips all report through this same path,
    and none of them get special-cased into an automatic fault just for
    running warm once.
    """
    candidates: list[dict] = []

    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, OSError) as exc:
        logger.info("Temperature sensors unavailable on this host: %s", exc)
        return candidates

    if not temps:
        return candidates

    for chip_name, entries in temps.items():
        for entry in entries:
            current = entry.current
            if current is None:
                continue

            # Prefer the sensor's own high/critical thresholds when the
            # kernel reports them; fall back to our configured defaults
            # otherwise so hosts without per-sensor thresholds still get
            # meaningful alerts.
            warning_threshold = entry.high if entry.high else _TEMP_WARNING_CELSIUS
            critical_threshold = entry.critical if entry.critical else _TEMP_CRITICAL_CELSIUS

            if current < warning_threshold:
                continue

            label = entry.label or chip_name
            severity = "critical" if current >= critical_threshold else "warning"
            evidence = {
                "sensor": chip_name,
                "label": label,
                "current_celsius": current,
                "warning_threshold_celsius": warning_threshold,
                "critical_threshold_celsius": critical_threshold,
            }
            candidates.append(
                _make_candidate(
                    issue_id=f"hardware_fault:temp:{chip_name}:{label}",
                    title=f"High Temperature ({label})",
                    problem=(
                        f"Sensor '{label}' on '{chip_name}' reads {current}\u00b0C, "
                        f"at or above the {warning_threshold}\u00b0C warning threshold."
                    ),
                    evidence=evidence,
                    severity=severity,
                    fallback_command="sensors 2>/dev/null || cat /sys/class/thermal/thermal_zone*/temp",
                )
            )

    return candidates


# ---------------------------------------------------------------------------
# Fans
# ---------------------------------------------------------------------------

async def _detect_fan_failure() -> list[dict]:
    """Flag any reporting fan that has stalled (0 RPM) while a temperature

    reading is elevated - a stopped fan alone is common and often
    meaningless (fanless boards, idle-off fan curves), but a stopped fan
    combined with heat is a genuine early-failure signal.
    """
    candidates: list[dict] = []

    try:
        fans = psutil.sensors_fans()
    except (AttributeError, OSError) as exc:
        logger.info("Fan sensors unavailable on this host: %s", exc)
        return candidates

    if not fans:
        return candidates

    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        temps = {}

    max_temp = 0.0
    for entries in temps.values():
        for entry in entries:
            if entry.current is not None:
                max_temp = max(max_temp, entry.current)

    if max_temp < _TEMP_WARNING_CELSIUS:
        return candidates  # system isn't running hot - a stalled fan isn't urgent yet

    for chip_name, entries in fans.items():
        for entry in entries:
            if entry.current is None or entry.current > 0:
                continue

            label = entry.label or chip_name
            evidence = {
                "sensor": chip_name,
                "label": label,
                "current_rpm": entry.current,
                "max_temperature_celsius": max_temp,
                "temperature_warning_threshold_celsius": _TEMP_WARNING_CELSIUS,
            }
            candidates.append(
                _make_candidate(
                    issue_id=f"hardware_fault:fan:{chip_name}:{label}",
                    title=f"Fan Failure ({label})",
                    problem=(
                        f"Fan '{label}' on '{chip_name}' reports 0 RPM while system temperature "
                        f"is elevated ({max_temp}\u00b0C)."
                    ),
                    evidence=evidence,
                    severity="critical",
                    fallback_command="sensors 2>/dev/null",
                )
            )

    return candidates


# ---------------------------------------------------------------------------
# Disk SMART health
# ---------------------------------------------------------------------------

def _physical_disk_devices() -> list[str]:
    """Reduce the mounted partitions' device paths to their unique

    underlying physical disks (e.g. /dev/sda1 -> /dev/sda), skipping
    loop/tmpfs/overlay/snap pseudo-devices that have no SMART data.
    """
    try:
        partitions = system_monitor.get_disk_info().get("partitions", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hardware monitor could not read partitions for SMART check: %s", exc)
        return []

    devices: set[str] = set()
    for part in partitions:
        device = part.get("device") or ""
        mountpoint = part.get("mountpoint") or ""

        if mountpoint == "/snap" or mountpoint.startswith(_IGNORED_MOUNT_PREFIXES):
            continue
        if not device or device.startswith(_IGNORED_DEVICE_PREFIXES):
            continue

        match = _NVME_PARTITION_SUFFIX_RE.match(device) or _PARTITION_SUFFIX_RE.match(device)
        devices.add(match.group(1) if match else device)

    return sorted(devices)


def _run_smartctl_raw(args: list[str]) -> tuple[str, "int | None"]:
    """Invoke `smartctl` directly and return (stdout, exit_code).

    Deliberately bypasses the shared `run_command()` helper: that helper
    discards stdout whenever the process exits non-zero, which is the
    right behaviour for every other command in this project but wrong
    for `smartctl`, whose exit status is a bitmask of many independent,
    often non-fatal conditions (see module docstring). This function
    always returns whatever stdout was produced, regardless of exit
    code, so the caller can make its own evidence-based health call.

    Never raises: a missing binary, a timeout, or any other failure to
    even launch the process degrades to `("", None)` - "nothing to
    report from this check", same as every other collector here.
    """
    try:
        result = subprocess.run(
            ["smartctl", *args],
            capture_output=True,
            text=True,
            timeout=_SMARTCTL_TIMEOUT_SECONDS,
            check=False,
        )
        return result.stdout, result.returncode
    except FileNotFoundError:
        logger.info("smartctl not installed - skipping SMART check")
        return "", None
    except subprocess.TimeoutExpired:
        logger.info("smartctl timed out")
        return "", None
    except Exception as exc:  # noqa: BLE001 - defensive catch-all for an external tool
        logger.warning("smartctl invocation failed: %s", exc)
        return "", None


# Overall self-assessment verdict line. Covers both phrasings smartctl
# uses depending on drive type:
#   ATA/NVMe : "SMART overall-health self-assessment test result: PASSED"
#   older ATA: "SMART Health Status: OK"
_SMART_OVERALL_VERDICT_RE = re.compile(
    r"(?:overall-health self-assessment test result|smart health status)\s*:\s*([A-Za-z]+)",
    re.IGNORECASE,
)
# NVMe-only SMART/Health Information log fields.
_SMART_CRITICAL_WARNING_RE = re.compile(r"critical warning\s*:\s*(0x[0-9a-fA-F]+)", re.IGNORECASE)
_SMART_MEDIA_ERRORS_RE = re.compile(r"media and data integrity errors\s*:\s*(\d+)", re.IGNORECASE)


def _parse_smart_health(output: str) -> dict:
    """Pull just the handful of SMART fields we trust as real health

    evidence out of `smartctl -a` output - the overall pass/fail
    self-assessment, NVMe's Critical Warning bitmask, and NVMe's Media
    and Data Integrity Errors counter. Everything else in the output
    (unsupported optional log pages such as a failed self-test log read,
    vendor attribute noise, etc.) is ignored by construction, since only
    these specific labeled fields are ever looked for.
    """
    verdict_match = _SMART_OVERALL_VERDICT_RE.search(output)
    critical_warning_match = _SMART_CRITICAL_WARNING_RE.search(output)
    media_errors_match = _SMART_MEDIA_ERRORS_RE.search(output)

    return {
        # "PASSED" / "OK" / "FAILED" / "FAILING" / ... / None if not found
        "overall_verdict": verdict_match.group(1).strip().upper() if verdict_match else None,
        # e.g. "0x00", or None if not an NVMe drive / not found
        "critical_warning": critical_warning_match.group(1).lower() if critical_warning_match else None,
        # int, or None if not an NVMe drive / not found
        "media_errors": int(media_errors_match.group(1)) if media_errors_match else None,
    }


def _is_smart_healthy(health: dict) -> "bool | None":
    """Return True (healthy), False (genuine fault), or None (inconclusive).

    Healthy: an explicit passing verdict (PASSED/OK), and - when
    present - a zero NVMe Critical Warning bitmask and zero Media and
    Data Integrity Errors count. This matches the real-world case this
    fixes exactly: PASSED + Critical Warning 0x00 + Media and Data
    Integrity Errors 0.

    Fault: an explicit failing verdict (FAILED/FAILING), OR a non-zero
    Critical Warning bitmask, OR a non-zero Media/Data Integrity Errors
    count - each is direct evidence from the drive's own firmware, never
    just a non-zero smartctl process exit code.

    Inconclusive (None): no parseable verdict at all - e.g. smartctl
    couldn't open the device, lacks permission, or the drive doesn't
    expose these fields. Never reported as a fault, since there is no
    real evidence either way; matches this project's existing
    "missing/ambiguous data is not an alert" convention.
    """
    verdict = health["overall_verdict"]
    critical_warning = health["critical_warning"]
    media_errors = health["media_errors"]

    has_fault_evidence = (
        verdict in ("FAILED", "FAILING")
        or (critical_warning is not None and critical_warning != "0x00")
        or (media_errors is not None and media_errors > 0)
    )
    if has_fault_evidence:
        return False

    if verdict in ("PASSED", "OK"):
        return True

    return None


async def _detect_disk_smart_failures() -> list[dict]:
    """Query `smartctl -a` (from smartmontools) for each physical disk

    backing a mounted, non-pseudo filesystem, and judge health from the
    actual SMART fields in its output rather than the process exit code
    (see module docstring). Gracefully returns nothing if `smartctl`
    isn't installed, isn't permitted (needs root on most distros), or
    the disk doesn't support SMART (e.g. some virtualized block
    devices), or produces output with no parseable health verdict at
    all - each of those is a normal, expected outcome on many hosts, not
    a detected fault.
    """
    candidates: list[dict] = []
    devices = _physical_disk_devices()
    if not devices:
        return candidates

    for device in devices:
        output, exit_code = _run_smartctl_raw(["-a", device])
        if not output.strip():
            # smartctl not installed, no permission, device doesn't
            # support SMART, or timed out - none of these are a
            # detected fault, just missing data.
            logger.info("SMART health unavailable for %s (smartctl exit code: %s)", device, exit_code)
            continue

        health = _parse_smart_health(output)
        healthy = _is_smart_healthy(health)
        if healthy is not False:
            # True (explicitly healthy, e.g. PASSED + Critical Warning
            # 0x00 + zero Media/Data Integrity Errors) or None
            # (inconclusive - no parseable verdict) both mean "nothing
            # to report". In particular, a non-zero smartctl exit code
            # by itself (e.g. from an unsupported "Read Self-test Log"
            # page) is never treated as a failure here.
            continue

        evidence = {
            "device": device,
            "smartctl_exit_code": exit_code,
            "smart_overall_verdict": health["overall_verdict"],
            "smart_critical_warning": health["critical_warning"],
            "smart_media_and_data_integrity_errors": health["media_errors"],
            "smartctl_output": output.strip()[:500],
        }
        candidates.append(
            _make_candidate(
                issue_id=f"hardware_fault:smart:{device}",
                title=f"Disk SMART Health Failure ({device})",
                problem=f"SMART health data for '{device}' shows real evidence of a drive problem.",
                evidence=evidence,
                severity="critical",
                fallback_command=f"smartctl -a {device}",
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Current sensor readings (NEW - additive, read-only, no detection logic)
#
# `_detect_overheating()` above is a FAULT detector: it only reports a
# sensor when it is at/above its warning threshold, and says nothing at
# all about a normal reading - by design, since its job is "is anything
# wrong", not "what is the current reading". Nothing about it is changed
# here. The functions below are the opposite: a plain, unconditional
# snapshot of whatever temperature sensors this host currently exposes, no
# threshold comparison, no candidate/issue creation, no interaction with
# issue_alert_store or fix_engine. Not called from detect_hardware_issues()
# / detect_all_issues() and does not affect fault-detection behavior in any
# way - it exists purely so a direct "what is my current CPU/GPU
# temperature?" question has real numbers to answer from even when nothing
# is wrong.
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nvidia_smi_readings() -> list[dict]:
    """Best-effort current GPU temperature(s) via `nvidia-smi`, if present.

    `psutil.sensors_temperatures()` only sees a GPU's temperature when the
    kernel exposes it through `/sys/class/hwmon` - true for amdgpu, nouveau,
    and i915, but NOT for NVIDIA's proprietary driver. `nvidia-smi` is the
    standard, vendor-provided way to read an NVIDIA GPU's current
    temperature when that driver is in use. Never assumes NVIDIA is
    present: if the binary is missing, times out, or errors, this simply
    returns an empty list - "nothing to report" - exactly like every other
    collector in this module (e.g. `_run_smartctl_raw`).
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_SMARTCTL_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        logger.info("nvidia-smi not installed - skipping NVIDIA GPU temperature read")
        return []
    except subprocess.TimeoutExpired:
        logger.info("nvidia-smi timed out")
        return []
    except Exception as exc:  # noqa: BLE001 - defensive catch-all for an external tool
        logger.warning("nvidia-smi invocation failed: %s", exc)
        return []

    if result.returncode != 0 or not result.stdout.strip():
        return []

    readings: list[dict] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        index, name, temp_str = parts
        try:
            current_celsius = float(temp_str)
        except ValueError:
            continue
        readings.append(
            {
                "source": "nvidia-smi",
                "chip": f"nvidia:{index}",
                "label": name or f"NVIDIA GPU {index}",
                "current_celsius": current_celsius,
                "high_celsius": None,
                "critical_celsius": None,
            }
        )
    return readings


def get_current_sensor_readings() -> dict:
    """Return every current temperature sensor reading this host exposes, right now.

    A pure snapshot - no thresholds, no fault/candidate creation, no writes
    to `issue_alert_store` - safe to call on every chat request without
    affecting any detection state (unlike e.g. `failure_predictor`'s
    rolling sample history). Reports whatever sensor/vendor is actually
    present (amdgpu, nouveau, i915, NVMe, CPU package, etc. via psutil;
    NVIDIA via `nvidia-smi` if installed) - never assumes any particular
    vendor is present, and never fabricates a value for a sensor this host
    doesn't actually expose.
    """
    sensors: list[dict] = []
    psutil_error: str | None = None

    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, OSError) as exc:
        temps = {}
        psutil_error = str(exc)
        logger.info("Temperature sensors unavailable on this host: %s", exc)

    for chip_name, entries in (temps or {}).items():
        for entry in entries:
            if entry.current is None:
                continue
            sensors.append(
                {
                    "source": "psutil",
                    "chip": chip_name,
                    "label": entry.label or chip_name,
                    "current_celsius": entry.current,
                    "high_celsius": entry.high,
                    "critical_celsius": entry.critical,
                }
            )

    sensors.extend(_nvidia_smi_readings())

    if sensors:
        note = (
            "These are the REAL, current temperature sensor readings collected "
            "just now on this host - this is NOT a fault list. A sensor "
            "appearing here with a normal value does NOT mean a hardware "
            "fault; only a matching entry in 'hardware_faults' (from the "
            "actual overheating detector) indicates a confirmed fault."
        )
    else:
        note = (
            "No temperature sensors were available from this host at "
            "collection time (psutil found none via /sys/class/hwmon"
            + (f" [{psutil_error}]" if psutil_error else "")
            + ", and nvidia-smi is not installed or returned no data). This "
            "is common on VMs/containers/some cloud hosts. Do not invent a "
            "CPU or GPU temperature - state plainly that no sensor data is "
            "currently available."
        )

    return {
        "sensors": sensors,
        "sensor_count": len(sensors),
        "collected_at": _now_iso(),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Aggregate entry point - called from fix_engine.detect_all_issues()
# ---------------------------------------------------------------------------

async def detect_hardware_issues() -> list[dict]:
    """Run every hardware check once and return their combined candidates.

    Shaped to drop straight into `detect_all_issues()`'s existing
    `asyncio.gather` call, the same way every other detector in this
    project is wired in. Each check is isolated with its own try/except
    so a failure or unavailable sensor in one never prevents the others
    from reporting.
    """
    candidates: list[dict] = []

    async def _safe(label: str, coro):
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001
            logger.error("Hardware monitor check '%s' failed: %s", label, exc)
            return []

    overheating, fan_failures, smart_failures = await asyncio.gather(
        _safe("overheating", _detect_overheating()),
        _safe("fan_failure", _detect_fan_failure()),
        _safe("disk_smart", _detect_disk_smart_failures()),
    )

    candidates.extend(overheating or [])
    candidates.extend(fan_failures or [])
    candidates.extend(smart_failures or [])

    return candidates