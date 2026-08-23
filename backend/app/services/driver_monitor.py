"""
Driver Anomaly Monitor - fault detection for kernel/driver-level problems
surfaced in the kernel ring buffer.

Follows the exact same "detector" shape as everything else feeding the
AI One-Click Fix Engine (see `fix_engine.py`): a read-only, defensive
collector that turns raw evidence (here, matching kernel log lines) into
candidate dicts shaped like `fix_engine._make_candidate()` produces
(issue_id/issue_type/title/problem/evidence/severity/fallback_command).
Nothing in this module diagnoses (that's `fix_engine._ai_diagnose`),
dedupes (that's `issue_alert_store.reconcile()`), or executes anything
(that's the existing Safe Command Execution pipeline, reached only after
an explicit user confirmation) - this module only reads kernel logs and
reports patterns found in them, exactly like `system_monitor.py`'s
`get_recent_logs()` already does for general journal entries.

Integration point: `fix_engine.detect_all_issues()` calls
`detect_driver_anomalies()` once per scan alongside its other detectors
via `asyncio.gather`, and folds the returned candidates into the same
candidate list handed to `issue_alert_store.reconcile()`.

Source of log lines: `journalctl -k` (the kernel ring buffer via
systemd-journald), with a `dmesg` fallback for hosts where journald
either isn't storing kernel logs or the invoking user can't read them.
Both are read-only, non-interactive, and already used elsewhere in this
project (`system_monitor.get_recent_logs` uses `journalctl` the same
way) - this module never executes anything beyond reading logs. A
missing/unreadable log source degrades to "nothing to report", not an
error, matching every other collector in this project.

Severity classification.

`IssueSeverity` (schemas/fixes.py) only has two values - "warning" and
"critical" - so "lower severity" for a softened match means "warning",
not a new tier. A category's tuple below still carries that same
`default_severity`, but instead of using it verbatim, `detect_driver_anomalies()`
runs each category's matches through a small classifier keyed by
`classifier_kind`:

  * "fixed"    - unchanged behaviour: always `default_severity`. Used for
    signatures that are unambiguous even as a single line (kernel panics,
    machine-check errors, storage I/O errors, hung tasks, thermal
    shutdown/throttle events) - these stay exactly as before.
  * "gpu"      - a lone GPU driver error line immediately surrounded by
    the driver's own "initialized successfully" messages reads as
    initialization-time noise, not a hardware fault, so it is reported
    at "warning" instead of "critical". Known-serious phrasing (bus
    fallen off, GPU recovery, ring/fence timeout, device lost/hang) or
    repeated occurrences still escalate straight to "critical", exactly
    as this category always has.
  * "network"  - an isolated "Link is Down" is normal (cable unplug,
    suspend/resume, DHCP renegotiation) and is not reported at all
    unless it either repeats enough to look like flapping or is paired
    with a genuinely concerning signature (tx queue timeout, carrier
    lost, watchdog timeout).
  * "generic_escalate" - the existing warning-level categories
    (firmware/PCIe/USB anomalies) are unchanged for a handful of
    occurrences, but escalate to "critical" once the same signature
    keeps recurring in one scan window, per the same "repeat implies
    persistent problem" rule applied to GPU/network above.

This keeps every previously-critical, unambiguous signature exactly as
critical as before, and only softens/suppresses the two categories that
were shown to produce false positives (isolated GPU init chatter, benign
link-down events), while still escalating on repetition or stronger
evidence - it does not add any new dependency, execute anything, or
change the candidate shape.
"""

import re
from datetime import datetime, timezone

from app.config import get_settings
from app.logger import get_logger
from app.utils import run_command

settings = get_settings()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
#
# Not in Settings yet, so read via getattr() with sane defaults - same
# approach as failure_predictor.py / hardware_monitor.py: this module
# works standalone today and will automatically pick up real config.py
# values once those settings are added there, with no further change
# needed here.
# ---------------------------------------------------------------------------
_LOG_LINES_SCANNED = int(getattr(settings, "driver_monitor_log_lines", 500))
_MAX_EXAMPLE_LINES = int(getattr(settings, "driver_monitor_max_examples", 5))
_MAX_LINE_LENGTH = 300  # truncate any single log line before it goes into evidence

# ---------------------------------------------------------------------------
# Severity-classification tuning.
#
# All thresholds are simple counts/regexes on the already-collected match
# list for one scan - no new log reads, no ML, no heavy dependencies.
# ---------------------------------------------------------------------------

# GPU: how many lines of surrounding context (before/after the matched
# line, in the same scanned window) to inspect for a reassuring
# "initialized successfully" style message from the same driver.
_GPU_CONTEXT_WINDOW = 6

# GPU: phrasing that always means a real, active hardware problem -
# never softened, regardless of context or count.
_GPU_STRONG_EVIDENCE_PATTERN = re.compile(
    r"fallen off the bus|amdgpu_device_gpu_recover|failed to reset|"
    r"gpu reset|ring \S+ timeout|dma fence.*timeout|device lost|\bhang\b",
    re.IGNORECASE,
)

# GPU: reassuring context indicating the matched line was part of normal
# driver bring-up rather than a live failure (e.g. "SMU is initialized
# successfully", "Initialized amdgpu ...").
_GPU_INIT_CONTEXT_PATTERN = re.compile(
    r"initializ\w*\s+successfully|\binitialized\s+\S*(?:amdgpu|nouveau|i915|radeon|nvidia)\b|"
    r"firmware.*loaded",
    re.IGNORECASE,
)

# GPU: this many occurrences (or more) in one scan window is treated as a
# persistent problem and always escalates to critical, even without the
# strong-evidence phrasing above.
_GPU_REPEAT_ESCALATE_COUNT = 3

# Network: benign, standalone interface state change - normal on cable
# unplug, suspend/resume, roaming, etc.
_NETWORK_BENIGN_PATTERN = re.compile(r"\blink is down\b", re.IGNORECASE)

# Network: phrasing that indicates an actual driver/hardware problem
# rather than a routine link state change.
_NETWORK_CONCERNING_PATTERN = re.compile(
    r"tx queue.*timed out|carrier lost|watchdog timeout|nic error|reset adapter",
    re.IGNORECASE,
)

# Network: an isolated/occasional "Link is Down" (at or below this count,
# with no concerning phrasing) is suppressed entirely rather than
# reported, per the false-positive fix. Above this, repeated flapping is
# reported as "warning"; at/above the escalate count below it's treated
# as a likely real connectivity/driver problem.
_NETWORK_SUPPRESS_AT_OR_BELOW_COUNT = 2
_NETWORK_REPEAT_ESCALATE_COUNT = 4

# Generic warning-level categories (firmware/PCIe/USB): unchanged at a
# handful of occurrences, but escalate to critical once the same
# signature clearly keeps recurring in one scan window.
_GENERIC_REPEAT_ESCALATE_COUNT = 5

# ---------------------------------------------------------------------------
# Anomaly categories.
#
# Each category is a (pattern, category_key, title, default_severity,
# fallback_command, classifier_kind) tuple. Patterns are intentionally
# broad, well-known kernel log signatures rather than an exhaustive
# driver database - the same "cheap, deterministic candidate now, AI
# explains the specifics later" split used by every other detector in
# this project. A line is assigned to the FIRST category whose pattern
# matches, so more specific/severe patterns are ordered first.
#
# `default_severity` is what "fixed" categories always use, and what
# every other classifier_kind uses as its starting point before
# softening/escalating (see the classifier functions below).
# ---------------------------------------------------------------------------
_ANOMALY_CATEGORIES: list[tuple["re.Pattern[str]", str, str, str, str, str]] = [
    (
        re.compile(r"\bhung_task\b|blocked for more than \d+ seconds", re.IGNORECASE),
        "hung_task",
        "Hung Kernel Task Detected",
        "critical",
        "journalctl -k -p err -n 100 --no-pager",
        "fixed",
    ),
    (
        re.compile(r"\bkernel (?:panic|oops)\b|\bOops:", re.IGNORECASE),
        "kernel_panic",
        "Kernel Panic / Oops Detected",
        "critical",
        "journalctl -k -p err -n 100 --no-pager",
        "fixed",
    ),
    (
        re.compile(r"\bmce\b.*hardware error|machine check", re.IGNORECASE),
        "machine_check",
        "Hardware Machine-Check Error",
        "critical",
        "journalctl -k -p err -n 100 --no-pager",
        "fixed",
    ),
    (
        re.compile(r"\bnvme\b.*\b(?:timeout|reset|error)\b|\bata\d+.*\b(?:exception|error|failed command)\b", re.IGNORECASE),
        "storage_driver_error",
        "Storage Driver Error (NVMe/ATA)",
        "critical",
        "journalctl -k -p err -n 100 --no-pager",
        "fixed",
    ),
    (
        re.compile(r"\bI/O error\b|\bcritical medium error\b|\bunable to read capacity\b", re.IGNORECASE),
        "io_error",
        "Disk I/O Error",
        "critical",
        "journalctl -k -p err -n 100 --no-pager",
        "fixed",
    ),
    (
        re.compile(r"\b(?:nouveau|amdgpu|i915|radeon|nvidia)\b.*\b(?:error|fault|failed|hang|timeout|gpu has fallen off the bus)\b", re.IGNORECASE),
        "gpu_driver_error",
        "GPU Driver Error",
        "critical",
        "journalctl -k -p err -n 100 --no-pager",
        "gpu",
    ),
    (
        re.compile(r"\bfirmware\b.*\b(?:failed|error|crash|timeout)\b", re.IGNORECASE),
        "firmware_error",
        "Device Firmware Error",
        "warning",
        "journalctl -k -p warning -n 100 --no-pager",
        "generic_escalate",
    ),
    (
        re.compile(r"\bpcieport\b.*\berror\b|\bAER\b.*\berror\b|\bcorrected error\b", re.IGNORECASE),
        "pcie_error",
        "PCIe Bus Error",
        "warning",
        "journalctl -k -p warning -n 100 --no-pager",
        "generic_escalate",
    ),
    (
        re.compile(r"\busb \d+-[\d.]+.*\b(?:device descriptor read.*error|disconnect|reset.*failed)\b", re.IGNORECASE),
        "usb_anomaly",
        "USB Device Anomaly",
        "warning",
        "journalctl -k -p warning -n 100 --no-pager",
        "generic_escalate",
    ),
    (
        re.compile(r"\blink is down\b|\bnetwork.*\b(?:tx queue.*timed out|carrier lost)\b", re.IGNORECASE),
        "network_driver_error",
        "Network Driver / Link Error",
        "warning",
        "journalctl -k -p warning -n 100 --no-pager",
        "network",
    ),
    (
        re.compile(r"\bthermal\b.*\b(?:critical|shutdown|throttl)", re.IGNORECASE),
        "thermal_event",
        "Thermal Throttling / Shutdown Event",
        "critical",
        "journalctl -k -p warning -n 100 --no-pager",
        "fixed",
    ),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_candidate(
    issue_id: str,
    title: str,
    problem: str,
    evidence: dict,
    severity: str,
    fallback_command: str,
) -> dict:
    """Same shape as fix_engine._make_candidate() - issue_type is always

    "driver_anomaly" here; the specific anomaly category lives in
    `issue_id` and `evidence`, exactly like every other detector's
    candidates.
    """
    return {
        "issue_id": issue_id,
        "issue_type": "driver_anomaly",
        "title": title,
        "severity": severity,
        "problem": problem,
        "evidence": evidence,
        "fallback_command": fallback_command,
    }


# ---------------------------------------------------------------------------
# Log collection - journalctl -k with a dmesg fallback, both read-only
# and non-interactive, matching system_monitor.get_recent_logs's pattern.
# ---------------------------------------------------------------------------

def _read_kernel_log_lines() -> tuple[list[str], str | None]:
    """Return (lines, error). `error` is set only if BOTH journalctl -k

    and dmesg failed - a single source failing over to the other is not
    reported as an error, matching this project's "one collector's
    failure never breaks the response" convention.
    """
    command = [
        "journalctl",
        "-k",
        "-n",
        str(_LOG_LINES_SCANNED),
        "--no-pager",
        "-o",
        "cat",  # plain message text, one per line - no JSON parsing needed
    ]
    ok, output = run_command(command, timeout=10)
    if ok:
        return [line for line in output.splitlines() if line.strip()], None

    logger.info("journalctl -k unavailable (%s), falling back to dmesg", output)

    ok, output = run_command(["dmesg", "--ctime"], timeout=10)
    if ok:
        lines = [line for line in output.splitlines() if line.strip()]
        return lines[-_LOG_LINES_SCANNED:], None

    error = f"Could not read kernel logs via journalctl -k or dmesg: {output}"
    logger.warning(error)
    return [], error


# ---------------------------------------------------------------------------
# Installed/loaded driver inventory - `lsmod`.
#
# Root-cause fix: this module previously only ever reported *anomalies*
# (kernel-log error patterns). A plain "which drivers are installed?"
# question has no anomaly to find - if nothing is currently misbehaving,
# `detect_driver_anomalies()` correctly returns an empty list, but that
# emptiness was then being read by context_builder as "no evidence of
# ANY driver info at all", so the LLM had nothing to answer an inventory
# question from. `get_loaded_kernel_modules()` is a separate, plain
# snapshot of every currently loaded kernel module (`lsmod`) - read-only,
# deterministic, and safe to call on every chat request (no rolling
# state, unlike failure_predictor). It is intentionally kept separate
# from the anomaly detector above: anomalies are still the only thing
# that ever creates a `driver_anomaly` issue/alert; this is purely
# descriptive inventory data for "what's installed/loaded" questions.
# ---------------------------------------------------------------------------

def get_loaded_kernel_modules(limit: int = 50) -> dict:
    """Return the currently loaded kernel modules/drivers via `lsmod`.

    A module here is "loaded", which is the practical, checkable meaning
    of "installed driver" on a running Linux system (there is no single
    authoritative list of every driver ever installed vs. in active use -
    `lsmod` is the standard, real-time answer to "what's loaded right
    now"). Never raises: a missing/unreadable `lsmod` degrades to an
    empty list plus an explanatory error, matching every other collector
    in this project.
    """
    errors: list[str] = []
    ok, output = run_command(["lsmod"], timeout=10)

    if not ok:
        errors.append(f"Could not list kernel modules via lsmod: {output}")
        logger.warning("get_loaded_kernel_modules: %s", errors[-1])
        return {
            "modules": [],
            "total_modules": 0,
            "collected_at": _now_iso(),
            "errors": errors,
        }

    modules: list[dict] = []
    lines = [l for l in output.splitlines() if l.strip()]
    # First line is the header ("Module  Size  Used by") - skip it.
    for line in lines[1:]:
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        name, size_str, used_by_count_str = parts[0], parts[1], parts[2]
        used_by = parts[3].strip() if len(parts) > 3 else ""
        try:
            size_bytes = int(size_str)
        except ValueError:
            size_bytes = None
        try:
            used_by_count = int(used_by_count_str)
        except ValueError:
            used_by_count = None
        modules.append(
            {
                "name": name,
                "size_bytes": size_bytes,
                "used_by_count": used_by_count,
                "used_by": [m for m in used_by.split(",") if m] if used_by else [],
            }
        )

    total = len(modules)
    return {
        "modules": modules[: max(limit, 0)],
        "total_modules": total,
        "collected_at": _now_iso(),
        "errors": errors,
    }


def _truncate(line: str) -> str:
    line = line.strip()
    return line if len(line) <= _MAX_LINE_LENGTH else line[: _MAX_LINE_LENGTH - 3] + "..."


# ---------------------------------------------------------------------------
# Severity classifiers.
#
# Each takes the category's `default_severity`, the full (uncapped) list
# of matching line indices for this scan, and the raw scanned lines (for
# context lookups), and returns (severity, note) - or (None, note) when
# the match set should be suppressed entirely (not reported this scan).
# `note` is a short, human-readable reason folded into evidence for the
# AI diagnosis step and the user-facing explanation, matching this
# project's "explainable" (XAI) framing - it never changes behaviour.
# ---------------------------------------------------------------------------

def _classify_fixed(default_severity: str, match_indices: list[int], lines: list[str]) -> tuple[str | None, str]:
    return default_severity, "Signature is unambiguous on its own; severity unchanged."


def _classify_gpu(default_severity: str, match_indices: list[int], lines: list[str]) -> tuple[str | None, str]:
    occurrence_count = len(match_indices)

    for idx in match_indices:
        if _GPU_STRONG_EVIDENCE_PATTERN.search(lines[idx]):
            return "critical", "Log line contains explicit hardware-failure phrasing (bus/reset/hang/timeout)."

    if occurrence_count >= _GPU_REPEAT_ESCALATE_COUNT:
        return "critical", f"Pattern recurred {occurrence_count} times in this scan - treated as a persistent driver problem."

    if occurrence_count == 1:
        idx = match_indices[0]
        lo = max(0, idx - _GPU_CONTEXT_WINDOW)
        hi = min(len(lines), idx + _GPU_CONTEXT_WINDOW + 1)
        context = lines[lo:hi]
        if any(_GPU_INIT_CONTEXT_PATTERN.search(l) for l in context):
            return (
                "warning",
                "Single occurrence immediately surrounded by successful driver-initialization "
                "messages - likely init-time noise, not an active hardware fault.",
            )

    return default_severity, "No reassuring init context and/or more than one occurrence; kept at default severity."


def _classify_network(default_severity: str, match_indices: list[int], lines: list[str]) -> tuple[str | None, str]:
    occurrence_count = len(match_indices)
    concerning = any(_NETWORK_CONCERNING_PATTERN.search(lines[idx]) for idx in match_indices)

    if concerning:
        if occurrence_count >= 2:
            return "critical", "Concerning driver phrasing (tx timeout/carrier lost/watchdog) seen multiple times."
        return default_severity, "Concerning driver phrasing seen once; reported at default severity."

    # From here on, every matched line is a benign "link is down"-style message.
    if occurrence_count >= _NETWORK_REPEAT_ESCALATE_COUNT:
        return "critical", f"Link state changed {occurrence_count} times in this scan - looks like flapping, not a normal disconnect."
    if occurrence_count <= _NETWORK_SUPPRESS_AT_OR_BELOW_COUNT:
        return None, "Isolated link-down event with no concerning phrasing - consistent with a normal disconnect, not reported."
    return default_severity, "Link went down a few times without concerning phrasing; reported but not escalated."


def _classify_generic_escalate(default_severity: str, match_indices: list[int], lines: list[str]) -> tuple[str | None, str]:
    occurrence_count = len(match_indices)
    if default_severity != "critical" and occurrence_count >= _GENERIC_REPEAT_ESCALATE_COUNT:
        return "critical", f"Pattern recurred {occurrence_count} times in this scan - treated as a persistent problem."
    return default_severity, "Occurrence count within normal range; severity unchanged."


_CLASSIFIERS = {
    "fixed": _classify_fixed,
    "gpu": _classify_gpu,
    "network": _classify_network,
    "generic_escalate": _classify_generic_escalate,
}


# ---------------------------------------------------------------------------
# Aggregate entry point - called from fix_engine.detect_all_issues()
# ---------------------------------------------------------------------------

async def detect_driver_anomalies() -> list[dict]:
    """Scan the most recent kernel log lines for known driver/hardware

    anomaly signatures and return one candidate per category observed in
    this pass (not one per matching line - a flapping USB device or a
    storage controller logging repeated errors should surface as a
    single "USB Device Anomaly" / "Storage Driver Error" card, exactly
    like `fix_engine._detect_disk_almost_full` consolidates multiple
    breaching mounts into one "Storage Incident" rather than one card
    per mount).

    Each category's severity is then resolved by its classifier (see
    `_CLASSIFIERS` above): unambiguous signatures behave exactly as
    before, while GPU/network/warning-level categories may be softened
    for isolated/init-time noise, escalated on repetition or stronger
    evidence, or - for a clearly benign, isolated link-down - suppressed
    entirely so it isn't reported as an anomaly at all.

    Never raises: a missing/unreadable log source, or zero matches,
    simply results in an empty list.
    """
    lines, read_error = _read_kernel_log_lines()
    if read_error or not lines:
        return []

    # category_key -> list of line indices (into `lines`) that matched,
    # in scan order. Uncapped, so classifiers see the true occurrence
    # count and can look at any matched line's surrounding context.
    match_indices: dict[str, list[int]] = {}

    for i, line in enumerate(lines):
        for pattern, category_key, _title, _severity, _fallback, _classifier_kind in _ANOMALY_CATEGORIES:
            if pattern.search(line):
                match_indices.setdefault(category_key, []).append(i)
                break  # first matching category wins - avoid double-counting one line

    if not match_indices:
        return []

    candidates: list[dict] = []
    for pattern, category_key, title, default_severity, fallback_command, classifier_kind in _ANOMALY_CATEGORIES:
        indices = match_indices.get(category_key)
        if not indices:
            continue

        classifier = _CLASSIFIERS[classifier_kind]
        severity, note = classifier(default_severity, indices, lines)
        if severity is None:
            # Classifier judged this match set to be a non-issue (e.g. an
            # isolated, benign link-down) - nothing to report this scan.
            continue

        occurrence_count = len(indices)
        examples = [_truncate(lines[i]) for i in indices[:_MAX_EXAMPLE_LINES]]

        evidence = {
            "category": category_key,
            "occurrence_count": occurrence_count,
            "log_lines_scanned": len(lines),
            "example_log_lines": examples,
            "classification_note": note,
        }
        candidates.append(
            _make_candidate(
                issue_id=f"driver_anomaly:{category_key}",
                title=title,
                problem=(
                    f"Kernel log shows {occurrence_count} occurrence(s) of a '{title.lower()}' "
                    f"pattern in the last {len(lines)} scanned kernel log lines."
                ),
                evidence=evidence,
                severity=severity,
                fallback_command=fallback_command,
            )
        )

    return candidates