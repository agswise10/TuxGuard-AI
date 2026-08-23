"""
Context builder for the AI Ops Assistant.

Given a classified intent, gathers just the system data relevant to it
(rather than dumping the entire system-info payload into every prompt).
This keeps prompts small, fast, and focused - and keeps the mapping from
"kind of question" to "data we fetch" in one obvious place.

Sprint 9 fix (routing audit): the GENERAL fallback used to be a 4-field
stub (version/uptime/cpu%/mem%) with no visibility into active issues at
all. Multi-topic and "active fault" style questions that don't cleanly
match one specific Intent (DRIVER_HEALTH / HARDWARE_HEALTH /
FAILURE_PREDICTION / PERFORMANCE / ...) land here, so GENERAL now also
carries the full, real active-issue set (driver anomalies, hardware
faults, predicted failures - each with severity/evidence/occurrence
data straight from `issue_alert_store`) instead of leaving the LLM to
invent generic advice ("check /var/log/syslog") when it has no better
information. PERFORMANCE now also includes disk, since "CPU, memory,
and disk usage" is a single, common, multi-topic question.
"""

from app.logger import get_logger
from app.services import (
    docker_monitor,
    driver_monitor,
    file_search,
    hardware_monitor,
    issue_alert_store,
    system_monitor,
)
from app.services.intent_classifier import Intent

logger = get_logger(__name__)

# Maps a detector-backed Intent to the exact `issue_type` its candidates are
# tagged with (see driver_monitor._make_candidate / hardware_monitor.
# _make_candidate / failure_predictor._make_candidate - all fixed strings,
# never modified here).
_DETECTOR_ISSUE_TYPE = {
    Intent.DRIVER_HEALTH: "driver_anomaly",
    Intent.HARDWARE_HEALTH: "hardware_fault",
    Intent.FAILURE_PREDICTION: "predicted_failure",
}

# Same mapping, but for the GENERAL/multi-topic snapshot, which needs all
# three categories at once rather than picking a single one.
_ALL_DETECTOR_ISSUE_TYPES: dict[str, str] = {
    "driver_anomalies": "driver_anomaly",
    "hardware_faults": "hardware_fault",
    "predicted_failures": "predicted_failure",
}


def _active_issues_of_type(issue_type: str) -> list[dict]:
    """Ground-truth, currently-active issues of one category.

    Reads from `issue_alert_store`, the same reconciled state the
    dashboard itself displays - it is populated by the periodic
    `fix_engine.detect_all_issues()` scan. Deliberately does NOT call
    driver_monitor / hardware_monitor / failure_predictor directly here:
    those modules are off-limits to modify, and for failure_predictor in
    particular, calling it outside its normal scan cadence would record an
    extra, unwanted sample into its rolling trend history. Reading the
    already-reconciled store instead is a pure, side-effect-free read.
    """
    return [
        issue
        for issue in issue_alert_store.get_active_issues()
        if issue.get("issue_type") == issue_type
    ]


def _all_active_issues() -> dict:
    """Every currently-active issue, grouped by category, straight from the
    store - no reinterpretation, no dropped fields (severity, evidence,
    occurrence_count, classification_note, etc. all pass through as-is).

    Used by the GENERAL/multi-topic context so "active faults", "system
    health", and similar broad questions are grounded in the real,
    reconciled issue set instead of falling back to generic advice.
    """
    grouped = {
        key: _active_issues_of_type(issue_type)
        for key, issue_type in _ALL_DETECTOR_ISSUE_TYPES.items()
    }
    total = sum(len(v) for v in grouped.values())
    grouped["total_active_issue_count"] = total
    grouped["ground_truth_note"] = (
        "The lists above (driver_anomalies / hardware_faults / "
        "predicted_failures) are the COMPLETE and ONLY currently active "
        "issues from the real detectors, each with its own severity and "
        "evidence. If a category's list is empty, there is currently no "
        "evidence of that kind of issue - say so plainly instead of "
        "guessing, inventing an issue, or telling the user to manually "
        "inspect logs for something the system has already scanned for."
    )
    return grouped


def _safe(label: str, fn, *args, **kwargs):
    """Run a collector defensively; never let one failure break the context."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Context collector '%s' failed: %s", label, exc)
        return {"error": f"Could not collect {label}: {exc}"}


def _slim_process(p: dict) -> dict:
    """Keep only the fields the LLM actually needs to reason about a process.

    The full process dict (cmdline, created_at, num_threads, ...) is still
    what /api/system/processes returns to the frontend unchanged - this
    trimming only affects what gets serialized into the LLM prompt, to keep
    prompt tokens down for the CPU demo.
    """
    return {
        "pid": p.get("pid"),
        "name": p.get("name"),
        "cpu_percent": p.get("cpu_percent"),
        "memory_percent": p.get("memory_percent"),
    }


def build_context(intent: Intent, message: str) -> dict:
    """Gather the system data most relevant to the given intent."""
    if intent in _DETECTOR_ISSUE_TYPE:
        issue_type = _DETECTOR_ISSUE_TYPE[intent]
        active = _active_issues_of_type(issue_type)
        key = {
            Intent.DRIVER_HEALTH: "driver_anomalies",
            Intent.HARDWARE_HEALTH: "hardware_faults",
            Intent.FAILURE_PREDICTION: "predicted_failures",
        }[intent]
        context = {
            key: active,
            "ground_truth_note": (
                f"The list above under '{key}' is the COMPLETE and ONLY set of "
                f"currently active {issue_type} findings from the real detector. "
                "Do not mention any device, vendor, driver, service, or issue "
                "that is not literally present in this list. If the list is "
                "empty, say plainly that there is currently no evidence of this "
                "kind of issue instead of guessing or describing a generic/"
                "hypothetical one."
            ),
        }

        # DRIVER_HEALTH additionally gets a plain inventory of every
        # currently loaded kernel module (driver_monitor.
        # get_loaded_kernel_modules()), alongside - not instead of - the
        # driver_anomalies list above. Root-cause fix: 'driver_anomalies'
        # only ever contains ACTIVE FAULTS; a plain "which drivers are
        # installed?" question has no fault to find, so without this the
        # LLM only ever saw an empty list and (per the ground-truth note)
        # correctly but unhelpfully said "no evidence of an issue" instead
        # of answering the inventory question actually being asked.
        if intent == Intent.DRIVER_HEALTH:
            raw_modules = _safe(
                "loaded_kernel_modules", driver_monitor.get_loaded_kernel_modules, limit=30
            )
            # Trim to what the LLM actually needs for grounding (name +
            # size) - `used_by` is real data too but rarely relevant to
            # "which drivers are loaded"/"any driver problem" questions
            # and, across 30-50 modules, was a meaningful chunk of the
            # prompt's token budget for little grounding value. This only
            # changes what's sent to the LLM prompt, not what driver_
            # monitor itself collects or returns elsewhere.
            if isinstance(raw_modules, dict) and isinstance(raw_modules.get("modules"), list):
                trimmed_modules = dict(raw_modules)
                trimmed_modules["modules"] = [
                    {"name": m.get("name"), "size_bytes": m.get("size_bytes")}
                    for m in raw_modules["modules"]
                ]
                context["loaded_kernel_modules"] = trimmed_modules
            else:
                context["loaded_kernel_modules"] = raw_modules
            context["loaded_kernel_modules_note"] = (
                "'loaded_kernel_modules' above is the REAL, current output of `lsmod` - "
                "the actual list of drivers/kernel modules loaded on this host right now. "
                "'total_modules' is the true total count; 'modules' is a sample of real "
                "module names from it (not the full list if total_modules is larger). When "
                "asked which drivers are loaded, your explanation MUST state the real "
                "total_modules count and name several real modules from the 'modules' "
                "sample - do NOT just tell the user to run `lsmod` themselves, that data is "
                "already right here. It is NOT a fault list and is unrelated to whether "
                "anything is wrong. Only 'driver_anomalies' above represents a confirmed "
                "problem; an empty 'driver_anomalies' list means no active driver problem, "
                "NOT that no drivers are installed - never conflate the two."
            )

        # HARDWARE_HEALTH additionally gets the real current sensor
        # readings (hardware_monitor.get_current_sensor_readings()),
        # alongside - not instead of - the hardware_faults list above.
        # hardware_faults only reports a sensor that is AT/ABOVE its
        # warning threshold (see hardware_monitor._detect_overheating);
        # it says nothing about a normal reading. Without this, a direct
        # "what is my current CPU/GPU temperature?" question had no real
        # numbers to answer from on a healthy system. Purely additive:
        # hardware_faults, the ground-truth note above, and every other
        # field are unchanged.
        if intent == Intent.HARDWARE_HEALTH:
            context["current_sensor_readings"] = _safe(
                "current_sensor_readings", hardware_monitor.get_current_sensor_readings
            )
            # Cross-category counts (not the full lists - those belong to
            # DRIVER_HEALTH/FAILURE_PREDICTION's own dedicated context) so
            # a hardware-fault question can correctly say "no hardware
            # fault, but 1 driver anomaly is active" instead of an empty
            # hardware_faults list being misread as "the system is
            # healthy" when a real, active issue exists in a different
            # category. Root-cause fix for exactly that reported behavior.
            other_issues = {
                "driver_anomaly_count": len(_active_issues_of_type("driver_anomaly")),
                "predicted_failure_count": len(_active_issues_of_type("predicted_failure")),
            }
            context["other_active_issue_categories"] = other_issues
            context["other_active_issue_categories_note"] = (
                "'other_active_issue_categories' shows counts of currently active issues in "
                "categories OTHER than hardware_faults (driver anomalies, predicted failures) "
                "- it is real, ground-truth data, just not the detail for this intent. An "
                "empty 'hardware_faults' list means no confirmed HARDWARE fault specifically - "
                "it does NOT mean the system overall is healthy. If any count here is above "
                "0, your explanation must say so explicitly and distinguish it from a hardware "
                "fault (e.g. 'no hardware fault detected, but there is 1 active driver "
                "anomaly - that's a separate category'), rather than implying full system "
                "health from hardware_faults alone."
            )
            context["sensor_reading_note"] = (
                "'current_sensor_readings' above is a plain, unconditional "
                "snapshot of every temperature sensor available on this host "
                "right now - it is NOT a fault list, and a value appearing "
                "there does not mean anything is wrong. Only 'hardware_faults' "
                "above represents a confirmed hardware fault (a reading "
                "at/above its warning threshold). When asked for a current "
                "temperature, report the real value(s) from "
                "'current_sensor_readings' and separately state whether "
                "'hardware_faults' contains an active overheating fault - "
                "never invent a temperature value that is not present in "
                "'current_sensor_readings'."
            )

        return context

    if intent == Intent.PERFORMANCE:
        top_cpu = _safe(
            "top_processes", system_monitor.get_processes, limit=5, sort_by="cpu_percent"
        ).get("processes", [])
        top_mem = _safe(
            "top_processes", system_monitor.get_processes, limit=5, sort_by="memory_percent"
        ).get("processes", [])
        return {
            "cpu": _safe("cpu", system_monitor.get_cpu_info),
            "memory": _safe("memory", system_monitor.get_memory_info),
            # Included alongside CPU/memory so "CPU, memory, and disk usage"
            # style questions are fully answerable from a single PERFORMANCE
            # context instead of disk silently being left out.
            "disk": _safe("disk", system_monitor.get_disk_info),
            "top_processes_by_cpu": [_slim_process(p) for p in top_cpu],
            "top_processes_by_memory": [_slim_process(p) for p in top_mem],
            "uptime_seconds": _safe("uptime", system_monitor.get_uptime_seconds),
        }

    if intent == Intent.SERVICE_MANAGEMENT:
        services_data = _safe("services", system_monitor.get_services, limit=200)
        target_services = _extract_mentioned_services(message, services_data.get("services", []))
        return {
            "matched_services": target_services,
            "total_services_found": services_data.get("total_services", 0),
            "collector_errors": services_data.get("errors", []),
        }

    if intent == Intent.DOCKER:
        return {"docker": _safe("docker", docker_monitor.get_docker_containers)}

    if intent == Intent.FILE_SEARCH:
        return {
            "large_files": _safe("large_files", file_search.find_large_files, min_size_mb=100, limit=10),
            "disk_usage": _safe("disk", system_monitor.get_disk_info),
        }

    if intent == Intent.LOG_ANALYSIS:
        logs_data = _safe("logs", system_monitor.get_recent_logs, lines=20, priority="warning")
        return {
            "recent_warning_and_error_logs": logs_data.get("entries", []),
            "collector_errors": logs_data.get("errors", []),
        }

    if intent == Intent.NETWORK:
        return {"network": _safe("network", system_monitor.get_network_info)}

    if intent == Intent.USERS_SESSIONS:
        return {"logged_in_users": _safe("users", system_monitor.get_logged_in_users)}

    # GENERAL fallback: catches multi-topic questions ("what's the current
    # CPU, memory, and disk usage?"), broad/ambiguous health questions
    # ("are there any active faults right now?", "give me overall system
    # health"), hardware-identity questions ("which CPU is being used?" -
    # answered from cpu.model_name/vendor below), and anything else
    # ("show me running processes", "what happened to my system recently?")
    # that doesn't cleanly match one narrower Intent above. This is now a
    # real, fairly complete snapshot - not just a 4-field stub -
    # specifically so the LLM never has to say information is "unavailable"
    # or fall back to generic advice for data that is actually already
    # being collected.
    version = _safe("version", system_monitor.get_system_version)
    cpu_info = _safe("cpu", system_monitor.get_cpu_info)
    memory_info = _safe("memory", system_monitor.get_memory_info)
    # Root-cause fix: a plain "show me running processes" (or similar
    # phrasing with no cpu/memory/disk/service keyword) didn't match any
    # narrower Intent and landed here, but this snapshot never included a
    # process list at all - only cpu/memory/disk *aggregate* percentages.
    # A small top-5-by-CPU snapshot (same slimming as the PERFORMANCE
    # branch above) closes that gap without duplicating the full process
    # list machinery.
    top_processes = _safe(
        "top_processes", system_monitor.get_processes, limit=5, sort_by="cpu_percent"
    ).get("processes", [])
    return {
        "hostname": version.get("hostname") if isinstance(version, dict) else None,
        "version": version,
        "uptime_seconds": _safe("uptime", system_monitor.get_uptime_seconds),
        "cpu": cpu_info,
        "cpu_usage_percent": cpu_info.get("usage_percent") if isinstance(cpu_info, dict) else None,
        "memory": memory_info,
        "memory_usage_percent": memory_info.get("usage_percent") if isinstance(memory_info, dict) else None,
        "disk": _safe("disk", system_monitor.get_disk_info),
        "top_processes_by_cpu": [_slim_process(p) for p in top_processes],
        "active_issues": _all_active_issues(),
    }


def _extract_mentioned_services(message: str, services: list[dict]) -> list[dict]:
    """Best-effort match of service names mentioned in the user's message.

    Falls back to returning nothing (rather than the whole service list) if
    no name matches, so the LLM is nudged to ask a clarifying question
    instead of guessing which service the user means.
    """
    text = message.lower()
    matched = []
    for service in services:
        name = service.get("name", "")
        base_name = name.replace(".service", "").lower()
        if base_name and base_name in text:
            matched.append(service)
    return matched