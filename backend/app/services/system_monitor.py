"""
Ubuntu System Monitor.

Collects system telemetry using `psutil` for cross-cutting metrics (CPU,
memory, disk, network, processes, uptime, logged-in users) and native
Ubuntu commands (`lsb_release`, `uname`, `systemctl`, `journalctl`) for
OS-specific information (distro version, services, journal logs).

Design notes:
- Every public function returns plain dicts/lists (JSON-serializable) so
  the route layer can validate them against the Pydantic schemas.
- Every collector is defensive: a failure in one metric (e.g. journalctl
  not being available in a sandbox) must not break the rest of the
  response. Errors are collected and returned alongside the data instead
  of raising, so callers - including a future AI layer - always get a
  usable response.
"""

import json
import platform
import socket
from datetime import datetime, timezone

import psutil

from app.logger import get_logger
from app.utils import run_command

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def get_cpu_identity() -> dict:
    """Collect the CPU's model name and vendor, deterministically.

    Root-cause fix: previously nothing in this module ever read "which CPU
    is actually in this machine" - `get_cpu_info()` only had usage/core-
    count/frequency figures. A question like "which CPU is being used?"
    therefore had no real hardware-identity data anywhere in the context
    the LLM sees, so it could only answer generically or refuse. This
    reads /proc/cpuinfo directly (present on every Linux host, no shell,
    no external binary) with an `lscpu` fallback for the rare case
    /proc/cpuinfo is unreadable/absent (e.g. some restricted containers).
    Never raises - an unreadable source just yields None fields, exactly
    like every other collector in this module.
    """
    model_name: str | None = None
    vendor: str | None = None

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key == "model name" and not model_name:
                    model_name = value
                elif key == "vendor_id" and not vendor:
                    vendor = value
                if model_name and vendor:
                    break
    except OSError as exc:
        logger.warning("Could not read /proc/cpuinfo: %s", exc)

    if not model_name:
        # Fallback for hosts/containers without a readable /proc/cpuinfo
        # "model name" line (e.g. some ARM/virtualized environments).
        ok, output = run_command(["lscpu"])
        if ok:
            for line in output.splitlines():
                if line.lower().startswith("model name:"):
                    model_name = line.split(":", 1)[1].strip()
                elif line.lower().startswith("vendor id:") and not vendor:
                    vendor = line.split(":", 1)[1].strip()

    return {"model_name": model_name, "vendor": vendor}


def get_cpu_info() -> dict:
    """Collect CPU usage, core counts, frequency, load average, and identity."""
    identity = get_cpu_identity()
    data = {
        "model_name": identity["model_name"],
        "vendor": identity["vendor"],
        "logical_cores": psutil.cpu_count(logical=True) or 0,
        "physical_cores": psutil.cpu_count(logical=False),
        "usage_percent": psutil.cpu_percent(interval=0.3),
        "per_core_percent": psutil.cpu_percent(interval=0.1, percpu=True),
        "frequency_mhz": None,
        "load_avg_1m": None,
        "load_avg_5m": None,
        "load_avg_15m": None,
    }

    try:
        freq = psutil.cpu_freq()
        if freq:
            data["frequency_mhz"] = round(freq.current, 2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read CPU frequency: %s", exc)

    try:
        load1, load5, load15 = psutil.getloadavg()
        data["load_avg_1m"] = round(load1, 2)
        data["load_avg_5m"] = round(load5, 2)
        data["load_avg_15m"] = round(load15, 2)
    except (OSError, AttributeError) as exc:
        logger.warning("Could not read load average: %s", exc)

    return data


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def get_memory_info() -> dict:
    """Collect RAM and swap usage."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "total_bytes": vm.total,
        "available_bytes": vm.available,
        "used_bytes": vm.used,
        "free_bytes": vm.free,
        "usage_percent": vm.percent,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
        "swap_free_bytes": swap.free,
        "swap_usage_percent": swap.percent,
    }


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def get_disk_info() -> dict:
    """Collect usage for every mounted disk partition."""
    partitions = []
    for part in psutil.disk_partitions(all=False):
        entry = {
            "device": part.device,
            "mountpoint": part.mountpoint,
            "filesystem_type": part.fstype,
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "usage_percent": 0.0,
            "error": None,
        }
        try:
            usage = psutil.disk_usage(part.mountpoint)
            entry.update(
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
                usage_percent=usage.percent,
            )
        except (PermissionError, OSError) as exc:
            entry["error"] = f"Could not read usage: {exc}"
            logger.warning("Disk usage error for %s: %s", part.mountpoint, exc)
        partitions.append(entry)

    return {"partitions": partitions}


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def get_network_info() -> dict:
    """Collect per-interface network I/O counters and addresses."""
    interfaces = []
    io_counters = psutil.net_io_counters(pernic=True)
    stats = psutil.net_if_stats()
    addrs = psutil.net_if_addrs()

    for name, counters in io_counters.items():
        stat = stats.get(name)
        interface_addrs = [a.address for a in addrs.get(name, []) if a.address]
        interfaces.append(
            {
                "name": name,
                "bytes_sent": counters.bytes_sent,
                "bytes_received": counters.bytes_recv,
                "packets_sent": counters.packets_sent,
                "packets_received": counters.packets_recv,
                "errors_in": counters.errin,
                "errors_out": counters.errout,
                "drops_in": counters.dropin,
                "drops_out": counters.dropout,
                "is_up": stat.isup if stat else None,
                "speed_mbps": stat.speed if stat else None,
                "addresses": interface_addrs,
            }
        )

    active_connections = None
    try:
        active_connections = len(psutil.net_connections(kind="inet"))
    except (PermissionError, psutil.AccessDenied) as exc:
        logger.warning("Could not enumerate network connections: %s", exc)

    return {
        "hostname": socket.gethostname(),
        "interfaces": interfaces,
        "active_connections": active_connections,
    }


# ---------------------------------------------------------------------------
# Logged-in users
# ---------------------------------------------------------------------------

def get_logged_in_users() -> list[dict]:
    """Collect currently logged-in users via psutil."""
    users = []
    try:
        for user in psutil.users():
            users.append(
                {
                    "username": user.name,
                    "terminal": user.terminal,
                    "host": user.host or None,
                    "login_time": datetime.fromtimestamp(
                        user.started, tz=timezone.utc
                    ).isoformat(),
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read logged-in users: %s", exc)
    return users


# ---------------------------------------------------------------------------
# OS / kernel version
# ---------------------------------------------------------------------------

def get_system_version() -> dict:
    """Collect Ubuntu version, codename, kernel version, and architecture."""
    ubuntu_version = None
    ubuntu_codename = None

    ok, output = run_command(["lsb_release", "-a"])
    if ok:
        for line in output.splitlines():
            if line.startswith("Description:"):
                ubuntu_version = line.split(":", 1)[1].strip()
            elif line.startswith("Codename:"):
                ubuntu_codename = line.split(":", 1)[1].strip()
    else:
        # Fallback: /etc/os-release is present on virtually all modern distros
        try:
            with open("/etc/os-release", "r", encoding="utf-8") as f:
                os_release = dict(
                    line.strip().split("=", 1)
                    for line in f
                    if "=" in line and not line.startswith("#")
                )
            ubuntu_version = os_release.get("PRETTY_NAME", "").strip('"') or None
            ubuntu_codename = os_release.get("VERSION_CODENAME", "").strip('"') or None
        except OSError as exc:
            logger.warning("Could not read /etc/os-release: %s", exc)

    return {
        "ubuntu_version": ubuntu_version,
        "ubuntu_codename": ubuntu_codename,
        "kernel_version": platform.release(),
        "architecture": platform.machine(),
        "hostname": socket.gethostname(),
    }


# ---------------------------------------------------------------------------
# Uptime
# ---------------------------------------------------------------------------

def get_uptime_seconds() -> float:
    return datetime.now().timestamp() - psutil.boot_time()


def get_boot_time_iso() -> str:
    return datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Aggregate: /system-info
# ---------------------------------------------------------------------------

def get_system_info() -> dict:
    """Aggregate all system-level metrics into a single structured payload."""
    errors: list[str] = []

    def safe(label, fn, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to collect {label}: {exc}"
            logger.error(message)
            errors.append(message)
            return default

    return {
        "timestamp": _now_iso(),
        "version": safe("version", get_system_version, {}),
        "uptime_seconds": safe("uptime", get_uptime_seconds, 0.0),
        "boot_time": safe("boot_time", get_boot_time_iso, ""),
        "cpu": safe("cpu", get_cpu_info, {}),
        "memory": safe("memory", get_memory_info, {}),
        "disk": safe("disk", get_disk_info, {"partitions": []}),
        "network": safe("network", get_network_info, {"hostname": "", "interfaces": []}),
        "logged_in_users": safe("logged_in_users", get_logged_in_users, []),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# /processes
# ---------------------------------------------------------------------------

def get_processes(limit: int = 50, sort_by: str = "cpu_percent") -> dict:
    """Collect running processes, sorted by CPU or memory usage.

    Args:
        limit: maximum number of processes to return.
        sort_by: "cpu_percent" or "memory_percent".
    """
    errors: list[str] = []
    processes = []

    attrs = [
        "pid",
        "name",
        "username",
        "status",
        "cpu_percent",
        "memory_percent",
        "memory_info",
        "num_threads",
        "create_time",
        "cmdline",
    ]

    for proc in psutil.process_iter(attrs):
        try:
            info = proc.info
            processes.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name") or "",
                    "username": info.get("username"),
                    "status": info.get("status"),
                    "cpu_percent": info.get("cpu_percent"),
                    "memory_percent": (
                        round(info["memory_percent"], 2)
                        if info.get("memory_percent") is not None
                        else None
                    ),
                    "memory_rss_bytes": (
                        info["memory_info"].rss if info.get("memory_info") else None
                    ),
                    "num_threads": info.get("num_threads"),
                    "created_at": (
                        datetime.fromtimestamp(
                            info["create_time"], tz=timezone.utc
                        ).isoformat()
                        if info.get("create_time")
                        else None
                    ),
                    "cmdline": " ".join(info.get("cmdline") or []) or None,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Processes can exit or be inaccessible mid-scan - skip them silently
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Error reading process info: {exc}")

    reverse_sort_key = sort_by if sort_by in ("cpu_percent", "memory_percent") else "cpu_percent"
    processes.sort(key=lambda p: p.get(reverse_sort_key) or 0, reverse=True)

    total = len(processes)
    return {
        "timestamp": _now_iso(),
        "total_processes": total,
        "processes": processes[: max(limit, 0)],
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# /services
# ---------------------------------------------------------------------------

def get_services(limit: int = 100, only_running: bool = False) -> dict:
    """Collect systemd service unit statuses via `systemctl`."""
    errors: list[str] = []
    services: list[dict] = []

    command = [
        "systemctl",
        "list-units",
        "--type=service",
        "--all",
        "--no-pager",
        "--no-legend",
        "--plain",
    ]
    ok, output = run_command(command, timeout=10)

    if not ok:
        errors.append(f"Could not query systemd services: {output}")
        return {
            "timestamp": _now_iso(),
            "total_services": 0,
            "services": [],
            "errors": errors,
        }

    for line in output.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        unit, load_state, active_state, sub_state = parts[:4]
        description = parts[4] if len(parts) > 4 else None
        if only_running and active_state != "active":
            continue
        services.append(
            {
                "name": unit,
                "load_state": load_state,
                "active_state": active_state,
                "sub_state": sub_state,
                "description": description,
            }
        )

    return {
        "timestamp": _now_iso(),
        "total_services": len(services),
        "services": services[: max(limit, 0)],
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# /logs
# ---------------------------------------------------------------------------

def get_recent_logs(lines: int = 100, priority: str | None = None, unit: str | None = None) -> dict:
    """Collect recent journal entries via `journalctl -o json`.

    Args:
        lines: number of most recent entries to return.
        priority: optional syslog priority filter (e.g. "err", "warning").
        unit: optional systemd unit name filter (e.g. "ssh.service").
    """
    errors: list[str] = []
    entries: list[dict] = []

    command = ["journalctl", "-n", str(lines), "-o", "json", "--no-pager"]
    if priority:
        command += ["-p", priority]
    if unit:
        command += ["-u", unit]

    ok, output = run_command(command, timeout=10)

    if not ok:
        errors.append(f"Could not read journal logs: {output}")
        return {
            "timestamp": _now_iso(),
            "total_entries": 0,
            "entries": [],
            "errors": errors,
        }

    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue

        timestamp_iso = None
        realtime_us = raw.get("__REALTIME_TIMESTAMP")
        if realtime_us:
            try:
                timestamp_iso = datetime.fromtimestamp(
                    int(realtime_us) / 1_000_000, tz=timezone.utc
                ).isoformat()
            except (ValueError, OverflowError):
                timestamp_iso = None

        entries.append(
            {
                "timestamp": timestamp_iso,
                "unit": raw.get("_SYSTEMD_UNIT") or raw.get("SYSLOG_IDENTIFIER"),
                "priority": str(raw.get("PRIORITY")) if raw.get("PRIORITY") is not None else None,
                "message": str(raw.get("MESSAGE", "")),
            }
        )

    return {
        "timestamp": _now_iso(),
        "total_entries": len(entries),
        "entries": entries,
        "errors": errors,
    }
