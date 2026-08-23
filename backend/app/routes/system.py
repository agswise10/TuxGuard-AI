"""
System Monitor routes.

Exposes Ubuntu system telemetry as clean, structured JSON so it can be
consumed both by the frontend and, later, by an AI reasoning layer.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.logger import get_logger
from app.schemas.system import (
    LogsResponse,
    ProcessesResponse,
    ServicesResponse,
    SystemInfoResponse,
)
from app.services import system_monitor

router = APIRouter(prefix="/api", tags=["System Monitor"])
logger = get_logger(__name__)


@router.get("/system-info", response_model=SystemInfoResponse)
def system_info():
    """Return CPU, memory, disk, network, uptime, version, and user info."""
    try:
        data = system_monitor.get_system_info()
        return SystemInfoResponse(**data)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to build system-info response: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to collect system information") from exc


@router.get("/processes", response_model=ProcessesResponse)
def processes(
    limit: int = Query(50, ge=1, le=500, description="Max number of processes to return"),
    sort_by: str = Query(
        "cpu_percent",
        pattern="^(cpu_percent|memory_percent)$",
        description="Field to sort processes by, descending",
    ),
):
    """Return running processes sorted by CPU or memory usage."""
    try:
        data = system_monitor.get_processes(limit=limit, sort_by=sort_by)
        return ProcessesResponse(**data)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to build processes response: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to collect process information") from exc


@router.get("/services", response_model=ServicesResponse)
def services(
    limit: int = Query(100, ge=1, le=1000, description="Max number of services to return"),
    only_running: bool = Query(False, description="Only include services with active_state=active"),
):
    """Return systemd service statuses."""
    try:
        data = system_monitor.get_services(limit=limit, only_running=only_running)
        return ServicesResponse(**data)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to build services response: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to collect service information") from exc


@router.get("/logs", response_model=LogsResponse)
def logs(
    lines: int = Query(100, ge=1, le=2000, description="Number of recent journal entries to return"),
    priority: Optional[str] = Query(
        None, description="Filter by syslog priority, e.g. 'err', 'warning', 'info'"
    ),
    unit: Optional[str] = Query(None, description="Filter by systemd unit name, e.g. 'ssh.service'"),
):
    """Return recent systemd journal log entries."""
    try:
        data = system_monitor.get_recent_logs(lines=lines, priority=priority, unit=unit)
        return LogsResponse(**data)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to build logs response: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to collect journal logs") from exc
