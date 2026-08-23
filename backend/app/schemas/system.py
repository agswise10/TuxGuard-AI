"""
Pydantic response models for the System Monitor APIs.

These schemas exist so every endpoint returns a predictable, strongly-typed
JSON shape. That predictability is what will let an AI layer (added later)
reliably parse and reason about this data - so keep new fields additive and
avoid renaming/removing existing ones without good reason.
"""

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /system-info
# ---------------------------------------------------------------------------

class CPUInfo(BaseModel):
    model_name: Optional[str] = None
    vendor: Optional[str] = None
    logical_cores: int
    physical_cores: Optional[int] = None
    usage_percent: float = Field(..., description="Overall CPU utilization percentage")
    per_core_percent: list[float] = Field(default_factory=list)
    frequency_mhz: Optional[float] = None
    load_avg_1m: Optional[float] = None
    load_avg_5m: Optional[float] = None
    load_avg_15m: Optional[float] = None


class MemoryInfo(BaseModel):
    total_bytes: int
    available_bytes: int
    used_bytes: int
    free_bytes: int
    usage_percent: float
    swap_total_bytes: int
    swap_used_bytes: int
    swap_free_bytes: int
    swap_usage_percent: float


class DiskPartitionInfo(BaseModel):
    device: str
    mountpoint: str
    filesystem_type: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    usage_percent: float
    error: Optional[str] = None


class DiskInfo(BaseModel):
    partitions: list[DiskPartitionInfo] = Field(default_factory=list)


class NetworkInterfaceInfo(BaseModel):
    name: str
    bytes_sent: int
    bytes_received: int
    packets_sent: int
    packets_received: int
    errors_in: int
    errors_out: int
    drops_in: int
    drops_out: int
    is_up: Optional[bool] = None
    speed_mbps: Optional[int] = None
    addresses: list[str] = Field(default_factory=list)


class NetworkInfo(BaseModel):
    hostname: str
    interfaces: list[NetworkInterfaceInfo] = Field(default_factory=list)
    active_connections: Optional[int] = None


class LoggedInUser(BaseModel):
    username: str
    terminal: Optional[str] = None
    host: Optional[str] = None
    login_time: Optional[str] = None


class SystemVersionInfo(BaseModel):
    ubuntu_version: Optional[str] = None
    ubuntu_codename: Optional[str] = None
    kernel_version: Optional[str] = None
    architecture: Optional[str] = None
    hostname: Optional[str] = None


class SystemInfoResponse(BaseModel):
    timestamp: str
    version: SystemVersionInfo
    uptime_seconds: float
    boot_time: str
    cpu: CPUInfo
    memory: MemoryInfo
    disk: DiskInfo
    network: NetworkInfo
    logged_in_users: list[LoggedInUser] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list, description="Non-fatal collection errors")


# ---------------------------------------------------------------------------
# /processes
# ---------------------------------------------------------------------------

class ProcessInfo(BaseModel):
    pid: int
    name: str
    username: Optional[str] = None
    status: Optional[str] = None
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_rss_bytes: Optional[int] = None
    num_threads: Optional[int] = None
    created_at: Optional[str] = None
    cmdline: Optional[str] = None


class ProcessesResponse(BaseModel):
    timestamp: str
    total_processes: int
    processes: list[ProcessInfo] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /services
# ---------------------------------------------------------------------------

class ServiceInfo(BaseModel):
    name: str
    load_state: Optional[str] = None
    active_state: Optional[str] = None
    sub_state: Optional[str] = None
    description: Optional[str] = None


class ServicesResponse(BaseModel):
    timestamp: str
    total_services: int
    services: list[ServiceInfo] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /logs
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    timestamp: Optional[str] = None
    unit: Optional[str] = None
    priority: Optional[str] = None
    message: str


class LogsResponse(BaseModel):
    timestamp: str
    total_entries: int
    entries: list[LogEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
