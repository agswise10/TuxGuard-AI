"""
Tool Executor - the "hands" of the Linux Operations Copilot.

This is the ONLY module that turns an ops question into a real command run
on the host. It is deliberately narrow and paranoid:

  * Every runnable command is a fixed, hard-coded argv list defined in
    ``_TOOLS`` below. There is no code path that builds a command from user
    input, string-formats a shell string, or accepts a caller-supplied
    command. Callers can only ever ask for one of the named tools
    (``ToolName``) - there is nothing to inject.
  * Commands are executed with ``subprocess.run(argv, shell=False, ...)``.
    No shell is ever invoked, so shell metacharacters (`;`, `|`, `&&`, `$()`,
    backticks, ...) are inert even if they somehow ended up in output we
    later interpolate anywhere.
  * Every tool is read-only: process/service/log/disk/network *listing* or
    *status* commands only. Nothing here starts, stops, restarts, deletes,
    or writes anything on the host.
  * Every execution is wrapped in exception handling and a timeout. A
    missing binary, a permissions error, a timeout, or any other failure is
    captured into a structured ``ToolResult`` and returned - never raised
    up to the caller and never left to crash the request.
  * Every result is JSON-serializable (`ToolResult.to_dict()`), including a
    parsed/structured view of the output so the LLM (and the frontend) get
    real fields to reason about, not just a wall of raw text.

This module knows nothing about intents, prompts, or the LLM - it only
knows how to safely run one of a fixed set of commands and structure what
comes back. See ``ops_intent_classifier.py`` for "which tool does this
question need" and ``ops_prompts.py`` / ``ops_assistant.py`` for how the
result is turned into an answer.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from app.config import get_settings
from app.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ToolExecutionError(Exception):
    """Raised only for programmer errors (e.g. unknown tool name).

    Actual host/command failures (timeout, missing binary, non-zero exit,
    ...) are never raised - they come back as a ``ToolResult`` with
    ``success=False`` so callers always get a JSON-able result.
    """


class ToolName(str, Enum):
    """The fixed, whitelisted set of tools this copilot may execute.

    This enum IS the whitelist. Nothing outside `_TOOLS` (keyed by these
    members) can ever be run - there is no way to construct a tool call
    from an arbitrary string.
    """

    CPU_TOP = "cpu_top"                    # ps -eo pid,comm,%cpu --sort=-%cpu | head -6
    MEMORY_TOP = "memory_top"              # ps -eo pid,comm,%mem --sort=-%mem | head -6
    DISK_USAGE = "disk_usage"              # df -h
    SERVICES_RUNNING = "services_running"  # systemctl list-units --type=service --state=running
    SERVICES_FAILED = "services_failed"    # systemctl --failed
    LOGS_RECENT = "logs_recent"            # journalctl -n 50 --no-pager
    LOGS_ERROR = "logs_error"              # journalctl -p err -n 50 --no-pager
    NETWORK_PORTS = "network_ports"        # ss -tuln


@dataclass
class ToolResult:
    """Structured, JSON-safe result of running one whitelisted tool."""

    tool: str
    display_command: str
    success: bool
    exit_code: int | None
    duration_seconds: float
    parsed: Any                     # structured data - list[dict] / dict, LLM + UI friendly
    raw_output: str                 # raw stdout (truncated), kept for transparency/debugging
    row_count: int | None = None
    truncated: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "command": self.display_command,
            "success": self.success,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "parsed": self.parsed,
            "raw_output": self.raw_output,
            "error": self.error,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# Output parsers - turn raw stdout into structured, LLM/UI-friendly data.
# Each parser is defensive: malformed/unexpected lines are skipped rather
# than raising, since a parsing hiccup should never take down the tool call.
# --------------------------------------------------------------------------

def _parse_ps_table(stdout: str, value_key: str, limit: int) -> list[dict]:
    """Parse `ps -eo pid,comm,%cpu|%mem --sort=-...` output into rows.

    Equivalent to piping through `head -(limit+1)` (header + limit rows),
    done in Python so the tool never needs a shell pipe.
    """
    lines = [l for l in stdout.splitlines() if l.strip()]
    if not lines:
        return []
    rows: list[dict] = []
    # lines[0] is the header ("PID COMMAND %CPU" / "PID COMMAND %MEM")
    for line in lines[1 : limit + 1]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_str, command, value_str = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        try:
            value = float(value_str)
        except ValueError:
            value = 0.0
        rows.append({"pid": pid, "process": command, value_key: value})
    return rows


def _parse_df(stdout: str) -> list[dict]:
    """Parse `df -h` output into per-filesystem rows."""
    lines = [l for l in stdout.splitlines() if l.strip()]
    if not lines:
        return []
    rows: list[dict] = []
    for line in lines[1:]:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        filesystem, size, used, avail, use_percent, mounted_on = parts
        try:
            use_percent_value = float(use_percent.rstrip("%"))
        except ValueError:
            use_percent_value = None
        rows.append(
            {
                "filesystem": filesystem,
                "size": size,
                "used": used,
                "available": avail,
                "use_percent": use_percent_value,
                "mounted_on": mounted_on,
            }
        )
    rows.sort(key=lambda r: (r["use_percent"] is None, -(r["use_percent"] or 0)))
    return rows


_UNIT_LINE_RE = re.compile(
    r"^\s*(?P<unit>\S+\.service)\s+(?P<load>\S+)\s+(?P<active>\S+)\s+(?P<sub>\S+)\s*(?P<description>.*)$"
)


def _parse_systemctl_units(stdout: str) -> list[dict]:
    """Parse `systemctl list-units --type=service ...` / `--failed` tables.

    Skips the header row and the summary/legend lines systemctl prints
    after the table (blank line, "N loaded units listed.", ...).
    """
    rows: list[dict] = []
    for line in stdout.splitlines():
        match = _UNIT_LINE_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "unit": match.group("unit"),
                "load": match.group("load"),
                "active": match.group("active"),
                "sub": match.group("sub"),
                "description": match.group("description").strip(),
            }
        )
    return rows


def _parse_journal_lines(stdout: str) -> list[str]:
    """journalctl output is already one log entry per line - just clean it up."""
    return [line.strip() for line in stdout.splitlines() if line.strip()]


_SS_HEADER_AND_STATE = re.compile(r"^\S+")


def _parse_ss(stdout: str) -> list[dict]:
    """Parse `ss -tuln` into rows of {protocol, state, local_address}."""
    lines = [l for l in stdout.splitlines() if l.strip()]
    if not lines:
        return []
    rows: list[dict] = []
    for line in lines[1:]:  # first line is the header (Netid State Recv-Q ...)
        parts = line.split()
        if len(parts) < 5:
            continue
        netid, state, recv_q, send_q, local_address = parts[0], parts[1], parts[2], parts[3], parts[4]
        peer_address = parts[5] if len(parts) > 5 else ""
        rows.append(
            {
                "protocol": netid,
                "state": state,
                "local_address": local_address,
                "peer_address": peer_address,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Tool registry - the single source of truth for what may be executed.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _ToolSpec:
    argv: tuple[str, ...]          # fixed argv - NEVER built from user input
    description: str
    parser: Callable[[str], Any]
    timeout_seconds: int = 10


_TOOLS: dict[ToolName, _ToolSpec] = {
    ToolName.CPU_TOP: _ToolSpec(
        argv=("ps", "-eo", "pid,comm,%cpu", "--sort=-%cpu"),
        description="Top CPU-consuming processes (equivalent to `ps -eo pid,comm,%cpu --sort=-%cpu | head -6`)",
        parser=lambda out: _parse_ps_table(out, "cpu_percent", limit=5),
    ),
    ToolName.MEMORY_TOP: _ToolSpec(
        argv=("ps", "-eo", "pid,comm,%mem", "--sort=-%mem"),
        description="Top memory-consuming processes (equivalent to `ps -eo pid,comm,%mem --sort=-%mem | head -6`)",
        parser=lambda out: _parse_ps_table(out, "memory_percent", limit=5),
    ),
    ToolName.DISK_USAGE: _ToolSpec(
        argv=("df", "-h"),
        description="Disk usage per mounted filesystem (`df -h`)",
        parser=_parse_df,
    ),
    ToolName.SERVICES_RUNNING: _ToolSpec(
        argv=("systemctl", "list-units", "--type=service", "--state=running", "--no-pager", "--no-legend"),
        description="Currently running systemd services (`systemctl list-units --type=service --state=running`)",
        parser=_parse_systemctl_units,
        timeout_seconds=15,
    ),
    ToolName.SERVICES_FAILED: _ToolSpec(
        argv=("systemctl", "--failed", "--no-pager", "--no-legend"),
        description="Failed systemd services (`systemctl --failed`)",
        parser=_parse_systemctl_units,
        timeout_seconds=15,
    ),
    ToolName.LOGS_RECENT: _ToolSpec(
        argv=("journalctl", "-n", "50", "--no-pager"),
        description="Most recent 50 system journal entries (`journalctl -n 50 --no-pager`)",
        parser=_parse_journal_lines,
        timeout_seconds=15,
    ),
    ToolName.LOGS_ERROR: _ToolSpec(
        argv=("journalctl", "-p", "err", "-n", "50", "--no-pager"),
        description="Most recent 50 error-level journal entries (`journalctl -p err -n 50 --no-pager`)",
        parser=_parse_journal_lines,
        timeout_seconds=15,
    ),
    ToolName.NETWORK_PORTS: _ToolSpec(
        argv=("ss", "-tuln"),
        description="Listening TCP/UDP sockets (`ss -tuln`)",
        parser=_parse_ss,
    ),
}


def list_tools() -> list[dict]:
    """Describe every whitelisted tool - used by the /api/assistant/tools endpoint."""
    return [
        {
            "tool": name.value,
            "command": " ".join(spec.argv),
            "description": spec.description,
        }
        for name, spec in _TOOLS.items()
    ]


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n[... {omitted} more characters truncated ...]", True


def execute_tool(tool_name: ToolName | str) -> ToolResult:
    """Run exactly one whitelisted, read-only tool and return a structured result.

    Never raises for host-level failures (missing binary, timeout, non-zero
    exit, permission denied, ...) - those all come back as
    ``ToolResult(success=False, error=...)``. Only raises ``ToolExecutionError``
    if ``tool_name`` is not a recognized, whitelisted tool - i.e. a
    programmer error, not something a user request could ever trigger.
    """
    try:
        tool = ToolName(tool_name)
    except ValueError as exc:
        raise ToolExecutionError(f"Unknown tool: {tool_name!r} is not a whitelisted tool") from exc

    spec = _TOOLS[tool]
    display_command = " ".join(spec.argv)
    max_chars = settings.execution_max_output_chars
    started = time.monotonic()

    try:
        result = subprocess.run(
            list(spec.argv),          # fixed argv, never shell=True, never user-built
            shell=False,
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
            check=False,
            stdin=subprocess.DEVNULL,  # never block on an interactive prompt
        )
        duration = round(time.monotonic() - started, 3)
        stdout, truncated = _truncate(result.stdout or "", max_chars)

        warnings: list[str] = []
        if result.returncode != 0:
            # Non-zero exit isn't necessarily fatal for these tools (e.g.
            # `systemctl --failed` exits non-zero when failed units exist -
            # that's the answer, not an error). Surface it as a warning with
            # stderr attached rather than treating it as a hard failure.
            stderr_snippet = (result.stderr or "").strip()
            if stderr_snippet:
                warnings.append(f"Command exited with code {result.returncode}: {stderr_snippet}")

        try:
            parsed = spec.parser(stdout)
            row_count = len(parsed) if isinstance(parsed, list) else None
        except Exception as exc:  # noqa: BLE001 - parsing must never break the tool call
            logger.warning("Parser for tool '%s' failed: %s", tool.value, exc)
            parsed = None
            row_count = None
            warnings.append(f"Could not parse command output into structured data: {exc}")

        return ToolResult(
            tool=tool.value,
            display_command=display_command,
            success=True,
            exit_code=result.returncode,
            duration_seconds=duration,
            parsed=parsed,
            raw_output=stdout,
            row_count=row_count,
            truncated=truncated,
            error=None,
            warnings=warnings,
        )

    except subprocess.TimeoutExpired as exc:
        duration = round(time.monotonic() - started, 3)
        logger.warning("Tool '%s' timed out after %ss", tool.value, spec.timeout_seconds)
        return ToolResult(
            tool=tool.value,
            display_command=display_command,
            success=False,
            exit_code=None,
            duration_seconds=duration,
            parsed=None,
            raw_output=(exc.stdout or ""),
            row_count=None,
            truncated=False,
            error=f"Timed out after {spec.timeout_seconds} seconds",
        )
    except FileNotFoundError as exc:
        logger.error("Tool '%s' binary not found: %s", tool.value, exc)
        return ToolResult(
            tool=tool.value,
            display_command=display_command,
            success=False,
            exit_code=None,
            duration_seconds=round(time.monotonic() - started, 3),
            parsed=None,
            raw_output="",
            row_count=None,
            truncated=False,
            error=f"Command not available on this host: {exc}",
        )
    except PermissionError as exc:
        logger.error("Tool '%s' permission denied: %s", tool.value, exc)
        return ToolResult(
            tool=tool.value,
            display_command=display_command,
            success=False,
            exit_code=None,
            duration_seconds=round(time.monotonic() - started, 3),
            parsed=None,
            raw_output="",
            row_count=None,
            truncated=False,
            error=f"Permission denied running this command: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - last-resort safety net, never let a tool crash the request
        logger.error("Unexpected error running tool '%s': %s", tool.value, exc)
        return ToolResult(
            tool=tool.value,
            display_command=display_command,
            success=False,
            exit_code=None,
            duration_seconds=round(time.monotonic() - started, 3),
            parsed=None,
            raw_output="",
            row_count=None,
            truncated=False,
            error=str(exc),
        )


def execute_tools(tool_names: list[ToolName | str]) -> dict[str, ToolResult]:
    """Run several whitelisted tools and return their results keyed by tool name.

    Used when a single question needs more than one data source (e.g. "why
    is my system slow?" -> CPU_TOP + MEMORY_TOP). Each tool is executed and
    isolated the same way as ``execute_tool`` - one tool failing never
    prevents the others from running.
    """
    results: dict[str, ToolResult] = {}
    for name in tool_names:
        tool = ToolName(name)
        results[tool.value] = execute_tool(tool)
    return results
