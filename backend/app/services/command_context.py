"""
Context gathering for AI command generation and explanation (Safe Command
Execution).

Commands are far more useful (and safer) when grounded in real facts about
the machine they will run on - the actual home directory, whether a
mentioned folder exists, its current size, disk space, and OS/kernel
version. This module collects a small, targeted snapshot of that instead
of dumping the entire system-info payload into every prompt.

Deliberately read-only: `du`/`stat` only report sizes, they never modify
anything.
"""

import os

from app.logger import get_logger
from app.services import system_monitor
from app.utils import run_command

logger = get_logger(__name__)

# Common user folders we know how to recognize by name in a free-text
# description. Extendable without touching the gathering logic below.
_KNOWN_FOLDERS = [
    "Downloads",
    "Documents",
    "Desktop",
    "Pictures",
    "Videos",
    "Music",
    "Templates",
    "Public",
]


def _home_directory() -> str:
    return os.path.expanduser("~")


def _folder_snapshot(path: str) -> dict:
    """Return existence, item count, and human-readable size for a directory."""
    snapshot = {"path": path, "exists": os.path.isdir(path)}
    if not snapshot["exists"]:
        return snapshot

    try:
        snapshot["item_count"] = len(os.listdir(path))
    except OSError as exc:
        snapshot["item_count"] = None
        logger.warning("Could not list %s: %s", path, exc)

    ok, output = run_command(["du", "-sh", path], timeout=10)
    if ok and output:
        # `du -sh <path>` output looks like: "128M\t/home/user/Downloads"
        size_str = output.split()[0] if output.split() else None
        snapshot["size_human"] = size_str
    else:
        snapshot["size_human"] = None

    return snapshot


def _mentioned_folders(description: str) -> list[str]:
    """Best-effort match of known folder names mentioned in the description."""
    text = description.lower()
    return [name for name in _KNOWN_FOLDERS if name.lower() in text]


def gather_command_context(description: str) -> dict:
    """Collect a compact, relevant snapshot of system facts for command
    generation/explanation.

    Always includes home directory, OS/kernel version, and overall disk
    usage (useful for nearly any command). Additionally inspects any known
    folder names mentioned in the description (e.g. "Downloads",
    "Documents") so the LLM can generate commands with correct, real paths
    instead of guessing.
    """
    context: dict = {
        "home_directory": _home_directory(),
        "os_version": _safe(system_monitor.get_system_version),
        "disk_usage": _safe(system_monitor.get_disk_info),
    }

    home = _home_directory()
    folders = {}
    for name in _mentioned_folders(description):
        folder_path = os.path.join(home, name)
        folders[name] = _folder_snapshot(folder_path)

    if folders:
        context["mentioned_folders"] = folders

    return context


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Command context collector failed: %s", exc)
        return default if default is not None else {"error": str(exc)}
