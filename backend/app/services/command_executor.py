"""
Command Executor.

The only place in this project where a shell command is actually run.
It is deliberately dumb: it does not decide whether a command is safe -
that is `command_safety.py`'s job, and callers MUST run that check
immediately before calling `execute()`. This module just runs the command
with a timeout and faithfully captures whatever happens.
"""

import subprocess
import time

from app.config import get_settings
from app.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


def _truncate(text: str, max_chars: int) -> str:
    """Cap captured output so a runaway command can't blow up the DB/response."""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n[... {omitted} more characters truncated ...]"


def execute(command: str, timeout: int | None = None) -> dict:
    """Run `command` in a shell and capture its result.

    Never raises: timeouts, missing binaries, and other failures are all
    captured into the returned dict instead of propagating, so the caller
    can always log and return a structured result.
    """
    timeout = timeout or settings.execution_timeout_seconds
    max_chars = settings.execution_max_output_chars
    started = time.monotonic()

    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            # Without this, the child inherits this process's stdin/controlling
            # TTY. If `command` contains `sudo` and passwordless sudo isn't
            # configured on the host, sudo will print its password prompt to
            # that inherited TTY and block reading stdin - which nothing will
            # ever answer, since this runs headless. That blocking read looks
            # identical to a genuine command hang and is only ever cleared by
            # the timeout below. Explicitly closing stdin makes sudo fail fast
            # with a clear "a password is required"/"no askpass" error instead,
            # so a real privilege problem surfaces immediately in stderr rather
            # than being masked as a 30s timeout.
            stdin=subprocess.DEVNULL,
        )
        duration = round(time.monotonic() - started, 3)
        return {
            "stdout": _truncate(result.stdout, max_chars),
            "stderr": _truncate(result.stderr, max_chars),
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "timed_out": False,
            "error": None,
        }
    except subprocess.TimeoutExpired as exc:
        duration = round(time.monotonic() - started, 3)
        logger.warning("Command timed out after %ss: %s", timeout, command)
        return {
            "stdout": _truncate(exc.stdout or "", max_chars),
            "stderr": _truncate(exc.stderr or "", max_chars)
            + f"\n[Execution timed out after {timeout}s and was terminated.]",
            "exit_code": None,
            "duration_seconds": duration,
            "timed_out": True,
            "error": f"Timed out after {timeout} seconds",
        }
    except FileNotFoundError as exc:
        logger.error("Could not run command (bash not found?): %s", exc)
        return {
            "stdout": "",
            "stderr": str(exc),
            "exit_code": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "timed_out": False,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error executing command: %s", exc)
        return {
            "stdout": "",
            "stderr": str(exc),
            "exit_code": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "timed_out": False,
            "error": str(exc),
        }