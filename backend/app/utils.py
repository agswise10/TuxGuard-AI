"""
Safe subprocess execution helper.

Centralizes running external Ubuntu commands (systemctl, journalctl,
lsb_release, who, ...) with a timeout and consistent error handling, so the
system monitor module never crashes the API if a command is missing, times
out, or fails on a given machine.
"""

import subprocess

from app.logger import get_logger

logger = get_logger(__name__)


def run_command(command: list[str], timeout: int = 5) -> tuple[bool, str]:
    """Run a shell command and return (success, output_or_error_message).

    Never raises - any failure (missing binary, timeout, non-zero exit) is
    captured and returned as a descriptive error string instead.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or f"exit code {result.returncode}"
            logger.warning("Command %s failed: %s", " ".join(command), stderr)
            return False, stderr
        return True, result.stdout.strip()
    except FileNotFoundError:
        message = f"Command not found: {command[0]}"
        logger.warning(message)
        return False, message
    except subprocess.TimeoutExpired:
        message = f"Command timed out: {' '.join(command)}"
        logger.warning(message)
        return False, message
    except Exception as exc:  # noqa: BLE001 - defensive catch-all for external calls
        message = f"Unexpected error running {' '.join(command)}: {exc}"
        logger.error(message)
        return False, message
