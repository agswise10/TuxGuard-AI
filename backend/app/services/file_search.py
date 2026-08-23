"""
Large-file search.

Read-only wrapper around `find`, scoped to a safe default search path and
bounded by a timeout, so a "find large files" query can never hang the API
or scan the entire filesystem by surprise.
"""

import subprocess

from app.logger import get_logger

logger = get_logger(__name__)

# Directories most likely to accumulate large user/application files.
# Kept intentionally narrower than "/" for speed and to avoid permission
# noise from pseudo-filesystems like /proc and /sys.
_DEFAULT_SEARCH_PATHS = ["/home", "/var", "/opt", "/tmp"]


def find_large_files(
    min_size_mb: int = 100,
    limit: int = 20,
    search_paths: list[str] | None = None,
) -> dict:
    """Find files larger than `min_size_mb` under the given search paths.

    Returns a dict with `files` (list of {path, size_bytes, size_human})
    sorted largest-first, and `error` (str | None).
    """
    paths = search_paths or _DEFAULT_SEARCH_PATHS

    command = [
        "find",
        *paths,
        "-xdev",
        "-type",
        "f",
        "-size",
        f"+{min_size_mb}M",
        "-printf",
        "%s %p\n",
    ]

    output = ""
    error = None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        output = result.stdout
        # `find` commonly exits non-zero purely from "Permission denied" on a
        # handful of directories while still producing valid stdout for
        # everything it *could* read - so we treat that as a soft warning,
        # not a hard failure, as long as we got some output back.
        if result.returncode != 0 and not output.strip():
            error = result.stderr.strip() or f"find exited with code {result.returncode}"
        elif result.returncode != 0:
            logger.info("find reported partial errors (likely permission-denied dirs): %s",
                        result.stderr.strip()[:300])
    except FileNotFoundError:
        error = "Command not found: find"
    except subprocess.TimeoutExpired:
        error = "find command timed out"
    except Exception as exc:  # noqa: BLE001
        error = f"Unexpected error running find: {exc}"

    if error:
        logger.warning("find_large_files command issue: %s", error)
        return {"files": [], "error": error}

    entries = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            size_str, path = line.split(" ", 1)
            size_bytes = int(size_str)
            entries.append(
                {
                    "path": path,
                    "size_bytes": size_bytes,
                    "size_human": _human_readable(size_bytes),
                }
            )
        except ValueError:
            continue

    entries.sort(key=lambda e: e["size_bytes"], reverse=True)
    return {"files": entries[:limit], "error": None}


def _human_readable(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
