"""
Docker container inspection.

Read-only wrapper around `docker ps`. Never starts/stops/modifies
containers - this project never executes actions on the user's behalf,
only observes and reports.
"""

import json

from app.logger import get_logger
from app.utils import run_command

logger = get_logger(__name__)

_FORMAT = (
    '{"id":"{{.ID}}","name":"{{.Names}}","image":"{{.Image}}",'
    '"status":"{{.Status}}","state":"{{.State}}","ports":"{{.Ports}}"}'
)


def get_docker_containers(all_containers: bool = True) -> dict:
    """Return running (or all) Docker containers via `docker ps`.

    Returns a dict with `available` (bool - is Docker installed/reachable),
    `containers` (list), and `error` (str | None) so callers can always
    render a sensible message even when Docker isn't present.
    """
    command = ["docker", "ps", "--format", _FORMAT]
    if all_containers:
        command.append("--all")

    ok, output = run_command(command, timeout=10)

    if not ok:
        return {"available": False, "containers": [], "error": output}

    containers = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            containers.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse docker ps line: %s (%s)", line, exc)

    return {"available": True, "containers": containers, "error": None}
