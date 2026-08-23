"""
Active-issue store for the AI One-Click Fix Engine.

Pure state + deduplication - no detection logic and no AI calls live here
(that's `fix_engine.py`). A detection scan calls `reconcile()` once per
pass with every issue currently observed on the host; this module is the
single source of truth for whether each one is:

    - brand new                -> create one active alert
    - already active, unchanged -> a duplicate scan, ignored (no new
                                    diagnosis, no new alert - just a
                                    freshness/occurrence bump)
    - already active, changed   -> update the existing alert in place
    - no longer observed        -> resolved, dropped from the active set

`issue_id` (e.g. "high_cpu", "disk_full:/home", "apache_down:apache2")
already encodes both the issue type and the affected resource, so keying
the store on it is exactly what "one active alert per issue" and "new
alert only if the issue type or affected resource changes" require: a
different type or a different resource always produces a different
`issue_id`, and therefore a different alert.

In-memory and process-local by design, matching this PoC's existing
pattern (see the old `alert_monitor.py`'s background task) - no database
table is needed for "currently active issues", only for permanent
history, which this module intentionally does not keep.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Awaitable, Callable

_active_issues: dict[str, dict] = {}
_lock = asyncio.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evidence_signature(evidence: dict) -> str:
    """Order-independent fingerprint of an issue's evidence.

    Two scans of the same issue with identical evidence produce the same
    signature - that's what tells a genuine update apart from a duplicate
    scan of an unchanged problem.
    """
    return json.dumps(evidence, sort_keys=True, default=str)


async def reconcile(
    candidates: list[dict],
    diagnose: Callable[[dict], Awaitable[dict]],
) -> list[dict]:
    """Merge one scan's worth of candidate issues into the active-issue set.

    candidates: one dict per issue currently observed, each with at least
        issue_id, issue_type, title, problem, evidence, severity,
        fallback_command (see fix_engine._make_candidate).
    diagnose: async callable(candidate) -> dict with reason,
        confidence_score, recommended_command, recommended_action. Called
        ONLY for a candidate that is brand new or whose evidence changed -
        never for an unchanged duplicate scan, and never for a resolved
        issue.

    Returns the full, current list of active alerts (one per issue_id),
    each tagged with a "status" of "new", "updated", or "duplicate" for
    this pass, plus "first_detected_at", "last_seen_at", and
    "occurrence_count". Issues not present in `candidates` this pass are
    treated as resolved and removed from the active set.
    """
    async with _lock:
        seen_ids: set[str] = set()

        for candidate in candidates:
            issue_id = candidate["issue_id"]
            seen_ids.add(issue_id)
            signature = _evidence_signature(candidate["evidence"])
            now = _now_iso()
            existing = _active_issues.get(issue_id)

            if existing is None:
                # New issue type/resource combination - the only case that
                # creates a new active alert.
                diagnosis = await diagnose(candidate)
                _active_issues[issue_id] = {
                    **candidate,
                    **diagnosis,
                    "evidence_signature": signature,
                    "first_detected_at": now,
                    "last_seen_at": now,
                    "occurrence_count": 1,
                    "status": "new",
                }
                continue

            if existing["evidence_signature"] == signature:
                # Same issue, same evidence as last time: a duplicate scan.
                # Ignore it - update freshness bookkeeping only, never
                # create a second alert or re-run the (expensive) AI
                # diagnosis for something that hasn't changed.
                existing["last_seen_at"] = now
                existing["occurrence_count"] += 1
                existing["status"] = "duplicate"
                continue

            # Same issue_id, evidence changed materially (e.g. CPU climbed
            # further, a different process is now the top consumer): update
            # the single active alert in place instead of creating another.
            diagnosis = await diagnose(candidate)
            existing.update(candidate)
            existing.update(diagnosis)
            existing["evidence_signature"] = signature
            existing["last_seen_at"] = now
            existing["occurrence_count"] += 1
            existing["status"] = "updated"

        # Anything active but not observed this scan has resolved itself
        # (metric dropped back under threshold, service recovered, container
        # restarted, ...) - it no longer belongs in the active set.
        for resolved_id in list(_active_issues.keys()):
            if resolved_id not in seen_ids:
                del _active_issues[resolved_id]

        return [dict(issue) for issue in _active_issues.values()]


def get_active_issues() -> list[dict]:
    """Current active alerts without running a new scan."""
    return [dict(issue) for issue in _active_issues.values()]


def clear() -> None:
    """Drop all active alerts. Mainly useful for tests."""
    _active_issues.clear()
