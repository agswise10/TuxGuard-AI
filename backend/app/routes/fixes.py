"""
AI One-Click Fix Engine routes.

    GET  /api/fixes/detect    -> run all detectors now, return every issue
                                  found with Problem / Reason / Evidence /
                                  Confidence Score / Recommended Command
    POST /api/fixes/generate  -> turn one issue's recommended command into
                                  a pending entry in the EXISTING Safe
                                  Command Execution pipeline (never runs it)

Actually running a fix is intentionally NOT a route in this file: once
`/api/fixes/generate` returns an `execution_id`, the frontend confirms and
executes it through the existing, unmodified

    POST /api/commands/{execution_id}/confirm

exactly like any other AI-generated command. This guarantees a One-Click
Fix can never bypass the deterministic safety analyzer or run without an
explicit user confirmation.
"""

from fastapi import APIRouter, HTTPException

from app.logger import get_logger
from app.schemas.commands import CommandPreviewResponse
from app.schemas.fixes import FixDetectionResponse, FixGenerateRequest
from app.services import fix_engine

router = APIRouter(prefix="/api/fixes", tags=["AI One-Click Fix Engine"])
logger = get_logger(__name__)


@router.get("/detect", response_model=FixDetectionResponse)
async def detect():
    """Detect High CPU, High Memory, Disk Almost Full, Apache Down, Docker

    Container Stopped, and Failed Service issues right now. Gathers live
    system telemetry and reconciles it against the active-issue store: an
    already-active issue is updated (or, if nothing changed, left alone) in
    place rather than duplicated, and a resolved issue is dropped. Never
    executes anything.
    """
    try:
        result = await fix_engine.detect_all_issues()
        return FixDetectionResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.error("Fix engine detection failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to detect issues") from exc


@router.post("/generate", response_model=CommandPreviewResponse)
def generate(request: FixGenerateRequest):
    """Prepare a One-Click Fix's recommended command for confirmation.

    Reuses the existing Safe Command Execution pipeline end to end: the
    deterministic safety analyzer runs immediately and the result is
    persisted exactly like any other previewed command. The command is
    only ever executed if the user explicitly confirms via the existing
    POST /api/commands/{execution_id}/confirm endpoint.
    """
    try:
        result = fix_engine.prepare_fix(
            session_id=request.session_id,
            issue_title=request.issue_title,
            command=request.command,
            explanation=request.explanation,
            confidence_score=request.confidence_score,
        )
        return CommandPreviewResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.error("Fix engine command preparation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to prepare fix command") from exc
