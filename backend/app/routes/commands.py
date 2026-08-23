"""
Safe Command Execution routes.

Implements the full requested workflow over REST:

    User -> AI generates command -> Preview -> Explain -> User confirms
    -> Execute -> Return output

  POST   /api/commands/generate         natural language -> AI-generated command (preview)
  POST   /api/commands/explain          user's own command -> AI explanation (preview)
  POST   /api/commands/{id}/confirm     confirm (or reject) a previewed command -> execute -> result
  GET    /api/commands/{id}             a single execution record
  GET    /api/commands/history/{sid}    full audit trail for a session
  DELETE /api/commands/history/{sid}    clear a session's audit trail
  GET    /api/commands/safety-rules     transparency: every block/warning rule this system enforces

There is no way to execute a command through this router without first
generating or explaining it (which runs the deterministic safety
analyzer) AND then explicitly confirming via /confirm. Blocked commands
can never be executed, confirmation or not.
"""

from fastapi import APIRouter, HTTPException

from app.logger import get_logger
from app.schemas.commands import (
    CommandConfirmRequest,
    CommandExecutionResult,
    CommandExplainRequest,
    CommandGenerateRequest,
    CommandHistoryRecord,
    CommandHistoryResponse,
    CommandPreviewResponse,
    SafetyRulesResponse,
)
from app.services import command_console, execution_store
from app.services.command_safety import list_rules
from app.services.ollama_client import OllamaUnavailableError

router = APIRouter(prefix="/api/commands", tags=["Safe Command Execution"])
logger = get_logger(__name__)


@router.get("/safety-rules", response_model=SafetyRulesResponse)
def safety_rules():
    """List every deterministic block/warning rule this system enforces."""
    return SafetyRulesResponse(**list_rules())


@router.post("/generate", response_model=CommandPreviewResponse)
async def generate(request: CommandGenerateRequest):
    """Step 1-2: AI generates a command from a plain-language description, then Preview."""
    try:
        result = await command_console.generate_command(
            description=request.description, session_id=request.session_id
        )
        return CommandPreviewResponse(**result)
    except OllamaUnavailableError as exc:
        logger.error("Ollama unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Command generation pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate command") from exc


@router.post("/explain", response_model=CommandPreviewResponse)
async def explain(request: CommandExplainRequest):
    """Step 2-3: explain a user-provided command (Preview + Explain), without altering it."""
    try:
        result = await command_console.explain_command(
            command=request.command, session_id=request.session_id
        )
        return CommandPreviewResponse(**result)
    except OllamaUnavailableError as exc:
        logger.error("Ollama unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Command explanation pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to explain command") from exc


@router.post("/{execution_id}/confirm", response_model=CommandExecutionResult)
async def confirm(execution_id: int, request: CommandConfirmRequest):
    """Step 4-6: User confirms -> Execute -> Return output.

    Safety is re-checked here on the final command text (in case it was
    edited after preview) before anything runs. Passing confirm=false
    safely cancels the pending command instead of running it.
    """
    try:
        result = await command_console.confirm_and_execute(
            execution_id=execution_id,
            confirm=request.confirm,
            edited_command=request.edited_command,
        )
        return CommandExecutionResult(**result)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Command execution failed (id=%s): %s", execution_id, exc)
        raise HTTPException(status_code=500, detail="Failed to execute command") from exc


@router.get("/history/{session_id}", response_model=CommandHistoryResponse)
def get_history(session_id: str):
    """Return the full audit trail (pending, blocked, rejected, and executed) for a session."""
    try:
        records = execution_store.get_history(session_id)
        return CommandHistoryResponse(session_id=session_id, total=len(records), executions=records)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch command history: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch command history") from exc


@router.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Delete all stored command executions for a session."""
    try:
        deleted = execution_store.clear_session(session_id)
        return {"session_id": session_id, "deleted_executions": deleted}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to clear command history: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to clear command history") from exc


@router.get("/{execution_id}", response_model=CommandHistoryRecord)
def get_execution(execution_id: int):
    """Return a single execution record by id (any status: pending, blocked, rejected, executed)."""
    record = execution_store.get_by_id(execution_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No command execution found with id {execution_id}")
    return record
