"""
AI Ops Assistant routes.

Exposes the natural-language assistant pipeline over REST. For most
questions this is still explanation-only: an explanation plus a list of
*recommended* commands for the user to review and run manually. For a
specific set of ops questions (see `GET /api/assistant/tools`), the
assistant instead executes a real, whitelisted, read-only command on the
host and grounds its answer in that live data - see
`app/services/ops_assistant.py` and `app/services/tool_executor.py`.
"""

from fastapi import APIRouter, HTTPException

from app.logger import get_logger
from app.schemas.assistant import (
    AssistantHealthResponse,
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
)
from app.services import ai_assistant, conversation_store, ollama_client
from app.services.ollama_client import OllamaUnavailableError
from app.services.tool_executor import list_tools

router = APIRouter(prefix="/api/assistant", tags=["AI Assistant"])
logger = get_logger(__name__)


@router.get("/tools")
def get_tools():
    """List the whitelisted, read-only tools the copilot can execute live.

    Purely informational - lets the frontend (or a curious user) see
    exactly which commands `/chat` may run on the host and why, before
    ever sending a question.
    """
    return {"tools": list_tools()}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask the AI assistant a natural-language question about this system.

    For the specific ops questions the copilot recognizes (top CPU/memory
    processes, disk usage, running/failed services, recent/error logs,
    listening ports), this executes a real whitelisted command on the host
    and grounds the answer in that live data (`tool_used` /
    `detailed_results` will be populated). Everything else runs the
    original pipeline: intent classification -> system context gathering ->
    LLM reasoning -> structured explanation + recommended commands, with no
    command execution.
    """
    try:
        result = await ai_assistant.process_query(
            message=request.message, session_id=request.session_id
        )
        return ChatResponse(**result)
    except OllamaUnavailableError as exc:
        logger.error("Ollama unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Assistant pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail="The assistant failed to process your request") from exc


@router.get("/history/{session_id}", response_model=ConversationHistoryResponse)
def get_history(session_id: str):
    """Return the full conversation history for a session."""
    try:
        messages = conversation_store.get_all_messages(session_id)
        return ConversationHistoryResponse(
            session_id=session_id, total_messages=len(messages), messages=messages
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch conversation history: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch conversation history") from exc


@router.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Delete all stored conversation history for a session."""
    try:
        deleted = conversation_store.clear_session(session_id)
        return {"session_id": session_id, "deleted_messages": deleted}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to clear conversation history: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to clear conversation history") from exc


@router.get("/health", response_model=AssistantHealthResponse)
async def assistant_health():
    """Check whether Ollama is reachable and the configured model is pulled."""
    result = await ollama_client.check_health()
    return AssistantHealthResponse(**result)
