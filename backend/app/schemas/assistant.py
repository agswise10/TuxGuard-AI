"""
Pydantic schemas for the AI Ops Assistant.

Defines the request/response contract for /api/assistant/* endpoints,
including the structured shape the LLM itself is instructed to produce
(explanation, recommended_commands, confidence_score, reasoning).
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's natural-language question")
    session_id: Optional[str] = Field(
        None, description="Conversation session id. Omit to start a new session."
    )


class RecommendedCommand(BaseModel):
    command: str
    description: str
    risk_level: Literal["low", "medium", "high"] = "low"


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    explanation: str
    recommended_commands: list[RecommendedCommand] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    context_summary: dict = Field(
        default_factory=dict, description="Summary of the system data used to answer this query"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal issues encountered while answering"
    )
    timestamp: str


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    intent: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: str


class ConversationHistoryResponse(BaseModel):
    session_id: str
    total_messages: int
    messages: list[ConversationMessage] = Field(default_factory=list)


class AssistantHealthResponse(BaseModel):
    reachable: bool
    configured_model: str
    model_available: bool
    installed_models: list[str] = Field(default_factory=list)
    error: Optional[str] = None
