"""
Pydantic schemas for Safe Command Execution.

Covers the full requested workflow's request/response contracts:
generate/explain (-> preview), confirm (-> execute -> result), and the
audit-trail history endpoints.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

RiskLevel = Literal["safe", "low", "medium", "high", "blocked"]
ExecutionStatus = Literal["pending", "blocked", "rejected", "executed"]
CommandSource = Literal["ai_generated", "user_provided"]


class MatchedRule(BaseModel):
    rule: str
    description: str
    severity: str


# ---------------------------------------------------------------------------
# Step 1-3: generate or explain -> preview
# ---------------------------------------------------------------------------

class CommandGenerateRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=1,
        description="Plain-language description of the command you want, "
        "e.g. 'show me the 10 largest files in my home directory'",
    )
    session_id: Optional[str] = Field(
        None, description="Groups related commands together. Omit to start a new session."
    )


class CommandExplainRequest(BaseModel):
    command: str = Field(
        ..., min_length=1, description="A shell command you typed yourself, to be explained before running"
    )
    session_id: Optional[str] = Field(None, description="Groups related commands together.")


class CommandPreviewResponse(BaseModel):
    execution_id: int = Field(..., description="Use this id to confirm/execute or reject this command")
    session_id: str
    description: Optional[str] = None
    source: CommandSource
    command: str
    explanation: str
    risk_level: RiskLevel
    blocked: bool = Field(..., description="If true, this command can NEVER be executed via /confirm")
    matched_rules: list[MatchedRule] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    status: ExecutionStatus
    context_summary: dict = Field(default_factory=dict)
    timestamp: str


# ---------------------------------------------------------------------------
# Step 4-6: confirm -> execute -> return output
# ---------------------------------------------------------------------------

class CommandConfirmRequest(BaseModel):
    confirm: bool = Field(
        ..., description="Must be true to run the command. False (or omitted) safely cancels it."
    )
    edited_command: Optional[str] = Field(
        None,
        description="If the user edited the command after preview, pass the final text here - "
        "it will be re-validated by the safety analyzer before running.",
    )


class CommandExecutionResult(BaseModel):
    execution_id: int
    session_id: str
    command: str
    status: ExecutionStatus
    risk_level: RiskLevel
    matched_rules: list[MatchedRule] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    duration_seconds: float = 0.0
    timed_out: bool = False
    executed_at: Optional[str] = None
    message: Optional[str] = Field(
        None, description="Explains why nothing ran, when status is 'blocked' or 'rejected'"
    )


# ---------------------------------------------------------------------------
# History / audit trail
# ---------------------------------------------------------------------------

class CommandHistoryRecord(BaseModel):
    id: int
    session_id: str
    description: Optional[str] = None
    command: str
    source: CommandSource
    explanation: Optional[str] = None
    risk_level: RiskLevel
    blocked: bool
    matched_rules: list[MatchedRule] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    status: ExecutionStatus
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    duration_seconds: Optional[float] = None
    timed_out: bool = False
    created_at: str
    executed_at: Optional[str] = None


class CommandHistoryResponse(BaseModel):
    session_id: str
    total: int
    executions: list[CommandHistoryRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Safety rules transparency
# ---------------------------------------------------------------------------

class SafetyRule(BaseModel):
    name: str
    description: str
    severity: str


class SafetyRulesResponse(BaseModel):
    block_rules: list[SafetyRule] = Field(default_factory=list)
    warning_rules: list[SafetyRule] = Field(default_factory=list)
