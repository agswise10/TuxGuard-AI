"""
Pydantic schemas for the AI One-Click Fix Engine.

Detection responses are defined here. Preparing a fix reuses
`CommandPreviewResponse` from `schemas/commands.py` directly (the exact
same shape returned by /api/commands/generate and /api/commands/explain),
and confirming/executing a fix reuses the existing /api/commands endpoints
unchanged - no new execution schema is needed.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

IssueType = Literal[
    "high_cpu",
    "high_memory",
    "disk_full",
    "apache_down",
    "docker_stopped",
    "failed_service",
    "predicted_failure",
    "hardware_fault",
    "driver_anomaly",
]
IssueSeverity = Literal["warning", "critical"]
AlertStatus = Literal["new", "updated", "duplicate"]


class DetectedIssue(BaseModel):
    issue_id: str = Field(..., description="Stable id for this issue; pass back to /api/fixes/generate")
    issue_type: IssueType
    title: str
    severity: IssueSeverity
    problem: str = Field(..., description="What was detected")
    reason: str = Field(..., description="Why it's happening (AI diagnosis, with deterministic fallback)")
    evidence: dict = Field(default_factory=dict, description="Raw metrics/facts backing the diagnosis")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    recommended_command: str
    recommended_action: str
    first_detected_at: str = Field(..., description="When this issue_id first became active")
    last_seen_at: str = Field(..., description="Most recent scan that observed this issue")
    occurrence_count: int = Field(..., ge=1, description="Number of scans this active alert has been seen in")
    status: AlertStatus = Field(
        ..., description="'new' (just created), 'updated' (evidence changed), or 'duplicate' (unchanged scan)"
    )


class FixDetectionResponse(BaseModel):
    checked_at: str
    total_issues: int
    issues: list[DetectedIssue] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list, description="Non-fatal detector failures, if any")


class FixGenerateRequest(BaseModel):
    issue_id: str = Field(..., min_length=1)
    issue_title: str = Field(..., min_length=1)
    command: str = Field(..., min_length=1, description="The recommended_command from /api/fixes/detect")
    explanation: str = Field(..., min_length=1, description="Problem + Reason, shown as the command's explanation")
    confidence_score: float = Field(0.7, ge=0.0, le=1.0)
    session_id: Optional[str] = Field(None, description="Groups this fix with the rest of the session's activity")
