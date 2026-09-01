"""Contracts shared by the tool-calling provider, registry, and agent loop."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import Field

from analysis.domain.models import DomainModel, Platform


class AgentRequest(DomainModel):
    question: str = Field(min_length=1, max_length=2_000)
    platform: Optional[Platform] = None
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    semantic_run_id: Optional[UUID] = None
    max_steps: int = Field(default=5, ge=1, le=8)


class ProviderToolCall(DomainModel):
    id: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ProviderTurn(DomainModel):
    content: Optional[str] = None
    tool_calls: list[ProviderToolCall] = Field(default_factory=list)
    assistant_message: dict[str, Any]


class AgentToolTrace(DomainModel):
    step: int = Field(ge=1)
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    result_characters: int = Field(ge=0)
    error: Optional[str] = None


class AgentResponse(DomainModel):
    answer: str
    model: str
    provider: str
    steps: int = Field(ge=1)
    tool_calls: list[AgentToolTrace] = Field(default_factory=list)
    finish_reason: str


class ToolDefinition(DomainModel):
    name: str
    description: str
    parameters: dict[str, Any]
