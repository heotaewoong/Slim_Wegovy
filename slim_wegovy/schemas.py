from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CitableItem(BaseModel):
    cite_uid: str
    relevance_score: float


class CitationSelection(BaseModel):
    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[CitableItem] = Field(default_factory=list)
    note: str = ""


def finalize_retrieval(
    status: Literal["sufficient", "partial", "no_evidence"],
    items: list[dict[str, Any]] | list[CitableItem],
    note: str = "",
) -> CitationSelection:
    """Submit final citation selection and end the retrieval phase."""
    parsed = [item if isinstance(item, CitableItem) else CitableItem(**item) for item in items]
    return CitationSelection(status=status, items=parsed, note=note)


class ToolEvent(BaseModel):
    phase: Literal["retrieval", "generation"]
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    error: str | None = None


class HarnessResult(BaseModel):
    answer: str
    retrieval: CitationSelection | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    generation_messages: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
