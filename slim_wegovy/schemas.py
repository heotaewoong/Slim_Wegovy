from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CitableItem(BaseModel):
    cite_uid: str
    relevance_score: float
    memory: str = Field(default="", max_length=1_200)


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
    parsed = []
    for item in items:
        if isinstance(item, CitableItem):
            parsed.append(item)
            continue
        normalized = dict(item)
        normalized["memory"] = str(normalized.get("memory", ""))[:1_200]
        parsed.append(CitableItem(**normalized))
    return CitationSelection(status=status, items=parsed, note=note)


class ToolEvent(BaseModel):
    phase: Literal["retrieval", "generation"]
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    error: str | None = None


class HarnessResult(BaseModel):
    answer: str
    finish_reason: str = "stop"
    retrieval: CitationSelection | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    generation_messages: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
