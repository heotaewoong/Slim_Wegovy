from __future__ import annotations

import json
from typing import Any

from slim_wegovy.config import Settings, load_settings
from slim_wegovy.mcp_client import StreamableHttpMcpClient
from slim_wegovy.openai_compat import LunitChatClient
from slim_wegovy.prompts import GENERATION_SYSTEM_PROMPT, RETRIEVAL_SYSTEM_PROMPT, retrieval_user_prompt
from slim_wegovy.schemas import CitationSelection, HarnessResult, ToolEvent, finalize_retrieval
from slim_wegovy.tools import FINALIZE_RETRIEVAL_TOOL, RETRIEVE_RELEVANT_CONTENT_TOOL, mcp_tool_to_openai_tool


class L2Harness:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.chat = LunitChatClient(self.settings)
        self.mcp = StreamableHttpMcpClient(self.settings)
        self._mcp_tools_cache: list[dict[str, Any]] | None = None

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> HarnessResult:
        tool_events: list[ToolEvent] = []
        retrieval_result: CitationSelection | None = None
        evidence: list[dict[str, Any]] = []
        retrieval_messages: list[dict[str, Any]] = []

        def retrieve_relevant_content(query: str) -> str:
            nonlocal retrieval_result, evidence, retrieval_messages
            retrieval = self.retrieve(query)
            retrieval_result = retrieval["selection"]
            evidence = retrieval["evidence"]
            retrieval_messages = retrieval["messages"]
            tool_events.extend(retrieval["tool_events"])
            return _format_retrieval_for_generation(retrieval_result, evidence)

        messages: list[dict[str, Any]] = [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}]
        for item in history or []:
            if item.get("role") in {"user", "assistant"}:
                messages.append({"role": item["role"], "content": item.get("content", "")})
        messages.append({"role": "user", "content": question})

        for _ in range(self.settings.generation_max_turns):
            response = self.chat.complete(messages, tools=[RETRIEVE_RELEVANT_CONTENT_TOOL])
            msg = response.choices[0].message
            messages.append(_assistant_message_to_dict(msg))

            if not msg.tool_calls:
                return HarnessResult(
                    answer=msg.content or "",
                    retrieval=retrieval_result,
                    evidence=evidence,
                    generation_messages=messages,
                    retrieval_messages=retrieval_messages,
                    tool_events=tool_events,
                )

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = _parse_tool_args(tool_call.function.arguments)
                if name != "retrieve_relevant_content":
                    content = f"[ERROR] Unknown generation tool: {name}"
                    tool_events.append(ToolEvent(phase="generation", name=name, arguments=args, result=content, error=content))
                else:
                    content = retrieve_relevant_content(str(args.get("query", question)))
                    tool_events.append(ToolEvent(phase="generation", name=name, arguments=args, result=content))
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

        return HarnessResult(
            answer="죄송합니다. 답변 생성 단계가 최대 tool 호출 횟수를 초과해 중단되었습니다.",
            retrieval=retrieval_result,
            evidence=evidence,
            generation_messages=messages,
            retrieval_messages=retrieval_messages,
            tool_events=tool_events,
        )

    def retrieve(self, query: str) -> dict[str, Any]:
        tool_events: list[ToolEvent] = []
        evidence_by_uid: dict[str, dict[str, Any]] = {}
        tools = [mcp_tool_to_openai_tool(tool) for tool in self._mcp_tools()] + [FINALIZE_RETRIEVAL_TOOL]
        valid_tool_names = {tool["function"]["name"] for tool in tools}

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {"role": "user", "content": retrieval_user_prompt(query)},
        ]

        for _ in range(self.settings.retrieval_max_turns):
            response = self.chat.complete(messages, tools=tools)
            msg = response.choices[0].message
            messages.append(_assistant_message_to_dict(msg))

            if not msg.tool_calls:
                messages.append(
                    {
                        "role": "system",
                        "content": "Retrieval mode must end by calling finalize_retrieval. Call it now.",
                    }
                )
                continue

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = _parse_tool_args(tool_call.function.arguments)
                if name == "finalize_retrieval":
                    selection = finalize_retrieval(
                        status=args.get("status", "partial"),
                        items=args.get("items", []),
                        note=args.get("note", ""),
                    )
                    return {
                        "selection": selection,
                        "evidence": _select_evidence(selection, evidence_by_uid),
                        "messages": messages,
                        "tool_events": tool_events,
                    }

                if name not in valid_tool_names:
                    content = f"[ERROR] Unknown retrieval tool: {name}"
                    tool_events.append(ToolEvent(phase="retrieval", name=name, arguments=args, result=content, error=content))
                else:
                    try:
                        content = self.mcp.call_tool(name, args)
                        content = _clip(content, self.settings.max_tool_result_chars)
                        _harvest_cite_uids(content, name, args, evidence_by_uid)
                        tool_events.append(ToolEvent(phase="retrieval", name=name, arguments=args, result=content))
                    except Exception as exc:
                        content = f"[ERROR] {type(exc).__name__}: {exc}"
                        tool_events.append(ToolEvent(phase="retrieval", name=name, arguments=args, result=content, error=content))
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

        selection = CitationSelection(status="partial" if evidence_by_uid else "no_evidence", items=[], note="retrieval tool-call budget exhausted")
        return {
            "selection": selection,
            "evidence": list(evidence_by_uid.values())[:8],
            "messages": messages,
            "tool_events": tool_events,
        }

    def _mcp_tools(self) -> list[dict[str, Any]]:
        if self._mcp_tools_cache is None:
            self._mcp_tools_cache = self.mcp.list_tools()
        return self._mcp_tools_cache


def _assistant_message_to_dict(msg: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        data["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
    return data


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _harvest_cite_uids(content: str, tool_name: str, arguments: dict[str, Any], evidence_by_uid: dict[str, dict[str, Any]]) -> None:
    for uid in _find_cite_uids(content):
        evidence_by_uid.setdefault(
            uid,
            {
                "cite_uid": uid,
                "tool": tool_name,
                "arguments": arguments,
                "content": _clip(content, 8_000),
            },
        )


def _find_cite_uids(text: str) -> list[str]:
    import re

    patterns = [
        r'"cite_uid"\s*:\s*"([^"]+)"',
        r"'cite_uid'\s*:\s*'([^']+)'",
        r"cite_uid\s*[:=]\s*([A-Za-z0-9_.:/#-]+)",
    ]
    seen: set[str] = set()
    result: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            uid = match.group(1).strip()
            if uid and uid not in seen:
                seen.add(uid)
                result.append(uid)
    return result


def _select_evidence(selection: CitationSelection, evidence_by_uid: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for item in selection.items:
        selected.append(
            evidence_by_uid.get(
                item.cite_uid,
                {"cite_uid": item.cite_uid, "content": "(selected by model; source text not captured)"},
            )
        )
    return selected


def _format_retrieval_for_generation(selection: CitationSelection, evidence: list[dict[str, Any]]) -> str:
    payload = {
        "status": selection.status,
        "note": selection.note,
        "selected_items": [item.model_dump() for item in selection.items],
        "evidence": evidence,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"
