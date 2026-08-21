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
        latest_retrieval_content = ""

        def retrieve_relevant_content(query: str) -> str:
            nonlocal retrieval_result, evidence, retrieval_messages, latest_retrieval_content
            retrieval = self.retrieve(query)
            retrieval_result = retrieval["selection"]
            evidence = retrieval["evidence"]
            retrieval_messages = retrieval["messages"]
            tool_events.extend(retrieval["tool_events"])
            latest_retrieval_content = _format_retrieval_for_generation(retrieval_result, evidence)
            return latest_retrieval_content

        external_instructions = [
            str(item.get("content", ""))
            for item in history or []
            if item.get("role") in {"system", "developer"} and item.get("content")
        ]
        system_prompt = GENERATION_SYSTEM_PROMPT
        if external_instructions:
            system_prompt += "\n\nAdditional conversation instructions:\n" + "\n".join(external_instructions)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for item in history or []:
            if item.get("role") in {"user", "assistant"}:
                messages.append({"role": item["role"], "content": item.get("content", "")})
        messages.append({"role": "user", "content": question})

        for _ in range(self.settings.generation_max_turns):
            response = self.chat.complete(messages, tools=[RETRIEVE_RELEVANT_CONTENT_TOOL])
            msg = response.choices[0].message
            messages.append(_assistant_message_to_dict(msg))

            if not msg.tool_calls:
                if not (msg.content or "").strip():
                    msg = self._complete_final_answer(
                        question, history, system_prompt, latest_retrieval_content
                    )
                    messages.append(_assistant_message_to_dict(msg))
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

        final_message = self._complete_final_answer(
            question, history, system_prompt, latest_retrieval_content
        )
        messages.append(_assistant_message_to_dict(final_message))
        return HarnessResult(
            answer=final_message.content or "",
            retrieval=retrieval_result,
            evidence=evidence,
            generation_messages=messages,
            retrieval_messages=retrieval_messages,
            tool_events=tool_events,
        )

    def _complete_final_answer(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        system_prompt: str,
        retrieval_content: str,
    ) -> Any:
        """Get a non-empty final answer from L2 using a compact, tool-free context."""
        final_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for item in (history or [])[-6:]:
            if item.get("role") in {"user", "assistant"}:
                final_messages.append(
                    {"role": item["role"], "content": _clip(str(item.get("content", "")), 2_000)}
                )
        final_messages.append({"role": "user", "content": question})
        if retrieval_content:
            final_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Use the retrieved material below for source-specific factual claims. "
                        "You may supplement it with established general medical knowledge only when "
                        "confident and useful, but do not fill a source-specific gap by guessing. "
                        "Clearly state any limitation or uncertainty and never invent a citation. "
                        "No more tools are available. Return only a complete, non-empty final answer "
                        "in the user's language.\n\n"
                        + retrieval_content
                    ),
                }
            )
        else:
            final_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Write the final answer now without tools. Address every part of the request, "
                        "follow the clinical communication rules, and return a complete, non-empty "
                        "answer in the user's language."
                    ),
                }
            )

        for attempt in range(2):
            response = self.chat.complete(final_messages, tools=None, tool_choice=None)
            msg = response.choices[0].message
            if (msg.content or "").strip():
                return msg
            final_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous response was empty (retry {attempt + 1}/2). "
                        "Return the complete final answer now."
                    ),
                }
            )
        raise RuntimeError("L2 returned an empty final answer after 2 attempts")

    def retrieve(self, query: str) -> dict[str, Any]:
        tool_events: list[ToolEvent] = []
        evidence_by_uid: dict[str, dict[str, Any]] = {}
        selected_mcp_tools = _select_mcp_tools(query, self._mcp_tools())
        tools = [mcp_tool_to_openai_tool(tool) for tool in selected_mcp_tools] + [FINALIZE_RETRIEVAL_TOOL]
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
                        "role": "user",
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

        fallback_items = [
            {"cite_uid": uid, "relevance_score": 0.5}
            for uid in list(evidence_by_uid)[:4]
        ]
        selection = finalize_retrieval(
            status="partial" if fallback_items else "no_evidence",
            items=fallback_items,
            note="retrieval tool-call time budget exhausted",
        )
        return {
            "selection": selection,
            "evidence": _select_evidence(selection, evidence_by_uid),
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


_TOOL_ROUTES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (
        ("법", "법령", "규정", "law", "legal"),
        ("openapi_law_search", "openapi_law_list_articles", "openapi_law_get_article"),
    ),
    (
        ("kcd", "질병코드", "상병코드", "진단코드", "청구코드"),
        ("kcd_search_codes", "kcd_get_name", "openapi_hira_disease_check_code"),
    ),
    (
        ("급여", "보험", "심평원", "hira", "약가", "수가", "비급여"),
        (
            "hira_updates_search",
            "openapi_hira_get_drug_price",
            "index_list_documents",
            "index_get_relevant_nodes",
            "index_get_page_content",
            "index_keyword_search",
        ),
    ),
    (
        ("가이드라인", "진료지침", "권고", "guideline", "목표치"),
        (
            "index_list_documents",
            "index_get_relevant_nodes",
            "index_get_page_content",
            "index_get_document_structure",
            "index_keyword_search",
        ),
    ),
    (
        ("약", "의약품", "위고비", "허가", "부작용", "이상반응", "용량", "금기", "상호작용", "성분", "drug", "medication"),
        (
            "adr_retrieve_drug_info",
            "openapi_mfds_get_drug_indication",
            "openapi_mfds_check_drug_permission",
            "openapi_mfds_find_drugs_by_ingredient",
        ),
    ),
    (
        ("논문", "연구", "pubmed", "faers", "안전성 신호", "문헌"),
        (
            "rag_get_all_data_sources",
            "rag_get_data_source_detail",
            "rag_vector_query",
            "rag_sql_query",
        ),
    ),
]

_DEFAULT_TOOL_NAMES = (
    "rag_get_all_data_sources",
    "rag_get_data_source_detail",
    "rag_vector_query",
    "index_list_documents",
    "index_get_relevant_nodes",
    "index_get_page_content",
)


def _select_mcp_tools(
    query: str, available: list[dict[str, Any]], max_tools: int = 8
) -> list[dict[str, Any]]:
    """Route a query to a compact MCP tool subset to stay within L2's input limit."""
    by_name = {tool.get("name"): tool for tool in available if isinstance(tool.get("name"), str)}
    lowered = query.casefold()
    ordered_names: list[str] = []
    for keywords, names in _TOOL_ROUTES:
        if any(keyword.casefold() in lowered for keyword in keywords):
            ordered_names.extend(names)
    if not ordered_names:
        ordered_names.extend(_DEFAULT_TOOL_NAMES)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in ordered_names:
        if name in seen or name not in by_name:
            continue
        selected.append(by_name[name])
        seen.add(name)
        if len(selected) >= max_tools:
            break
    if not selected:
        selected = available[:max_tools]
    return selected


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
                "content": _clip(content, 2_000),
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
    for item in selection.items[:4]:
        selected.append(
            evidence_by_uid.get(
                item.cite_uid,
                {"cite_uid": item.cite_uid, "content": "(selected by model; source text not captured)"},
            )
        )
    return selected


def _format_retrieval_for_generation(selection: CitationSelection, evidence: list[dict[str, Any]]) -> str:
    numbered_evidence = [
        {"citation_index": index, **item} for index, item in enumerate(evidence, start=1)
    ]
    payload = {
        "status": selection.status,
        "note": selection.note,
        "selected_items": [item.model_dump() for item in selection.items],
        "evidence": numbered_evidence if selection.items else [],
        "uncited_fallback_evidence": evidence if not selection.items else [],
    }
    return _clip(json.dumps(payload, ensure_ascii=False, indent=2), 10_000)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"
