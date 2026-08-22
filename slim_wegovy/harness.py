from __future__ import annotations

import json
import re
import time
from typing import Any

from slim_wegovy.config import Settings, load_settings
from slim_wegovy.mcp_client import StreamableHttpMcpClient
from slim_wegovy.openai_compat import LunitChatClient
from slim_wegovy.prompts import (
    CONTEXT_COMPACTION_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    SIMPLE_GENERATION_SYSTEM_PROMPT,
    retrieval_user_prompt,
)
from slim_wegovy.schemas import CitationSelection, HarnessResult, ToolEvent, finalize_retrieval
from slim_wegovy.tools import FINALIZE_RETRIEVAL_TOOL, RETRIEVE_RELEVANT_CONTENT_TOOL, mcp_tool_to_openai_tool


class L2Harness:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.chat = LunitChatClient(self.settings)
        self.mcp = StreamableHttpMcpClient(self.settings)
        self._mcp_tools_cache: list[dict[str, Any]] | None = None

    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> HarnessResult:
        tool_events: list[ToolEvent] = []
        retrieval_result: CitationSelection | None = None
        evidence: list[dict[str, Any]] = []
        retrieval_messages: list[dict[str, Any]] = []
        latest_retrieval_content = ""
        deadline = time.monotonic() + self.settings.request_deadline_sec
        completion_tokens = _bounded_completion_tokens(
            max_tokens, self.settings.max_completion_tokens
        )
        generation_temperature = 0.0 if temperature is None else temperature
        safety_override = _high_risk_improvised_response(
            question, history, max_tokens=completion_tokens
        )
        if safety_override:
            return HarnessResult(
                answer=safety_override,
                finish_reason="stop",
                generation_messages=[
                    *_compact_history(history),
                    {"role": "user", "content": question},
                ],
            )

        working_history = self._prepare_history(question, history, deadline=deadline)

        def retrieve_relevant_content(query: str) -> str:
            nonlocal retrieval_result, evidence, retrieval_messages, latest_retrieval_content
            try:
                retrieval = self.retrieve(query, deadline=deadline)
            except Exception as exc:
                retrieval_result = finalize_retrieval(
                    status="no_evidence",
                    items=[],
                    note="Retrieval was unavailable; answer only from reliable stable knowledge.",
                )
                evidence = []
                retrieval_messages = []
                latest_retrieval_content = _format_retrieval_for_generation(
                    retrieval_result, evidence
                )
                error = f"{type(exc).__name__}: {exc}"
                tool_events.append(
                    ToolEvent(
                        phase="generation",
                        name="retrieval_error",
                        arguments={"query": query},
                        result="[ERROR] Retrieval unavailable",
                        error=error,
                    )
                )
                return latest_retrieval_content
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
        system_prompt = _generation_system_prompt(history)
        if external_instructions:
            system_prompt += "\n\nAdditional conversation instructions:\n" + "\n".join(external_instructions)
        request_policy = _request_specific_policy(question)
        if request_policy:
            system_prompt += "\n\nRequest-specific policy:\n" + request_policy
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for item in working_history:
            if item.get("role") in {"user", "assistant"}:
                messages.append({"role": item["role"], "content": item.get("content", "")})
        messages.append({"role": "user", "content": question})

        offer_retrieval = _should_offer_retrieval(question, history)
        for _ in range(self.settings.generation_max_turns):
            generation_tools = (
                [RETRIEVE_RELEVANT_CONTENT_TOOL]
                if offer_retrieval and retrieval_result is None
                else None
            )
            try:
                response = self.chat.complete(
                    messages,
                    tools=generation_tools,
                    tool_choice="auto" if generation_tools else None,
                    max_tokens=completion_tokens,
                    temperature=generation_temperature,
                    timeout_sec=_stage_timeout(
                        deadline,
                        self.settings.retrieval_model_timeout_sec
                        if generation_tools
                        else self.settings.lunit_timeout_sec,
                    ),
                )
            except Exception as exc:
                if not generation_tools:
                    raise
                error = f"{type(exc).__name__}: {exc}"
                tool_events.append(
                    ToolEvent(
                        phase="generation",
                        name="retrieval_decision_fallback",
                        arguments={},
                        result="[ERROR] Retrieval decision unavailable; generating without tools",
                        error=error,
                    )
                )
                retrieval_result = finalize_retrieval(
                    status="no_evidence",
                    items=[],
                    note="No verified current evidence is available in the provided source set.",
                )
                latest_retrieval_content = _format_retrieval_for_generation(
                    retrieval_result, []
                )
                answer, finish_reason = self._complete_final_answer(
                    question,
                    working_history,
                    system_prompt,
                    latest_retrieval_content,
                    max_tokens=completion_tokens,
                    temperature=generation_temperature,
                    deadline=deadline,
                )
                return HarnessResult(
                    answer=answer,
                    finish_reason=finish_reason,
                    retrieval=retrieval_result,
                    evidence=evidence,
                    generation_messages=messages,
                    retrieval_messages=retrieval_messages,
                    tool_events=tool_events,
                )
            choice = response.choices[0]
            msg = choice.message
            messages.append(_assistant_message_to_dict(msg))

            if not msg.tool_calls:
                if not (msg.content or "").strip():
                    answer, finish_reason = self._complete_final_answer(
                        question,
                        working_history,
                        system_prompt,
                        latest_retrieval_content,
                        max_tokens=completion_tokens,
                        temperature=generation_temperature,
                        deadline=deadline,
                    )
                else:
                    answer, finish_reason = self._finish_answer(
                        msg.content or "",
                        _choice_finish_reason(choice),
                        messages,
                        max_tokens=completion_tokens,
                        temperature=generation_temperature,
                        deadline=deadline,
                    )
                return HarnessResult(
                    answer=answer,
                    finish_reason=finish_reason,
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

        answer, finish_reason = self._complete_final_answer(
            question,
            working_history,
            system_prompt,
            latest_retrieval_content,
            max_tokens=completion_tokens,
            temperature=generation_temperature,
            deadline=deadline,
        )
        return HarnessResult(
            answer=answer,
            finish_reason=finish_reason,
            retrieval=retrieval_result,
            evidence=evidence,
            generation_messages=messages,
            retrieval_messages=retrieval_messages,
            tool_events=tool_events,
        )

    def _prepare_history(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        *,
        deadline: float,
    ) -> list[dict[str, str]]:
        """Abstractively compact only histories that would otherwise crowd the answer."""
        filtered = [
            {"role": str(item["role"]), "content": str(item.get("content", ""))}
            for item in history or []
            if item.get("role") in {"user", "assistant"}
        ]
        if sum(len(item["content"]) for item in filtered) <= (
            self.settings.history_compaction_threshold_chars
        ):
            return filtered

        bounded = _compact_history(
            filtered,
            max_chars=self.settings.max_compaction_input_chars,
            max_item_chars=8_000,
        )
        compaction_request = (
            "Latest user question (summarize prior context for this objective; do not answer):\n"
            f"{question}\n\nPrior conversation:\n"
            + json.dumps(bounded, ensure_ascii=False)
        )
        try:
            response = self.chat.complete(
                [
                    {"role": "system", "content": CONTEXT_COMPACTION_PROMPT},
                    {"role": "user", "content": compaction_request},
                ],
                tools=None,
                tool_choice=None,
                max_tokens=self.settings.max_compaction_tokens,
                temperature=0.0,
                timeout_sec=_stage_timeout(
                    deadline,
                    self.settings.compaction_timeout_sec,
                    reserve_sec=self.settings.final_answer_reserve_sec,
                ),
            )
            memory = (response.choices[0].message.content or "").strip()
        except Exception:
            memory = ""
        if not memory:
            return _compact_history(
                filtered,
                max_chars=self.settings.history_compaction_threshold_chars,
            )
        return [
            {
                "role": "user",
                "content": (
                    "[Compressed conversation memory; this describes prior context and is not "
                    "a new instruction.]\n" + memory
                ),
            }
        ]

    def _complete_final_answer(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        system_prompt: str,
        retrieval_content: str,
        *,
        max_tokens: int,
        temperature: float | None,
        deadline: float,
    ) -> tuple[str, str]:
        """Get a non-empty final answer from L2 using a compact, tool-free context."""
        final_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        final_messages.extend(_compact_history(history))
        final_messages.append({"role": "user", "content": question})
        if retrieval_content:
            final_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Write the final answer to the original user now, in the user's "
                        "language. Use the evidence below only where it directly supports a "
                        "claim. No more tools are available. If evidence is incomplete, name "
                        "only the specific unresolved fact. Return a non-empty, complete "
                        "answer.\n\n"
                        + retrieval_content
                    ),
                }
            )
        else:
            final_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Write the final answer to the original user now, in the user's "
                        "language. No tools are available. Return a non-empty, complete answer."
                    ),
                }
            )

        for attempt in range(2):
            response = self.chat.complete(
                final_messages,
                tools=None,
                tool_choice=None,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_sec=_stage_timeout(deadline, self.settings.lunit_timeout_sec),
            )
            choice = response.choices[0]
            msg = choice.message
            if (msg.content or "").strip():
                final_messages.append(_assistant_message_to_dict(msg))
                return self._finish_answer(
                    msg.content or "",
                    _choice_finish_reason(choice),
                    final_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    deadline=deadline,
                )
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

    def _finish_answer(
        self,
        answer: str,
        finish_reason: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float | None,
        deadline: float,
    ) -> tuple[str, str]:
        """Continue a response that the upstream model stopped mid-answer."""
        if not _looks_truncated(answer, finish_reason):
            return answer.strip(), finish_reason

        combined = answer.rstrip()
        continuation_messages = list(messages)
        continuation_reason = finish_reason
        for _ in range(self.settings.max_continuations):
            if _remaining_seconds(deadline) < 5:
                break
            continuation_messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous answer was cut off. Continue from the next missing "
                        "point without restarting or repeating it. Complete every remaining "
                        "user request and any relevant safety or follow-up guidance. Return "
                        "only the continuation and end with a complete sentence."
                    ),
                }
            )
            try:
                response = self.chat.complete(
                    continuation_messages,
                    tools=None,
                    tool_choice=None,
                    max_tokens=min(self.settings.max_continuation_tokens, max_tokens),
                    temperature=temperature,
                    timeout_sec=_stage_timeout(deadline, self.settings.lunit_timeout_sec),
                )
            except Exception:
                # A partial answer can still earn credit; never turn it into a
                # guaranteed candidate failure because a best-effort repair failed.
                break
            choice = response.choices[0]
            msg = choice.message
            segment = (msg.content or "").strip()
            continuation_reason = _choice_finish_reason(choice)
            if not segment:
                break
            combined = _join_continuation(combined, segment)
            continuation_messages.append(_assistant_message_to_dict(msg))
            if not _looks_truncated(segment, continuation_reason):
                break
        return combined.strip(), continuation_reason

    def retrieve(self, query: str, deadline: float | None = None) -> dict[str, Any]:
        tool_events: list[ToolEvent] = []
        evidence_by_uid: dict[str, dict[str, Any]] = {}
        if deadline is not None:
            setup_budget = (
                self.settings.mcp_request_timeout_sec * 3
                if self._mcp_tools_cache is None
                else 0
            )
            if _remaining_seconds(deadline) <= (
                self.settings.final_answer_reserve_sec + setup_budget
            ):
                return _retrieval_fallback(
                    evidence_by_uid,
                    [],
                    tool_events,
                    "retrieval skipped to preserve the final-answer deadline",
                )
        selected_mcp_tools = _select_mcp_tools(query, self._mcp_tools())
        if not selected_mcp_tools:
            return _retrieval_fallback(
                evidence_by_uid,
                [],
                tool_events,
                "no matching evidence source is available",
            )
        tools = [mcp_tool_to_openai_tool(tool) for tool in selected_mcp_tools] + [FINALIZE_RETRIEVAL_TOOL]
        valid_tool_names = {tool["function"]["name"] for tool in tools}

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {"role": "user", "content": retrieval_user_prompt(query)},
        ]

        for _ in range(self.settings.retrieval_max_turns):
            if deadline is not None and _remaining_seconds(deadline) <= (
                self.settings.final_answer_reserve_sec + 1
            ):
                break
            response = self.chat.complete(
                messages,
                tools=tools,
                max_tokens=self.settings.max_retrieval_tokens,
                timeout_sec=(
                    _stage_timeout(
                        deadline,
                        self.settings.retrieval_model_timeout_sec,
                        reserve_sec=self.settings.final_answer_reserve_sec,
                    )
                    if deadline is not None
                    else self.settings.retrieval_model_timeout_sec
                ),
            )
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
                    selection = _validated_selection(selection, evidence_by_uid)
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
                        tool_timeout = self.settings.mcp_tool_timeout_sec
                        if deadline is not None:
                            tool_timeout = _stage_timeout(
                                deadline,
                                self.settings.mcp_tool_timeout_sec,
                                reserve_sec=self.settings.final_answer_reserve_sec,
                            )
                        content = self.mcp.call_tool(name, args, timeout_sec=tool_timeout)
                        content = _clip(content, self.settings.max_tool_result_chars)
                        _harvest_cite_uids(content, name, args, evidence_by_uid)
                        tool_events.append(ToolEvent(phase="retrieval", name=name, arguments=args, result=content))
                    except Exception as exc:
                        content = f"[ERROR] {type(exc).__name__}: {exc}"
                        tool_events.append(ToolEvent(phase="retrieval", name=name, arguments=args, result=content, error=content))
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

        return _retrieval_fallback(
            evidence_by_uid,
            messages,
            tool_events,
            "retrieval tool-call time budget exhausted",
        )

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
        ("법령", "법률", "의료법", "국민건강보험법", "korean law"),
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
        (
            "가이드라인",
            "진료지침",
            "권고",
            "guideline",
            "guidelines",
            "protocol",
            "protocols",
            "clinical guidance",
            "consensus statement",
            "목표치",
        ),
        (
            "rag_get_all_data_sources",
            "rag_vector_query",
            "index_list_documents",
            "index_get_relevant_nodes",
            "index_get_page_content",
            "index_get_document_structure",
            "index_keyword_search",
        ),
    ),
    (
        ("의약품", "약물", "복용", "처방", "위고비", "부작용", "이상반응", "용량", "금기", "상호작용", "성분", "drug", "medication", "dose", "dosage", "contraindication", "interaction", "interactions", "interact"),
        (
            "adr_retrieve_drug_info",
        ),
    ),
    (
        ("식약처", "mfds", "국내 허가", "한국 허가"),
        (
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
    ordered_names: list[str] = []
    for keywords, names in _TOOL_ROUTES:
        if any(_keyword_matches(query, keyword) for keyword in keywords):
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
                "content": _citation_context(content, uid),
            },
        )


def _citation_context(content: str, cite_uid: str, max_chars: int = 2_000) -> str:
    """Keep the structured record containing a citation instead of a blind prefix."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    def find_record(value: Any) -> Any | None:
        if isinstance(value, dict):
            if str(value.get("cite_uid", "")) == cite_uid:
                return value
            for child in value.values():
                match = find_record(child)
                if match is not None:
                    return match
        elif isinstance(value, list):
            for child in value:
                match = find_record(child)
                if match is not None:
                    return match
        return None

    record = find_record(parsed)
    if record is not None:
        return _clip(json.dumps(record, ensure_ascii=False), max_chars)

    marker = content.find(cite_uid)
    if marker < 0 or len(content) <= max_chars:
        return _clip(content, max_chars)
    start = max(0, marker - max_chars // 3)
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "... [context before omitted]\n" if start else ""
    suffix = "\n... [context after omitted]" if end < len(content) else ""
    return prefix + content[start:end] + suffix


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


def _validated_selection(
    selection: CitationSelection, evidence_by_uid: dict[str, dict[str, Any]]
) -> CitationSelection:
    """Drop citation identifiers that were not present in actual MCP output."""
    valid_items = [
        item for item in selection.items if item.cite_uid in evidence_by_uid
    ][:4]
    if len(valid_items) == len(selection.items) and len(valid_items) <= 4:
        return selection
    missing_count = len(selection.items) - len(valid_items)
    note = selection.note.strip()
    if missing_count:
        note = (note + " " if note else "") + (
            f"Ignored {missing_count} citation id(s) not found in tool output."
        )
    status = selection.status
    if not valid_items:
        status = "no_evidence"
    elif status == "sufficient":
        status = "partial"
    return finalize_retrieval(status=status, items=valid_items, note=note)


def _format_retrieval_for_generation(selection: CitationSelection, evidence: list[dict[str, Any]]) -> str:
    numbered_evidence = [
        {"citation_index": index, **item} for index, item in enumerate(evidence, start=1)
    ]
    payload = {
        "status": selection.status,
        "note": selection.note,
        "answer_constraint": (
            "No verified current evidence was found in the provided source set. Do not mention retrieval or tool failure. Do not attribute any statement to a current or local guideline, and do not invent an exact schedule, price, policy, facility, product, contact, source, or citation. Give stable medical knowledge when confident, label it as general rather than current local policy, preserve decision-changing conditions, and ask only the highest-yield missing context."
            if selection.status == "no_evidence"
            else "First verify that each item matches the requested issuer, jurisdiction, population, and task. compact_evidence_memory is a lossy index, not independent evidence: verify decision-critical claims against the matching raw evidence and ignore a memory that conflicts with it. Ignore and do not summarize mismatched material unless it is explicitly useful; a nonlocal study is not a substitute for a requested local guideline. Use only evidence that directly addresses the question. If no item actually matches, follow the no-evidence policy: give stable general knowledge when confident, label the exact current or local rule as unresolved, and ask only decision-changing context."
        ),
        "compact_evidence_memory": [item.model_dump() for item in selection.items],
        "evidence": numbered_evidence if selection.items else [],
        "uncited_fallback_evidence": evidence if not selection.items else [],
    }
    return _clip(json.dumps(payload, ensure_ascii=False, indent=2), 10_000)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


_RETRIEVAL_PATTERNS = (
    r"\b(?:latest|most recent|up-to-date|updated|new|current|official)\s+(?:clinical\s+)?(?:guidelines?|guidance|recommendations?|evidence|labels?|protocols?|consensus statements?|schedules?)\b",
    r"\b(?:local|national|country-specific|russian|korean|japanese|canadian|australian|uk|u\.s\.|us)\s+(?:clinical\s+)?(?:guidelines?|protocols?|policy|schedule)\b",
    r"\b(?:cdc|who|fda|nice|esc|acog|aap|uspstf)\s+(?:catch-up\s+)?(?:guidelines?|guidance|recommendations?|protocols?|schedule|table|label)\b",
    r"\b(?:as of \d{4}|guideline updates?|new evidence)\b",
    r"\b(?:according to|per|under)\s+(?:(?:current|latest|official|standard)\s+)?(?:clinical\s+)?guidelines?\b",
    r"\b(?:please\s+)?(?:cite|provide|include|list|show|give)\s+(?:the\s+)?(?:sources?|citations?|references?|supporting evidence)\b",
    r"\b(?:drug labels?|prescribing information|dosing instructions?|drug interactions?)\b",
    r"\b(?:check|compare)\s+[a-z0-9][a-z0-9-]*\s+with\s+[a-z0-9][a-z0-9-]*\b",
    r"\b(?:current|latest|official)\s+(?:law|regulation|reimbursement|billing|coding)\b",
    r"(?:최신|최근|현행|공식|개정)\s*(?:가이드라인|진료지침|권고|근거|허가사항|용법|용량|법령|급여 기준|보험 기준)",
    r"(?:출처|인용|참고문헌|근거 문헌)(?:를|을)?\s*(?:제시|포함|알려|달라|주세요)",
    r"(?:약물 상호작용|심평원|약가|수가|질병코드|상병코드|청구코드|한국 법령|의료법)",
)

_EDITING_REQUEST = re.compile(
    r"\b(rewrite|reword|proofread|translate|summarize|summarise|condense|polish)\b|"
    r"(다시 써|고쳐 써|번역|요약|줄여|다듬어)",
    flags=re.IGNORECASE,
)
_EXPLICIT_EVIDENCE_SUBREQUEST = re.compile(
    r"\b(new evidence|find evidence|look up|fact[- ]?check|verify|cite|citations?|"
    r"provide (?:the )?(?:sources?|references?)|what do (?:the )?(?:current|latest|official)|"
    r"(?:according to|per|under) (?:current |latest |official |standard )?(?:clinical )?guidelines?)\b|"
    r"(최신 근거|근거를 찾아|사실 확인|검증해|출처를|인용해)",
    flags=re.IGNORECASE,
)
_FOLLOWUP_REFERENCE = re.compile(
    r"\b(this|that|it|those|above|earlier|same|scenario|what about|so for)\b|"
    r"(그 약|그 검사|그 경우|아까|위 내용|같은 경우)",
    flags=re.IGNORECASE,
)


_IMPROVISED_TERMS = re.compile(
    r"\b(improvised|makeshift|home[- ]?made|diy|jury[- ]rig(?:ged)?)\b|"
    r"(즉석|임시|자가 제작|수제)",
    flags=re.IGNORECASE,
)
_HIGH_RISK_MEDICAL_ACTION = re.compile(
    r"\b(invasive (?:procedure|treatment)|life[- ]?support|medical (?:device|circuit)|"
    r"dialysis|hemodialysis|peritoneal dialysis|ecmo|ventilat(?:or|ion)|intubat(?:e|ion)|"
    r"central (?:venous )?line|blood transfusion|infusion pump|defibrillat(?:or|ion)|"
    r"anesthe(?:sia|tic))\b|"
    r"(침습(?:적)? (?:시술|처치)|생명 ?유지|의료 (?:장비|회로)|투석|에크모|인공호흡기|"
    r"삽관|중심정맥관|수혈|주입 ?펌프|제세동기|마취)",
    flags=re.IGNORECASE,
)
_OPERATIONAL_BUILD_CONTEXT = re.compile(
    r"\b(how (?:can|do|to)|steps?|instructions?|build|assemble|construct|modify|operate|"
    r"set ?up|replace|substitute|convert|parts?|components?|circuit|settings?)\b|"
    r"(방법|단계|절차|만들|조립|제작|개조|작동|설정|대체|부품|회로)",
    flags=re.IGNORECASE,
)
_NON_OPERATIONAL_HIGH_RISK_CONTEXT = re.compile(
    r"\b(fiction|fictional|novel|screenplay|story|training|planning|simulation|"
    r"translate|translation|rewrite|summarize|explain why|why .* unsafe)\b|"
    r"(소설|각본|창작|훈련|계획|시뮬레이션|번역|다시 써|요약|왜 위험)",
    flags=re.IGNORECASE,
)


def _high_risk_improvised_response(
    question: str,
    history: list[dict[str, str]] | None = None,
    *,
    max_tokens: int | None = None,
) -> str | None:
    """Block operational requests to improvise dangerous medical care or equipment."""
    if max_tokens is not None and max_tokens < 96:
        return None
    if not (
        _IMPROVISED_TERMS.search(question)
        and _HIGH_RISK_MEDICAL_ACTION.search(question)
        and (
            _OPERATIONAL_BUILD_CONTEXT.search(question)
            or len(re.findall(r"\b\w+\b", question)) <= 12
        )
    ):
        return None
    intent_context = "\n".join(
        str(item.get("content", ""))
        for item in (history or [])[-4:]
        if item.get("role") in {"system", "developer", "user"}
    )
    if _NON_OPERATIONAL_HIGH_RISK_CONTEXT.search(f"{intent_context}\n{question}"):
        return None

    is_korean = re.search(r"[가-힣]", question) is not None
    if is_korean:
        return (
            "실제 상황이라면 생명을 위협할 수 있습니다. 직접 제작·개조한 침습 처치나 "
            "생명유지 장비를 만들거나 작동하지 말고, 즉시 지역 응급의료 서비스와 "
            "의료책임자에게 연락하세요. 오지라면 확립된 원격의료·후송 체계를 가동하세요.\n\n"
            "설계·조립·설정 단계는 안내할 수 없습니다. 비검증 부품이나 재료를 쓰면 "
            "상황에 따라 공기색전, 출혈, 감염, 조직 손상, 전기·압력·산소·용량 오류와 "
            "사망을 일으킬 수 있습니다. 구조를 기다리는 동안에는 훈련받은 의료진의 "
            "기존 프로토콜과 응급요원의 지시만 따르세요.\n\n"
            "현재 진행 중인 응급상황인가요, 아니면 창작·훈련·계획을 위한 질문인가요?"
        )

    return (
        "If this is real, it may be life-threatening. Do not build, modify, or operate an "
        "improvised invasive treatment or life-support device. Contact local emergency medical "
        "services and the responsible medical lead now; in a remote setting, activate the "
        "established telemedicine and evacuation pathway.\n\n"
        "I cannot provide design, assembly, or settings instructions. Unvalidated components or "
        "materials can cause, depending on the procedure, air embolism, bleeding, infection, "
        "tissue injury, electrical, pressure, oxygen, or dosing errors, and death. While waiting "
        "for rescue, follow only an established protocol directed by trained medical staff.\n\n"
        "Is this an active emergency, or a fiction, training, or planning scenario?"
    )


_LOCAL_REMEDY_QUERY = re.compile(
    r"\b(local|traditional|folk|indigenous)(?:\s+[a-z-]+){0,3}\s+remed(?:y|ies)\b|"
    r"(현지 (?:치료법|민간요법|전통요법)|전통요법|민간요법)",
    flags=re.IGNORECASE,
)
_ALTITUDE_TRAVEL_QUERY = re.compile(
    r"\b(altitude|high altitude|mountain sickness|acute mountain sickness|"
    r"cusco|la paz|himalaya|andes|ascent|hace|hape)\b|"
    r"(고산|고도병|산악병|쿠스코|라파스|등반)",
    flags=re.IGNORECASE,
)


def _generation_system_prompt(
    history: list[dict[str, str]] | None,
) -> str:
    """Use the compact core policy only when no conversational memory is needed."""
    has_conversation_history = any(
        item.get("role") in {"user", "assistant"}
        and bool(str(item.get("content", "")).strip())
        for item in history or []
    )
    return (
        GENERATION_SYSTEM_PROMPT
        if has_conversation_history
        else SIMPLE_GENERATION_SYSTEM_PROMPT
    )


def _request_specific_policy(question: str) -> str:
    policies: list[str] = []
    if _LOCAL_REMEDY_QUERY.search(question):
        policies.append(
            "This is a local or traditional remedy request. Start with proven measures and "
            "urgent red flags. Ask only the most important context that changes safety. "
            "Describe a cultural practice as common use, not as an effective treatment, unless "
            "good evidence supports the benefit. Explicitly state limited or absent evidence "
            "and material risks, interactions, legal or drug-testing implications when relevant. "
            "Do not invent local products, availability, physiology, or benefit claims."
        )
    if _ALTITUDE_TRAVEL_QUERY.search(question):
        policies.append(
            "This is an altitude-related request. Distinguish mild acute mountain sickness "
            "from cerebral or pulmonary red flags. Ask about current symptoms, altitude, speed "
            "of ascent, cardiopulmonary history, medications, and prior episodes only as they "
            "change safety. State when to stop ascent, descend, use available oxygen or standard "
            "care, and seek urgent evacuation."
        )
    return "\n".join(policies)


def _should_offer_retrieval(
    question: str, history: list[dict[str, str]] | None
) -> bool:
    if _EDITING_REQUEST.search(question) and not _EXPLICIT_EVIDENCE_SUBREQUEST.search(
        question
    ):
        return False
    if any(
        re.search(pattern, question, flags=re.IGNORECASE)
        for pattern in _RETRIEVAL_PATTERNS
    ):
        return True
    if not _FOLLOWUP_REFERENCE.search(question):
        return False
    recent_user_context = "\n".join(
        str(item.get("content", ""))
        for item in (history or [])[-8:]
        if item.get("role") == "user"
    )
    text = f"{recent_user_context}\n{question}"
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in _RETRIEVAL_PATTERNS
    )


def _keyword_matches(query: str, keyword: str) -> bool:
    lowered_query = query.casefold()
    lowered_keyword = keyword.casefold()
    if lowered_keyword.isascii():
        return re.search(rf"\b{re.escape(lowered_keyword)}\b", lowered_query) is not None
    return lowered_keyword in lowered_query


def _bounded_completion_tokens(requested: int | None, maximum: int) -> int:
    if requested is None:
        return maximum
    return max(1, min(int(requested), maximum))


def _remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _stage_timeout(
    deadline: float, cap_sec: float, *, reserve_sec: float = 0
) -> float:
    available = _remaining_seconds(deadline) - reserve_sec
    if available < 1:
        raise TimeoutError("request deadline exhausted before the next model/tool stage")
    return max(1.0, min(float(cap_sec), available))


def _retrieval_fallback(
    evidence_by_uid: dict[str, dict[str, Any]],
    messages: list[dict[str, Any]],
    tool_events: list[ToolEvent],
    note: str,
) -> dict[str, Any]:
    non_evidence_tools = {
        "rag_get_all_data_sources",
        "rag_get_data_source_detail",
        "index_list_documents",
        "index_get_document_structure",
    }
    fallback_items = [
        {"cite_uid": uid, "relevance_score": 0.5}
        for uid, item in evidence_by_uid.items()
        if item.get("tool") not in non_evidence_tools
    ][:4]
    selection = finalize_retrieval(
        status="partial" if fallback_items else "no_evidence",
        items=fallback_items,
        note=note,
    )
    return {
        "selection": selection,
        "evidence": _select_evidence(selection, evidence_by_uid),
        "messages": messages,
        "tool_events": tool_events,
    }


def _choice_finish_reason(choice: Any) -> str:
    reason = getattr(choice, "finish_reason", None)
    return str(reason) if reason else "stop"


def _looks_truncated(content: str, finish_reason: str) -> bool:
    if finish_reason == "length":
        return True
    stripped = content.rstrip()
    return bool(stripped) and stripped.endswith(("-", "–", "—", ":", "(", "[", "/"))


def _join_continuation(partial: str, continuation: str) -> str:
    if not partial:
        return continuation
    if partial.endswith((" ", "\n", "\t", "(", "[", "/")):
        separator = ""
    elif partial.endswith(("–", "—")):
        separator = " "
    elif partial.endswith("-"):
        separator = " " if len(partial) > 1 and partial[-2].isspace() else ""
    elif continuation.startswith((".", ",", ";", ":", "!", "?", ")", "]")):
        separator = ""
    else:
        separator = "\n"
    return partial + separator + continuation


def _compact_history(
    history: list[dict[str, str]] | None,
    *,
    max_chars: int = 14_000,
    max_item_chars: int = 4_000,
) -> list[dict[str, str]]:
    """Preserve complete evaluation history whenever it fits, including long chats."""
    filtered = [
        {
            "role": str(item["role"]),
            "content": _clip(str(item.get("content", "")), max_item_chars),
        }
        for item in history or []
        if item.get("role") in {"user", "assistant"}
    ]
    if sum(len(item["content"]) for item in filtered) <= max_chars:
        return filtered
    if not filtered:
        return []

    first = filtered[0]
    budget = max(0, max_chars - len(first["content"]))
    recent: list[dict[str, str]] = []
    for item in reversed(filtered[1:]):
        if len(item["content"]) > budget:
            continue
        recent.append(item)
        budget -= len(item["content"])
    return [first, *reversed(recent)]
