import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock

from slim_wegovy.config import Settings
from slim_wegovy.harness import (
    L2Harness,
    _compact_history,
    _format_retrieval_for_generation,
    _select_mcp_tools,
    _should_offer_retrieval,
    _request_specific_policy,
    _validated_selection,
)
from slim_wegovy.openai_compat import LunitChatClient
from slim_wegovy.prompts import GENERATION_SYSTEM_PROMPT
from slim_wegovy.schemas import CitationSelection, finalize_retrieval


class FakeToolCall:
    def __init__(self, name="unexpected_tool", arguments="{}"):
        self.id = "call-1"
        self.function = SimpleNamespace(name=name, arguments=arguments)

    def model_dump(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


def chat_response(content, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


class HarnessTests(unittest.TestCase):
    def test_l2_client_disables_parallel_tool_calls(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        client = LunitChatClient(settings)
        client.client.chat.completions.create = Mock(return_value=chat_response("ok"))

        client.complete([{"role": "user", "content": "질문"}], tools=[{"type": "function"}])

        call = client.client.chat.completions.create.call_args
        self.assertFalse(call.kwargs["parallel_tool_calls"])
        self.assertEqual(call.kwargs["max_tokens"], 6144)
        self.assertEqual(settings.retrieval_max_turns, 2)
        self.assertEqual(settings.generation_max_turns, 2)
        self.assertEqual(settings.max_retrieval_tokens, 2048)
        self.assertEqual(settings.request_deadline_sec, 165)
        self.assertEqual(settings.retrieval_model_timeout_sec, 45)
        self.assertEqual(settings.final_answer_reserve_sec, 55)
        self.assertEqual(settings.lunit_timeout_sec, 120)
        self.assertEqual(settings.lunit_max_retries, 0)

    def test_l2_client_honors_stage_overrides(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        client = LunitChatClient(settings)
        client.client.chat.completions.create = Mock(return_value=chat_response("ok"))

        client.complete(
            [{"role": "user", "content": "question"}],
            max_tokens=777,
            temperature=0.0,
            timeout_sec=12,
        )

        call = client.client.chat.completions.create.call_args
        self.assertEqual(call.kwargs["max_tokens"], 777)
        self.assertEqual(call.kwargs["temperature"], 0.0)
        self.assertEqual(call.kwargs["timeout"], 12.0)

    def test_generation_prompt_is_language_adaptive_and_complete(self):
        self.assertIn("language used by the latest user message", GENERATION_SYSTEM_PROMPT)
        self.assertIn("Address every part", GENERATION_SYSTEM_PROMPT)
        self.assertIn("warning signs and timeframe", GENERATION_SYSTEM_PROMPT)
        self.assertIn("Calibrate uncertainty", GENERATION_SYSTEM_PROMPT)
        self.assertIn("ask a few targeted questions", GENERATION_SYSTEM_PROMPT)
        self.assertIn("no unsupported patient-specific claim", GENERATION_SYSTEM_PROMPT)

    def test_generation_evidence_payload_is_bounded(self):
        selection = CitationSelection(
            status="sufficient",
            items=[{"cite_uid": f"cite-{index}", "relevance_score": 0.9} for index in range(8)],
        )
        evidence = [
            {"cite_uid": f"cite-{index}", "content": "x" * 3000} for index in range(8)
        ]
        rendered = _format_retrieval_for_generation(selection, evidence)
        self.assertLessEqual(len(rendered), 10_040)

    def test_no_evidence_payload_forbids_unverified_local_claims(self):
        rendered = _format_retrieval_for_generation(
            finalize_retrieval("no_evidence", []), []
        )
        self.assertIn("Do not attribute any statement", rendered)
        self.assertIn("do not invent an exact schedule", rendered)

    def test_local_remedy_request_gets_specific_evidence_policy(self):
        policy = _request_specific_policy("altitude sickness in Cusco local remedy")
        self.assertIn("Start with proven measures", policy)
        self.assertIn("Do not invent local products", policy)
        self.assertIn("when to stop ascent", policy)

    def test_non_altitude_remedy_does_not_get_altitude_instructions(self):
        policy = _request_specific_policy("traditional ginger remedy for nausea")
        self.assertIn("local or traditional remedy", policy)
        self.assertNotIn("stop ascent", policy)

    def test_generation_prompt_records_requested_medical_ai_experience(self):
        self.assertIn("more than 10 years of experience", GENERATION_SYSTEM_PROMPT)
        self.assertIn("give the safest useful conditional guidance first", GENERATION_SYSTEM_PROMPT)

    def test_drug_query_routes_to_small_relevant_tool_set(self):
        available = [
            {"name": "adr_retrieve_drug_info"},
            {"name": "openapi_mfds_get_drug_indication"},
            {"name": "openapi_law_search"},
            {"name": "rag_vector_query"},
        ]
        selected = _select_mcp_tools("위고비의 공식 허가사항상 이상반응", available)
        names = [tool["name"] for tool in selected]
        self.assertEqual(names, ["adr_retrieve_drug_info"])
        self.assertNotIn("openapi_law_search", names)

    def test_korean_regulator_query_adds_mfds_tool(self):
        available = [
            {"name": "adr_retrieve_drug_info"},
            {"name": "openapi_mfds_get_drug_indication"},
            {"name": "openapi_law_search"},
        ]
        selected = _select_mcp_tools("식약처 위고비 허가사항", available)
        self.assertEqual(
            [tool["name"] for tool in selected],
            ["adr_retrieve_drug_info", "openapi_mfds_get_drug_indication"],
        )

    def test_router_does_not_treat_method_or_booking_as_law_or_drug(self):
        available = [
            {"name": "openapi_law_search"},
            {"name": "adr_retrieve_drug_info"},
            {"name": "rag_vector_query"},
        ]
        selected = _select_mcp_tools("병원 예약 방법을 알려줘", available)
        names = [tool["name"] for tool in selected]
        self.assertEqual(names, ["rag_vector_query"])

    def test_generic_or_emergency_question_does_not_offer_retrieval(self):
        self.assertFalse(_should_offer_retrieval("How can CBT help with anxiety?", []))
        self.assertFalse(
            _should_offer_retrieval(
                "I have chest pressure and I am short of breath. What should I do?",
                [{"role": "user", "content": "No recent travel."}],
            )
        )
        self.assertTrue(
            _should_offer_retrieval(
                "Please cite the latest guideline for this treatment.", []
            )
        )
        self.assertTrue(
            _should_offer_retrieval(
                "I live in Moscow and want the local Russian guidelines.", []
            )
        )
        self.assertFalse(
            _should_offer_retrieval("Altitude sickness in Cusco local remedy", [])
        )
        self.assertFalse(
            _should_offer_retrieval("How much is the IUD out of pocket?", [])
        )
        self.assertTrue(
            _should_offer_retrieval(
                "Rewrite this and say ultrasound is first per guidelines.", []
            )
        )
        self.assertTrue(
            _should_offer_retrieval(
                "Rewrite this, and also find new evidence on tumor markers.", []
            )
        )

    def test_plural_guideline_query_routes_to_generic_evidence_tools(self):
        available = [
            {"name": "rag_get_all_data_sources"},
            {"name": "rag_vector_query"},
            {"name": "index_list_documents"},
            {"name": "openapi_mfds_get_drug_indication"},
        ]
        selected = _select_mcp_tools("local Russian clinical guidelines", available)
        names = [tool["name"] for tool in selected]
        self.assertEqual(
            names,
            ["rag_get_all_data_sources", "rag_vector_query", "index_list_documents"],
        )
        self.assertNotIn("openapi_mfds_get_drug_indication", names)

    def test_retrieval_budget_preserves_collected_citations(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
            retrieval_max_turns=1,
        )
        harness = L2Harness(settings)
        harness._mcp_tools_cache = [
            {
                "name": "adr_retrieve_drug_info",
                "description": "drug info",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        harness.chat.complete = Mock(
            return_value=chat_response(
                None,
                [FakeToolCall("adr_retrieve_drug_info", '{"query":"위고비"}')],
            )
        )
        harness.mcp.call_tool = Mock(
            return_value='{"cite_uid":"cite-drug-1","content":"근거"}'
        )

        retrieval = harness.retrieve("위고비 이상반응")

        self.assertEqual(retrieval["selection"].status, "partial")
        self.assertEqual(retrieval["selection"].items[0].cite_uid, "cite-drug-1")
        self.assertEqual(retrieval["evidence"][0]["cite_uid"], "cite-drug-1")

    def test_tool_budget_fallback_is_still_generated_by_l2(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
            generation_max_turns=1,
        )
        harness = L2Harness(settings)
        harness.chat.complete = Mock(side_effect=[
            chat_response(None, [FakeToolCall()]),
            chat_response("L2가 생성한 최종 답변"),
        ])

        result = harness.answer("질문")

        self.assertEqual(result.answer, "L2가 생성한 최종 답변")
        final_call = harness.chat.complete.call_args_list[-1]
        self.assertIsNone(final_call.kwargs["tools"])

    def test_retrieval_is_offered_once_then_final_generation_is_tool_free(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        harness = L2Harness(settings)
        harness.retrieve = Mock(
            return_value={
                "selection": finalize_retrieval("no_evidence", []),
                "evidence": [],
                "messages": [],
                "tool_events": [],
            }
        )
        harness.chat.complete = Mock(
            side_effect=[
                chat_response(
                    None,
                    [
                        FakeToolCall(
                            "retrieve_relevant_content",
                            '{"query":"latest guideline"}',
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                chat_response("Complete sourced answer."),
            ]
        )

        result = harness.answer("Please cite the latest guideline.")

        self.assertEqual(result.answer, "Complete sourced answer.")
        self.assertIsNotNone(harness.chat.complete.call_args_list[0].kwargs["tools"])
        self.assertIsNone(harness.chat.complete.call_args_list[1].kwargs["tools"])
        harness.retrieve.assert_called_once_with("latest guideline", deadline=ANY)

    def test_retrieval_skips_network_when_final_answer_deadline_is_exhausted(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        harness = L2Harness(settings)
        harness.mcp.list_tools = Mock(side_effect=AssertionError("network should not run"))

        retrieval = harness.retrieve("latest guideline", deadline=0)

        self.assertEqual(retrieval["selection"].status, "no_evidence")
        harness.mcp.list_tools.assert_not_called()

    def test_length_finish_is_continued_without_tools(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        harness = L2Harness(settings)
        harness.chat.complete = Mock(
            side_effect=[
                chat_response("First part –", finish_reason="length"),
                chat_response("remaining complete sentence.", finish_reason="stop"),
            ]
        )

        result = harness.answer("Give me a complete three-month plan.")

        self.assertEqual(result.answer, "First part – remaining complete sentence.")
        self.assertEqual(result.finish_reason, "stop")
        continuation_call = harness.chat.complete.call_args_list[1]
        self.assertIsNone(continuation_call.kwargs["tools"])
        self.assertEqual(continuation_call.kwargs["max_tokens"], 2048)

    def test_failed_continuation_keeps_partial_answer_instead_of_502(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        harness = L2Harness(settings)
        harness.chat.complete = Mock(
            side_effect=[
                chat_response("Useful partial answer.", finish_reason="length"),
                TimeoutError("repair timed out"),
            ]
        )

        result = harness.answer("Give me a complete plan.")

        self.assertEqual(result.answer, "Useful partial answer.")
        self.assertEqual(result.finish_reason, "length")

    def test_improvised_invasive_procedure_uses_deterministic_safety_gate(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        harness = L2Harness(settings)
        harness.chat.complete = Mock(side_effect=AssertionError("model must not run"))

        result = harness.answer("makeshift blood transfusion device")

        self.assertIn("Do not build, modify, or operate", result.answer)
        self.assertIn("emergency medical services", result.answer)
        self.assertTrue(result.answer.endswith("planning scenario?"))
        harness.chat.complete.assert_not_called()

    def test_non_operational_improvised_question_uses_contextual_model_answer(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        harness = L2Harness(settings)
        harness.chat.complete = Mock(return_value=chat_response("Contextual safety explanation."))

        result = harness.answer(
            "For a fictional screenplay, explain why a homemade ventilator circuit is unsafe."
        )

        self.assertEqual(result.answer, "Contextual safety explanation.")
        harness.chat.complete.assert_called_once()

    def test_failed_retrieval_decision_falls_back_to_tool_free_answer(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        harness = L2Harness(settings)
        harness.chat.complete = Mock(
            side_effect=[
                TimeoutError("tool decision timed out"),
                chat_response("Fallback answer without retrieval."),
            ]
        )

        result = harness.answer("Please cite the latest guideline.")

        self.assertEqual(result.answer, "Fallback answer without retrieval.")
        self.assertEqual(
            result.tool_events[0].name, "retrieval_decision_fallback"
        )
        self.assertIsNone(harness.chat.complete.call_args_list[1].kwargs["tools"])

    def test_requested_budget_and_temperature_reach_generation(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        harness = L2Harness(settings)
        harness.chat.complete = Mock(return_value=chat_response("Done."))

        harness.answer("Question", max_tokens=777, temperature=0.0)

        call = harness.chat.complete.call_args
        self.assertEqual(call.kwargs["max_tokens"], 777)
        self.assertEqual(call.kwargs["temperature"], 0.0)

    def test_generation_defaults_to_zero_temperature_for_stability(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        harness = L2Harness(settings)
        harness.chat.complete = Mock(return_value=chat_response("Done."))

        harness.answer("Question")

        self.assertEqual(harness.chat.complete.call_args.kwargs["temperature"], 0.0)

    def test_invalid_citation_ids_are_removed(self):
        selection = finalize_retrieval(
            "sufficient",
            [
                {"cite_uid": "real", "relevance_score": 0.9},
                {"cite_uid": "invented", "relevance_score": 0.8},
            ],
        )
        validated = _validated_selection(
            selection, {"real": {"cite_uid": "real", "content": "evidence"}}
        )
        self.assertEqual([item.cite_uid for item in validated.items], ["real"])
        self.assertEqual(validated.status, "partial")

    def test_compact_history_preserves_public_eval_length_when_short(self):
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(17)
        ]
        self.assertEqual(_compact_history(history), history)

    def test_empty_generation_is_retried_by_l2(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        harness = L2Harness(settings)
        harness.chat.complete = Mock(side_effect=[
            chat_response(""),
            chat_response("재시도로 생성한 완결된 답변"),
        ])

        result = harness.answer("질문")

        self.assertEqual(result.answer, "재시도로 생성한 완결된 답변")
        self.assertIsNone(harness.chat.complete.call_args_list[-1].kwargs["tools"])


if __name__ == "__main__":
    unittest.main()
