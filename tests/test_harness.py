import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from slim_wegovy.config import Settings
from slim_wegovy.harness import L2Harness, _format_retrieval_for_generation, _select_mcp_tools
from slim_wegovy.openai_compat import LunitChatClient
from slim_wegovy.prompts import GENERATION_SYSTEM_PROMPT
from slim_wegovy.schemas import CitationSelection


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


def chat_response(content, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


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
        self.assertEqual(call.kwargs["max_tokens"], 1024)
        self.assertEqual(settings.retrieval_max_turns, 4)
        self.assertEqual(settings.generation_max_turns, 2)
        self.assertEqual(settings.lunit_timeout_sec, 60)
        self.assertEqual(settings.lunit_max_retries, 1)

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

    def test_drug_query_routes_to_small_relevant_tool_set(self):
        available = [
            {"name": "adr_retrieve_drug_info"},
            {"name": "openapi_mfds_get_drug_indication"},
            {"name": "openapi_law_search"},
            {"name": "rag_vector_query"},
        ]
        selected = _select_mcp_tools("위고비의 공식 허가사항상 이상반응", available)
        names = [tool["name"] for tool in selected]
        self.assertEqual(names, ["adr_retrieve_drug_info", "openapi_mfds_get_drug_indication"])
        self.assertNotIn("openapi_law_search", names)

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
