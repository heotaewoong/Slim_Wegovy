import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from slim_wegovy.web import ChatCompletionRequest, app, chat_completions
from slim_wegovy.config import load_settings


class SubmissionApiTests(unittest.TestCase):
    def test_health_and_models_start_without_api_key(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json()["status"], "ok")
            response = client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "Lunit/L2-preview")

    def test_chat_completion_preserves_multiturn_history(self):
        captured = {}

        class FakeHarness:
            def answer(self, question, history, **kwargs):
                captured["question"] = question
                captured["history"] = history
                captured.update(kwargs)
                return SimpleNamespace(answer="17입니다.", finish_reason="stop")

        messages = [
            {"role": "user", "content": "17을 기억해."},
            {"role": "assistant", "content": "기억하겠습니다."},
            {"role": "user", "content": "그 숫자가 뭐였지?"},
        ]
        with patch("slim_wegovy.web._harness", return_value=FakeHarness()):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={"messages": messages, "max_tokens": 6144, "temperature": 0},
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "17입니다.")
        self.assertEqual(response.json()["usage"]["total_tokens"], 0)
        self.assertEqual(captured["question"], messages[-1]["content"])
        self.assertEqual(captured["history"], messages[:-1])
        self.assertEqual(captured["max_tokens"], 6144)
        self.assertEqual(captured["temperature"], 0.0)

    def test_streaming_wraps_final_l2_answer(self):
        fake = SimpleNamespace(
            answer=lambda question, history, **kwargs: SimpleNamespace(
                answer="안녕하세요.", finish_reason="length"
            )
        )
        with patch("slim_wegovy.web._harness", return_value=fake):
            with TestClient(app) as client:
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "안녕"}], "stream": True},
                ) as response:
                    content = "".join(response.iter_text())
        self.assertEqual(response.status_code, 200)
        self.assertIn("안녕하세요.", content)
        self.assertIn('"finish_reason": "length"', content)
        self.assertIn('"usage"', content)
        self.assertIn("data: [DONE]", content)

    def test_accepts_common_evaluation_api_key_alias(self):
        with patch.dict(
            os.environ,
            {"LUNIT_FM_API_KEY": "", "LUNIT_API_KEY": "", "OPENAI_API_KEY": "alias-key"},
            clear=False,
        ):
            settings = load_settings()
        self.assertEqual(settings.lunit_api_key, "alias-key")

    def test_uses_inbound_bearer_when_environment_key_is_missing(self):
        fake = SimpleNamespace(
            answer=lambda question, history, **kwargs: SimpleNamespace(
                answer="ok", finish_reason="stop"
            )
        )
        with patch("slim_wegovy.web._harness", side_effect=RuntimeError("Missing required environment variable(s): LUNIT_FM_API_KEY")):
            with patch("slim_wegovy.web._harness_with_key", return_value=fake) as keyed:
                with TestClient(app) as client:
                    response = client.post(
                        "/v1/chat/completions",
                        headers={"Authorization": "Bearer injected-key"},
                        json={"messages": [{"role": "user", "content": "안녕"}]},
                    )
        self.assertEqual(response.status_code, 200)
        keyed.assert_called_once_with("injected-key")

    def test_chat_completions_are_not_globally_serialized(self):
        state_lock = threading.Lock()
        start_barrier = threading.Barrier(4)
        state = {"active": 0, "peak": 0}

        class ConcurrentHarness:
            def answer(self, question, history, **kwargs):
                start_barrier.wait(timeout=2)
                with state_lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.05)
                with state_lock:
                    state["active"] -= 1
                return SimpleNamespace(answer="ok", finish_reason="stop")

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "동시 요청"}]
        )
        with patch("slim_wegovy.web._request_harness", return_value=ConcurrentHarness()):
            with ThreadPoolExecutor(max_workers=4) as executor:
                responses = list(
                    executor.map(lambda _: chat_completions(request), range(4))
                )

        self.assertEqual(len(responses), 4)
        self.assertEqual(state["peak"], 4)

    def test_rejects_empty_messages(self):
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")


if __name__ == "__main__":
    unittest.main()
