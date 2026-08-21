import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from slim_wegovy.web import app
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
            def answer(self, question, history):
                captured["question"] = question
                captured["history"] = history
                return SimpleNamespace(answer="17입니다.")

        messages = [
            {"role": "user", "content": "17을 기억해."},
            {"role": "assistant", "content": "기억하겠습니다."},
            {"role": "user", "content": "그 숫자가 뭐였지?"},
        ]
        with patch("slim_wegovy.web._harness", return_value=FakeHarness()):
            with TestClient(app) as client:
                response = client.post("/v1/chat/completions", json={"messages": messages})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "17입니다.")
        self.assertEqual(response.json()["usage"]["total_tokens"], 0)
        self.assertEqual(captured["question"], messages[-1]["content"])
        self.assertEqual(captured["history"], messages[:-1])

    def test_streaming_wraps_final_l2_answer(self):
        fake = SimpleNamespace(answer=lambda question, history: SimpleNamespace(answer="안녕하세요."))
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
        fake = SimpleNamespace(answer=lambda question, history: SimpleNamespace(answer="ok"))
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

    def test_rejects_empty_messages(self):
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")


if __name__ == "__main__":
    unittest.main()
