from __future__ import annotations

import time
from typing import Any

import openai

from slim_wegovy.config import Settings


class PatientSimulator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = openai.OpenAI(
            api_key=settings.lunit_api_key,
            base_url=f"{settings.patient_api_url}/v1",
            max_retries=0,
            timeout=60,
        )

    def next_question(self, messages: list[dict[str, Any]]) -> str:
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.patient_model,
                    messages=messages,
                )
                return response.choices[0].message.content or ""
            except openai.NotFoundError:
                response = self.client.chat.completions.create(
                    model=self.settings.patient_model,
                    messages=[],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                if "502" not in str(exc) or attempt == 3:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise last_exc  # type: ignore[misc]
