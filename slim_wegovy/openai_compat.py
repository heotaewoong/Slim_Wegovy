from __future__ import annotations

import time
from typing import Any

import openai

from slim_wegovy.config import Settings


class LunitChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = openai.OpenAI(
            api_key=settings.lunit_api_key,
            base_url=f"{settings.lunit_api_url}/v1",
            max_retries=0,
            timeout=120,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        max_retries: int = 2,
    ) -> Any:
        params: dict[str, Any] = {
            "model": self.settings.lunit_model,
            "messages": messages,
            "max_tokens": self.settings.max_completion_tokens,
        }
        if tools is not None:
            params["tools"] = tools
            params["tool_choice"] = tool_choice or "auto"
            # L2 has a deliberately small input window. Serial tool calls keep
            # multiple large MCP responses from entering one follow-up turn.
            params["parallel_tool_calls"] = False

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.client.chat.completions.create(**params)
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                time.sleep(1.5 * (2**attempt))
        raise last_exc  # type: ignore[misc]
