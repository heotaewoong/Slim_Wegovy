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
            timeout=settings.lunit_timeout_sec,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        max_retries: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_sec: float | None = None,
    ) -> Any:
        if max_retries is None:
            max_retries = self.settings.lunit_max_retries
        if max_tokens is None:
            max_tokens = self.settings.max_completion_tokens
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        params: dict[str, Any] = {
            "model": self.settings.lunit_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if timeout_sec is not None:
            params["timeout"] = max(1.0, float(timeout_sec))
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
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.RateLimitError,
            ) as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                time.sleep(1.5 * (2**attempt))
        raise last_exc  # type: ignore[misc]
