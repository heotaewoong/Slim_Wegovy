from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    lunit_api_url: str
    lunit_api_key: str
    lunit_model: str
    mcp_url: str
    patient_api_url: str = "https://patient.hackathon.lunit.io"
    patient_model: str = "patient-simulator-ko"
    mcp_protocol_version: str = "2025-06-18"
    # One evidence call followed by a finalize call keeps current-fact lookups
    # useful without consuming CoEval's 180-second request budget.
    retrieval_max_turns: int = 2
    generation_max_turns: int = 2
    max_tool_result_chars: int = 3_500
    # CoEval's Conquer splits request 6,144 tokens because L2 reasoning tokens
    # share the same budget. A smaller internal cap silently truncates otherwise
    # valid HealthBench answers and the evaluator grades that partial text.
    max_completion_tokens: int = 6_144
    max_retrieval_tokens: int = 2_048
    max_continuation_tokens: int = 2_048
    max_continuations: int = 1
    # Leave response-serialization headroom inside CoEval's 180-second timeout.
    request_deadline_sec: int = 165
    retrieval_model_timeout_sec: int = 45
    final_answer_reserve_sec: int = 55
    lunit_timeout_sec: int = 120
    # CoEval already retries candidate inference. Nested long retries can exceed
    # its 180-second request timeout and amplify load under 16-way concurrency.
    lunit_max_retries: int = 0
    mcp_tool_timeout_sec: int = 30
    mcp_request_timeout_sec: int = 15


def load_settings(api_key_override: str | None = None) -> Settings:
    load_dotenv(ROOT_DIR / ".env")

    api_url = os.environ.get("LUNIT_FM_API_URL", "https://model.hackathon.lunit.io").rstrip("/")
    api_key = (
        api_key_override
        or os.environ.get("LUNIT_FM_API_KEY", "")
        or os.environ.get("LUNIT_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    model = os.environ.get("LUNIT_FM_MODEL", "Lunit/L2-preview")
    mcp_url = os.environ.get("LUNIT_MCP_URL", "https://mcp.hackathon.lunit.io/mcp")
    mcp_protocol_version = os.environ.get("MCP_PROTOCOL_VERSION", "2025-06-18")

    missing = [
        name
        for name, value in (
            ("LUNIT_FM_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    return Settings(
        lunit_api_url=api_url,
        lunit_api_key=api_key,
        lunit_model=model,
        mcp_url=mcp_url,
        mcp_protocol_version=mcp_protocol_version,
    )
