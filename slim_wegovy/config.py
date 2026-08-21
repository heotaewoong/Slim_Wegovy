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
    retrieval_max_turns: int = 6
    generation_max_turns: int = 3
    max_tool_result_chars: int = 3_500
    max_completion_tokens: int = 1_024


def load_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")

    api_url = os.environ.get("LUNIT_FM_API_URL", "https://model.hackathon.lunit.io").rstrip("/")
    api_key = os.environ.get("LUNIT_FM_API_KEY", "")
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
