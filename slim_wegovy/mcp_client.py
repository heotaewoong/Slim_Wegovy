from __future__ import annotations

import json
import threading
from typing import Any

import requests

from slim_wegovy.config import Settings


class StreamableHttpMcpClient:
    """Small MCP Streamable HTTP JSON-RPC client for the Lunit hackathon server."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._id = 0
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {settings.lunit_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
        )
        self._session_id: str | None = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "slim-wegovy", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized")
        self._initialized = True

    def list_tools(self) -> list[dict[str, Any]]:
        self.initialize()
        result = self._request("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.initialize()
        result = self._request("tools/call", {"name": name, "arguments": arguments}, timeout=90)
        text = stringify_mcp_content(result.get("content", []))
        if result.get("isError"):
            return f"[ERROR] {text}"
        return text

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _request(self, method: str, params: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id(), "method": method}
        if params is not None:
            payload["params"] = params
        response = self._session.post(self.settings.mcp_url, json=payload, timeout=timeout)
        response.raise_for_status()
        self._capture_session_id(response)
        message = _parse_mcp_response(response)
        if "error" in message:
            err = message["error"]
            raise RuntimeError(f"MCP error ({method}): {err.get('message', err)}")
        return message.get("result", {})

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        response = self._session.post(self.settings.mcp_url, json=payload, timeout=30)
        response.raise_for_status()
        self._capture_session_id(response)

    def _capture_session_id(self, response: requests.Response) -> None:
        session_id = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
        if session_id and session_id != self._session_id:
            self._session_id = session_id
            self._session.headers.update({"Mcp-Session-Id": session_id})


def stringify_mcp_content(content: Any) -> str:
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content or "")

    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            chunks.append(str(item))
        elif item.get("type") == "text" and isinstance(item.get("text"), str):
            chunks.append(item["text"])
        else:
            chunks.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _parse_mcp_response(response: requests.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type or "application/json" in content_type:
        response.encoding = "utf-8"
    text = response.text.strip()
    if "text/event-stream" in content_type or text.startswith("event:") or text.startswith("data:"):
        events: list[list[str]] = []
        data_lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
            elif not line.strip() and data_lines:
                events.append(data_lines)
                data_lines = []
        if data_lines:
            events.append(data_lines)
        if not events:
            raise RuntimeError(f"Empty MCP SSE response: {text[:300]}")
        last = events[-1]
        try:
            return json.loads("\n".join(last))
        except json.JSONDecodeError:
            return json.loads("".join(last))
    return response.json()
