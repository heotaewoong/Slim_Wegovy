from __future__ import annotations

import json
import threading
from typing import Any

import requests

from slim_wegovy.config import Settings


class StreamableHttpMcpClient:
    """MCP Streamable HTTP JSON-RPC client for the Lunit server."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._id = 0
        self._id_lock = threading.Lock()
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
        self._protocol_version = settings.mcp_protocol_version

    def initialize(self) -> None:
        if self._initialized:
            return
        result = self._request(
            "initialize",
            {
                "protocolVersion": self._protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "slim-wegovy", "version": "0.1.0"},
            },
        )
        negotiated = result.get("protocolVersion")
        if isinstance(negotiated, str):
            self._protocol_version = negotiated
        self._initialized = True
        self._notify("notifications/initialized")

    def list_tools(self) -> list[dict[str, Any]]:
        self.initialize()
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            result = self._request("tools/list", {"cursor": cursor} if cursor else {})
            page = result.get("tools", [])
            if not isinstance(page, list):
                raise RuntimeError("MCP tools/list returned an invalid tools field")
            tools.extend(tool for tool in page if isinstance(tool, dict))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(
        self, name: str, arguments: dict[str, Any], timeout_sec: float | None = None
    ) -> str:
        self.initialize()
        result = self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=timeout_sec or self.settings.mcp_tool_timeout_sec,
        )
        chunks: list[str] = []
        content = stringify_mcp_content(result.get("content", []))
        if content:
            chunks.append(content)
        if result.get("structuredContent") is not None:
            chunks.append(json.dumps(result["structuredContent"], ensure_ascii=False))
        rendered = "\n".join(chunks).strip()
        if result.get("isError"):
            return f"[ERROR] {rendered}"
        return rendered

    def _next_id(self) -> int:
        with self._id_lock:
            self._id += 1
            return self._id

    def _message_headers(self, method: str, name: str | None = None) -> dict[str, str]:
        headers = {"Mcp-Method": method}
        if self._initialized:
            headers["MCP-Protocol-Version"] = self._protocol_version
        if name:
            headers["Mcp-Name"] = name
        return headers

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if timeout is None:
            timeout = self.settings.mcp_request_timeout_sec
        request_id = self._next_id()
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        name = params.get("name") if method == "tools/call" and params else None
        response = self._session.post(
            self.settings.mcp_url,
            json=payload,
            headers=self._message_headers(method, name),
            timeout=timeout,
        )
        response.raise_for_status()
        self._capture_session_id(response)
        message = _parse_mcp_response(response, expected_id=request_id)
        if "error" in message:
            err = message["error"]
            raise RuntimeError(f"MCP error ({method}): {err.get('message', err)}")
        result = message.get("result", {})
        if not isinstance(result, dict):
            raise RuntimeError(f"MCP returned an invalid result for {method}")
        return result

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        response = self._session.post(
            self.settings.mcp_url,
            json=payload,
            headers=self._message_headers(method),
            timeout=min(self.settings.mcp_request_timeout_sec, 15),
        )
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


def _parse_mcp_response(
    response: requests.Response, expected_id: int | str | None = None
) -> dict[str, Any]:
    if response.status_code in {202, 204} or not response.content:
        return {}
    content_type = response.headers.get("content-type", "").lower()
    response.encoding = "utf-8"
    text = response.text.strip()
    messages: list[dict[str, Any]] = []

    if "text/event-stream" in content_type or text.startswith(("event:", "data:")):
        data_lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
            elif not line.strip() and data_lines:
                try:
                    value = json.loads("\n".join(data_lines))
                    if isinstance(value, dict):
                        messages.append(value)
                except json.JSONDecodeError:
                    pass
                data_lines = []
        if data_lines:
            try:
                value = json.loads("\n".join(data_lines))
                if isinstance(value, dict):
                    messages.append(value)
            except json.JSONDecodeError:
                pass
    else:
        value = response.json()
        if isinstance(value, dict):
            messages.append(value)

    message = next(
        (item for item in messages if expected_id is None or item.get("id") == expected_id), None
    )
    if message is None:
        raise RuntimeError(f"MCP response did not contain JSON-RPC id {expected_id}: {text[:300]}")
    return message
