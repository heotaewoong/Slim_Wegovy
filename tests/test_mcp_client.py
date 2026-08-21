import json
import unittest

import requests

from slim_wegovy.config import Settings
from slim_wegovy.mcp_client import StreamableHttpMcpClient


def response(status=200, payload=None, headers=None, sse=False):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    if payload is None:
        result._content = b""
    elif sse:
        result.headers["content-type"] = "text/event-stream"
        result._content = f"event: message\ndata: {json.dumps(payload)}\n\n".encode()
    else:
        result.headers["content-type"] = "application/json"
        result._content = json.dumps(payload).encode()
    return result


class MCPClientTests(unittest.TestCase):
    def test_sse_session_protocol_and_tool_pagination(self):
        settings = Settings(
            lunit_api_url="https://model.example",
            lunit_api_key="lunit_test",
            lunit_model="Lunit/L2-preview",
            mcp_url="https://mcp.example/mcp",
        )
        client = StreamableHttpMcpClient(settings)
        calls = []

        def post(url, json, headers, timeout):
            calls.append((json["method"], dict(headers)))
            method = json["method"]
            if method == "initialize":
                return response(
                    payload={
                        "jsonrpc": "2.0",
                        "id": json["id"],
                        "result": {"protocolVersion": "2025-06-18", "capabilities": {}},
                    },
                    headers={"mcp-session-id": "session-1"},
                    sse=True,
                )
            if method == "notifications/initialized":
                return response(status=202)
            if json.get("params", {}).get("cursor") is None:
                return response(payload={
                    "jsonrpc": "2.0", "id": json["id"],
                    "result": {"tools": [{"name": "first"}], "nextCursor": "page-2"},
                })
            return response(payload={
                "jsonrpc": "2.0", "id": json["id"],
                "result": {"tools": [{"name": "second"}]},
            })

        client._session.post = post
        tools = client.list_tools()

        self.assertEqual([tool["name"] for tool in tools], ["first", "second"])
        self.assertEqual(client._session.headers["Mcp-Session-Id"], "session-1")
        self.assertEqual(calls[1][1]["MCP-Protocol-Version"], "2025-06-18")
        self.assertEqual([call[0] for call in calls], [
            "initialize", "notifications/initialized", "tools/list", "tools/list"
        ])


if __name__ == "__main__":
    unittest.main()
