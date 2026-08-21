from __future__ import annotations

from typing import Any


def mcp_tool_to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
        },
    }


FINALIZE_RETRIEVAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finalize_retrieval",
        "description": "Submit your final citation selection and end the retrieval phase.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["sufficient", "partial", "no_evidence"],
                    "description": "Whether enough evidence was found.",
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cite_uid": {"type": "string"},
                            "relevance_score": {"type": "number"},
                        },
                        "required": ["cite_uid", "relevance_score"],
                    },
                    "description": "Citation-capable items selected from tool results.",
                },
                "note": {"type": "string"},
            },
            "required": ["status", "items"],
        },
    },
}


RETRIEVE_RELEVANT_CONTENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_relevant_content",
        "description": "Retrieve relevant content to ground your answer. Pass a single, self-contained query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A standalone retrieval query containing all necessary context.",
                }
            },
            "required": ["query"],
        },
    },
}
