"""Integration tests for the MCP server mounted at /mcp."""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.mcp_app import build_mcp


def _unwrap_tool_result(result):
    """Normalize a FastMCP call_tool result into a dict.

    Different SDK versions return either ``(content_blocks, structured)`` tuples,
    a bare ``list[ContentBlock]``, or the structured dict directly. The text
    blocks carry JSON-encoded payloads matching the tool's return value.
    """
    if isinstance(result, tuple):
        _content, structured = result
        if isinstance(structured, dict):
            return structured
        result = _content
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        for block in result:
            text = getattr(block, "text", None)
            if text is not None:
                return json.loads(text)
    raise AssertionError(f"cannot unwrap tool result: {type(result)}")


@pytest.fixture
def client(patch_model):
    # TestClient runs the app's lifespan, which is what starts the MCP
    # session manager that backs the /mcp mount.
    with TestClient(create_app(load_at_startup=False)) as c:
        yield c


def test_mcp_initialize_handshake(client):
    """POST /mcp should speak JSON-RPC and respond to `initialize`."""
    r = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
        headers={"Accept": "application/json, text/event-stream"},
    )
    # Streamable-HTTP returns 200 (JSON body) or 202/200 (SSE stream).
    assert r.status_code in (200, 202), f"got {r.status_code}: {r.text[:400]}"


def test_tools_are_registered():
    """The FastMCP instance must expose the two documented tools."""
    mcp = build_mcp()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert "detect_pii" in names
    assert "mask_pii" in names


def test_detect_pii_tool_dispatches_to_core(patch_model):
    """Invoking the FastMCP-registered detect_pii tool must go through _do_detect."""
    mcp = build_mcp()
    result = asyncio.run(
        mcp.call_tool(
            "detect_pii",
            {"text": "contact alice@example.com for details"},
        )
    )
    structured = _unwrap_tool_result(result)
    assert structured["entity_count"] == 1
    assert structured["entities"][0]["label"] == "private_email"


def test_mask_pii_tool_dispatches_to_core(patch_model):
    """mask_pii must replace the detected span with mask_char."""
    mcp = build_mcp()
    result = asyncio.run(
        mcp.call_tool(
            "mask_pii",
            {"text": "contact alice@example.com", "mask_char": "[X]"},
        )
    )
    structured = _unwrap_tool_result(result)
    assert structured["entity_count"] == 1
    assert "[X]" in structured["masked_text"]
    assert "alice@example.com" not in structured["masked_text"]
