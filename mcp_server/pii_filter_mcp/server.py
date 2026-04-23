"""MCP server exposing pii-filter REST endpoints as tools."""
from __future__ import annotations

import os
from typing import Optional

import httpx
from mcp.server import Server
from mcp.types import TextContent, Tool


PII_FILTER_URL = os.environ.get("PII_FILTER_URL", "http://localhost:8080")
TIMEOUT_SEC = float(os.environ.get("PII_FILTER_TIMEOUT_SEC", "60"))


def build_server() -> Server:
    server = Server("pii-filter")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="detect_pii",
                description=(
                    "Detect PII spans in the given text. Returns structured "
                    "entities with label, character offsets, surface text, and score."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "text": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["precise", "balanced", "recall"],
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            ),
            Tool(
                name="mask_pii",
                description=(
                    "Mask PII in the given text and return the redacted string."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "text": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["precise", "balanced", "recall"],
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "mask_char": {"type": "string"},
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "detect_pii":
            return await _detect(arguments)
        if name == "mask_pii":
            return await _mask(arguments)
        raise ValueError(f"unknown tool: {name}")

    return server


async def _detect(args: dict) -> list[TextContent]:
    payload = {"text": args["text"]}
    if "mode" in args:
        payload["mode"] = args["mode"]
    if "labels" in args:
        payload["labels"] = args["labels"]

    async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
        r = await client.post(f"{PII_FILTER_URL}/detect", json=payload)
        r.raise_for_status()
        body = r.json()
    import json
    return [TextContent(
        type="text",
        text=json.dumps({
            "entities": body["entities"],
            "entity_count": body["meta"]["entity_count"],
        }),
    )]


async def _mask(args: dict) -> list[TextContent]:
    payload = {"text": args["text"]}
    for k in ("mode", "labels", "mask_char"):
        if k in args:
            payload[k] = args[k]

    async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
        r = await client.post(f"{PII_FILTER_URL}/mask", json=payload)
        r.raise_for_status()
        body = r.json()
    import json
    return [TextContent(
        type="text",
        text=json.dumps({
            "masked_text": body["masked_text"],
            "entity_count": body["entity_count"],
        }),
    )]
