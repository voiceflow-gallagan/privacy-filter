"""Entrypoint: `python -m pii_filter_mcp` runs the stdio MCP server."""
from __future__ import annotations

import asyncio

from mcp.server.stdio import stdio_server

from pii_filter_mcp.server import build_server


async def _run() -> None:
    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
