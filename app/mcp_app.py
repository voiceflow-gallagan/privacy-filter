"""In-process MCP server mounted into the main FastAPI app.

Tools dispatch to ``_do_detect`` directly — no HTTP self-loopback. The server
is exposed over streamable-HTTP at ``/mcp`` on the same uvicorn process that
serves the REST API.
"""
from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.routes.detect import _do_detect
from app.schemas import DetectRequest


def build_mcp() -> FastMCP:
    # The SDK's default DNS-rebinding protection only allows
    # localhost/127.0.0.1 Host headers, which breaks this service when fronted
    # by a reverse proxy (the public domain's Host header is rejected). This
    # service is documented as needing network-level access controls; disable
    # the built-in Host check so any proxied hostname is accepted.
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    # streamable_http_path="/" because the resulting Starlette app is mounted
    # at "/mcp" in the parent FastAPI app — otherwise the route would end up
    # double-prefixed at "/mcp/mcp".
    mcp = FastMCP(
        "pii-filter",
        streamable_http_path="/",
        transport_security=security,
    )

    @mcp.tool()
    async def detect_pii(
        text: str,
        mode: str = "balanced",
        labels: Optional[list[str]] = None,
    ) -> dict:
        """Detect PII spans in ``text``.

        Returns an object with ``entities`` (list of label/start/end/text/score)
        and ``entity_count``.
        """
        req = DetectRequest(text=text, mode=mode, labels=labels)
        resp = await _do_detect(req)
        return {
            "entities": [e.model_dump() for e in resp.entities],
            "entity_count": resp.meta.entity_count,
        }

    @mcp.tool()
    async def mask_pii(
        text: str,
        mode: str = "balanced",
        labels: Optional[list[str]] = None,
        mask_char: str = "[REDACTED]",
    ) -> dict:
        """Mask PII in ``text``.

        Returns ``masked_text`` (the redacted string) and ``entity_count``.
        """
        req = DetectRequest(
            text=text,
            mode=mode,
            labels=labels,
            mask=True,
            mask_char=mask_char,
        )
        resp = await _do_detect(req)
        return {
            "masked_text": resp.masked_text or text,
            "entity_count": resp.meta.entity_count,
        }

    return mcp
