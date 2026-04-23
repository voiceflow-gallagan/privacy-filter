"""
Test the MCP tool layer in isolation by stubbing httpx.

Run from repo root:  pytest mcp_server/tests/ -v
"""
from __future__ import annotations

import json
import pytest
import httpx

from pii_filter_mcp import server as srv


class _FakeClient:
    def __init__(self, response_body: dict, *args, **kwargs):
        self._body = response_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def post(self, url, json):
        self.last_url = url
        self.last_json = json
        return _FakeResp(self._body)


class _FakeResp:
    def __init__(self, body): self._body = body
    def raise_for_status(self): pass
    def json(self): return self._body


@pytest.fixture
def fake_detect_response():
    return {
        "text": "alice@example.com",
        "entities": [{
            "label": "private_email", "start": 0, "end": 17,
            "text": "alice@example.com", "score": 0.99,
        }],
        "meta": {"model": "x", "mode": "balanced",
                 "entity_count": 1, "processing_ms": 10},
    }


@pytest.fixture
def fake_mask_response():
    return {"masked_text": "[REDACTED]", "entity_count": 1, "processing_ms": 10}


async def test_detect_calls_correct_endpoint(monkeypatch, fake_detect_response):
    fakes: list[_FakeClient] = []
    def factory(*a, **kw):
        c = _FakeClient(fake_detect_response, *a, **kw)
        fakes.append(c)
        return c
    monkeypatch.setattr(httpx, "AsyncClient", factory)

    out = await srv._detect({"text": "alice@example.com"})

    assert fakes[0].last_url.endswith("/detect")
    assert fakes[0].last_json == {"text": "alice@example.com"}
    body = json.loads(out[0].text)
    assert body["entity_count"] == 1
    assert body["entities"][0]["label"] == "private_email"


async def test_mask_passes_optional_args(monkeypatch, fake_mask_response):
    captured: list[_FakeClient] = []
    def factory(*a, **kw):
        c = _FakeClient(fake_mask_response, *a, **kw)
        captured.append(c)
        return c
    monkeypatch.setattr(httpx, "AsyncClient", factory)

    out = await srv._mask({
        "text": "alice@example.com",
        "mode": "recall",
        "labels": ["private_email"],
        "mask_char": "***",
    })
    assert captured[0].last_url.endswith("/mask")
    assert captured[0].last_json == {
        "text": "alice@example.com",
        "mode": "recall",
        "labels": ["private_email"],
        "mask_char": "***",
    }
    body = json.loads(out[0].text)
    assert body["masked_text"] == "[REDACTED]"
