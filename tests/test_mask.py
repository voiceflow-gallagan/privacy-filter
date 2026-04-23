from fastapi.testclient import TestClient
import pytest
from app.main import create_app


@pytest.fixture
def client(patch_model):
    return TestClient(create_app(load_at_startup=False))


def test_mask_returns_only_masked_text(client):
    r = client.post("/mask", json={"text": "Call alice@example.com tonight."})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"masked_text", "entity_count", "processing_ms"}
    assert body["masked_text"] == "Call [REDACTED] tonight."
    assert body["entity_count"] == 1


def test_mask_respects_label_filter(client):
    r = client.post("/mask", json={
        "text": "Alice Smith wrote alice@example.com",
        "labels": ["private_email"],
    })
    body = r.json()
    # Person not in label filter, so left alone; email masked.
    assert body["masked_text"] == "Alice Smith wrote [REDACTED]"
    assert body["entity_count"] == 1


def test_mask_unknown_label_is_422(client):
    r = client.post("/mask", json={"text": "x", "labels": ["bogus"]})
    assert r.status_code == 422
