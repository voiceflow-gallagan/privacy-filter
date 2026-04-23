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


def test_mask_with_overlapping_entities_does_not_leak(client, patch_model):
    """
    Regression: when two overlapping spans are both returned (e.g. after word-
    boundary expansion pushes adjacent spans into overlap), the mask must
    produce a single [REDACTED] over their union, not partial-token leakage
    like '[REDACTEDED]' or 'TED]'.
    """
    # Swap fake_inference to return overlapping spans by hand
    overlapping_text = "chez FNAC Champs-Élysées and more"
    fake = patch_model.run_inference

    def _mock(text, mode):
        if text == overlapping_text:
            return [
                {"label": "account_number",  "start": 5, "end": 24, "text": "FNAC Champs-Élysées", "score": 0.7},
                {"label": "private_address", "start": 17, "end": 24, "text": "Élysées",             "score": 0.6},
            ]
        return fake(text, mode)
    fake.impl = _mock

    r = client.post("/mask", json={"text": overlapping_text})
    body = r.json()
    # Exactly one mask token covers the union, rest of text untouched.
    assert body["masked_text"] == "chez [REDACTED] and more"
    # But two entities still reported — overlap resolution is a mask-only concern.
    assert body["entity_count"] == 2


def test_mask_unknown_label_is_422(client):
    r = client.post("/mask", json={"text": "x", "labels": ["bogus"]})
    assert r.status_code == 422
