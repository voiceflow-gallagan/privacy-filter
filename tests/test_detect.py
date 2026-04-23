from fastapi.testclient import TestClient
import pytest

from app.main import create_app


@pytest.fixture
def client(patch_model):
    return TestClient(create_app(load_at_startup=False))


def test_detect_returns_entities(client):
    r = client.post("/detect", json={"text": "My name is Alice Smith."})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "My name is Alice Smith."
    labels = [e["label"] for e in body["entities"]]
    assert "private_person" in labels
    assert body["meta"]["entity_count"] == len(body["entities"])
    assert body["meta"]["model"] == "test/fake"
    assert body["masked_text"] is None


def test_detect_with_mask_returns_masked_text(client):
    r = client.post("/detect", json={
        "text": "My name is Alice Smith.",
        "mask": True,
    })
    body = r.json()
    assert body["masked_text"] == "My name is [REDACTED]."


def test_detect_with_custom_mask_char(client):
    r = client.post("/detect", json={
        "text": "Email me at alice@example.com please.",
        "mask": True,
        "mask_char": "***",
    })
    assert r.json()["masked_text"] == "Email me at *** please."


def test_detect_with_label_filter_returns_only_requested(client):
    r = client.post("/detect", json={
        "text": "Alice Smith wrote alice@example.com",
        "labels": ["private_email"],
    })
    body = r.json()
    labels = {e["label"] for e in body["entities"]}
    assert labels == {"private_email"}


def test_detect_unknown_label_is_422(client):
    r = client.post("/detect", json={"text": "x", "labels": ["bogus"]})
    assert r.status_code == 422
    assert "bogus" in r.text


def test_detect_returns_503_when_model_not_loaded(monkeypatch):
    from app import model as model_module
    monkeypatch.setattr(model_module, "_state", None)
    client = TestClient(create_app(load_at_startup=False))
    r = client.post("/detect", json={"text": "x"})
    assert r.status_code == 503


def test_detect_text_too_long_is_422(client, monkeypatch):
    from app.config import Settings
    # Force a tiny limit for this test
    monkeypatch.setattr("app.routes.detect.get_settings",
                        lambda: Settings(max_text_length=10))
    r = client.post("/detect", json={"text": "x" * 50})
    assert r.status_code == 422
