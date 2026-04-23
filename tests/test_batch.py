from fastapi.testclient import TestClient
import pytest
from app.main import create_app


@pytest.fixture
def client(patch_model):
    return TestClient(create_app(load_at_startup=False))


def test_batch_happy_path(client):
    r = client.post("/detect/batch", json={
        "items": [
            {"text": "Hello Alice Smith"},
            {"text": "alice@example.com"},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["batch_size"] == 2
    assert len(body["results"]) == 2
    assert body["results"][0]["status"] == "ok"
    assert body["results"][1]["status"] == "ok"
    assert {e["label"] for e in body["results"][0]["entities"]} == {"private_person"}


def test_batch_with_mask_top_level(client):
    r = client.post("/detect/batch", json={
        "items": [{"text": "alice@example.com"}],
        "mask": True,
    })
    body = r.json()
    assert body["results"][0]["masked_text"] == "[REDACTED]"


def test_batch_per_item_label_filter(client):
    r = client.post("/detect/batch", json={
        "items": [
            {"text": "Alice Smith and alice@example.com",
             "labels": ["private_email"]},
        ],
    })
    body = r.json()
    labels = {e["label"] for e in body["results"][0]["entities"]}
    assert labels == {"private_email"}


def test_batch_size_exceeds_limit(client, monkeypatch):
    from app.config import Settings
    monkeypatch.setattr("app.routes.batch.get_settings",
                        lambda: Settings(max_batch_size=2))
    r = client.post("/detect/batch", json={
        "items": [{"text": "x"}, {"text": "y"}, {"text": "z"}],
    })
    assert r.status_code == 422
    assert "batch" in r.text.lower()


def test_batch_total_token_budget_exceeded(client, monkeypatch):
    from app.config import Settings
    monkeypatch.setattr("app.routes.batch.get_settings",
                        lambda: Settings(max_batch_total_tokens=10))
    r = client.post("/detect/batch", json={
        "items": [{"text": "the quick brown fox " * 50}],
    })
    assert r.status_code == 422
    assert "token" in r.text.lower()


def test_batch_per_item_too_long_returns_partial_error(client, monkeypatch):
    """An item exceeding chunk size yields a per-item 'item_too_long' error;
    other items still succeed."""
    from app.config import Settings
    monkeypatch.setattr(
        "app.routes.batch.get_settings",
        lambda: Settings(chunk_size_tokens=20,
                         max_batch_total_tokens=10_000),
    )
    r = client.post("/detect/batch", json={
        "items": [
            {"text": "alice@example.com"},                # short
            {"text": "the quick brown fox " * 100},       # too long
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["status"] == "ok"
    assert body["results"][1]["status"] == "error"
    assert body["results"][1]["error"]["code"] == "item_too_long"


def test_batch_503_when_model_not_loaded(monkeypatch):
    from app import model as model_module
    monkeypatch.setattr(model_module, "_state", None)
    client = TestClient(create_app(load_at_startup=False))
    r = client.post("/detect/batch", json={"items": [{"text": "x"}]})
    assert r.status_code == 503


def test_batch_unknown_label_is_per_item_error_not_request_error(client):
    """Per spec: per-item label issues return as item-level errors, not 422 for the whole batch."""
    r = client.post("/detect/batch", json={
        "items": [{"text": "x", "labels": ["bogus"]}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["status"] == "error"
    assert "bogus" in body["results"][0]["error"]["message"]
