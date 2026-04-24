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


def _low_conf_inference(text: str, mode: str) -> list[dict]:
    """Single low-confidence (0.50) span covering the whole text."""
    return [{"label": "private_person", "start": 0, "end": len(text),
             "text": text, "score": 0.50}]


def test_mode_precise_drops_low_confidence(client, patch_model):
    patch_model.run_inference.impl = _low_conf_inference
    r = client.post("/detect", json={"text": "Bob", "mode": "precise"})
    body = r.json()
    # 0.50 < precise threshold (0.85) → no entities
    assert body["entities"] == []
    assert body["meta"]["mode"] == "precise"


def test_mode_balanced_also_drops_sub_fifty_five(client, patch_model):
    patch_model.run_inference.impl = _low_conf_inference
    r = client.post("/detect", json={"text": "Bob", "mode": "balanced"})
    body = r.json()
    # 0.50 < balanced threshold (0.55) → no entities
    assert body["entities"] == []
    assert body["meta"]["mode"] == "balanced"


def test_mode_recall_keeps_low_confidence(client, patch_model):
    patch_model.run_inference.impl = _low_conf_inference
    r = client.post("/detect", json={"text": "Bob", "mode": "recall"})
    body = r.json()
    assert len(body["entities"]) == 1
    assert body["entities"][0]["score"] == 0.50
    assert body["meta"]["mode"] == "recall"


def test_default_mode_env_applies_when_request_omits_mode(client, patch_model, monkeypatch):
    from app.config import Settings
    monkeypatch.setattr("app.routes.detect.get_settings",
                        lambda: Settings(default_mode="recall"))
    patch_model.run_inference.impl = _low_conf_inference
    r = client.post("/detect", json={"text": "Bob"})
    body = r.json()
    # DEFAULT_MODE=recall → low-confidence span survives
    assert len(body["entities"]) == 1
    assert body["meta"]["mode"] == "recall"


def test_explicit_mode_overrides_default_mode_env(client, patch_model, monkeypatch):
    from app.config import Settings
    monkeypatch.setattr("app.routes.detect.get_settings",
                        lambda: Settings(default_mode="recall"))
    patch_model.run_inference.impl = _low_conf_inference
    r = client.post("/detect", json={"text": "Bob", "mode": "precise"})
    body = r.json()
    assert body["entities"] == []
    assert body["meta"]["mode"] == "precise"


def test_detect_masks_spoken_card_number(client):
    # Luhn-valid 16-digit spoken card. If this test fails with "spoken run
    # still visible", _scan_spoken is not wired into regex_spans yet.
    text = (
        "Caller: The card number is four, seven, one, six, "
        "three, eight, five, two, nine, four, oh, one, "
        "two, eight, eight, seven. Expiry is zero nine."
    )
    r = client.post("/detect", json={"text": text, "mask": True})
    body = r.json()
    # The spoken digits must not survive redaction.
    assert "four, seven, one, six" not in body["masked_text"]
    assert "two, eight, eight, seven" not in body["masked_text"]
    labels = {e["label"] for e in body["entities"]}
    assert "credit_card_number" in labels
