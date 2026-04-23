from fastapi.testclient import TestClient
import pytest

from app.main import create_app


@pytest.fixture
def client_limited(patch_model, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_IP", "3/10minutes")
    return TestClient(create_app(load_at_startup=False))


@pytest.fixture
def client_unlimited(patch_model, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    return TestClient(create_app(load_at_startup=False))


def test_rate_limit_returns_429_after_limit(client_limited):
    for i in range(3):
        r = client_limited.post("/detect", json={"text": "x"})
        assert r.status_code == 200, f"expected 200 on call {i+1}, got {r.status_code}"
    r = client_limited.post("/detect", json={"text": "x"})
    assert r.status_code == 429


def test_health_exempt_from_rate_limit(client_limited):
    for _ in range(20):
        r = client_limited.get("/health")
        assert r.status_code == 200


def test_rate_limit_disabled_means_unlimited(client_unlimited):
    for _ in range(20):
        r = client_unlimited.post("/detect", json={"text": "x"})
        assert r.status_code == 200
