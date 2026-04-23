from fastapi.testclient import TestClient
import asyncio
import pytest

from app.main import create_app
from app import model as model_module


@pytest.fixture
def client_unloaded(monkeypatch):
    monkeypatch.setattr(model_module, "_state", None)
    monkeypatch.setattr(model_module, "_semaphore", asyncio.Semaphore(2))
    app = create_app(load_at_startup=False)
    return TestClient(app)


@pytest.fixture
def client_loaded(patch_model):
    app = create_app(load_at_startup=False)
    return TestClient(app)


def test_health_always_200_when_unloaded(client_unloaded):
    r = client_unloaded.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_health_reports_loaded_when_ready(client_loaded):
    r = client_loaded.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"


def test_ready_503_when_unloaded(client_unloaded):
    r = client_unloaded.get("/ready")
    assert r.status_code == 503


def test_ready_200_when_loaded(client_loaded):
    r = client_loaded.get("/ready")
    assert r.status_code == 200
