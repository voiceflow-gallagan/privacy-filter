import pytest
from pydantic import ValidationError
from app.schemas import (
    DetectRequest, DetectResponse, MaskRequest, MaskResponse,
    Entity, Mode, BatchRequest, BatchItem,
)


def test_detect_request_minimal():
    r = DetectRequest(text="hello")
    assert r.text == "hello"
    # Unset by design — the route resolves it from settings.default_mode
    # (see app/modes.py). The schema default is None.
    assert r.mode is None
    assert r.mask is False
    assert r.mask_char == "[REDACTED]"
    assert r.labels is None


def test_detect_request_rejects_invalid_mode():
    with pytest.raises(ValidationError):
        DetectRequest(text="x", mode="aggressive")


def test_entity_shape():
    e = Entity(label="private_email", start=0, end=5, text="hello", score=0.9)
    assert e.label == "private_email"
    assert e.score == 0.9


def test_batch_request_requires_items():
    with pytest.raises(ValidationError):
        BatchRequest(items=[])


def test_batch_item_inherits_detect_fields():
    item = BatchItem(text="hi", labels=["private_email"])
    assert item.labels == ["private_email"]
    assert item.mode is None
