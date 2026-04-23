"""
Shared test fixtures.

Strategy:
- Use real `gpt2` fast tokenizer for offset_mapping correctness in chunker tests.
  It is small, downloads once, then uses HF cache.
- Mock the inference function so tests don't need the real 3GB privacy-filter model.
"""
from __future__ import annotations

import asyncio
from typing import Callable, Optional

import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def fast_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    assert tok.is_fast
    return tok


@pytest.fixture
def fake_inference():
    """
    Returns a callable matching the real `run_inference(text, mode) -> list[dict]`
    signature. Default behavior: detect every 'alice@example.com' as private_email
    and every 'Alice Smith' as private_person. Tests can replace .impl for custom logic.
    """
    class _Fake:
        impl: Callable[[str, str], list[dict]] = staticmethod(
            lambda text, mode: _default_detect(text)
        )

        def __call__(self, text: str, mode: str = "balanced") -> list[dict]:
            return self.impl(text, mode)

    return _Fake()


def _default_detect(text: str) -> list[dict]:
    out = []
    needles = [
        ("Alice Smith", "private_person", 0.97),
        ("alice@example.com", "private_email", 0.99),
        ("+33 6 12 34 56 78", "private_phone", 0.95),
    ]
    for needle, label, score in needles:
        start = 0
        while True:
            idx = text.find(needle, start)
            if idx == -1:
                break
            out.append({
                "label": label,
                "start": idx,
                "end": idx + len(needle),
                "text": needle,
                "score": score,
            })
            start = idx + len(needle)
    return out


@pytest.fixture
def patch_model(monkeypatch, fast_tokenizer, fake_inference):
    """
    Patches app.model so the API layer thinks the model is loaded
    and routes inference calls through fake_inference.
    """
    from app import model as model_module

    state = model_module.ModelState(
        tokenizer=fast_tokenizer,
        run_inference=fake_inference,
        device="cpu",
        model_name="test/fake",
        loaded=True,
    )
    monkeypatch.setattr(model_module, "_state", state)
    # Reset semaphore for each test
    monkeypatch.setattr(model_module, "_semaphore", asyncio.Semaphore(2))
    return state
