import asyncio
import pytest
from app import model as model_module


@pytest.fixture
def reset_state(monkeypatch):
    monkeypatch.setattr(model_module, "_state", None)
    monkeypatch.setattr(model_module, "_semaphore", asyncio.Semaphore(2))


def test_get_state_raises_when_not_loaded(reset_state):
    with pytest.raises(model_module.ModelNotLoadedError):
        model_module.get_state()


def test_state_is_set_and_returned(monkeypatch, fast_tokenizer):
    state = model_module.ModelState(
        tokenizer=fast_tokenizer,
        run_inference=lambda t, m: [],
        device="cpu",
        model_name="x/y",
        loaded=True,
    )
    monkeypatch.setattr(model_module, "_state", state)
    assert model_module.get_state() is state


async def test_run_inference_async_uses_semaphore(monkeypatch, fast_tokenizer):
    """Verify the semaphore caps in-flight calls."""
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(model_module, "_semaphore", sem)

    in_flight = 0
    max_in_flight = 0

    def slow(text, mode):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        import time; time.sleep(0.05)
        in_flight -= 1
        return []

    state = model_module.ModelState(
        tokenizer=fast_tokenizer,
        run_inference=slow,
        device="cpu",
        model_name="x/y",
        loaded=True,
    )
    monkeypatch.setattr(model_module, "_state", state)

    await asyncio.gather(*[
        model_module.run_inference_async("foo", "balanced")
        for _ in range(5)
    ])
    assert max_in_flight == 1
