from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional

from app.config import get_settings


class ModelNotLoadedError(RuntimeError):
    pass


@dataclass
class ModelState:
    tokenizer: object
    run_inference: Callable[[str, str], list[dict]]
    device: str
    model_name: str
    loaded: bool


_state: Optional[ModelState] = None
_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    get_settings().max_concurrent_inferences
)


def get_state() -> ModelState:
    if _state is None or not _state.loaded:
        raise ModelNotLoadedError("Model not yet loaded")
    return _state


def is_loaded() -> bool:
    return _state is not None and _state.loaded


async def run_inference_async(text: str, mode: str) -> list[dict]:
    state = get_state()
    async with _semaphore:
        return await asyncio.to_thread(state.run_inference, text, mode)


def load_model() -> None:
    """
    Load the real privacy-filter model. Called from FastAPI startup.
    Kept thin so tests can swap the global state directly.
    """
    global _state, _semaphore
    settings = get_settings()

    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"Tokenizer for {settings.model_name} is not a fast tokenizer; "
            "offset_mapping is required for chunker correctness."
        )

    pipe = pipeline(
        "token-classification",
        model=settings.model_name,
        tokenizer=tokenizer,
        device=0 if settings.device == "cuda" else -1,
        aggregation_strategy="simple",
    )

    def _run(text: str, mode: str) -> list[dict]:
        raw = pipe(text)
        return [
            {
                "label": r["entity_group"],
                "start": int(r["start"]),
                "end": int(r["end"]),
                "text": r["word"],
                "score": float(r["score"]),
            }
            for r in raw
        ]

    _state = ModelState(
        tokenizer=tokenizer,
        run_inference=_run,
        device=settings.device,
        model_name=settings.model_name,
        loaded=True,
    )
    _semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
