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


def postprocess_spans(text: str, spans: list[dict],
                      max_whitespace_gap: int = 4) -> list[dict]:
    """Clean up raw pipeline spans.

    1. Trim leading/trailing whitespace from each span (prevents the mask from
       eating adjacent spaces, so "Hi Alice" masks to "Hi [REDACTED]" not "Hi[REDACTED]").
    2. Merge same-label spans that overlap, touch, or are separated only by
       whitespace. The model's BIOES decoding often fragments a single entity
       into multiple adjacent sub-spans ("Marie Dubois" → [" Marie", "Dub", "ois"]);
       this stitches them back together.

    Does NOT merge across punctuation or word characters — two emails separated
    by ", " stay separate. `max_whitespace_gap` caps how much whitespace a merge
    can bridge; 4 chars accommodates IBAN/phone/credit-card grouping spaces
    while rejecting larger structural gaps.
    """
    # 1. Trim whitespace
    trimmed: list[dict] = []
    for s in spans:
        start, end = int(s["start"]), int(s["end"])
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            trimmed.append({**s,
                            "start": start, "end": end,
                            "text": text[start:end]})

    if not trimmed:
        return trimmed

    # 2. Sort and merge same-label overlapping / whitespace-only-gap spans
    trimmed.sort(key=lambda s: (s["start"], s["end"]))
    merged: list[dict] = [dict(trimmed[0])]
    for s in trimmed[1:]:
        last = merged[-1]
        if s["label"] == last["label"]:
            overlap_or_touch = s["start"] <= last["end"]
            gap = text[last["end"]:s["start"]]
            bridge = (len(gap) <= max_whitespace_gap and gap.isspace())
            if overlap_or_touch or bridge:
                new_end = max(last["end"], s["end"])
                last["end"] = new_end
                last["text"] = text[last["start"]:new_end]
                last["score"] = max(last["score"], s["score"])
                continue
        merged.append(dict(s))
    return merged


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
        spans = [
            {
                "label": r["entity_group"],
                "start": int(r["start"]),
                "end": int(r["end"]),
                "text": r["word"],
                "score": float(r["score"]),
            }
            for r in raw
        ]
        return postprocess_spans(text, spans)

    _state = ModelState(
        tokenizer=tokenizer,
        run_inference=_run,
        device=settings.device,
        model_name=settings.model_name,
        loaded=True,
    )
    _semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
