from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException

from app.chunker import detect_with_chunking
from app.config import get_settings
from app.labels import UnknownLabelError, validate_labels
from app.model import ModelNotLoadedError, get_state
from app.schemas import (
    DetectMeta,
    DetectRequest,
    DetectResponse,
    Entity,
)


router = APIRouter()


def _apply_mask(text: str, entities: list[Entity], mask_char: str) -> str:
    parts = sorted(entities, key=lambda e: e.start, reverse=True)
    out = text
    for e in parts:
        out = out[: e.start] + mask_char + out[e.end :]
    return out


def _filter_by_labels(spans: list[dict], allowed: frozenset[str] | None) -> list[dict]:
    if allowed is None:
        return spans
    return [s for s in spans if s["label"] in allowed]


@router.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest) -> DetectResponse:
    settings = get_settings()
    if len(req.text) > settings.max_text_length:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds MAX_TEXT_LENGTH ({settings.max_text_length})",
        )
    try:
        allowed = validate_labels(req.labels)
    except UnknownLabelError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        state = get_state()
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded")

    started = time.perf_counter()

    import asyncio

    def _do():
        return detect_with_chunking(
            text=req.text,
            tokenizer=state.tokenizer,
            run_inference=lambda t, m: state.run_inference(t, m),
            mode=req.mode,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            smart_split=settings.smart_split,
        )

    from app import model as model_module
    async with model_module._semaphore:
        result = await asyncio.to_thread(_do)

    spans = _filter_by_labels(result.entities, allowed)
    entities = [Entity(**s) for s in spans]
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    meta = DetectMeta(
        model=state.model_name,
        mode=req.mode,
        entity_count=len(entities),
        processing_ms=elapsed_ms,
        chunks_processed=result.chunks_processed if result.chunks_processed > 1 else None,
        input_tokens=result.input_tokens if result.chunks_processed > 1 else None,
    )

    masked_text = _apply_mask(req.text, entities, req.mask_char) if req.mask else None

    return DetectResponse(
        text=req.text,
        entities=entities,
        masked_text=masked_text,
        meta=meta,
    )
