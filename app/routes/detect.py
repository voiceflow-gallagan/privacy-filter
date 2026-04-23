import asyncio
import time
from fastapi import APIRouter, HTTPException, Request

from app import model as model_module
from app.chunker import detect_with_chunking
from app.config import get_settings
from app.labels import UnknownLabelError, validate_labels
from app.model import ModelNotLoadedError, get_state
from app.ratelimit import current_limit, limiter
from app.schemas import (
    DetectMeta, DetectRequest, DetectResponse, Entity, MaskRequest, MaskResponse,
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


async def _do_detect(request: Request, req: DetectRequest) -> DetectResponse:
    """Core /detect logic without rate-limit decoration. Reused by /detect and /mask."""
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

    return DetectResponse(text=req.text, entities=entities,
                          masked_text=masked_text, meta=meta)


@router.post("/detect", response_model=DetectResponse)
@limiter.limit(current_limit)
async def detect(request: Request, req: DetectRequest) -> DetectResponse:
    return await _do_detect(request, req)


@router.post("/mask", response_model=MaskResponse)
@limiter.limit(current_limit)
async def mask(request: Request, req: MaskRequest) -> MaskResponse:
    detect_req = DetectRequest(
        text=req.text, mode=req.mode, mask=True,
        mask_char=req.mask_char, labels=req.labels,
    )
    detect_resp = await _do_detect(request, detect_req)
    return MaskResponse(
        masked_text=detect_resp.masked_text or req.text,
        entity_count=detect_resp.meta.entity_count,
        processing_ms=detect_resp.meta.processing_ms,
    )
