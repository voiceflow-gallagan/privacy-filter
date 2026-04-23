import asyncio
import time
from fastapi import APIRouter, HTTPException, Request

from app import model as model_module
from app.config import get_settings
from app.labels import UnknownLabelError, validate_labels
from app.model import ModelNotLoadedError, get_state
from app.ratelimit import current_limit, limiter
from app.schemas import (
    BatchItem,
    BatchItemError,
    BatchItemResult,
    BatchMeta,
    BatchRequest,
    BatchResponse,
    Entity,
)


router = APIRouter()


def _apply_mask(text: str, entities: list[Entity], mask_char: str) -> str:
    parts = sorted(entities, key=lambda e: e.start, reverse=True)
    out = text
    for e in parts:
        out = out[: e.start] + mask_char + out[e.end :]
    return out


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


@router.post("/detect/batch", response_model=BatchResponse)
@limiter.limit(current_limit)
async def detect_batch(request: Request, req: BatchRequest) -> BatchResponse:
    settings = get_settings()

    if len(req.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"batch size {len(req.items)} exceeds MAX_BATCH_SIZE "
                   f"({settings.max_batch_size})",
        )

    try:
        state = get_state()
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="model not loaded")

    token_counts = [_count_tokens(state.tokenizer, item.text) for item in req.items]
    total_tokens = sum(token_counts)
    if total_tokens > settings.max_batch_total_tokens:
        raise HTTPException(
            status_code=422,
            detail=f"batch total tokens {total_tokens} exceeds "
                   f"MAX_BATCH_TOTAL_TOKENS ({settings.max_batch_total_tokens})",
        )

    started = time.perf_counter()
    results: list[BatchItemResult] = []

    for item, n_tok in zip(req.items, token_counts):
        results.append(await _process_item(item, n_tok, req.mask, settings, state))

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return BatchResponse(
        results=results,
        meta=BatchMeta(
            model=state.model_name,
            batch_size=len(req.items),
            processing_ms=elapsed_ms,
        ),
    )


async def _process_item(item: BatchItem, n_tok: int, mask_top: bool,
                        settings, state) -> BatchItemResult:
    if n_tok > settings.chunk_size_tokens:
        return BatchItemResult(
            status="error",
            error=BatchItemError(
                code="item_too_long",
                message=(f"item has {n_tok} tokens which exceeds "
                         f"CHUNK_SIZE_TOKENS ({settings.chunk_size_tokens}); "
                         "chunking is not allowed in batch - use POST /detect"),
            ),
        )

    try:
        allowed = validate_labels(item.labels)
    except UnknownLabelError as e:
        return BatchItemResult(
            status="error",
            error=BatchItemError(code="unknown_label", message=str(e)),
        )

    try:
        async with model_module._semaphore:
            spans = await asyncio.to_thread(state.run_inference, item.text, item.mode)
    except Exception as exc:  # noqa: BLE001 - surface inference failure per item
        return BatchItemResult(
            status="error",
            error=BatchItemError(code="inference_failed", message=str(exc)),
        )

    if allowed is not None:
        spans = [s for s in spans if s["label"] in allowed]
    entities = [Entity(**s) for s in spans]
    masked_text = _apply_mask(item.text, entities, item.mask_char) if mask_top else None

    return BatchItemResult(
        status="ok",
        entities=entities,
        masked_text=masked_text,
        meta={"entity_count": len(entities)},
    )
