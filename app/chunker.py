from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ChunkResult:
    entities: list[dict]
    chunks_processed: int
    input_tokens: int


def detect_with_chunking(
    text: str,
    tokenizer,
    run_inference: Callable[[str, str], list[dict]],
    mode: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    smart_split: bool,
) -> ChunkResult:
    """
    Detect PII with automatic chunking for long inputs.

    Uses tokenizer.offset_mapping so chunk slicing happens against the
    original text — no decode round-trip, no offset drift.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding["input_ids"]
    n_tokens = len(tokens)

    if n_tokens <= chunk_size_tokens:
        entities = run_inference(text, mode)
        return ChunkResult(entities=entities, chunks_processed=1, input_tokens=n_tokens)

    raise NotImplementedError("Multi-chunk path lands in Task 8")
