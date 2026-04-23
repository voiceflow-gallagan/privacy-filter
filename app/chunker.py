from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


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
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    n_tokens = len(tokens)

    if n_tokens <= chunk_size_tokens:
        entities = run_inference(text, mode)
        return ChunkResult(entities=entities, chunks_processed=1, input_tokens=n_tokens)

    step = chunk_size_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("chunk_size_tokens must be greater than overlap_tokens")

    all_entities: list[dict] = []
    chunks_processed = 0

    start_tok = 0
    while start_tok < n_tokens:
        end_tok = min(start_tok + chunk_size_tokens, n_tokens)
        char_start = offsets[start_tok][0]
        char_end = offsets[end_tok - 1][1]

        if smart_split:
            char_start, char_end = _nudge_to_paragraph(text, offsets,
                                                      start_tok, end_tok,
                                                      char_start, char_end)

        chunk_text = text[char_start:char_end]
        spans = run_inference(chunk_text, mode)
        for s in spans:
            all_entities.append({
                **s,
                "start": s["start"] + char_start,
                "end": s["end"] + char_start,
            })
        chunks_processed += 1
        if end_tok >= n_tokens:
            break
        start_tok += step

    deduped = deduplicate_spans(all_entities)
    return ChunkResult(entities=deduped, chunks_processed=chunks_processed,
                       input_tokens=n_tokens)


def deduplicate_spans(entities: Iterable[dict]) -> list[dict]:
    """
    Collapse spans that share a label and overlap in character range,
    keeping the one with the higher score.
    """
    sorted_ents = sorted(entities, key=lambda e: (e["start"], -e["score"]))
    kept: list[dict] = []
    for e in sorted_ents:
        absorbed = False
        for k in kept:
            if k["label"] == e["label"] and e["start"] < k["end"] and e["end"] > k["start"]:
                if e["score"] > k["score"]:
                    k.update(e)
                absorbed = True
                break
        if not absorbed:
            kept.append(dict(e))
    kept.sort(key=lambda e: e["start"])
    return kept


def _nudge_to_paragraph(text: str, offsets, start_tok: int, end_tok: int,
                        char_start: int, char_end: int,
                        token_window: int = 200) -> tuple[int, int]:
    """
    Try to align the chunk's end (and next-chunk's start) to the nearest "\n\n"
    within +/- token_window tokens of the calculated split. Best-effort:
    if no paragraph break exists in range, falls back to the calculated boundaries.
    """
    win_start_tok = max(0, end_tok - token_window)
    win_end_tok = min(len(offsets), end_tok + token_window)
    if win_end_tok > win_start_tok:
        win_char_start = offsets[win_start_tok][0]
        win_char_end = offsets[win_end_tok - 1][1]
        best = -1
        idx = text.find("\n\n", win_char_start, win_char_end)
        while idx != -1:
            if best == -1 or abs(idx - char_end) < abs(best - char_end):
                best = idx
            idx = text.find("\n\n", idx + 2, win_char_end)
        if best != -1:
            char_end = best
    return char_start, char_end
