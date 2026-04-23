from app.chunker import detect_with_chunking


def test_short_text_takes_fast_path(fast_tokenizer, fake_inference):
    text = "Hello, my name is Alice Smith."
    result = detect_with_chunking(
        text=text,
        tokenizer=fast_tokenizer,
        run_inference=fake_inference,
        mode="balanced",
        chunk_size_tokens=120_000,
        overlap_tokens=512,
        smart_split=False,
    )
    assert result.chunks_processed == 1
    assert result.input_tokens == len(fast_tokenizer.encode(text, add_special_tokens=False))
    assert any(e["label"] == "private_person" and e["text"] == "Alice Smith"
               for e in result.entities)


def test_short_text_offsets_match_source(fast_tokenizer, fake_inference):
    text = "Email me at alice@example.com please."
    result = detect_with_chunking(
        text=text, tokenizer=fast_tokenizer, run_inference=fake_inference,
        mode="balanced", chunk_size_tokens=120_000, overlap_tokens=512,
        smart_split=False,
    )
    e = next(x for x in result.entities if x["label"] == "private_email")
    assert text[e["start"]:e["end"]] == "alice@example.com"


def _make_long_text(filler_tokens: int, fast_tokenizer, sentinel: str) -> str:
    """Build text with > N tokens by repeating a phrase, with `sentinel` at a known location."""
    base = "The quick brown fox jumps over the lazy dog. "
    chunk_text = base * (filler_tokens // 10)  # ~10 tokens per repeat
    inserted = chunk_text + sentinel + " " + chunk_text
    return inserted


def test_multi_chunk_processes_more_than_one_chunk(fast_tokenizer, fake_inference):
    text = _make_long_text(2_000, fast_tokenizer, "Email me at alice@example.com.")
    # Force chunking with a small chunk size
    result = detect_with_chunking(
        text=text, tokenizer=fast_tokenizer, run_inference=fake_inference,
        mode="balanced", chunk_size_tokens=1_000, overlap_tokens=64,
        smart_split=False,
    )
    assert result.chunks_processed >= 2
    # The email should still be found and offsets should map back to source
    emails = [e for e in result.entities if e["label"] == "private_email"]
    assert emails, "Expected at least one private_email entity"
    for e in emails:
        assert text[e["start"]:e["end"]] == "alice@example.com"


def test_overlap_dedup_collapses_duplicate_spans(fast_tokenizer, fake_inference):
    """A PII span sitting in the overlap window between two chunks must
    appear exactly once in the merged result."""
    # Build text where the email lives near a chunk boundary
    head = "x " * 500
    tail = " y" * 500
    text = head + "alice@example.com" + tail
    result = detect_with_chunking(
        text=text, tokenizer=fast_tokenizer, run_inference=fake_inference,
        mode="balanced", chunk_size_tokens=600, overlap_tokens=200,
        smart_split=False,
    )
    emails = [e for e in result.entities if e["label"] == "private_email"]
    assert len(emails) == 1
    e = emails[0]
    assert text[e["start"]:e["end"]] == "alice@example.com"


def test_dedup_keeps_higher_score():
    from app.chunker import deduplicate_spans
    spans = [
        {"label": "private_email", "start": 0, "end": 5, "text": "x", "score": 0.7},
        {"label": "private_email", "start": 0, "end": 5, "text": "x", "score": 0.95},
    ]
    out = deduplicate_spans(spans)
    assert len(out) == 1
    assert out[0]["score"] == 0.95


def test_offsets_are_correct_with_cjk_and_emoji(fast_tokenizer, fake_inference):
    """Spans returned for chunked input must slice the original text correctly
    even when chunks straddle multibyte characters."""
    head = "你好世界 " * 200            # CJK
    middle = "🎉🚀✨ " * 100             # emoji
    text = head + "alice@example.com" + middle + head
    result = detect_with_chunking(
        text=text, tokenizer=fast_tokenizer, run_inference=fake_inference,
        mode="balanced", chunk_size_tokens=400, overlap_tokens=64,
        smart_split=False,
    )
    emails = [e for e in result.entities if e["label"] == "private_email"]
    assert emails, "Expected the email to be detected"
    for e in emails:
        assert text[e["start"]:e["end"]] == "alice@example.com", (
            f"Offset drift: text[{e['start']}:{e['end']}]="
            f"{text[e['start']:e['end']]!r}"
        )


def test_dedup_keeps_different_labels_overlapping():
    from app.chunker import deduplicate_spans
    spans = [
        {"label": "private_email", "start": 0, "end": 10, "text": "x", "score": 0.9},
        {"label": "private_phone", "start": 5, "end": 12, "text": "y", "score": 0.9},
    ]
    out = deduplicate_spans(spans)
    assert len(out) == 2
