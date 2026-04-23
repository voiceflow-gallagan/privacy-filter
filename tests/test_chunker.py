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
