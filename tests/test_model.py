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


# ---------- postprocess_spans ----------

def _span(label, start, end, score=0.9):
    return {"label": label, "start": start, "end": end,
            "text": None, "score": score}


def test_postprocess_trims_leading_whitespace():
    text = "Hi Alice Smith, welcome"
    spans = [_span("private_person", 2, 14)]   # includes leading space: " Alice Smith"
    out = model_module.postprocess_spans(text, spans)
    assert len(out) == 1
    assert out[0]["start"] == 3
    assert out[0]["end"] == 14
    assert out[0]["text"] == "Alice Smith"


def test_postprocess_trims_trailing_whitespace():
    text = "call +33 6 42 18 and done"
    spans = [_span("private_phone", 5, 17)]  # " 6 42 18 " → trim leading/trailing spaces
    out = model_module.postprocess_spans(text, spans)
    assert text[out[0]["start"]:out[0]["end"]] == "+33 6 42 18"


def test_postprocess_merges_overlapping_fragments():
    """Classic BIOES fragmentation: three overlapping spans for 'Marie Dubois'."""
    text = "Welcome, Marie Dubois!"
    spans = [
        _span("private_person", 8, 14, 0.92),   # " Marie"
        _span("private_person", 8, 18, 0.88),   # " Marie Dub"
        _span("private_person", 18, 21, 0.85),  # "ois"
    ]
    out = model_module.postprocess_spans(text, spans)
    assert len(out) == 1
    assert out[0]["text"] == "Marie Dubois"
    assert out[0]["score"] == 0.92  # highest of the merged fragments


def test_postprocess_merges_across_single_space_gap():
    """IBAN/phone/credit-card: fragments separated by a space should merge."""
    text = "IBAN: FR76 3000 4000 0312"
    spans = [
        _span("account_number", 6, 10),   # "FR76"
        _span("account_number", 11, 15),  # "3000"
        _span("account_number", 16, 20),  # "4000"
        _span("account_number", 21, 25),  # "0312"
    ]
    out = model_module.postprocess_spans(text, spans)
    assert len(out) == 1
    assert out[0]["text"] == "FR76 3000 4000 0312"


def test_postprocess_does_not_merge_across_punctuation():
    """Two emails separated by ', ' must stay separate."""
    text = "emails: a@x.com, b@y.com here"
    spans = [
        _span("private_email", 8, 15),   # "a@x.com"
        _span("private_email", 17, 24),  # "b@y.com"
    ]
    out = model_module.postprocess_spans(text, spans)
    assert len(out) == 2
    assert out[0]["text"] == "a@x.com"
    assert out[1]["text"] == "b@y.com"


def test_postprocess_does_not_merge_different_labels():
    """Adjacent spans with different labels stay separate."""
    text = "Alice alice@x.com"
    spans = [
        _span("private_person", 0, 5),
        _span("private_email", 6, 17),
    ]
    out = model_module.postprocess_spans(text, spans)
    assert len(out) == 2
    assert {o["label"] for o in out} == {"private_person", "private_email"}


def test_postprocess_drops_all_whitespace_spans():
    text = "hello   world"
    spans = [_span("private_person", 5, 8)]  # all spaces
    out = model_module.postprocess_spans(text, spans)
    assert out == []


def test_postprocess_expands_subword_span_to_full_word():
    """Fixes 'dou[REDACTED]' leakage when the model tags 'ze' inside 'douze'."""
    text = "zéro six, douze, trente"
    # 'ze' is the last 2 chars of 'douze' — indices [13, 15]
    assert text[13:15] == "ze"
    spans = [_span("private_phone", 13, 15)]
    out = model_module.postprocess_spans(text, spans)
    assert len(out) == 1
    assert out[0]["text"] == "douze"


def test_postprocess_expands_subword_from_interior():
    """A span wholly inside a word should grow to cover the whole word."""
    text = "welcome Alice Smith!"
    # 'lic' is interior chars of 'Alice' — indices [9, 12]
    assert text[9:12] == "lic"
    spans = [_span("private_person", 9, 12)]
    out = model_module.postprocess_spans(text, spans)
    assert out[0]["text"] == "Alice"


def test_postprocess_does_not_expand_if_boundary_is_clean():
    """If the span is already aligned to word boundaries, don't touch it."""
    text = "call +33 6 42 18 and done"
    # span captures " +33 6 42 18 " with whitespace around — clean boundaries
    # (text[4]=' ' and text[17]='a' with text[16]=' '), shouldn't grow left.
    spans = [_span("private_phone", 5, 17)]
    assert text[5:17] == "+33 6 42 18 "
    out = model_module.postprocess_spans(text, spans)
    # Trimming should give "+33 6 42 18" — NOT "call +33 6 42 18"
    assert out[0]["text"] == "+33 6 42 18"


def test_postprocess_expand_does_not_cross_hyphen():
    """Hyphens are not alnum, so expansion stops at them (preserves segmentation)."""
    text = "Welcome, Laurent-Mercier here"
    # model tags just 'Laur' — indices [9, 13]
    assert text[9:13] == "Laur"
    spans = [_span("private_person", 9, 13)]
    out = model_module.postprocess_spans(text, spans)
    # Expands to 'Laurent' only; hyphen is a word boundary
    assert out[0]["text"] == "Laurent"
