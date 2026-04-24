from __future__ import annotations

from app.postprocess import (
    _aba_checksum_ok,
    merge_with_model_spans,
    regex_spans,
)


def _labels(spans, label):
    return [s for s in spans if s["label"] == label]


def _texts(spans, label):
    return [s["text"] for s in _labels(spans, label)]


def test_bullet_masked_last4():
    text = "Primary card under dispute: Visa •••• •••• •••• 3421"
    spans = regex_spans(text)
    assert _texts(spans, "credit_card_last4") == ["3421"]


def test_x_masked_last4():
    text = "Backup card: XXXX-XXXX-XXXX-9982 (Mastercard, last 4 only)"
    assert _texts(regex_spans(text), "credit_card_last4") == ["9982"]


def test_asterisk_prefix_last4():
    text = "card ****4444 was charged"
    assert _texts(regex_spans(text), "credit_card_last4") == ["4444"]


def test_hash_dots_last4():
    text = "Starbucks $6.75, card #...1117 (Discover partial)"
    assert _texts(regex_spans(text), "credit_card_last4") == ["1117"]


def test_ending_variants():
    text = (
        "Mastercard ending 8842 "
        "Best Buy card ending 1111 (Visa) "
        "a card ending in 7821 that wasn't used"
    )
    assert _texts(regex_spans(text), "credit_card_last4") == ["8842", "1111", "7821"]


def test_cvv_patterns():
    text = (
        "exp 09/27, CVV 328. "
        "CVC: 901. "
        "CID: 1234. "
        "CVV: 555. "
        "security code is 999."
    )
    assert _texts(regex_spans(text), "secret") == ["328", "901", "1234", "555", "999"]


def test_aba_routing_valid():
    text = "Chase checking linked: routing 021000021, account 9876543210"
    assert _texts(regex_spans(text), "account_number") == ["021000021"]


def test_aba_routing_checksum_rejects_garbage():
    text = "routing 123456789, account 1"
    assert _texts(regex_spans(text), "account_number") == []


def test_email_detection():
    text = "Contact: fraud-ops@chase.com or r.kim.1979@gmail.com"
    assert sorted(_texts(regex_spans(text), "private_email")) == [
        "fraud-ops@chase.com",
        "r.kim.1979@gmail.com",
    ]


def test_no_false_positive_on_plain_text_numbers():
    text = "See page 4. 1234 people attended. Chapter 7821 begins here."
    spans = regex_spans(text)
    assert _labels(spans, "credit_card_last4") == []
    assert _labels(spans, "secret") == []


def test_merge_dedupes_exact_matches():
    model = [{"label": "private_email", "start": 0, "end": 10,
              "text": "a@b.co.com", "score": 0.9}]
    extra = [{"label": "private_email", "start": 0, "end": 10,
              "text": "a@b.co.com", "score": 1.0}]
    merged = merge_with_model_spans(model, extra)
    assert len(merged) == 1
    assert merged[0]["score"] == 0.9  # model wins exact dupes


def test_merge_keeps_distinct_spans():
    model = [{"label": "private_email", "start": 0, "end": 10,
              "text": "a@b.co.com", "score": 0.9}]
    extra = [{"label": "private_email", "start": 20, "end": 30,
              "text": "c@d.co.com", "score": 1.0}]
    merged = merge_with_model_spans(model, extra)
    assert len(merged) == 2
    assert merged[0]["start"] < merged[1]["start"]


def test_aba_checksum_known_values():
    # Chase NY ABA
    assert _aba_checksum_ok("021000021")
    # Invalid
    assert not _aba_checksum_ok("123456789")
    assert not _aba_checksum_ok("000000000") is False  # trivially valid mathematically
    # Non-digit / wrong length
    assert not _aba_checksum_ok("02100002")
    assert not _aba_checksum_ok("abcdefghi")


def test_confidence_is_one():
    text = "Visa •••• 1234"
    spans = regex_spans(text)
    assert all(s["score"] == 1.0 for s in spans)


def test_adversarial_inputs_no_redos():
    """Quadratic-backtracking regression guard.

    The naive greedy patterns that preceded the anchor-on-needle rewrite
    were O(N²) on these inputs and would hang the event loop near the
    524288-char MAX_TEXT_LENGTH. 500k × 0.5s is a generous ceiling —
    measured real cost on the current implementation is ~15ms.
    """
    import time

    payloads = {
        "mask_chars": ("xX" * 250_000) + " no digits here",
        "hash_dots": "#" + ("." * 100_000) + " nope",
        "dots_only": "." * 500_000,
        "word_chars_no_at": "a" * 500_000,
    }
    for name, payload in payloads.items():
        t0 = time.perf_counter()
        spans = regex_spans(payload)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"{name} took {elapsed:.2f}s (ReDoS regression?)"
        assert not any(s["label"] == "credit_card_last4" for s in spans)
        assert not any(s["label"] == "private_email" for s in spans)


def test_scan_spoken_luhn_card_emits_credit_card_number():
    from app.postprocess import _scan_spoken
    text = (
        "The card number is four, seven, one, six, "
        "three, eight, five, two, "
        "nine, four, oh, one, "
        "two, eight, eight, seven."
    )
    spans = list(_scan_spoken(text))
    cards = [s for s in spans if s["label"] == "credit_card_number"]
    assert len(cards) == 1
    # The span covers the full spoken run — from "four" to "seven".
    first_four = text.index("four")
    last_seven_end = text.rindex("seven") + len("seven")
    assert cards[0]["start"] == first_four
    assert cards[0]["end"] == last_seven_end
    assert cards[0]["score"] == 1.0


def test_scan_spoken_luhn_invalid_16_run_still_emits_card():
    from app.postprocess import _scan_spoken
    # Regression: transcripts contain fixture / misspoken cards that fail
    # Luhn. A 13-19 consecutive spoken digit-word run is overwhelmingly a
    # card in any realistic context, so we still emit credit_card_number
    # (Luhn is informational, not a gate).
    text = (
        "one one one one one one one one "
        "one one one one one one one one"
    )
    spans = list(_scan_spoken(text))
    labels = [s["label"] for s in spans]
    assert "credit_card_number" in labels
    # And NOT a phone, to guard against the old 16 > 15 fall-through.
    assert "private_phone" not in labels


def test_scan_spoken_real_transcript_card_with_ellipsis_separators():
    from app.postprocess import _scan_spoken
    # The exact pattern from a real transcript that leaked before the
    # Luhn-optional fix: 4716 3852 9401 2867 (not Luhn-valid), spoken with
    # "..." group separators and newlines.
    text = (
        "The card number is four, seven, one, six... \n"
        "three, eight, five, two... nine, four, oh, one... "
        "two, eight, six, seven. Expiry is zero nine."
    )
    spans = list(_scan_spoken(text))
    cards = [s for s in spans if s["label"] == "credit_card_number"]
    assert len(cards) == 1
    # Span covers the full 16-digit spoken run.
    assert "four, seven, one, six" in text[cards[0]["start"]:cards[0]["end"]]
    assert "two, eight, six, seven" in text[cards[0]["start"]:cards[0]["end"]]


def test_scan_spoken_phone_double_oh_nine_hundred():
    from app.postprocess import _scan_spoken
    text = (
        "Callback number is zero seven seven double-oh, "
        "nine hundred, seven eight three."
    )
    spans = list(_scan_spoken(text))
    phones = [s for s in spans if s["label"] == "private_phone"]
    assert len(phones) == 1
    # Span covers the full spoken run
    assert "zero seven seven" in text[phones[0]["start"]:phones[0]["end"]]
    assert "seven eight three" in text[phones[0]["start"]:phones[0]["end"]]


def test_scan_spoken_card_beats_phone_on_same_group():
    from app.postprocess import _scan_spoken
    # Valid 16-digit Luhn card — must emit ONLY credit_card_number, not phone.
    # NOTE: use a card whose digit sequence is Luhn-valid. The full-card
    # fixture in test_scan_spoken_luhn_card_emits_credit_card_number (earlier
    # in this file) is known-Luhn-valid; reuse its digit pattern.
    text = (
        "four, seven, one, six, three, eight, five, two, "
        "nine, four, oh, one, two, eight, eight, seven"
    )
    spans = list(_scan_spoken(text))
    labels = [s["label"] for s in spans]
    assert "credit_card_number" in labels
    assert "private_phone" not in labels


def test_scan_spoken_short_run_is_not_phone():
    from app.postprocess import _scan_spoken
    text = "Room one two three four five six seven for the meeting."
    # 7 digits — below the 10-digit phone threshold → no phone span.
    assert not any(s["label"] == "private_phone"
                   for s in _scan_spoken(text))


def test_scan_spoken_cvv_keyword_captures_spoken_digits():
    from app.postprocess import _scan_spoken
    text = "Security code is three, nine, two."
    spans = list(_scan_spoken(text))
    cvv = [s for s in spans if s["label"] == "secret"]
    assert len(cvv) == 1
    assert "three" in text[cvv[0]["start"]:cvv[0]["end"]]
    assert "two" in text[cvv[0]["start"]:cvv[0]["end"]]


def test_scan_spoken_ending_keyword_captures_spoken_last4():
    from app.postprocess import _scan_spoken
    text = "ending in four four five one on file"
    spans = list(_scan_spoken(text))
    last4 = [s for s in spans if s["label"] == "credit_card_last4"]
    assert len(last4) == 1
    run = text[last4[0]["start"]:last4[0]["end"]]
    assert "four four five one" in run


def test_ipv4_public_is_flagged():
    text = "Source IP: 185.220.101.47 (Tor exit node, DE)"
    assert _texts(regex_spans(text), "ip_address") == ["185.220.101.47"]


def test_ipv4_private_ranges_skipped():
    text = (
        "internal 10.0.0.1 and 172.16.5.4 and 192.168.1.10 "
        "and 127.0.0.1 and 169.254.169.254"
    )
    assert _texts(regex_spans(text), "ip_address") == []


def test_ipv4_multicast_and_zero_skipped():
    text = "multicast 224.0.0.1 or 0.0.0.0"
    assert _texts(regex_spans(text), "ip_address") == []


def test_ipv4_out_of_range_octets_skipped():
    text = "not an IP: 999.1.2.3 or 1.2.3.999 or 300.300.300.300"
    assert _texts(regex_spans(text), "ip_address") == []


def test_ipv4_not_adjacent_to_version_strings():
    # "v4.2.1.0" should NOT match (leading alphanum breaks the boundary).
    text = "running on v4.2.1.0 build"
    assert _texts(regex_spans(text), "ip_address") == []


def test_spoken_digits_no_redos():
    """The tokenizer is O(N); this guards against future refactors that
    introduce backtracking. 500 KB of noise with sporadic digit-words must
    process in well under half a second, regardless of whether any labels
    are emitted (one huge 50k-digit group lands outside every rule's length
    bounds and produces no spans — that's fine, the point is the tokenizer
    didn't stall)."""
    import time
    payload = ("zero " * 50_000) + ("noise " * 50_000)  # ~500 KB
    t0 = time.perf_counter()
    spans = regex_spans(payload)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"regex_spans took {elapsed:.2f}s (spoken ReDoS?)"
    # Sanity: regex_spans returned without raising and the result is a list.
    assert isinstance(spans, list)


def test_scan_spoken_ending_keyword_after_group_does_not_trigger():
    from app.postprocess import _scan_spoken
    # Keyword appears AFTER the digit group — must NOT bind.
    text = "four four five one is the ending number on file"
    spans = list(_scan_spoken(text))
    assert not any(s["label"] == "credit_card_last4" for s in spans)


def test_scan_spoken_cvv_keyword_after_group_does_not_trigger():
    from app.postprocess import _scan_spoken
    text = "three nine two is the security code"
    spans = list(_scan_spoken(text))
    assert not any(s["label"] == "secret" for s in spans)
