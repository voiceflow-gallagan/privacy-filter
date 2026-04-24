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
