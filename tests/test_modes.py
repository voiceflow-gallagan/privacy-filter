from __future__ import annotations

from app.modes import MODE_THRESHOLDS, apply_mode_threshold, threshold_for


def _span(score: float, label: str = "private_email", start: int = 0, end: int = 1) -> dict:
    return {"label": label, "start": start, "end": end, "text": "x", "score": score}


def test_threshold_for_known_modes():
    assert threshold_for("precise") == 0.85
    assert threshold_for("balanced") == 0.55
    assert threshold_for("recall") == 0.0


def test_threshold_for_unknown_mode_is_zero():
    assert threshold_for("aggressive") == 0.0


def test_precise_drops_low_confidence():
    spans = [_span(0.49), _span(0.70), _span(0.86), _span(0.95)]
    kept = apply_mode_threshold(spans, "precise")
    assert [s["score"] for s in kept] == [0.86, 0.95]


def test_balanced_keeps_test_fixtures():
    # These scores come from tests/test_mask.py fixtures and must survive
    # in the default mode, otherwise real overlap-mask behaviour breaks.
    spans = [_span(0.60, label="private_address"),
             _span(0.70, label="account_number")]
    kept = apply_mode_threshold(spans, "balanced")
    assert len(kept) == 2


def test_balanced_drops_sub_fifty_five():
    spans = [_span(0.49), _span(0.54), _span(0.55), _span(0.60)]
    kept = apply_mode_threshold(spans, "balanced")
    assert [s["score"] for s in kept] == [0.55, 0.60]


def test_recall_keeps_everything():
    spans = [_span(0.1), _span(0.49), _span(0.99)]
    kept = apply_mode_threshold(spans, "recall")
    assert len(kept) == 3


def test_regex_spans_always_survive():
    # Regex postprocess spans are score 1.0 by construction.
    spans = [_span(1.0, label="credit_card_last4")]
    for mode in MODE_THRESHOLDS:
        assert len(apply_mode_threshold(spans, mode)) == 1


def test_missing_score_treated_as_one():
    spans = [{"label": "private_email", "start": 0, "end": 1, "text": "x"}]
    assert len(apply_mode_threshold(spans, "precise")) == 1
