from __future__ import annotations

# Thresholds picked from observed model scores in fixtures and real docs:
# model emits spans in roughly [0.45, 0.99]. Regex postprocess spans are 1.0.
#
# - precise (0.85): keep high-confidence spans only. Drops fragmented
#   expiration-date sub-spans, weak whitespace-only sub-tokens, and most
#   model noise. Some true positives may be lost.
# - balanced (0.55): default. Keeps the weakest legitimate signals observed
#   in the test corpus (0.6 address, 0.7 account) while dropping the
#   sub-0.5 fragments the BIOES decoder sometimes emits.
# - recall (0.0): no threshold. Every model span flows through.
MODE_THRESHOLDS: dict[str, float] = {
    "precise": 0.85,
    "balanced": 0.55,
    "recall": 0.0,
}


def threshold_for(mode: str) -> float:
    return MODE_THRESHOLDS.get(mode, 0.0)


def apply_mode_threshold(spans: list[dict], mode: str) -> list[dict]:
    """Drop spans whose score falls below the mode's threshold.

    Regex post-processor spans (score 1.0) always survive. Spans missing a
    score are treated as confidence 1.0 (defensive — the pipeline always
    attaches one, but we don't want a typo to silently nuke results).
    """
    t = threshold_for(mode)
    if t <= 0.0:
        return list(spans)
    return [s for s in spans if s.get("score", 1.0) >= t]
