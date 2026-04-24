from __future__ import annotations

import re
from typing import Iterator

from app.spoken_digits import extract_groups


# The naive patterns (e.g. `[mask]{2,}(?:...)*[\s\-]*\d{4}`) blow up to O(N²)
# on adversarial mask-char-only input (`xXxX…` * 250k) because `finditer`
# tries every starting position and at each one the engine walks forward to
# discover no trailing digits exist. Possessive quantifiers *at a single
# position* don't help — the cost is per-position-scan, not per-backtrack.
#
# Fix: anchor on the rarest token. For partial cards that's `\d{4}\b`; for
# emails it's `@`. We locate those cheaply in one pass, then validate a
# bounded preceding/following context. Worst-case work is O(N + k·window)
# where k is the hit count.
_MASK_CHARS_CLASS = r"[•·●*xX✕×]"

_FOUR_DIGITS_WORD = re.compile(r"(?<!\d)\d{4}\b")

# All prefix patterns end with `$` so they must land immediately before the
# 4-digit group when applied to a bounded text[pos-WINDOW:pos] slice.
_MASK_PREFIX = re.compile(
    rf"{_MASK_CHARS_CLASS}{{2,}}(?:[\s\-]{{0,3}}{_MASK_CHARS_CLASS}+){{0,8}}"
    rf"[\s\-]{{0,3}}$"
)
_HASH_DOTS_PREFIX = re.compile(r"#\s*\.{3,}\s*$")
_ENDING_PREFIX = re.compile(r"\bending(?:\s+in)?\s+$", re.IGNORECASE)

_CVV = re.compile(
    r"\b(?:CVV|CVC|CID|security\s+code)\b\s*(?:is\s+|[:=]\s*)?(\d{3,4})\b",
    re.IGNORECASE,
)
_SPOKEN_CVV_KEYWORD = re.compile(
    r"\b(?:CVV|CVC|CID|security\s+code)\b[\s:=]*(?:is\s+)?",
    re.IGNORECASE,
)
_SPOKEN_ENDING_KEYWORD = re.compile(
    r"\bending(?:\s+in)?\s+",
    re.IGNORECASE,
)
_KEYWORD_BRIDGE_GAP = 20
_ROUTING = re.compile(
    r"\brouting(?:\s+(?:number|no\.?))?[\s:#]*(\d{9})\b", re.IGNORECASE
)

_AT_SIGN = re.compile(r"@")
_LOCAL_PART_SUFFIX = re.compile(r"[\w.+\-]+$")
_DOMAIN_PREFIX = re.compile(r"^[\w\-]+(?:\.[\w\-]+)+")

_LAST4_WINDOW = 64
_LOCAL_PART_WINDOW = 64
_DOMAIN_WINDOW = 255


def _luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _aba_checksum_ok(digits: str) -> bool:
    if len(digits) != 9 or not digits.isdigit():
        return False
    d = [int(c) for c in digits]
    total = 3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])
    return total % 10 == 0


def _scan_last4(text: str) -> Iterator[dict]:
    """4-digit groups preceded by a partial-card shorthand.

    Covers bullet/X/asterisk masks (`•••• 3421`, `XXXX-9982`, `****4444`),
    `#...1117` shorthand, and `ending [in] NNNN`.
    """
    for m in _FOUR_DIGITS_WORD.finditer(text):
        start, end = m.start(), m.end()
        ctx = text[max(0, start - _LAST4_WINDOW):start]
        if (_MASK_PREFIX.search(ctx)
                or _HASH_DOTS_PREFIX.search(ctx)
                or _ENDING_PREFIX.search(ctx)):
            yield {
                "label": "credit_card_last4",
                "start": start,
                "end": end,
                "text": m.group(0),
                "score": 1.0,
            }


def _scan_emails(text: str) -> Iterator[dict]:
    for m in _AT_SIGN.finditer(text):
        at = m.start()
        left_ctx = text[max(0, at - _LOCAL_PART_WINDOW):at]
        right_ctx = text[at + 1:at + 1 + _DOMAIN_WINDOW]
        lm = _LOCAL_PART_SUFFIX.search(left_ctx)
        rm = _DOMAIN_PREFIX.match(right_ctx)
        if lm and rm:
            start = at - len(lm.group(0))
            end = at + 1 + len(rm.group(0))
            yield {
                "label": "private_email",
                "start": start,
                "end": end,
                "text": text[start:end],
                "score": 1.0,
            }


def _scan_spoken(text: str) -> Iterator[dict]:
    """Emit spans for PII expressed as spelled-out digits.

    Rule 1: Luhn-validated credit card (13-19 digits).
    Rule 2: phone / long numeric ID (10-15 digits), only if Rule 1 missed.
    Rule 3: keyword-anchored spoken CVV / ending-last-4. The keyword runs
            in the original text, the digits in the spoken group; if the
            group starts within 20 chars after a keyword hit, emit the
            mapped label.
    """
    for group in extract_groups(text):
        if _luhn_valid(group.digits):
            start = group.spans[0][0]
            end = group.spans[-1][1]
            yield {
                "label": "credit_card_number",
                "start": start,
                "end": end,
                "text": text[start:end],
                "score": 1.0,
            }
            continue

        if 10 <= len(group.digits) <= 15:
            start = group.spans[0][0]
            end = group.spans[-1][1]
            yield {
                "label": "private_phone",
                "start": start,
                "end": end,
                "text": text[start:end],
                "score": 1.0,
            }
            continue

        group_start = group.spans[0][0]
        span_start = group.spans[0][0]
        span_end = group.spans[-1][1]

        if 3 <= len(group.digits) <= 4:
            for km in _SPOKEN_CVV_KEYWORD.finditer(text):
                if 0 <= group_start - km.end() <= _KEYWORD_BRIDGE_GAP:
                    yield {
                        "label": "secret",
                        "start": span_start,
                        "end": span_end,
                        "text": text[span_start:span_end],
                        "score": 1.0,
                    }
                    break

        if len(group.digits) == 4:
            for km in _SPOKEN_ENDING_KEYWORD.finditer(text):
                if 0 <= group_start - km.end() <= _KEYWORD_BRIDGE_GAP:
                    yield {
                        "label": "credit_card_last4",
                        "start": span_start,
                        "end": span_end,
                        "text": text[span_start:span_end],
                        "score": 1.0,
                    }
                    break


def regex_spans(text: str) -> list[dict]:
    """Run deterministic regex rules across the full text.

    Emits score-1.0 spans for: partial credit-card shorthands
    (credit_card_last4), CVV/CVC/CID/security codes (secret), ABA routing
    numbers (account_number, checksum-validated), and every email
    (private_email).
    """
    out: list[dict] = list(_scan_last4(text))

    for m in _CVV.finditer(text):
        start, end = m.span(1)
        out.append({"label": "secret", "start": start, "end": end,
                    "text": text[start:end], "score": 1.0})

    for m in _ROUTING.finditer(text):
        start, end = m.span(1)
        digits = text[start:end]
        if _aba_checksum_ok(digits):
            out.append({"label": "account_number", "start": start, "end": end,
                        "text": digits, "score": 1.0})

    out.extend(_scan_emails(text))
    return out


def merge_with_model_spans(model_spans: list[dict], extra: list[dict]) -> list[dict]:
    """Add regex spans to model spans, deduping exact (start, end, label) matches.

    Model spans keep their original score on conflict; regex adds only genuinely
    new ranges. Mask-time range union (in routes/detect.py) handles overlaps
    across different labels.
    """
    seen: set[tuple[int, int, str]] = {
        (s["start"], s["end"], s["label"]) for s in model_spans
    }
    out = list(model_spans)
    for s in extra:
        key = (s["start"], s["end"], s["label"])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out
