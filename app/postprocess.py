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
_SPOKEN_EXPIRY_KEYWORD = re.compile(
    r"\b(?:expir(?:y|es|ed|ation)|exp\.?)\b[\s:=]*(?:is\s+)?",
    re.IGNORECASE,
)
_KEYWORD_BRIDGE_GAP = 20
_ROUTING = re.compile(
    r"\brouting(?:\s+(?:number|no\.?))?[\s:#]*(\d{9})\b", re.IGNORECASE
)

_AT_SIGN = re.compile(r"@")
_LOCAL_PART_SUFFIX = re.compile(r"[\w.+\-]+$")
_DOMAIN_PREFIX = re.compile(r"^[\w\-]+(?:\.[\w\-]+)+")

# key=value partial-card / partial-phone shorthands in structured log lines:
# `last4=2867`, `card_last4: 1234`, `phone_last4=0783`. Anchored on the
# keyword so it can't ReDoS on adversarial input.
_KV_CARD_LAST4 = re.compile(
    r"\b(?:card_last4|card_last_4|last4|last_4)\s*[:=]\s*(\d{4})\b",
    re.IGNORECASE,
)
_KV_PHONE_LAST4 = re.compile(
    r"\b(?:phone_last4|phone_last_4|mobile_last4)\s*[:=]\s*(\d{4})\b",
    re.IGNORECASE,
)

# Public IPv4: four dot-separated integers, each 0-255. Octet-range check
# runs after the regex. Private / loopback / link-local ranges are skipped
# so internal infrastructure identifiers don't clog results.
_IPV4 = re.compile(
    r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.])"
)

# US-style parenthesised phone: (XXX) XXX-XXXX with common separator
# variants. The model often catches "XXX) XXX-XXXX" but drops the leading
# paren and a stray "(" is left in the redacted output.
_US_PHONE_PAREN = re.compile(
    r"\(\d{3}\)\s?\d{3}[-.\s]\d{4}\b"
)

# ISO-8601 timestamps including optional fractional seconds and timezone.
# The model often fragments these ("…18.", "443Z", "443Z] session_start…"),
# each fragment picking up a different label. Emitting one deterministic
# span over the whole timestamp lets the mask-time union subsume the
# fragments so the final output is a single clean [REDACTED].
_ISO8601 = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"[T ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
)

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


def _is_public_ipv4(octets: tuple[int, int, int, int]) -> bool:
    """Return True for globally-routable IPv4 addresses.

    Skips RFC 1918 private ranges, loopback, link-local, multicast, and
    reserved blocks — the stuff that's never PII but shows up in internal
    logs all the time.
    """
    a, b, c, d = octets
    if not all(0 <= x <= 255 for x in octets):
        return False
    if a == 10:
        return False
    if a == 172 and 16 <= b <= 31:
        return False
    if a == 192 and b == 168:
        return False
    if a == 127:
        return False
    if a == 169 and b == 254:
        return False
    if a == 0:
        return False
    if a >= 224:  # multicast + reserved
        return False
    return True


def _scan_us_phone_paren(text: str) -> "Iterator[dict]":
    for m in _US_PHONE_PAREN.finditer(text):
        yield {
            "label": "private_phone",
            "start": m.start(),
            "end": m.end(),
            "text": m.group(0),
            "score": 1.0,
        }


def _scan_iso8601(text: str) -> "Iterator[dict]":
    for m in _ISO8601.finditer(text):
        yield {
            "label": "private_date",
            "start": m.start(),
            "end": m.end(),
            "text": m.group(0),
            "score": 1.0,
        }


def _scan_ipv4(text: str) -> "Iterator[dict]":
    for m in _IPV4.finditer(text):
        octets = tuple(int(m.group(i)) for i in (1, 2, 3, 4))
        if not _is_public_ipv4(octets):
            continue
        yield {
            "label": "ip_address",
            "start": m.start(),
            "end": m.end(),
            "text": m.group(0),
            "score": 1.0,
        }


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

    Rule 1: credit card (13-19 digits). Luhn validity is a nice-to-have,
            not a gate — real call transcripts include misspoken digits and
            fixture numbers that fail Luhn, but a 13-19 consecutive spoken
            digit-word run is overwhelmingly a card in any realistic context.
    Rule 2: phone / long numeric ID (10-12 digits), only if the group length
            didn't land in the credit-card range.
    Rule 3: keyword-anchored spoken CVV / ending-last-4. The keyword runs
            in the original text, the digits in the spoken group; if the
            group starts within 20 chars after a keyword hit, emit the
            mapped label.
    """
    cvv_keyword_ends: list[int] = [m.end() for m in _SPOKEN_CVV_KEYWORD.finditer(text)]
    ending_keyword_ends: list[int] = [m.end() for m in _SPOKEN_ENDING_KEYWORD.finditer(text)]
    expiry_keyword_ends: list[int] = [m.end() for m in _SPOKEN_EXPIRY_KEYWORD.finditer(text)]

    for group in extract_groups(text):
        n = len(group.digits)
        if 13 <= n <= 19:
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

        if 10 <= n <= 12:
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
            for k_end in cvv_keyword_ends:
                if 0 <= group_start - k_end <= _KEYWORD_BRIDGE_GAP:
                    yield {
                        "label": "secret",
                        "start": span_start,
                        "end": span_end,
                        "text": text[span_start:span_end],
                        "score": 1.0,
                    }
                    break

        if len(group.digits) == 4:
            for k_end in ending_keyword_ends:
                if 0 <= group_start - k_end <= _KEYWORD_BRIDGE_GAP:
                    yield {
                        "label": "credit_card_last4",
                        "start": span_start,
                        "end": span_end,
                        "text": text[span_start:span_end],
                        "score": 1.0,
                    }
                    break

        # Rule 4: spoken card expiry (MMYY packed). "Expiry is zero nine,
        # twenty-eight" → digits "0928" → private_date. Requires the teens
        # / compound-tens grammar in spoken_digits to reach 4 digits.
        if len(group.digits) == 4:
            for k_end in expiry_keyword_ends:
                if 0 <= group_start - k_end <= _KEYWORD_BRIDGE_GAP:
                    yield {
                        "label": "private_date",
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
    numbers (account_number, checksum-validated), every email
    (private_email), and spoken-digit runs (credit_card_number,
    private_phone, credit_card_last4, secret).
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

    for m in _KV_CARD_LAST4.finditer(text):
        start, end = m.span(1)
        out.append({"label": "credit_card_last4", "start": start, "end": end,
                    "text": text[start:end], "score": 1.0})

    for m in _KV_PHONE_LAST4.finditer(text):
        start, end = m.span(1)
        out.append({"label": "private_phone", "start": start, "end": end,
                    "text": text[start:end], "score": 1.0})

    out.extend(_scan_emails(text))
    out.extend(_scan_ipv4(text))
    out.extend(_scan_iso8601(text))
    out.extend(_scan_us_phone_paren(text))
    out.extend(_scan_spoken(text))
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
