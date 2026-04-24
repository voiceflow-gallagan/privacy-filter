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
# Phone-context cue words within the wider lookback window. When present,
# a "ending in NNNN" match should be labeled phone_last4 instead of the
# default credit_card_last4 — transcripts often say "phone number on file
# ending in 4421" where the last-4 belongs to the phone, not a card.
_PHONE_CONTEXT_CUE = re.compile(
    r"\b(?:phone|mobile|cell|telephone|contact\s+number|"
    r"callback\s+number|numéro\s+de\s+téléphone|portable)\b",
    re.IGNORECASE,
)
# SSN-context cue words. "last four digits of your Social Security Number"
# → ssn_last4, not credit_card_last4.
_SSN_CONTEXT_CUE = re.compile(
    r"\b(?:SSN|SS#|social\s+security(?:\s+(?:number|no\.?))?)\b",
    re.IGNORECASE,
)
_LAST4_PHONE_CONTEXT_WINDOW = 128

# Full US SSN: 3-2-4 with hyphens. Unambiguous format, emit ssn directly.
_SSN_FULL = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# SSN last-4 is an "last four" + "Social Security" pair followed by a
# 4-digit group within a bounded window. Phrasings like "please confirm
# the last four digits of your Social Security Number 7742" bridge the
# keyword and the digits across ~30 chars, so the tight ending-prefix
# anchor isn't enough — we scan for the keyword pair first, then look
# ahead for the first 4-digit group within the post-keyword budget.
_SSN_LAST4_KEYWORDS = re.compile(
    r"\blast\s+four(?:\s+digits)?\b[^\n]{0,80}?\bsocial\s+security\b",
    re.IGNORECASE,
)
_FOUR_DIGITS_PLAIN = re.compile(r"\b(\d{4})\b")
_SSN_LAST4_POST_WINDOW = 40

# OTP / verification code: keyword then a 4-8 digit group within a short
# post-keyword window. Same two-step structure as SSN: match the cue,
# look ahead for the digits. Tolerates connectives like "to you:",
# "sent to your phone", "sent was" that a single fused regex can't cover.
_OTP_KEYWORDS = re.compile(
    r"\b(?:OTP|verification\s+code|one[-\s]?time\s+(?:password|code|pin)"
    r"|code\s+we\s+sent|passcode|confirmation\s+code)\b",
    re.IGNORECASE,
)
_OTP_DIGITS = re.compile(r"\b(\d{4,8})\b")
_OTP_POST_WINDOW = 40

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
    `#...1117` shorthand, and `ending [in] NNNN`. The default label is
    `credit_card_last4`, but `ending [in] NNNN` with a phone cue word
    ("phone", "mobile", "cell", …) in the wider 128-char lookback is
    routed to `phone_last4` so downstream consumers don't mislabel.
    """
    for m in _FOUR_DIGITS_WORD.finditer(text):
        start, end = m.start(), m.end()
        ctx = text[max(0, start - _LAST4_WINDOW):start]
        is_masked = _MASK_PREFIX.search(ctx) or _HASH_DOTS_PREFIX.search(ctx)
        is_ending = _ENDING_PREFIX.search(ctx)
        if not (is_masked or is_ending):
            continue

        label = "credit_card_last4"
        if is_ending and not is_masked:
            wide_ctx = text[max(0, start - _LAST4_PHONE_CONTEXT_WINDOW):start]
            if _SSN_CONTEXT_CUE.search(wide_ctx):
                label = "ssn_last4"
            elif _PHONE_CONTEXT_CUE.search(wide_ctx):
                label = "phone_last4"

        yield {
            "label": label,
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

    for m in _SSN_FULL.finditer(text):
        out.append({"label": "ssn", "start": m.start(), "end": m.end(),
                    "text": m.group(0), "score": 1.0})

    for m in _SSN_LAST4_KEYWORDS.finditer(text):
        tail = text[m.end():m.end() + _SSN_LAST4_POST_WINDOW]
        dm = _FOUR_DIGITS_PLAIN.search(tail)
        if dm:
            start = m.end() + dm.start(1)
            end = m.end() + dm.end(1)
            out.append({"label": "ssn_last4", "start": start, "end": end,
                        "text": text[start:end], "score": 1.0})

    for m in _OTP_KEYWORDS.finditer(text):
        tail = text[m.end():m.end() + _OTP_POST_WINDOW]
        dm = _OTP_DIGITS.search(tail)
        if dm:
            start = m.end() + dm.start(1)
            end = m.end() + dm.end(1)
            out.append({"label": "otp", "start": start, "end": end,
                        "text": text[start:end], "score": 1.0})

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


_NAME_WORD_MIN = 3
# Letter runs — accented Latin included so FR/ES/DE names tokenize properly.
_NAME_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")


def _ranges_overlap(span: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    a, b = span
    for s, e in ranges:
        if s < b and e > a:
            return True
    return False


def augment_person_coverage(text: str, entities: list[dict]) -> list[dict]:
    """Find uncovered occurrences of names already identified by the model.

    The model sometimes misses short isolated mentions of a name when a
    longer multi-token span covered the same surface elsewhere ("David
    Chen" caught early, bare "David" missed later). This scans the text
    for word-boundary case-sensitive occurrences of every 3+ character
    name word seen in existing private_person spans, and emits new
    private_person spans for matches not already covered by any entity.

    Case-sensitive matching prevents false positives on accidental
    collisions like "mark" → "Mark", while still catching every casing
    the model itself has validated.
    """
    words: set[str] = set()
    for e in entities:
        if e.get("label") != "private_person":
            continue
        for w in _NAME_WORD_RE.findall(e.get("text", "") or ""):
            if len(w) >= _NAME_WORD_MIN:
                words.add(w)

    if not words:
        return []

    covered = [(int(e["start"]), int(e["end"])) for e in entities]
    extra: list[dict] = []
    for word in words:
        pattern = re.compile(rf"\b{re.escape(word)}\b")
        for m in pattern.finditer(text):
            if _ranges_overlap((m.start(), m.end()), covered):
                continue
            extra.append({
                "label": "private_person",
                "start": m.start(),
                "end": m.end(),
                "text": m.group(0),
                "score": 1.0,
            })
            covered.append((m.start(), m.end()))
    return extra


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
