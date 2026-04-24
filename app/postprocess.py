from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class RegexRule:
    name: str
    pattern: re.Pattern[str]
    label: str
    group: int = 1
    validator: Optional[Callable[[str], bool]] = None


_MASK_CHARS = r"[•·●*xX✕×]"

_LAST4_MASKED = re.compile(
    rf"{_MASK_CHARS}{{2,}}(?:[\s\-]*{_MASK_CHARS}+)*[\s\-]*(\d{{4}})\b"
)
_LAST4_HASH_DOTS = re.compile(r"#\s*\.{3,}\s*(\d{4})\b")
_LAST4_ENDING = re.compile(
    r"\bending(?:\s+in)?\s+(\d{4})\b", re.IGNORECASE
)
_CVV = re.compile(
    r"\b(?:CVV|CVC|CID|security\s+code)\b\s*(?:is\s+|[:=]\s*)?(\d{3,4})\b",
    re.IGNORECASE,
)
_ROUTING = re.compile(
    r"\brouting(?:\s+(?:number|no\.?))?[\s:#]*(\d{9})\b", re.IGNORECASE
)
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")


def _aba_checksum_ok(digits: str) -> bool:
    if len(digits) != 9 or not digits.isdigit():
        return False
    d = [int(c) for c in digits]
    total = 3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])
    return total % 10 == 0


RULES: tuple[RegexRule, ...] = (
    RegexRule("last4_masked", _LAST4_MASKED, "credit_card_last4"),
    RegexRule("last4_hash_dots", _LAST4_HASH_DOTS, "credit_card_last4"),
    RegexRule("last4_ending", _LAST4_ENDING, "credit_card_last4"),
    RegexRule("cvv", _CVV, "secret"),
    RegexRule("aba_routing", _ROUTING, "account_number", validator=_aba_checksum_ok),
    RegexRule("email", _EMAIL, "private_email", group=0),
)


def regex_spans(text: str) -> list[dict]:
    """Run deterministic regex rules across the full text.

    Returned spans are confidence-1.0 because they encode hard patterns
    (mask-prefix + 4 digits, ABA checksum, labelled CVV, literal @).
    """
    out: list[dict] = []
    for rule in RULES:
        for m in rule.pattern.finditer(text):
            start, end = m.span(rule.group)
            if start == end:
                continue
            surface = text[start:end]
            if rule.validator and not rule.validator(surface):
                continue
            out.append({
                "label": rule.label,
                "start": start,
                "end": end,
                "text": surface,
                "score": 1.0,
            })
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
