from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DigitGroup:
    """A run of consecutive spoken-digit words extracted from text.

    digits: the normalized digit string, e.g. "07700900783"
    spans: one (start, end) per emitted digit, pointing back into the
           original text. Compound forms ("double-oh", "nine hundred")
           emit multiple digits that all share the same compound span.
    """
    digits: str
    spans: list[tuple[int, int]]


# Merged dict across EN/FR/ES/DE. Populated incrementally; EN first.
DIGIT_WORDS: dict[str, int] = {
    # English
    "zero": 0, "oh": 0, "o": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

# Multiplier prefixes: word → repeat count.
MULT_WORDS: dict[str, int] = {
    "double": 2, "triple": 3,
}

# "hundred" equivalents across languages.
HUNDRED_WORDS: frozenset[str] = frozenset({
    "hundred",
})


import re
from enum import Enum
from typing import Iterator, Optional


class TokenKind(str, Enum):
    DIGIT = "DIGIT"
    MULT = "MULT"
    HUNDRED = "HUNDRED"
    SEP = "SEP"
    OTHER = "OTHER"


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    kind: TokenKind
    value: Optional[int] = None
    text: str = ""


_TOKEN_RE = re.compile(
    r"[A-Za-zÀ-ÿ]+"
    r"|[^\w\s]+"
    r"|\s+"
)


def _classify_word(word: str) -> tuple[TokenKind, Optional[int]]:
    lw = word.lower()
    if lw in DIGIT_WORDS:
        return TokenKind.DIGIT, DIGIT_WORDS[lw]
    if lw in MULT_WORDS:
        return TokenKind.MULT, MULT_WORDS[lw]
    if lw in HUNDRED_WORDS:
        return TokenKind.HUNDRED, None
    return TokenKind.OTHER, None


def tokenize(text: str) -> Iterator[Token]:
    """Yield Tokens covering every character of `text`. Non-alphabetic runs
    (whitespace + punctuation) are emitted as SEP; alphabetic runs are
    classified via the digit / multiplier / hundred dictionaries."""
    for m in _TOKEN_RE.finditer(text):
        chunk = m.group(0)
        start, end = m.start(), m.end()
        if chunk[0].isalpha():
            kind, value = _classify_word(chunk)
            yield Token(start=start, end=end, kind=kind,
                        value=value, text=chunk)
        else:
            yield Token(start=start, end=end, kind=TokenKind.SEP, text=chunk)


_MAX_SEP_CHARS = 6
_MIN_GROUP_DIGITS = 3


def extract_groups(text: str) -> list[DigitGroup]:
    """Parse consecutive digit-word runs into DigitGroup objects.

    A group is a maximal sequence of DIGIT tokens separated by SEP runs of
    <= 6 characters. OTHER, MULT, HUNDRED, or a longer SEP close the group.
    Groups with fewer than 3 emitted digits are discarded.

    MULT and HUNDRED close the group in v1 of this function; Tasks 5 and 6
    extend the behaviour to handle compound forms.
    """
    groups: list[DigitGroup] = []
    current_digits: list[str] = []
    current_spans: list[tuple[int, int]] = []

    def flush() -> None:
        if len(current_digits) >= _MIN_GROUP_DIGITS:
            groups.append(DigitGroup(
                digits="".join(current_digits),
                spans=list(current_spans),
            ))
        current_digits.clear()
        current_spans.clear()

    tokens = list(tokenize(text))
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.kind == TokenKind.DIGIT:
            # Look ahead for HUNDRED (optionally via one short SEP).
            la = i + 1
            if (la < len(tokens)
                    and tokens[la].kind == TokenKind.SEP
                    and len(tokens[la].text) <= _MAX_SEP_CHARS):
                la += 1
            if la < len(tokens) and tokens[la].kind == TokenKind.HUNDRED:
                compound_span = (tok.start, tokens[la].end)
                # Emit DIGIT + "00" (exactly 3 chars)
                current_digits.extend([str(tok.value), "0", "0"])
                current_spans.extend([compound_span] * 3)
                i = la + 1
                continue
            current_digits.append(str(tok.value))
            current_spans.append((tok.start, tok.end))
            i += 1
            continue
        elif tok.kind == TokenKind.SEP:
            if len(tok.text) > _MAX_SEP_CHARS:
                flush()
            i += 1
        elif tok.kind == TokenKind.MULT:
            # MULT must be followed (optionally via one short SEP) by a DIGIT.
            nxt_idx = i + 1
            if (nxt_idx < len(tokens)
                    and tokens[nxt_idx].kind == TokenKind.SEP
                    and len(tokens[nxt_idx].text) <= _MAX_SEP_CHARS):
                nxt_idx += 1
            if nxt_idx < len(tokens) and tokens[nxt_idx].kind == TokenKind.DIGIT:
                digit = tokens[nxt_idx]
                # Compound span covers MULT start through DIGIT end.
                compound_span = (tok.start, digit.end)
                for _ in range(tok.value or 1):
                    current_digits.append(str(digit.value))
                    current_spans.append(compound_span)
                i = nxt_idx + 1
                continue
            # Lone MULT → close group.
            flush()
            i += 1
        else:
            # OTHER / HUNDRED — still close the group in this task.
            # Task 6 will replace the HUNDRED handling.
            flush()
            i += 1

    flush()
    return groups
