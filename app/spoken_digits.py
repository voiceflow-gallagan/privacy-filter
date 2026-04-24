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
