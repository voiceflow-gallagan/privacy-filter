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
