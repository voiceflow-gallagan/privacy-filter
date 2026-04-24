from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional


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


DIGIT_WORDS: dict[str, int] = {
    # English
    "zero": 0, "oh": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
    # French
    "zéro": 0,
    "un": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "sept": 7, "huit": 8, "neuf": 9,
    # (French "six" collides with English → already 6; OK)
    # Spanish
    "cero": 0,
    "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    # German
    "null": 0,
    "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
}

MULT_WORDS: dict[str, int] = {
    # English
    "double": 2, "triple": 3,
    # French
    "doublé": 2,
    # Spanish
    "doble": 2,
    # German
    "doppel": 2, "dreifach": 3,
}

HUNDRED_WORDS: frozenset[str] = frozenset({
    "hundred",         # EN
    "cent",            # FR
    "cien", "ciento",  # ES
    "hundert",         # DE
})

# Teens (10-19): each emits exactly two digits. Used for spoken 4-digit
# expiries, ages, and two-word years ("nineteen eighty-two" → 19 then 82).
# FR/DE use hyphenated or agglutinative forms for 17-19 — those get parsed
# via the TENS + DIGIT compound path at the grammar level, or via German
# single-word compounds (siebzehn = 17) already listed.
TEEN_WORDS: dict[str, int] = {
    # English
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    # French
    "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14,
    "quinze": 15, "seize": 16,
    # Spanish
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    # German
    "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14,
    "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
}

# Tens (20, 30, …, 90). Combined with a trailing DIGIT via hyphen or short
# SEP to build compounds like "twenty-eight" → 28, "eighty-two" → 82.
TENS_WORDS: dict[str, int] = {
    # English
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    # French
    "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50,
    "soixante": 60,
    # Spanish
    "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
    # German
    "zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50,
    "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90,
}


class TokenKind(str, Enum):
    DIGIT = "DIGIT"
    MULT = "MULT"
    HUNDRED = "HUNDRED"
    TEEN = "TEEN"     # 10-19, emits 2 digits
    TENS = "TENS"     # 20/30/…/90, emits 2 digits (compounds with DIGIT)
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
    if lw in TEEN_WORDS:
        return TokenKind.TEEN, TEEN_WORDS[lw]
    if lw in TENS_WORDS:
        return TokenKind.TENS, TENS_WORDS[lw]
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
        elif tok.kind == TokenKind.TEEN:
            # Teens emit exactly 2 digits (10-19). The compound span covers
            # just this word.
            current_digits.extend(list(f"{tok.value:02d}"))
            current_spans.extend([(tok.start, tok.end)] * 2)
            i += 1
            continue
        elif tok.kind == TokenKind.TENS:
            # Tens (20/30/…/90) optionally combine with a trailing DIGIT via
            # a short SEP (hyphen or space) to form compounds: "twenty-eight"
            # → 28. Without a trailing DIGIT, emit the bare ten ("20").
            la = i + 1
            if (la < len(tokens)
                    and tokens[la].kind == TokenKind.SEP
                    and len(tokens[la].text) <= _MAX_SEP_CHARS):
                la += 1
            if la < len(tokens) and tokens[la].kind == TokenKind.DIGIT:
                combined = (tok.value or 0) + (tokens[la].value or 0)
                compound_span = (tok.start, tokens[la].end)
                current_digits.extend(list(f"{combined:02d}"))
                current_spans.extend([compound_span] * 2)
                i = la + 1
                continue
            current_digits.extend(list(f"{(tok.value or 0):02d}"))
            current_spans.extend([(tok.start, tok.end)] * 2)
            i += 1
            continue
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
