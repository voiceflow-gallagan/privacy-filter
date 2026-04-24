from __future__ import annotations

from app.spoken_digits import DIGIT_WORDS, DigitGroup


def test_en_digits_mapped_to_values():
    assert DIGIT_WORDS["zero"] == 0
    assert DIGIT_WORDS["oh"] == 0
    for i, w in enumerate(
        ["zero", "one", "two", "three", "four",
         "five", "six", "seven", "eight", "nine"]
    ):
        assert DIGIT_WORDS[w] == i


def test_digit_group_is_dataclass_like():
    g = DigitGroup(digits="123", spans=[(0, 3), (4, 7), (8, 11)])
    assert g.digits == "123"
    assert len(g.spans) == 3


from app.spoken_digits import TokenKind, tokenize


def test_tokenize_basic_digits_and_seps():
    tokens = list(tokenize("zero one two"))
    kinds = [t.kind for t in tokens]
    assert kinds == [
        TokenKind.DIGIT, TokenKind.SEP,
        TokenKind.DIGIT, TokenKind.SEP,
        TokenKind.DIGIT,
    ]
    values = [t.value for t in tokens if t.kind == TokenKind.DIGIT]
    assert values == [0, 1, 2]
    # Spans point back into the original text
    starts = [t.start for t in tokens if t.kind == TokenKind.DIGIT]
    assert starts == [0, 5, 9]


def test_tokenize_emits_other_for_non_digit_words():
    tokens = list(tokenize("hello one world"))
    kinds = [t.kind for t in tokens]
    assert TokenKind.OTHER in kinds
    assert TokenKind.DIGIT in kinds


def test_tokenize_case_insensitive():
    tokens = list(tokenize("ZERO ONE"))
    digit_values = [t.value for t in tokens if t.kind == TokenKind.DIGIT]
    assert digit_values == [0, 1]


def test_tokenize_word_boundary_respected():
    # "zerox" must NOT match "zero"
    tokens = list(tokenize("zerox"))
    assert all(t.kind != TokenKind.DIGIT for t in tokens)


def test_tokenize_bounded_separator():
    text = "one" + (",.-,.-,") + "two"
    tokens = list(tokenize(text))
    sep_tokens = [t for t in tokens if t.kind == TokenKind.SEP]
    assert sep_tokens, "expected at least one SEP between the digit words"
