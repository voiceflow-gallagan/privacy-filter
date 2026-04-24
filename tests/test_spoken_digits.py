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
