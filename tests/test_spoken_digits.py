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


from app.spoken_digits import extract_groups


def test_extract_groups_simple_run():
    text = "one two three four five"
    groups = extract_groups(text)
    assert len(groups) == 1
    assert groups[0].digits == "12345"
    assert len(groups[0].spans) == 5
    # First digit span covers "one"
    assert text[groups[0].spans[0][0]:groups[0].spans[0][1]] == "one"


def test_extract_groups_drops_short_runs():
    # Two digits is below the >=3 threshold -> no group.
    assert extract_groups("one two") == []


def test_extract_groups_breaks_on_other():
    text = "one two three apple four five six"
    groups = extract_groups(text)
    # "apple" (OTHER) closes the first group; second group starts after.
    assert len(groups) == 2
    assert groups[0].digits == "123"
    assert groups[1].digits == "456"


def test_extract_groups_respects_long_separator():
    # 7-char SEP run closes the group.
    seven = ",.-,.-,"
    assert len(seven) == 7
    text = f"one two three{seven}four five six"
    groups = extract_groups(text)
    assert len(groups) == 2


def test_extract_groups_keeps_short_separator():
    # 6-char SEP run is fine.
    six = ", ... "
    assert len(six) == 6
    text = f"one two three{six}four five six"
    groups = extract_groups(text)
    assert len(groups) == 1
    assert groups[0].digits == "123456"


def test_extract_groups_double_oh_emits_two_zeros():
    # "zero seven seven double-oh eight nine" → "0770089" (7 digits)
    groups = extract_groups("zero seven seven double-oh eight nine")
    assert len(groups) == 1
    assert groups[0].digits == "0770089"
    # The two emitted zeros from "double-oh" share the same compound span
    # (covering just the literal "double-oh" word-pair).
    zero_spans = [groups[0].spans[3], groups[0].spans[4]]
    assert zero_spans[0] == zero_spans[1]


def test_extract_groups_triple_four():
    # "one two triple-four five six" → "1244456"
    groups = extract_groups("one two triple-four five six")
    assert len(groups) == 1
    assert groups[0].digits == "1244456"


def test_extract_groups_nine_hundred_emits_900():
    groups = extract_groups("zero seven nine hundred four five")
    assert len(groups) == 1
    assert groups[0].digits == "0790045"
    # The three emitted digits from "nine hundred" (9, 0, 0) all share the
    # compound span covering "nine hundred".
    nine_hundred_spans = groups[0].spans[2:5]
    assert len(set(nine_hundred_spans)) == 1


def test_extract_groups_hundred_does_not_absorb_trailing_digits():
    # Phone pattern: "nine hundred, seven eight three" → 900 then 783.
    groups = extract_groups(
        "zero seven seven double-oh nine hundred seven eight three"
    )
    assert len(groups) == 1
    assert groups[0].digits == "07700900783"


def test_fr_digits():
    groups = extract_groups("zéro un deux trois quatre cinq")
    assert len(groups) == 1
    assert groups[0].digits == "012345"


def test_es_digits():
    groups = extract_groups("cero uno dos tres cuatro cinco")
    assert groups[0].digits == "012345"


def test_de_digits():
    groups = extract_groups("null eins zwei drei vier fünf")
    assert groups[0].digits == "012345"


def test_mixed_language_digits_in_one_group():
    # Merged dict means EN + FR + DE + ES in the same run parse cleanly.
    groups = extract_groups("zero uno zwei three four five")
    assert groups[0].digits == "012345"


def test_fr_hundred_cent():
    groups = extract_groups("zéro sept neuf cent quatre cinq")
    assert groups[0].digits == "0790045"


def test_teens_emit_two_digits():
    # "one two three fifteen" → "123" + "15" = "12315" (5 digits)
    groups = extract_groups("one two three fifteen")
    assert len(groups) == 1
    assert groups[0].digits == "12315"


def test_compound_tens_twenty_eight():
    # "zero nine, twenty-eight" → "0928"
    groups = extract_groups("zero nine, twenty-eight")
    assert len(groups) == 1
    assert groups[0].digits == "0928"


def test_year_pattern_nineteen_eighty_two():
    groups = extract_groups("nineteen eighty-two")
    assert len(groups) == 1
    assert groups[0].digits == "1982"


def test_bare_ten_without_digit():
    # "one two thirty" → "12" + "30" = "1230"
    groups = extract_groups("one two thirty")
    assert len(groups) == 1
    assert groups[0].digits == "1230"


def test_de_teen_siebzehn():
    # German 17 is a single word; no hyphen parsing needed.
    # null + eins + siebzehn → 0 + 1 + 17 → "0117"
    groups = extract_groups("null eins siebzehn")
    assert groups[0].digits == "0117"
