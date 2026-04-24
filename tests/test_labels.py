import pytest
from app.labels import KNOWN_LABELS, validate_labels, UnknownLabelError


def test_known_labels():
    # Eight model-native labels plus credit_card_last4 from the regex
    # postprocessor (see app/postprocess.py).
    assert KNOWN_LABELS == frozenset({
        "private_person", "private_email", "private_phone",
        "private_address", "account_number", "private_url",
        "private_date", "secret",
        "credit_card_last4",
    })


def test_validate_labels_accepts_none():
    assert validate_labels(None) is None


def test_validate_labels_accepts_empty_list_as_none():
    assert validate_labels([]) is None


def test_validate_labels_returns_frozen_subset():
    out = validate_labels(["private_email", "private_phone"])
    assert out == frozenset({"private_email", "private_phone"})


def test_validate_labels_rejects_unknown():
    with pytest.raises(UnknownLabelError) as exc:
        validate_labels(["private_email", "bogus_label"])
    assert "bogus_label" in str(exc.value)
