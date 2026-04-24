from typing import Optional, Sequence


KNOWN_LABELS: frozenset[str] = frozenset({
    "private_person",
    "private_email",
    "private_phone",
    "private_address",
    "account_number",
    "private_url",
    "private_date",
    "secret",
    "credit_card_last4",
    "credit_card_number",
    "ip_address",
})


class UnknownLabelError(ValueError):
    def __init__(self, unknown: list[str]) -> None:
        super().__init__(
            f"Unknown label(s): {sorted(unknown)}. "
            f"Allowed: {sorted(KNOWN_LABELS)}"
        )
        self.unknown = unknown


def validate_labels(labels: Optional[Sequence[str]]) -> Optional[frozenset[str]]:
    if not labels:
        return None
    requested = set(labels)
    unknown = sorted(requested - KNOWN_LABELS)
    if unknown:
        raise UnknownLabelError(unknown)
    return frozenset(requested)
