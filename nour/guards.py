"""PII detection and redaction for pharmacist notes. You write this file."""
from __future__ import annotations


def find_pii(text: str) -> list[dict]:
    """Find Egyptian national IDs and mobile phone numbers in `text`.

    Returns a list of `{"type": "NATIONAL_ID" | "PHONE", "start": int, "end": int}`, offsets
    into the original `text`, sorted by `start`. Digits may be ASCII, Arabic-Indic (٠-٩) or
    Persian (۰-۹), in any mix within one match.

    NATIONAL_ID: exactly 14 consecutive digits not touching other digits on either side; the
    first digit is 2 or 3 (selects the century); digits 2-7 are a real calendar date YYMMDD in
    that century; digits 8-9 are a known governorate code (01-04, 11-19, 21-29, 31-35, 88). The
    14th digit is not checked.

    PHONE: an Egyptian mobile number: `0`, then `1`, then one of `0 1 2 5`, then eight more
    digits; optionally preceded by a country prefix `+2`, `002` or `2`; single spaces or dashes
    may separate digit groups; not preceded by a digit or `+` that is not part of the match, and
    not followed by a digit.

    A national ID match wins over an overlapping phone match.
    """
    raise NotImplementedError


def redact(text: str) -> tuple[str, list[dict]]:
    """Replace each PII match with `[NATIONAL_ID]` or `[PHONE]`.

    Returns `(new_text, find_pii(text))`. Text without PII comes back unchanged.
    """
    raise NotImplementedError
