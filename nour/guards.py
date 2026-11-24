"""PII detection and redaction for pharmacist notes. You write this file."""
from __future__ import annotations

from datetime import date
import re


_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ID = re.compile(r"(?<!\d)[23][0-9]{13}(?!\d)")
_PHONE = re.compile(r"(?<![\d+])(?:(?:\+2|0[ -]?0[ -]?2|2)[ -]?)?0[ -]?1[ -]?[0125](?:[ -]?[0-9]){8}(?!\d)")
_GOVERNORATES = {1, 2, 3, 4, 88, *range(11, 20), *range(21, 30), *range(31, 36)}


def _valid_id(number: str) -> bool:
    if int(number[7:9]) not in _GOVERNORATES:
        return False
    century = 1900 if number[0] == "2" else 2000
    try:
        date(century + int(number[1:3]), int(number[3:5]), int(number[5:7]))
    except ValueError:
        return False
    return True


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
    normalized = text.translate(_DIGITS)
    ids = [match.span() for match in _ID.finditer(normalized) if _valid_id(match.group())]
    results = [{"type": "NATIONAL_ID", "start": start, "end": end} for start, end in ids]
    for match in _PHONE.finditer(normalized):
        start, end = match.span()
        if all(end <= id_start or start >= id_end for id_start, id_end in ids):
            results.append({"type": "PHONE", "start": start, "end": end})
    return sorted(results, key=lambda item: item["start"])


def redact(text: str) -> tuple[str, list[dict]]:
    """Replace each PII match with `[NATIONAL_ID]` or `[PHONE]`.

    Returns `(new_text, find_pii(text))`. Text without PII comes back unchanged.
    """
    findings = find_pii(text)
    parts = []
    cursor = 0
    for finding in findings:
        parts.extend((text[cursor:finding["start"]], f"[{finding['type']}]"))
        cursor = finding["end"]
    parts.append(text[cursor:])
    return "".join(parts), findings
