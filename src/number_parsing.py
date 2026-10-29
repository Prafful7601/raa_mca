"""Robust number parsing shared by claim-type tagging, near-miss perturbation,
and the JSON-check executor. Strips commas/units/%, handles simple dates."""
from __future__ import annotations
import re

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "dozen": 12, "hundred": 100, "thousand": 1000, "million": 1000000,
}


def parse_number(text) -> float | None:
    """Parse a single scalar number out of a cell/token, or None if absent."""
    if text is None:
        return None
    s = str(text).strip()
    if s == "":
        return None
    s = s.replace(",", "")
    s = re.sub(r"[%$£€]", "", s)
    s = re.sub(r"\b(kg|km|lb|lbs|m|ft|kmh|mph|pts|points?|years?|yrs?)\b", "", s, flags=re.IGNORECASE).strip()
    m = re.match(r"^-?\d+\.?\d*$", s)
    if m:
        try:
            return float(s)
        except ValueError:
            return None
    m = _NUMBER_RE.search(s)
    if m:
        try:
            return float(m.group().replace(",", ""))
        except ValueError:
            return None
    return None


def contains_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))


def contains_number_word(text: str) -> bool:
    tokens = re.findall(r"[a-z]+", text.lower())
    return any(t in _NUMBER_WORDS for t in tokens)


def extract_numbers(text: str) -> list[float]:
    out = []
    for tok in re.findall(r"-?\d[\d,]*\.?\d*", text):
        v = parse_number(tok)
        if v is not None:
            out.append(v)
    return out
