import datetime as dt
import re
from typing import Literal

_CURLY = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
_SPACE = re.compile(r"\s+")


def normalise(s: str) -> str:
    """Lowercase, map curly quotes to straight, and collapse runs of whitespace."""
    return _SPACE.sub(" ", s.translate(_CURLY)).strip().lower()


def quote_in_document(quote: str, text: str) -> bool:
    """True when the normalised quote appears in the normalised text. Empty quote is False."""
    needle = normalise(quote)
    if not needle:
        return False
    return needle in normalise(text)


def compare_dates(
    a: dt.date | None, b: dt.date | None
) -> Literal["a_newer", "b_newer", "same", "unknown"]:
    if a is None or b is None:
        return "unknown"
    if a > b:
        return "a_newer"
    if b > a:
        return "b_newer"
    return "same"
