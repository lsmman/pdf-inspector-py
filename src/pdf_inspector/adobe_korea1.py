"""Adobe-Korea1 CID-to-Unicode mapping table."""

from __future__ import annotations

from ._adobe_korea1_data import ADOBE_KOREA1_CID_TO_UNICODE

__all__ = ["ADOBE_KOREA1_CID_TO_UNICODE", "lookup_korea1"]


def lookup_korea1(cid: int) -> str | None:
    """Look up a CID in the Adobe-Korea1 table and return the Unicode character."""
    code = ADOBE_KOREA1_CID_TO_UNICODE.get(cid)
    if code is None or 0xD800 <= code <= 0xDFFF:
        return None
    return chr(code)
