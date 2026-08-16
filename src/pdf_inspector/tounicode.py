"""ToUnicode CMap parsing for PDF text extraction.

Only the pieces the detector depends on are ported so far; CMap parsing itself
is still to come. See the port status table in the README.
"""

from __future__ import annotations

from typing import Any

from .pdfdoc import Document

#: Range expansion stops after this many CID visits, so a font that declares a
#: full-width range cannot make the scan quadratic.
MAX_CID_W_EXPANSION = 65_536


def cid_values_look_like_unicode(cid_font_dict: Any, doc: Document) -> bool:
    """Whether a CID font's ``/W`` array suggests CIDs *are* Unicode codepoints.

    Many PDF generators (Chromium, wkhtmltopdf) emit Identity-H fonts where the
    CID happens to equal the Unicode codepoint, which makes passthrough decoding
    work even with no ToUnicode CMap. GID-based subsets instead number from
    zero, so the median CID separates the two cases.
    """
    w_arr = doc.resolve(cid_font_dict.get("/W")) if cid_font_dict else None
    if not isinstance(w_arr, (list, tuple)):
        return False

    # The /W array format is [cid [w1 w2 ...]] or [cid_start cid_end w].
    # Only unique CIDs are collected: repeating a full-width range must not grow
    # the working set by the range length on every copy.
    seen: set[int] = set()
    i = 0
    while i < len(w_arr) and len(seen) < MAX_CID_W_EXPANSION:
        start = _as_int(doc.resolve(w_arr[i]))
        if start is None:
            i += 1
            continue
        start &= 0xFFFF

        if i + 1 < len(w_arr):
            following = doc.resolve(w_arr[i + 1])
            if isinstance(following, (list, tuple)):
                # [cid [w1 w2 ...]] — CIDs are cid, cid+1, ..., cid+len-1
                for j in range(len(following)):
                    if len(seen) >= MAX_CID_W_EXPANSION:
                        break
                    seen.add((start + j) & 0xFFFF)
                i += 2
            else:
                # [cid_start cid_end w] — an inclusive range of CIDs
                if i + 2 < len(w_arr):
                    cid_end = _as_int(following)
                    if cid_end is not None:
                        _record_unique_cid_range(start, cid_end & 0xFFFF, seen)
                    i += 3
                else:
                    i += 1
        else:
            seen.add(start)
            i += 1

    if not seen:
        return False

    cids = sorted(seen)
    median = cids[len(cids) // 2]
    # Unicode text CIDs are typically >= 0x20 (space) with letters at 0x41+.
    # GID-based subsets typically start at low values (0-based).
    return median >= 0x41


def _record_unique_cid_range(start: int, end: int, seen: set[int]) -> None:
    if start > end:
        return
    for cid in range(start, end + 1):
        if len(seen) >= MAX_CID_W_EXPANSION:
            return
        seen.add(cid)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
