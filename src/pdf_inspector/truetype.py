"""Minimal TrueType/OpenType ``cmap`` inspection.

Upstream uses the ``ttf-parser`` crate for one question: does this embedded font
carry a ``cmap`` subtable that maps codepoints to glyphs? Pulling in a full font
library for that would be a heavy dependency — and would not survive the
pure-Python constraint that lets this package run under Pyodide — so the table
directory and ``cmap`` header are read directly here.

Only the parts needed to answer that question are implemented: enough of the
table directory to locate ``cmap``, and enough of each ``cmap`` subtable to
decide whether it is a Unicode mapping with at least one entry.
"""

from __future__ import annotations

import struct

#: cmap platform IDs (OpenType spec, 'cmap' table).
_PLATFORM_UNICODE = 0
_PLATFORM_WINDOWS = 3
#: Windows encoding IDs that address Unicode.
_WINDOWS_SYMBOL = 0
_WINDOWS_BMP = 1
_WINDOWS_FULL = 10


def font_has_unicode_cmap(data: bytes) -> bool:
    """Whether an embedded font has a usable Unicode ``cmap`` subtable.

    Mirrors upstream's check: a subtable counts when it is a Unicode mapping (or
    Windows/Symbol, which upstream accepts explicitly) *and* it actually
    contains at least one codepoint. A present-but-empty ``cmap`` does not make
    the font decodable.
    """
    cmap = _find_table(data, b"cmap")
    if cmap is None:
        return False

    offset, length = cmap
    if length < 4 or offset + 4 > len(data):
        return False

    try:
        _version, num_tables = struct.unpack_from(">HH", data, offset)
    except struct.error:
        return False

    for index in range(num_tables):
        record = offset + 4 + index * 8
        if record + 8 > len(data):
            break
        try:
            platform_id, encoding_id, subtable_offset = struct.unpack_from(
                ">HHI", data, record
            )
        except struct.error:
            break

        if not _is_unicode_subtable(platform_id, encoding_id):
            continue

        if _subtable_has_codepoints(data, offset + subtable_offset):
            return True

    return False


def _is_unicode_subtable(platform_id: int, encoding_id: int) -> bool:
    if platform_id == _PLATFORM_UNICODE:
        return True
    if platform_id == _PLATFORM_WINDOWS:
        # Upstream accepts Windows/Symbol (encoding 0) alongside true Unicode
        # encodings, because symbol fonts map into the F0xx private-use block
        # that the glyph-name path knows how to unshift.
        return encoding_id in (_WINDOWS_SYMBOL, _WINDOWS_BMP, _WINDOWS_FULL)
    return False


def _subtable_has_codepoints(data: bytes, offset: int) -> bool:
    """Whether a ``cmap`` subtable maps at least one codepoint."""
    if offset + 2 > len(data):
        return False
    try:
        (fmt,) = struct.unpack_from(">H", data, offset)
    except struct.error:
        return False

    if fmt == 0:
        # Byte encoding table: 256 single-byte glyph indices.
        if offset + 6 + 256 > len(data):
            return False
        return any(data[offset + 6 : offset + 6 + 256])

    if fmt == 4:
        # Segment mapping to delta values. seg_count_x2 > 2 means at least one
        # real segment beyond the mandatory 0xFFFF terminator.
        try:
            seg_count_x2 = struct.unpack_from(">H", data, offset + 6)[0]
        except struct.error:
            return False
        return seg_count_x2 > 2

    if fmt == 6:
        # Trimmed table mapping.
        try:
            entry_count = struct.unpack_from(">H", data, offset + 8)[0]
        except struct.error:
            return False
        return entry_count > 0

    if fmt in (12, 13):
        # Segmented coverage / many-to-one range mappings.
        try:
            num_groups = struct.unpack_from(">I", data, offset + 12)[0]
        except struct.error:
            return False
        return num_groups > 0

    # Formats 2, 8, 10, 14 are rare in PDF-embedded fonts. Treating an
    # unrecognised-but-present Unicode subtable as usable matches upstream's
    # bias: the check exists to avoid condemning a decodable font to OCR.
    return True


def _find_table(data: bytes, tag: bytes) -> tuple[int, int] | None:
    """Locate a table by tag, returning ``(offset, length)``."""
    if len(data) < 12:
        return None

    sfnt_version = data[:4]
    base = 0
    if sfnt_version == b"ttcf":
        # TrueType Collection: use the first font, as upstream's Face::parse
        # with index 0 does.
        try:
            first_offset = struct.unpack_from(">I", data, 12)[0]
        except struct.error:
            return None
        if first_offset + 12 > len(data):
            return None
        base = first_offset

    try:
        num_tables = struct.unpack_from(">H", data, base + 4)[0]
    except struct.error:
        return None

    for index in range(num_tables):
        record = base + 12 + index * 16
        if record + 16 > len(data):
            break
        table_tag = data[record : record + 4]
        if table_tag != tag:
            continue
        try:
            offset, length = struct.unpack_from(">II", data, record + 8)
        except struct.error:
            return None
        if offset >= len(data):
            return None
        return offset, length

    return None
