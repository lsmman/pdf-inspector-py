"""Minimal TrueType/OpenType reading.

Upstream uses the ``ttf-parser`` crate to answer three questions about an
embedded font: does it have a usable ``cmap``, what does that ``cmap`` map, and
what are the glyph names in ``post``. Pulling in a full font library for that
would be a heavy dependency — and a compiled one, which would cost the
pure-Python install that lets this package run under Pyodide — so the tables are
read directly here.

Only what those questions need is implemented: the table directory, the ``cmap``
subtable formats that appear in PDF-embedded fonts (0, 4, 6, 12), ``maxp`` for
the glyph count, and version 2.0 of ``post`` for glyph names.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

#: cmap platform IDs (OpenType spec, 'cmap' table).
PLATFORM_UNICODE = 0
PLATFORM_MACINTOSH = 1
PLATFORM_WINDOWS = 3
#: Windows encoding IDs that address Unicode.
WINDOWS_SYMBOL = 0
WINDOWS_BMP = 1
WINDOWS_FULL = 10

#: The 258 standard Macintosh glyph names that `post` format 2.0 indexes below
#: 258. Only the ones the Adobe Glyph List can resolve matter here, but the list
#: has to be complete for the indices to line up.
MAC_GLYPH_NAMES: tuple[str, ...] = (
    ".notdef", ".null", "nonmarkingreturn", "space", "exclam", "quotedbl",
    "numbersign", "dollar", "percent", "ampersand", "quotesingle", "parenleft",
    "parenright", "asterisk", "plus", "comma", "hyphen", "period", "slash",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "colon", "semicolon", "less", "equal", "greater", "question", "at",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "bracketleft",
    "backslash", "bracketright", "asciicircum", "underscore", "grave", "a", "b",
    "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
    "r", "s", "t", "u", "v", "w", "x", "y", "z", "braceleft", "bar",
    "braceright", "asciitilde", "Adieresis", "Aring", "Ccedilla", "Eacute",
    "Ntilde", "Odieresis", "Udieresis", "aacute", "agrave", "acircumflex",
    "adieresis", "atilde", "aring", "ccedilla", "eacute", "egrave",
    "ecircumflex", "edieresis", "iacute", "igrave", "icircumflex", "idieresis",
    "ntilde", "oacute", "ograve", "ocircumflex", "odieresis", "otilde", "uacute",
    "ugrave", "ucircumflex", "udieresis", "dagger", "degree", "cent", "sterling",
    "section", "bullet", "paragraph", "germandbls", "registered", "copyright",
    "trademark", "acute", "dieresis", "notequal", "AE", "Oslash", "infinity",
    "plusminus", "lessequal", "greaterequal", "yen", "mu", "partialdiff",
    "summation", "product", "pi", "integral", "ordfeminine", "ordmasculine",
    "Omega", "ae", "oslash", "questiondown", "exclamdown", "logicalnot",
    "radical", "florin", "approxequal", "Delta", "guillemotleft",
    "guillemotright", "ellipsis", "nonbreakingspace", "Agrave", "Atilde",
    "Otilde", "OE", "oe", "endash", "emdash", "quotedblleft", "quotedblright",
    "quoteleft", "quoteright", "divide", "lozenge", "ydieresis", "Ydieresis",
    "fraction", "currency", "guilsinglleft", "guilsinglright", "fi", "fl",
    "daggerdbl", "periodcentered", "quotesinglbase", "quotedblbase",
    "perthousand", "Acircumflex", "Ecircumflex", "Aacute", "Edieresis", "Egrave",
    "Iacute", "Icircumflex", "Idieresis", "Igrave", "Oacute", "Ocircumflex",
    "apple", "Ograve", "Uacute", "Ucircumflex", "Ugrave", "dotlessi",
    "circumflex", "tilde", "macron", "breve", "dotaccent", "ring", "cedilla",
    "hungarumlaut", "ogonek", "caron", "Lslash", "lslash", "Scaron", "scaron",
    "Zcaron", "zcaron", "brokenbar", "Eth", "eth", "Yacute", "yacute", "Thorn",
    "thorn", "minus", "multiply", "onesuperior", "twosuperior", "threesuperior",
    "onehalf", "onequarter", "threequarters", "franc", "Gbreve", "gbreve",
    "Idotaccent", "Scedilla", "scedilla", "Cacute", "cacute", "Ccaron", "ccaron",
    "dcroat",
)


@dataclass
class CmapSubtable:
    """One ``cmap`` subtable, with its codepoint-to-glyph mapping."""

    platform_id: int
    encoding_id: int
    mapping: dict[int, int] = field(default_factory=dict)

    def is_unicode(self) -> bool:
        """Whether this subtable addresses Unicode, as ttf-parser defines it."""
        if self.platform_id == PLATFORM_UNICODE:
            return True
        if self.platform_id == PLATFORM_WINDOWS:
            return self.encoding_id in (WINDOWS_BMP, WINDOWS_FULL)
        return False

    def glyph_index(self, codepoint: int) -> int | None:
        gid = self.mapping.get(codepoint)
        return gid if gid else None  # GID 0 is .notdef, treated as absent


class Face:
    """The parts of a font face this package reads."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._tables = _table_directory(data)
        self._subtables: list[CmapSubtable] | None = None
        self._glyph_names: dict[int, str] | None = None

    @classmethod
    def parse(cls, data: bytes) -> Face | None:
        try:
            face = cls(data)
        except Exception:
            return None
        return face if face._tables else None

    # ── cmap ─────────────────────────────────────────────────────────

    @property
    def cmap_subtables(self) -> list[CmapSubtable]:
        if self._subtables is None:
            try:
                self._subtables = _parse_cmap(self._data, self._tables.get("cmap"))
            except Exception:
                self._subtables = []
        return self._subtables

    def has_unicode_cmap(self) -> bool:
        """Whether a Unicode (or Windows/Symbol) subtable maps anything.

        A present-but-empty ``cmap`` does not make the font decodable, so the
        mapping has to be non-empty — that is what upstream's codepoint count
        checks.
        """
        for subtable in self.cmap_subtables:
            is_symbol = (
                subtable.platform_id == PLATFORM_WINDOWS
                and subtable.encoding_id == WINDOWS_SYMBOL
            )
            if (subtable.is_unicode() or is_symbol) and subtable.mapping:
                return True
        return False

    def find_subtable(self, platform_id: int, encoding_id: int) -> CmapSubtable | None:
        for subtable in self.cmap_subtables:
            if (
                subtable.platform_id == platform_id
                and subtable.encoding_id == encoding_id
            ):
                return subtable
        return None

    # ── post / maxp ──────────────────────────────────────────────────

    @property
    def number_of_glyphs(self) -> int:
        entry = self._tables.get("maxp")
        if entry is None:
            return 0
        offset, _ = entry
        try:
            return struct.unpack_from(">H", self._data, offset + 4)[0]
        except struct.error:
            return 0

    def glyph_name(self, gid: int) -> str | None:
        if self._glyph_names is None:
            try:
                self._glyph_names = _parse_post(self._data, self._tables.get("post"))
            except Exception:
                self._glyph_names = {}
        return self._glyph_names.get(gid)


def font_has_unicode_cmap(data: bytes) -> bool:
    """Whether an embedded font has a usable Unicode ``cmap`` subtable."""
    face = Face.parse(data)
    return face is not None and face.has_unicode_cmap()


# ── table directory ──────────────────────────────────────────────────


def _table_directory(data: bytes) -> dict[str, tuple[int, int]]:
    """Map table tag -> ``(offset, length)``."""
    if len(data) < 12:
        return {}

    base = 0
    if data[:4] == b"ttcf":
        # TrueType Collection: use the first font, matching Face::parse(_, 0).
        try:
            base = struct.unpack_from(">I", data, 12)[0]
        except struct.error:
            return {}
        if base + 12 > len(data):
            return {}

    try:
        num_tables = struct.unpack_from(">H", data, base + 4)[0]
    except struct.error:
        return {}

    tables: dict[str, tuple[int, int]] = {}
    for index in range(num_tables):
        record = base + 12 + index * 16
        if record + 16 > len(data):
            break
        tag = data[record : record + 4].decode("latin-1")
        try:
            offset, length = struct.unpack_from(">II", data, record + 8)
        except struct.error:
            break
        if offset < len(data):
            tables[tag] = (offset, length)
    return tables


# ── cmap parsing ─────────────────────────────────────────────────────


def _parse_cmap(
    data: bytes, entry: tuple[int, int] | None
) -> list[CmapSubtable]:
    if entry is None:
        return []
    offset, length = entry
    if length < 4 or offset + 4 > len(data):
        return []

    _version, num_tables = struct.unpack_from(">HH", data, offset)

    subtables: list[CmapSubtable] = []
    for index in range(num_tables):
        record = offset + 4 + index * 8
        if record + 8 > len(data):
            break
        platform_id, encoding_id, sub_offset = struct.unpack_from(">HHI", data, record)
        mapping = _parse_cmap_subtable(data, offset + sub_offset)
        if mapping is None:
            continue
        subtables.append(CmapSubtable(platform_id, encoding_id, mapping))
    return subtables


def _parse_cmap_subtable(data: bytes, offset: int) -> dict[int, int] | None:
    if offset + 2 > len(data):
        return None
    try:
        (fmt,) = struct.unpack_from(">H", data, offset)
    except struct.error:
        return None

    if fmt == 0:
        return _cmap_format0(data, offset)
    if fmt == 4:
        return _cmap_format4(data, offset)
    if fmt == 6:
        return _cmap_format6(data, offset)
    if fmt == 12:
        return _cmap_format12(data, offset)
    # Formats 2, 8, 10, 13 and 14 are rare in PDF-embedded fonts. An empty
    # mapping keeps the subtable visible (so platform/encoding checks still see
    # it) without claiming coverage it cannot supply.
    return {}


def _cmap_format0(data: bytes, offset: int) -> dict[int, int]:
    """Byte encoding table: 256 single-byte glyph indices."""
    start = offset + 6
    if start + 256 > len(data):
        return {}
    return {code: gid for code, gid in enumerate(data[start : start + 256]) if gid}


def _cmap_format4(data: bytes, offset: int) -> dict[int, int]:
    """Segment mapping to delta values — the common BMP format."""
    try:
        seg_count_x2 = struct.unpack_from(">H", data, offset + 6)[0]
    except struct.error:
        return {}
    seg_count = seg_count_x2 // 2
    if seg_count == 0:
        return {}

    ends_at = offset + 14
    starts_at = ends_at + seg_count_x2 + 2  # +2 for reservedPad
    deltas_at = starts_at + seg_count_x2
    ranges_at = deltas_at + seg_count_x2
    if ranges_at + seg_count_x2 > len(data):
        return {}

    mapping: dict[int, int] = {}
    for i in range(seg_count):
        end = struct.unpack_from(">H", data, ends_at + i * 2)[0]
        start = struct.unpack_from(">H", data, starts_at + i * 2)[0]
        delta = struct.unpack_from(">h", data, deltas_at + i * 2)[0]
        range_offset_at = ranges_at + i * 2
        range_offset = struct.unpack_from(">H", data, range_offset_at)[0]

        if start > end or start == 0xFFFF:
            continue

        for code in range(start, end + 1):
            if range_offset == 0:
                gid = (code + delta) & 0xFFFF
            else:
                # glyphIdArray is addressed relative to the rangeOffset slot.
                glyph_at = range_offset_at + range_offset + (code - start) * 2
                if glyph_at + 2 > len(data):
                    continue
                gid = struct.unpack_from(">H", data, glyph_at)[0]
                if gid:
                    gid = (gid + delta) & 0xFFFF
            if gid:
                mapping[code] = gid
    return mapping


def _cmap_format6(data: bytes, offset: int) -> dict[int, int]:
    """Trimmed table mapping."""
    try:
        first_code, entry_count = struct.unpack_from(">HH", data, offset + 6)
    except struct.error:
        return {}
    start = offset + 10
    if start + entry_count * 2 > len(data):
        return {}
    mapping: dict[int, int] = {}
    for i in range(entry_count):
        gid = struct.unpack_from(">H", data, start + i * 2)[0]
        if gid:
            mapping[first_code + i] = gid
    return mapping


def _cmap_format12(data: bytes, offset: int) -> dict[int, int]:
    """Segmented coverage — needed for supplementary-plane characters."""
    try:
        num_groups = struct.unpack_from(">I", data, offset + 12)[0]
    except struct.error:
        return {}
    mapping: dict[int, int] = {}
    for i in range(num_groups):
        record = offset + 16 + i * 12
        if record + 12 > len(data):
            break
        start_char, end_char, start_gid = struct.unpack_from(">III", data, record)
        if start_char > end_char or end_char - start_char > 0x10FFFF:
            continue
        for index, code in enumerate(range(start_char, end_char + 1)):
            gid = start_gid + index
            if gid:
                mapping[code] = gid
    return mapping


# ── post parsing ─────────────────────────────────────────────────────


def _parse_post(data: bytes, entry: tuple[int, int] | None) -> dict[int, str]:
    """Glyph names from a version 2.0 ``post`` table."""
    if entry is None:
        return {}
    offset, length = entry
    if offset + 34 > len(data):
        return {}

    try:
        version = struct.unpack_from(">I", data, offset)[0]
    except struct.error:
        return {}
    if version != 0x00020000:
        # 1.0 is the standard Mac set (no custom names), 3.0 has no names at all.
        return {}

    try:
        num_glyphs = struct.unpack_from(">H", data, offset + 32)[0]
    except struct.error:
        return {}

    indices_at = offset + 34
    if indices_at + num_glyphs * 2 > len(data):
        return {}
    indices = [
        struct.unpack_from(">H", data, indices_at + i * 2)[0] for i in range(num_glyphs)
    ]

    # Pascal strings follow the index array, in order.
    names: list[str] = []
    pos = indices_at + num_glyphs * 2
    end = min(offset + length, len(data)) if length else len(data)
    while pos < end:
        name_len = data[pos]
        pos += 1
        if pos + name_len > end:
            break
        names.append(data[pos : pos + name_len].decode("latin-1"))
        pos += name_len

    glyph_names: dict[int, str] = {}
    for gid, index in enumerate(indices):
        if index < 258:
            if index < len(MAC_GLYPH_NAMES):
                glyph_names[gid] = MAC_GLYPH_NAMES[index]
        else:
            custom = index - 258
            if custom < len(names):
                glyph_names[gid] = names[custom]
    return glyph_names
