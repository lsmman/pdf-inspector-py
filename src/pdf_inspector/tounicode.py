"""Deciding which ToUnicode CMap a font should be decoded with.

:mod:`pdf_inspector.cmap` parses a CMap program. This module works one level up:
it walks a document's fonts, parses their ``/ToUnicode`` streams, and — for the
many fonts that ship a broken CMap or none at all — builds the fallbacks that
make their text decodable anyway (embedded TrueType ``cmap`` tables, glyph
names, predefined CID collections, subset-GID repair).

.. note::
   ``build_cmap_from_builtin_cmap`` is not implemented. Upstream loads pdf.js's
   bundled binary CMaps (``external/bcmaps``) to resolve the Adobe-Japan1,
   Adobe-GB1 and Adobe-CNS1 collections. Adobe-Korea1 does not need them — it
   comes from the generated table in :mod:`pdf_inspector.adobe_korea1` — so
   Korean CID fonts decode, while the other three predefined collections fall
   through to whatever other fallback applies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .adobe_korea1 import ADOBE_KOREA1_CID_TO_UNICODE
from .cmap import MAX_CID_W_EXPANSION, ToUnicodeCMap
from .glyph_names import glyph_to_char
from .pdfdoc import Document, ObjectId, PageRef, stream_bytes
from .truetype import (
    PLATFORM_MACINTOSH,
    PLATFORM_WINDOWS,
    WINDOWS_BMP,
    WINDOWS_SYMBOL,
    Face,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CMapEntry",
    "FontCMaps",
    "MAX_CID_W_EXPANSION",
    "build_cmap_from_truetype",
    "cid_values_look_like_unicode",
    "load_builtin_cmap_by_name",
]

#: Below this many mappings a primary CMap is treated as too sparse to trust,
#: and a fallback takes its place.
SPARSE_CMAP_THRESHOLD = 10


@dataclass
class CMapEntry:
    """A primary CMap plus the alternates extraction may fall back to."""

    primary: ToUnicodeCMap
    remapped: ToUnicodeCMap | None = None
    fallback: ToUnicodeCMap | None = None


# ── document walk ────────────────────────────────────────────────────


class FontCMaps:
    """CMaps for a document's fonts, indexed by object number.

    The key is the ``/ToUnicode`` stream's object number where one exists, and
    the embedded font file's (or CIDFont's) object number for the fallbacks —
    matching how the extractor looks a font's CMap up.
    """

    def __init__(self, by_obj_num: dict[int, CMapEntry] | None = None) -> None:
        self._by_obj_num: dict[int, CMapEntry] = by_obj_num or {}

    @classmethod
    def from_doc(
        cls,
        doc: Document,
        page_filter: set[int] | None = None,
        skip_truetype_fallback: bool = False,
    ) -> FontCMaps:
        """Build CMaps for every page, or only the pages in ``page_filter``.

        ``skip_truetype_fallback`` is upstream's fast mode: it skips parsing
        embedded font files, which is expensive on large fonts. Fonts that
        cannot be decoded from their ToUnicode CMap alone then produce empty or
        garbage text, which routes the page to OCR — the right trade for
        pipelines that always have OCR available.
        """
        by_obj_num: dict[int, CMapEntry] = {}

        for page_num, page in doc.get_pages().items():
            if page_filter is not None and page_num not in page_filter:
                continue

            fonts = _page_fonts(doc, page)
            _collect_cmaps_from_fonts(doc, fonts, by_obj_num, skip_truetype_fallback)

            if not skip_truetype_fallback:
                _collect_cmaps_from_xobjects(doc, page, by_obj_num)

        return cls(by_obj_num)

    def get_by_obj(self, obj_num: int) -> CMapEntry | None:
        return self._by_obj_num.get(obj_num)

    def __len__(self) -> int:
        return len(self._by_obj_num)

    def __contains__(self, obj_num: int) -> bool:
        return obj_num in self._by_obj_num


def _page_fonts(doc: Document, page: PageRef) -> list[Any]:
    """Every font dictionary reachable from a page's resources."""
    fonts: list[Any] = []
    seen: set[ObjectId] = set()
    own, ancestor_ids = doc.get_page_resources(page)
    resource_dicts = ([own] if own is not None else []) + [
        doc.dictionary_for_id(object_id) for object_id in ancestor_ids
    ]

    for resources in resource_dicts:
        if resources is None:
            continue
        font_dict = doc.get_dictionary(resources.get("/Font"))
        if font_dict is None:
            continue
        for name in list(font_dict.keys()):
            raw = font_dict.raw_get(name)
            object_id = Document.object_id(raw)
            if object_id is not None:
                if object_id in seen:
                    continue
                seen.add(object_id)
            font = doc.get_dictionary(raw)
            if font is not None:
                fonts.append(font)
    return fonts


def _collect_cmaps_from_fonts(
    doc: Document,
    fonts: list[Any],
    by_obj_num: dict[int, CMapEntry],
    skip_truetype_fallback: bool,
) -> None:
    """Three passes, matching upstream's ordering."""
    _pass_tounicode_streams(doc, fonts, by_obj_num, skip_truetype_fallback)
    if skip_truetype_fallback:
        return
    _pass_identity_h_without_tounicode(doc, fonts, by_obj_num)
    _pass_simple_fonts_without_tounicode(doc, fonts, by_obj_num)


def _pass_tounicode_streams(
    doc: Document,
    fonts: list[Any],
    by_obj_num: dict[int, CMapEntry],
    skip_truetype_fallback: bool,
) -> None:
    """First pass: parse the ``/ToUnicode`` streams that are present."""
    for font_dict in fonts:
        if "/ToUnicode" not in font_dict:
            continue
        raw = font_dict.raw_get("/ToUnicode")
        object_id = Document.object_id(raw)
        if object_id is None:
            continue
        obj_num = object_id[0]
        if obj_num in by_obj_num:
            continue
        stream = doc.get_stream(raw)
        if stream is None:
            continue

        cmap = ToUnicodeCMap.parse(stream_bytes(stream))
        if cmap is None:
            # ToUnicode present but unparseable; fall back so decoding is not
            # simply empty.
            fallback = (
                build_fallback_cmap_for_simple(doc, font_dict)
                if skip_truetype_fallback
                else (
                    build_fallback_cmap_for_type0(doc, font_dict)
                    or build_fallback_cmap_for_simple(doc, font_dict)
                )
            )
            if fallback is not None:
                logger.debug(
                    "ToUnicode CMap obj=%s parse failed; using fallback (entries=%s)",
                    obj_num,
                    len(fallback.char_map),
                )
                by_obj_num[obj_num] = CMapEntry(primary=fallback)
            continue

        primary, remapped = try_remap_subset_cmap(doc, cmap, font_dict, obj_num)
        primary_entries = primary.entry_count()

        # Building fallbacks is only worth it when the primary is sparse:
        # parsing a large embedded TrueType font is slow, so it is skipped
        # whenever the ToUnicode CMap already suffices.
        if primary_entries < SPARSE_CMAP_THRESHOLD and not skip_truetype_fallback:
            fallback = build_fallback_tounicode_from_encoding(
                doc, font_dict
            ) or build_fallback_cmap_for_simple(doc, font_dict)
            if fallback is None:
                fallback = build_fallback_cmap_for_type0(doc, font_dict)
        elif primary_entries < SPARSE_CMAP_THRESHOLD:
            fallback = build_fallback_tounicode_from_encoding(
                doc, font_dict
            ) or build_fallback_cmap_for_simple(doc, font_dict)
        else:
            fallback = build_fallback_tounicode_from_encoding(doc, font_dict)

        if primary_entries < SPARSE_CMAP_THRESHOLD and fallback is not None:
            logger.debug(
                "ToUnicode CMap obj=%s too sparse (%s entries); using fallback",
                obj_num,
                primary_entries,
            )
            remapped = primary
            primary = fallback
            fallback = None

        by_obj_num[obj_num] = CMapEntry(
            primary=primary, remapped=remapped, fallback=fallback
        )


def _pass_identity_h_without_tounicode(
    doc: Document, fonts: list[Any], by_obj_num: dict[int, CMapEntry]
) -> None:
    """Second pass: Identity-H/V fonts with no ToUnicode at all."""
    for font_dict in fonts:
        if "/ToUnicode" in font_dict:
            continue
        encoding = Document.name_of(doc.resolve(font_dict.get("/Encoding")))
        if encoding not in ("Identity-H", "Identity-V"):
            continue

        desc_fonts = doc.resolve(font_dict.get("/DescendantFonts"))
        if not isinstance(desc_fonts, (list, tuple)) or not desc_fonts:
            continue
        cid_font_dict = doc.get_dictionary(desc_fonts[0])
        if cid_font_dict is None:
            continue

        font_file_raw = _font_file_raw(doc, cid_font_dict)
        font_file_id = Document.object_id(font_file_raw) if font_file_raw else None

        # The key must match what the extractor looks up: the font file's object
        # number when there is one, otherwise the CIDFont dictionary's.
        if font_file_id is not None:
            lookup_key = font_file_id[0]
        else:
            descendant_id = Document.object_id(desc_fonts[0])
            lookup_key = descendant_id[0] if descendant_id else 0
        if lookup_key == 0 or lookup_key in by_obj_num:
            continue

        # 1. The embedded font's own cmap table.
        if font_file_raw is not None:
            stream = doc.get_stream(font_file_raw)
            if stream is not None:
                cmap = build_cmap_from_truetype(stream_bytes(stream))
                if cmap is not None:
                    logger.debug(
                        "TrueType CMap obj=%s (embedded font) char_map=%s",
                        lookup_key,
                        len(cmap.char_map),
                    )
                    by_obj_num[lookup_key] = CMapEntry(primary=cmap)
                    continue

        # 2. A predefined CID->Unicode mapping named by CIDSystemInfo.
        cmap = build_cmap_from_cid_system_info(doc, cid_font_dict)
        if cmap is not None:
            logger.debug(
                "Predefined CMap obj=%s (CIDSystemInfo) char_map=%s",
                lookup_key,
                len(cmap.char_map),
            )
            by_obj_num[lookup_key] = CMapEntry(primary=cmap)
            continue

        # 3. Last resort: CID-as-Unicode passthrough. Many generators (Chromium,
        # wkhtmltopdf) emit Identity-H where the CIDs *are* Unicode codepoints
        # but strip the cmap table and omit ToUnicode. The /W array tells the two
        # cases apart: real Unicode CIDs sit at 0x41+, GID-based subsets number
        # from zero.
        if cid_values_look_like_unicode(cid_font_dict, doc):
            logger.debug(
                "Identity-H font obj=%s: W array CIDs look like Unicode — passthrough",
                lookup_key,
            )
            passthrough = ToUnicodeCMap(code_byte_length=2, cid_passthrough=True)
            by_obj_num[lookup_key] = CMapEntry(primary=passthrough)
        else:
            logger.debug(
                "Identity-H font obj=%s: no decoding possible "
                "(stripped cmap, GID-based CIDs)",
                lookup_key,
            )


def _pass_simple_fonts_without_tounicode(
    doc: Document, fonts: list[Any], by_obj_num: dict[int, CMapEntry]
) -> None:
    """Third pass: simple fonts with neither ToUnicode nor an /Encoding."""
    for font_dict in fonts:
        if "/ToUnicode" in font_dict:
            continue
        # A font with an explicit encoding decodes through the standard
        # encoding path and needs no fallback CMap.
        if "/Encoding" in font_dict:
            continue
        subtype = Document.name_of(doc.resolve(font_dict.get("/Subtype")))
        if subtype is None or subtype == "Type0":
            continue

        font_file_raw = _font_file_raw(doc, font_dict)
        if font_file_raw is None:
            continue
        font_file_id = Document.object_id(font_file_raw)
        if font_file_id is None:
            continue
        lookup_key = font_file_id[0]
        if lookup_key in by_obj_num:
            continue

        stream = doc.get_stream(font_file_raw)
        if stream is None:
            continue
        cmap = build_simple_cmap_from_truetype(stream_bytes(stream))
        if cmap is not None:
            logger.debug(
                "Simple font cmap obj=%s (embedded font) char_map=%s",
                lookup_key,
                len(cmap.char_map),
            )
            by_obj_num[lookup_key] = CMapEntry(primary=cmap)


def _collect_cmaps_from_xobjects(
    doc: Document, page: PageRef, by_obj_num: dict[int, CMapEntry]
) -> None:
    """Walk Form XObjects in a page's resources and collect their font CMaps."""
    own, ancestor_ids = doc.get_page_resources(page)
    visited: set[ObjectId] = set()

    resource_dicts = ([own] if own is not None else []) + [
        doc.dictionary_for_id(object_id) for object_id in ancestor_ids
    ]
    for resources in resource_dicts:
        if resources is not None:
            _walk_xobject_fonts(doc, resources, by_obj_num, visited)


def _walk_xobject_fonts(
    doc: Document,
    resources: Any,
    by_obj_num: dict[int, CMapEntry],
    visited: set[ObjectId],
) -> None:
    xobject_dict = doc.get_dictionary(resources.get("/XObject"))
    if xobject_dict is None:
        return

    for name in list(xobject_dict.keys()):
        raw = xobject_dict.raw_get(name)
        object_id = Document.object_id(raw)
        if object_id is None or object_id in visited:
            continue
        visited.add(object_id)

        stream = doc.get_stream(raw)
        if stream is None:
            continue
        if Document.name_of(doc.resolve(stream.get("/Subtype"))) != "Form":
            continue

        form_resources = doc.get_dictionary(stream.get("/Resources"))
        if form_resources is None:
            continue

        font_dict = doc.get_dictionary(form_resources.get("/Font"))
        if font_dict is not None:
            fonts = []
            for font_name in list(font_dict.keys()):
                font = doc.get_dictionary(font_dict.raw_get(font_name))
                if font is not None:
                    fonts.append(font)
            _collect_cmaps_from_fonts(doc, fonts, by_obj_num, False)

        _walk_xobject_fonts(doc, form_resources, by_obj_num, visited)


# ── subset GID repair ────────────────────────────────────────────────


def try_remap_subset_cmap(
    doc: Document, cmap: ToUnicodeCMap, font_dict: Any, obj_num: int
) -> tuple[ToUnicodeCMap, ToUnicodeCMap | None]:
    """Detect and repair ToUnicode CMaps from subset fonts with a GID mismatch.

    Some generators subset-embed fonts by renumbering GIDs sequentially
    (1, 2, 3, ...) but leave the ToUnicode CMap pointing at the original GID
    values. Returns ``(primary, repaired_or_None)``.
    """
    encoding = Document.name_of(doc.resolve(font_dict.get("/Encoding")))
    if encoding not in ("Identity-H", "Identity-V"):
        return cmap, None

    # A minimum source CID above 2 is what marks the CMap as still using old,
    # non-sequential GIDs.
    min_cid = cmap.min_source_cid()
    if min_cid is None or min_cid <= 2:
        return cmap, None

    cid_font_dict = get_descendant_cid_font(doc, font_dict)
    if cid_font_dict is None:
        return cmap, None

    # Both repair paths assume CIDs are glyph indices a subsetter can renumber,
    # which holds only for CIDFontType2 (TrueType). For CIDFontType0 (CFF), CIDs
    # resolve through the CFF charset, so a valid CMap stays valid after
    # subsetting and renumbering it would corrupt correct text. A missing or
    # unresolvable /Subtype keeps the repair enabled; only an explicit
    # non-CIDFontType2 disables it.
    subtype = Document.name_of(doc.resolve(cid_font_dict.get("/Subtype")))
    if subtype is not None and subtype != "CIDFontType2":
        logger.debug(
            "Subset remap skipped for obj=%s: descendant is not CIDFontType2", obj_num
        )
        return cmap, None

    # An explicit CIDToGIDMap gives an exact repair.
    cid_to_gid = get_cid_to_gid_map(doc, cid_font_dict)
    if cid_to_gid is not None:
        repaired = build_cmap_with_cid_to_gid_map(cmap, cid_to_gid)
        if repaired is not None:
            logger.debug(
                "CIDToGIDMap repair applied for obj=%s: %s entries",
                obj_num,
                len(repaired.char_map),
            )
            return cmap, repaired
        # Fall through to the sequential remap when the repair produced nothing.

    # A W array starting at a low CID indicates sequential post-subset GIDs.
    w_start = get_w_array_start_cid(doc, cid_font_dict)
    if w_start is None or w_start > 2:
        return cmap, None

    # If the W array actually covers the CMap's highest source CID, the CMap is
    # aligned with the font and nothing was renumbered. A sparse W array
    # starting at CID 0 with additional high-CID entries matching the CMap is
    # the normal subset layout, not a mismatch.
    max_cid = cmap.max_source_cid()
    if max_cid is not None and w_array_covers_cid(doc, cid_font_dict, max_cid):
        logger.debug(
            "Subset remap skipped for obj=%s: W array covers CMap max CID %s",
            obj_num,
            max_cid,
        )
        return cmap, None

    logger.debug(
        "Subset GID mismatch for obj=%s: W starts at CID %s, CMap min CID %s. Remapping.",
        obj_num,
        w_start,
        min_cid,
    )
    return cmap, cmap.remap_to_sequential()


def get_descendant_cid_font(doc: Document, font_dict: Any) -> Any | None:
    """The first DescendantFont dictionary of a Type0 font."""
    desc_fonts = doc.resolve(font_dict.get("/DescendantFonts"))
    if not isinstance(desc_fonts, (list, tuple)) or not desc_fonts:
        return None
    return doc.get_dictionary(desc_fonts[0])


def get_w_array_start_cid(doc: Document, cid_font_dict: Any) -> int | None:
    """The first CID in a CIDFont's ``/W`` array."""
    arr = doc.resolve(cid_font_dict.get("/W"))
    if not isinstance(arr, (list, tuple)) or not arr:
        return None
    value = _as_int(doc.resolve(arr[0]))
    return value & 0xFFFF if value is not None else None


def w_array_covers_cid(doc: Document, cid_font_dict: Any, target: int) -> bool:
    """Whether the ``/W`` array explicitly covers ``target``.

    The array uses two formats (PDF 32000-1:2008, 9.7.4.3):
    ``c [w1 w2 ... wn]`` gives widths for CIDs c..c+n-1, and
    ``c_first c_last w`` gives one width for the whole range.
    """
    arr = doc.resolve(cid_font_dict.get("/W"))
    if not isinstance(arr, (list, tuple)):
        return False

    i = 0
    while i < len(arr):
        first = _as_int(doc.resolve(arr[i]))
        if first is None:
            break
        i += 1
        if i >= len(arr):
            break

        following = doc.resolve(arr[i])
        if isinstance(following, (list, tuple)):
            last = first + len(following) - 1
            if first <= target <= last:
                return True
            i += 1
        else:
            last = _as_int(following)
            if last is None:
                break  # Unknown token — stop rather than misread the rest.
            i += 1
            if i < len(arr):
                i += 1  # skip the width value
            if first <= target <= last:
                return True
    return False


def get_cid_to_gid_map(doc: Document, cid_font_dict: Any) -> list[int] | None:
    """``/CIDToGIDMap`` as a list of GIDs indexed by CID."""
    raw = cid_font_dict.raw_get("/CIDToGIDMap") if "/CIDToGIDMap" in cid_font_dict else None
    if raw is None:
        return None
    if Document.name_of(doc.resolve(raw)) == "Identity":
        return None
    stream = doc.get_stream(raw)
    if stream is None:
        return None
    return parse_cid_to_gid_stream(stream_bytes(stream))


def parse_cid_to_gid_stream(data: bytes) -> list[int] | None:
    if len(data) < 2:
        return None
    return [
        (data[i] << 8) | data[i + 1] for i in range(0, len(data) - len(data) % 2, 2)
    ]


def build_cmap_with_cid_to_gid_map(
    cmap: ToUnicodeCMap, cid_to_gid: list[int]
) -> ToUnicodeCMap | None:
    """Apply a CIDToGIDMap to a CMap that maps GID->Unicode, yielding CID->Unicode."""
    new_cmap = ToUnicodeCMap()
    for cid, gid in enumerate(cid_to_gid):
        mapped = cmap.lookup(gid)
        if mapped is not None:
            new_cmap.char_map[cid] = mapped
    if not new_cmap.char_map:
        return None
    new_cmap.code_byte_length = 2
    return new_cmap


# ── TrueType-derived CMaps ───────────────────────────────────────────


def build_cmap_from_truetype(font_data: bytes) -> ToUnicodeCMap | None:
    """Build a CMap from an embedded TrueType font's ``cmap`` table.

    For Identity-H CID fonts CID == GID, and the font's cmap maps
    Unicode -> GID, so reversing it gives CID -> Unicode.
    """
    face = Face.parse(font_data)
    if face is None:
        return None
    gid_to_unicode = build_gid_to_unicode(face)
    if not gid_to_unicode:
        return None

    logger.debug("TrueType cmap: %s GID->Unicode entries", len(gid_to_unicode))

    cmap = ToUnicodeCMap()
    for gid, char in gid_to_unicode.items():
        cmap.char_map[gid] = char
    cmap.code_byte_length = 2  # Identity-H uses 2-byte CIDs
    return cmap


def build_simple_cmap_from_truetype(font_data: bytes) -> ToUnicodeCMap | None:
    """Build a single-byte CMap for a simple font from its embedded font file.

    In subsetted TrueType fonts the GID is not the character code, so the
    font's own encoding subtable is used to translate the byte codes that
    appear in content streams into GIDs before mapping them to Unicode.
    """
    face = Face.parse(font_data)
    if face is None:
        return None
    gid_to_unicode = build_gid_to_unicode(face)
    if not gid_to_unicode:
        return None

    cmap = ToUnicodeCMap()
    used_encoding_cmap = False

    # Preference order matches upstream: Mac Roman maps byte codes directly,
    # Windows Symbol maps F000+byte, Windows BMP maps codepoints (which single-
    # byte OCR output often mislabels as WinAnsi).
    for platform_id, encoding_id, offset in (
        (PLATFORM_MACINTOSH, 0, 0),
        (PLATFORM_WINDOWS, WINDOWS_SYMBOL, 0xF000),
        (PLATFORM_WINDOWS, WINDOWS_BMP, 0),
    ):
        subtable = face.find_subtable(platform_id, encoding_id)
        if subtable is None:
            continue
        for code in range(0x20, 0x100):
            gid = subtable.glyph_index(code + offset)
            if gid is None:
                continue
            char = gid_to_unicode.get(gid)
            if char is not None:
                cmap.char_map.setdefault(code, strip_pua_char(char))
        used_encoding_cmap = True
        break

    if not used_encoding_cmap:
        # No encoding subtable — treat the GID as the code.
        for gid, char in gid_to_unicode.items():
            if gid <= 0xFF:
                cmap.char_map[gid] = char
        # Fill the remaining single-byte codes from glyph names, which recovers
        # ligatures such as "t_i".
        for gid in range(face.number_of_glyphs):
            if gid > 0xFF or gid in cmap.char_map:
                continue
            name = face.glyph_name(gid)
            if name is None:
                continue
            text = glyph_name_to_string(name)
            if text is not None:
                cmap.char_map[gid] = text

    if not cmap.char_map:
        return None
    logger.debug("TrueType simple cmap: %s code->Unicode entries", len(cmap.char_map))
    cmap.code_byte_length = 1
    return cmap


def build_cmap_from_glyph_names(face: Face) -> ToUnicodeCMap | None:
    """Build a CMap from a font's ``post`` glyph names via the Adobe Glyph List."""
    cmap = ToUnicodeCMap()
    for gid in range(face.number_of_glyphs):
        name = face.glyph_name(gid)
        if name is None:
            continue
        char = glyph_to_char(name)
        if char is not None:
            cmap.char_map[gid] = char

    if not cmap.char_map:
        return None
    logger.debug("TrueType post glyph names: %s entries", len(cmap.char_map))
    cmap.code_byte_length = 2
    return cmap


def build_gid_to_unicode(face: Face) -> dict[int, str]:
    """Reverse a font's Unicode subtables into GID -> character.

    The lowest codepoint wins where several map to the same glyph, matching
    upstream's first-write-wins behaviour.
    """
    gid_to_unicode: dict[int, str] = {}

    for subtable in face.cmap_subtables:
        is_symbol = (
            subtable.platform_id == PLATFORM_WINDOWS
            and subtable.encoding_id == WINDOWS_SYMBOL
        )
        if not subtable.is_unicode() and not is_symbol:
            continue
        for codepoint in sorted(subtable.mapping):
            gid = subtable.mapping[codepoint]
            if not gid or gid in gid_to_unicode:
                continue
            if 0xD800 <= codepoint <= 0xDFFF or codepoint > 0x10FFFF:
                continue
            gid_to_unicode[gid] = chr(codepoint)

    if not gid_to_unicode:
        from_names = build_cmap_from_glyph_names(face)
        if from_names is None:
            return {}
        return {
            gid: text[0] for gid, text in from_names.char_map.items() if text
        }

    return gid_to_unicode


def strip_pua_char(char: str) -> str:
    """Strip the Private Use Area F000 offset (Windows Symbol convention)."""
    code = ord(char)
    if 0xF000 <= code <= 0xF0FF:
        return chr(code - 0xF000)
    return char


def glyph_name_to_string(name: str) -> str | None:
    """Resolve a glyph name to text, including underscore-joined ligatures."""
    base = name.split(".")[0]
    char = glyph_to_char(base)
    if char is not None:
        return char

    if "_" in base:
        out: list[str] = []
        for part in base.split("_"):
            if not part:
                return None
            part_char = glyph_to_char(part)
            if part_char is not None:
                out.append(part_char)
            elif len(part) == 1:
                out.append(part)
            else:
                return None
        if out:
            return "".join(out)

    if base in ("ti", "tt", "tz"):
        return base
    return None


# ── predefined collections ───────────────────────────────────────────


def build_cmap_from_cid_system_info(
    doc: Document, cid_font_dict: Any
) -> ToUnicodeCMap | None:
    """Build a CMap from the predefined collection named by CIDSystemInfo."""
    ordering = _cid_system_info_ordering(doc, cid_font_dict)
    if ordering is None:
        return None

    if ordering == "Korea1":
        cmap = ToUnicodeCMap()
        for cid, code in ADOBE_KOREA1_CID_TO_UNICODE.items():
            if 0xD800 <= code <= 0xDFFF:
                continue
            cmap.char_map[cid] = chr(code)
        cmap.code_byte_length = 2
        logger.debug("Adobe-Korea1 predefined CMap: %s entries", len(cmap.char_map))
        return cmap

    if ordering in ("Japan1", "GB1", "CNS1"):
        return build_cmap_from_builtin_cmap(ordering)

    return None


def _cid_system_info_ordering(doc: Document, cid_font_dict: Any) -> str | None:
    csi = doc.get_dictionary(cid_font_dict.get("/CIDSystemInfo"))
    if csi is None:
        return None
    ordering = doc.resolve(csi.get("/Ordering"))
    return str(ordering) if isinstance(ordering, str) else None


def build_cmap_from_builtin_cmap(ordering: str) -> ToUnicodeCMap | None:
    """Load a predefined CID collection from pdf.js's bundled binary CMaps.

    Not implemented. Upstream ships ``external/bcmaps`` and parses its binary
    CMap format; that parser has not been ported yet, so Adobe-Japan1,
    Adobe-GB1 and Adobe-CNS1 fall through to the other fallbacks.
    Adobe-Korea1 is unaffected — it is served from the generated table in
    :mod:`pdf_inspector.adobe_korea1`.
    """
    logger.debug("builtin bcmap for ordering=%s is not available in this port", ordering)
    return None


def load_builtin_cmap_by_name(name: str) -> ToUnicodeCMap | None:
    """Resolve a ``usecmap`` reference. See :func:`build_cmap_from_builtin_cmap`."""
    logger.debug("usecmap target %s is not available in this port", name)
    return None


def build_fallback_tounicode_from_encoding(
    doc: Document, font_dict: Any
) -> ToUnicodeCMap | None:
    """Compose the font's /Encoding CMap with its collection's UCS2 CMap.

    Always returns None while :func:`build_cmap_from_builtin_cmap` is
    unimplemented: the UCS2 half of the composition is what is missing.
    """
    return None


# ── font-file fallbacks ──────────────────────────────────────────────


def build_fallback_cmap_for_type0(doc: Document, font_dict: Any) -> ToUnicodeCMap | None:
    """A fallback CMap for a Type0 Identity-H font from its embedded font data."""
    if Document.name_of(doc.resolve(font_dict.get("/Subtype"))) != "Type0":
        return None
    if Document.name_of(doc.resolve(font_dict.get("/Encoding"))) not in (
        "Identity-H",
        "Identity-V",
    ):
        return None

    cid_font_dict = get_descendant_cid_font(doc, font_dict)
    if cid_font_dict is None:
        return None

    font_file_raw = _font_file_raw(doc, cid_font_dict)
    if font_file_raw is not None:
        stream = doc.get_stream(font_file_raw)
        if stream is not None:
            cmap = build_cmap_from_truetype(stream_bytes(stream))
            if cmap is not None:
                cid_to_gid = get_cid_to_gid_map(doc, cid_font_dict)
                if cid_to_gid is not None:
                    repaired = build_cmap_with_cid_to_gid_map(cmap, cid_to_gid)
                    if repaired is not None:
                        logger.debug(
                            "Fallback TrueType CMap repaired with CIDToGIDMap: %s entries",
                            len(repaired.char_map),
                        )
                        return repaired
                logger.debug(
                    "Fallback TrueType CMap (Type0) char_map=%s", len(cmap.char_map)
                )
                return cmap

    cmap = build_cmap_from_cid_system_info(doc, cid_font_dict)
    if cmap is not None:
        logger.debug(
            "Fallback CIDSystemInfo CMap (Type0) char_map=%s", len(cmap.char_map)
        )
    return cmap


def build_fallback_cmap_for_simple(
    doc: Document, font_dict: Any
) -> ToUnicodeCMap | None:
    """A fallback CMap for a simple (non-Type0) font from its embedded font data."""
    subtype = Document.name_of(doc.resolve(font_dict.get("/Subtype")))
    if subtype is None or subtype == "Type0":
        return None

    font_file_raw = _font_file_raw(doc, font_dict)
    if font_file_raw is None:
        return None
    stream = doc.get_stream(font_file_raw)
    if stream is None:
        return None

    cmap = build_simple_cmap_from_truetype(stream_bytes(stream))
    if cmap is not None:
        logger.debug(
            "Fallback simple font cmap char_map=%s", len(cmap.char_map)
        )
    return cmap


def _font_file_raw(doc: Document, dictionary: Any) -> Any | None:
    """The raw ``/FontFile2`` or ``/FontFile3`` reference from a descriptor."""
    descriptor = doc.get_dictionary(dictionary.get("/FontDescriptor"))
    if descriptor is None:
        return None
    for key in ("/FontFile2", "/FontFile3"):
        if key in descriptor:
            return descriptor.raw_get(key)
    return None


# ── /W array heuristic ───────────────────────────────────────────────


def cid_values_look_like_unicode(cid_font_dict: Any, doc: Document) -> bool:
    """Whether a CID font's ``/W`` array suggests CIDs *are* Unicode codepoints.

    Many generators (Chromium, wkhtmltopdf) emit Identity-H fonts where the CID
    equals the Unicode codepoint, which makes passthrough decoding work with no
    ToUnicode CMap. GID-based subsets instead number from zero, so the median
    CID separates the two cases.
    """
    w_arr = doc.resolve(cid_font_dict.get("/W")) if cid_font_dict else None
    if not isinstance(w_arr, (list, tuple)):
        return False

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
                for j in range(len(following)):
                    if len(seen) >= MAX_CID_W_EXPANSION:
                        break
                    seen.add((start + j) & 0xFFFF)
                i += 2
            else:
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
