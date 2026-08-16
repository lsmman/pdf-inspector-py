"""The embedded-font reader, checked against fonts from upstream's fixtures.

Upstream delegates this to the ttf-parser crate, so there are no unit tests to
port. These pin the behaviour the ported code depends on instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pdf_inspector.pdfdoc import Document, stream_bytes
from pdf_inspector.tounicode import (
    build_cmap_from_truetype,
    glyph_name_to_string,
    strip_pua_char,
)
from pdf_inspector.truetype import Face, font_has_unicode_cmap

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def quiet_pypdf():
    logging.getLogger("pypdf").setLevel(logging.ERROR)


def embedded_fonts(pdf_name: str) -> list[bytes]:
    """Every embedded FontFile2/FontFile3 stream in a fixture."""
    doc = Document.from_path(FIXTURES / pdf_name)
    out: list[bytes] = []
    seen: set[tuple[int, int]] = set()

    for page in doc.get_pages().values():
        own, ancestors = doc.get_page_resources(page)
        resource_dicts = ([own] if own else []) + [
            doc.dictionary_for_id(a) for a in ancestors
        ]
        for resources in resource_dicts:
            if resources is None:
                continue
            fonts = doc.get_dictionary(resources.get("/Font"))
            if fonts is None:
                continue
            for key in list(fonts.keys()):
                font = doc.get_dictionary(fonts.raw_get(key))
                if font is None:
                    continue
                candidates = [font]
                descendants = doc.resolve(font.get("/DescendantFonts"))
                if isinstance(descendants, (list, tuple)) and descendants:
                    child = doc.get_dictionary(descendants[0])
                    if child is not None:
                        candidates.append(child)
                for candidate in candidates:
                    descriptor = doc.get_dictionary(candidate.get("/FontDescriptor"))
                    if descriptor is None:
                        continue
                    for name in ("/FontFile2", "/FontFile3"):
                        if name not in descriptor:
                            continue
                        raw = descriptor.raw_get(name)
                        object_id = Document.object_id(raw)
                        if object_id in seen:
                            continue
                        if object_id is not None:
                            seen.add(object_id)
                        stream = doc.get_stream(raw)
                        if stream is not None:
                            out.append(stream_bytes(stream))
    return out


def test_reads_cmap_from_a_real_embedded_font():
    fonts = embedded_fonts("2013-app2.pdf")
    assert fonts, "fixture should embed at least one font file"

    faces = [f for f in (Face.parse(data) for data in fonts) if f is not None]
    assert faces, "at least one embedded font should parse"

    face = faces[0]
    assert face.number_of_glyphs > 0
    assert face.has_unicode_cmap()

    # A Unicode subtable must actually map something — an empty cmap does not
    # make a font decodable, which is the distinction upstream's codepoint count
    # draws.
    unicode_tables = [s for s in face.cmap_subtables if s.is_unicode() and s.mapping]
    assert unicode_tables

    # Round-trip a letter through the subtable the CMap builder uses.
    cmap = build_cmap_from_truetype(fonts[0])
    assert cmap is not None
    assert cmap.code_byte_length == 2
    assert cmap.char_map


def test_font_without_cmap_is_reported_as_such():
    """shinagawa_identity_h embeds a subset font with its cmap stripped.

    That is exactly why the detector routes it to OCR, so the reader must not
    claim it is decodable.
    """
    fonts = embedded_fonts("shinagawa_identity_h.pdf")
    assert fonts

    face = Face.parse(fonts[0])
    assert face is not None
    assert face.number_of_glyphs > 0
    assert not face.has_unicode_cmap()
    assert not font_has_unicode_cmap(fonts[0])
    assert build_cmap_from_truetype(fonts[0]) is None


def test_bare_cff_is_rejected_like_upstream():
    """A FontFile3 holding bare CFF has no sfnt table directory.

    ttf-parser rejects it too, so the port stays faithful by returning None
    rather than inventing a mapping.
    """
    assert Face.parse(b"\x01\x00\x04\x01not-an-sfnt") is None
    assert not font_has_unicode_cmap(b"\x01\x00\x04\x01not-an-sfnt")


def test_malformed_font_data_is_rejected():
    assert Face.parse(b"") is None
    assert Face.parse(b"\x00" * 4) is None
    assert not font_has_unicode_cmap(b"garbage")


def test_strip_pua_char_unshifts_symbol_range():
    assert strip_pua_char("") == "A"
    assert strip_pua_char("") == "ÿ"
    # Outside F000-F0FF the character is left alone.
    assert strip_pua_char("") == ""
    assert strip_pua_char("A") == "A"


def test_glyph_name_to_string_resolves_names_and_ligatures():
    assert glyph_name_to_string("A") == "A"
    # The AGL suffix after '.' is dropped.
    assert glyph_name_to_string("zero.tf") == "0"
    # The full name is looked up before it is split, so an underscore-joined
    # name the Adobe Glyph List already knows resolves to its ligature.
    assert glyph_name_to_string("f_i") == "ﬁ"
    assert glyph_name_to_string("f_f_i") == "ﬃ"
    # Only names the list does not know get split and concatenated.
    assert glyph_name_to_string("t_i") == "ti"
    assert glyph_name_to_string("a_b") == "ab"
    # Names that resolve to nothing stay unresolved.
    assert glyph_name_to_string("notaglyphname") is None
