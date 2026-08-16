"""Ported from the ``#[cfg(test)]`` block in upstream src/glyph_names.rs,
plus table-integrity checks for the generated data module.
"""

from __future__ import annotations

from pdf_inspector.adobe_korea1 import ADOBE_KOREA1_CID_TO_UNICODE, lookup_korea1
from pdf_inspector.glyph_names import glyph_to_char


def test_uni_hex_parsing():
    assert glyph_to_char("uni0041") == "A"
    assert glyph_to_char("uni00e9") == "é"
    # PUA F0xx symbol-encoding offset is stripped.
    assert glyph_to_char("uniF041") == "A"


def test_u_hex_parsing():
    assert glyph_to_char("u0041") == "A"
    assert glyph_to_char("u1F600") == "😀"


def test_non_ascii_uni_name_does_not_panic():
    # A crafted /Differences name like `/uni#80#80#80#80` decodes into "uni"
    # followed by four U+FFFD replacements. Upstream's byte slicing would land
    # mid-character; it must be handled gracefully instead.
    assert glyph_to_char("uni" + "�" * 4) is None
    assert glyph_to_char("uni�bc") is None
    assert glyph_to_char("unié00") is None


def test_agl_suffix_is_stripped():
    # Per the Adobe Glyph List spec, the suffix after '.' is dropped.
    assert glyph_to_char("zero.tf") == "0"
    assert glyph_to_char("hyphen.case") == "-"


def test_local_overrides_present():
    assert glyph_to_char("C21") == "≥"
    assert glyph_to_char("C25") == "≈"
    assert glyph_to_char("C19") == "~"
    assert glyph_to_char("C24") == "~"


def test_generated_table_sizes_match_upstream():
    # Upstream ships 17056 Adobe-Korea1 entries; the codegen must not drop any.
    assert len(ADOBE_KOREA1_CID_TO_UNICODE) == 17056


def test_korea1_lookup():
    assert lookup_korea1(1) == " "
    assert lookup_korea1(18154) == "힣"
    assert lookup_korea1(999999) is None
