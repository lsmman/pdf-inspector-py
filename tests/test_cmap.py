"""Ported from the ``#[cfg(test)]`` block in upstream src/tounicode.rs."""

from __future__ import annotations

from pdf_inspector.cmap import ToUnicodeCMap, hex_to_unicode_string


def parse(text: str) -> ToUnicodeCMap:
    cmap = ToUnicodeCMap.parse(text.encode("utf-8"))
    assert cmap is not None
    return cmap


def test_parse_bfchar_2byte():
    cmap = parse(
        """
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
1 begincodespacerange
<0000><FFFF>
endcodespacerange
3 beginbfchar
<0003> <0020>
<0024> <0041>
<0025> <0042>
endbfchar
endcmap
"""
    )
    assert cmap.code_byte_length == 2
    assert cmap.lookup(0x0003) == " "
    assert cmap.lookup(0x0024) == "A"
    assert cmap.lookup(0x0025) == "B"


def test_hex_to_unicode_non_ascii_no_panic():
    # A destination containing a multi-byte char makes the byte length even
    # while a byte offset can land inside a char. Upstream must not panic
    # slicing it; both implementations reject it instead.
    assert hex_to_unicode_string("XéY") is None
    assert hex_to_unicode_string("�0") is None


def test_parse_bfchar_non_ascii_destination_no_panic():
    # Crafted /ToUnicode CMap: a non-hex, non-ASCII destination previously
    # triggered a char-boundary panic upstream. The malformed entry is skipped.
    ToUnicodeCMap.parse(b"beginbfchar <0041> <X\xc3\xa9Y> endbfchar")


def test_parse_bfchar_1byte():
    # The pattern that caused the CJK bug: the codespace says <0000><FFFF> but
    # every source code is 1-byte hex.
    cmap = parse(
        """
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
3 beginbfchar
<20> <0020>
<41> <0041>
<42> <0042>
endbfchar
"""
    )
    assert cmap.code_byte_length == 1
    assert cmap.lookup(0x0020) == " "
    assert cmap.lookup(0x0041) == "A"


def test_decode_cids_2byte():
    cmap = parse(
        """
1 begincodespacerange
<0000><FFFF>
endcodespacerange
3 beginbfchar
<0003> <0020>
<0024> <0041>
<0025> <0042>
endbfchar
"""
    )
    # "AB " in 2-byte CID encoding
    assert cmap.decode_cids(bytes([0x00, 0x24, 0x00, 0x25, 0x00, 0x03])) == "AB "


def test_decode_cids_1byte_no_cjk_garbage():
    cmap = parse(
        """
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
5 beginbfchar
<20> <0020>
<42> <0042>
<79> <0079>
<50> <0050>
<52> <0052>
endbfchar
"""
    )
    assert cmap.code_byte_length == 1

    # "By" must decode to "By", not to the CJK character 䉹 that reading the two
    # bytes as one code would produce.
    result = cmap.decode_cids(bytes([0x42, 0x79]))
    assert result == "By"
    assert "䉹" not in result

    assert cmap.decode_cids(bytes([0x50, 0x52])) == "PR"


def test_bfrange_array_format():
    cmap = parse(
        """
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 beginbfrange
<0003> <0005> [<0041> <0042> <0043>]
endbfrange
"""
    )
    assert cmap.lookup(0x0003) == "A"
    assert cmap.lookup(0x0004) == "B"
    assert cmap.lookup(0x0005) == "C"


def test_parse_bfchar_surrogate_pair_emoji():
    cmap = parse(
        """
1 begincodespacerange
<00> <FF>
endcodespacerange
2 beginbfchar
<16> <D83CDF1F>
<9D> <D83CDFAD>
endbfchar
"""
    )
    assert cmap.code_byte_length == 1
    assert cmap.lookup(0x16) == "🌟"
    assert cmap.lookup(0x9D) == "🎭"


def test_bfrange_scalar_base():
    cmap = parse(
        """
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 beginbfrange
<0010> <0012> <0041>
endbfrange
"""
    )
    assert cmap.lookup(0x0010) == "A"
    assert cmap.lookup(0x0011) == "B"
    assert cmap.lookup(0x0012) == "C"
    assert cmap.lookup(0x0013) is None


def test_remap_to_sequential_sorts_and_renumbers_from_one():
    cmap = parse(
        """
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
3 beginbfchar
<0064> <0043>
<0010> <0041>
<0032> <0042>
endbfchar
"""
    )
    remapped = cmap.remap_to_sequential()
    # Sorted old CIDs 0x10, 0x32, 0x64 become 1, 2, 3 — GID 0 is .notdef.
    assert remapped.char_map == {1: "A", 2: "B", 3: "C"}


def test_empty_cmap_parses_to_none():
    assert ToUnicodeCMap.parse(b"begincmap endcmap") is None
