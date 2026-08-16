"""Ported from the ``#[cfg(test)]`` block in upstream src/detector.rs.

These cover the byte-level content-stream scanner and the OCR reason
classification, which are the parts of detection that do not need a document.
"""

from __future__ import annotations

from pdf_inspector.detector import (
    PageAnalysis,
    distribute_pages,
    page_ocr_reasons,
    scan_content_for_text_operators,
)
from pdf_inspector.ocr_reasons import (
    OCR_REASON_NO_TEXT,
    OCR_REASON_SCANNED,
    OCR_REASON_SUSPECTED_GARBLED_TEXT,
    OCR_REASON_VECTOR_TEXT,
)


def scan(content: bytes) -> tuple[int, int, int, int, set[int]]:
    """Run the scanner and return its counters plus the collected characters."""
    unique_chars: set[int] = set()
    ops, imgs, paths, fonts = scan_content_for_text_operators(
        content, unique_chars, set()
    )
    return ops, imgs, paths, fonts, unique_chars


def test_page_ocr_reasons_classify():
    scanned = PageAnalysis(has_images=True)
    assert page_ocr_reasons(scanned) == [OCR_REASON_SCANNED]

    blank = PageAnalysis()
    assert page_ocr_reasons(blank) == [OCR_REASON_NO_TEXT]

    vector = PageAnalysis(has_vector_text=True)
    assert page_ocr_reasons(vector) == [OCR_REASON_VECTOR_TEXT]

    garbled = PageAnalysis(has_identity_h_no_tounicode=True)
    assert page_ocr_reasons(garbled) == [OCR_REASON_SUSPECTED_GARBLED_TEXT]

    type3 = PageAnalysis(has_only_type3_fonts=True)
    assert page_ocr_reasons(type3) == [OCR_REASON_SUSPECTED_GARBLED_TEXT]

    # Undecodable fonts and vector text are reported together, fonts first.
    both = PageAnalysis(has_identity_h_no_tounicode=True, has_vector_text=True)
    assert page_ocr_reasons(both) == [
        OCR_REASON_SUSPECTED_GARBLED_TEXT,
        OCR_REASON_VECTOR_TEXT,
    ]


def test_scan_content_operators():
    # Sample PDF content stream with text operators
    ops, imgs, _, _, uchars = scan(b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET")
    assert ops == 1
    assert imgs == 0
    # "Hello World" without space: H, e, l, o, W, r, d = 7 unique
    assert len(uchars) >= 7

    # Content with a TJ array
    ops2, _, _, _, uchars2 = scan(b"BT /F1 12 Tf 100 700 Td [(H) 10 (ello)] TJ ET")
    assert ops2 == 1
    # H, e, l, o = 4 unique
    assert len(uchars2) >= 4

    # Content with Do (XObject invocation — not counted as an image here; actual
    # image detection is handled by scan_xobjects_in_resources)
    ops3, imgs3, _, _, _ = scan(b"q 100 0 0 100 50 700 cm /Img1 Do Q")
    assert ops3 == 0
    assert imgs3 == 0


def test_scan_content_successive_tj_collects_each_operand():
    # Lookback is floored at the previous Tj/TJ/Tf so later operators must still
    # see their own operands.
    ops, _, _, _, uchars = scan(b"[(Hello)] TJ [(World)] TJ (More) Tj")
    assert ops == 3
    for ch in b"HeloWrdM":
        assert ch in uchars, f"missing char {chr(ch)}"


def test_scan_content_tj_inside_literal_is_not_an_operator():
    # `Tj` followed by a space inside a literal must not count as an operator or
    # pin the lookback floor; the real `Tj` still collects the string.
    ops, _, _, _, uchars = scan(b"BT (Hello Tj World) Tj ET")
    assert ops == 1
    for ch in b"HeloTjWrd":
        assert ch in uchars, f"missing char {chr(ch)}"


def test_scan_content_malformed_tj_lookback_stays_linear():
    # `] TJ` with no `[` used to walk the entire prefix for every operator
    # (quadratic). 30k repeats is enough that a prefix rescan would dominate the
    # runtime; with the floor it is a single linear pass.
    n = 30_000
    ops, _, _, _, uchars = scan(b"] TJ\n" * n)
    assert ops == n
    assert not uchars


def test_image_dominated_detection():
    # Do operators are not counted as images by scan_content_for_text_operators;
    # image-dominated detection relies on scan_xobjects_in_resources, which
    # checks XObject Subtype. Verify Do operators don't inflate image_count.
    content = b"".join(f"/Im{i} Do\n".encode() for i in range(50))
    content += b"BT (x) Tj ET\n" * 3

    ops, imgs, _, _, uchars = scan(content)
    assert ops == 3
    assert imgs == 0
    assert len(uchars) == 1


def test_normal_text_not_image_dominated():
    ops, imgs, _, _, uchars = scan(
        b"BT /F1 12 Tf (The quick brown fox jumps over the lazy dog) Tj ET\n"
        b"/Img1 Do\n/Img2 Do\n"
    )
    assert ops == 1
    assert imgs == 0
    assert len(uchars) >= 5


def test_path_heavy_detection():
    # Simulate vector-outlined text: many path ops, few text ops
    content = b"BT (Header) Tj ET\n"
    content += b"100 200 m 150 250 l 200 200 c h\n" * 500
    content += b"f\n"

    text, imgs, paths, _, _ = scan(content)
    assert text == 1
    assert imgs == 0
    # 500 * (m + l + c + h) + 1 f = 2001
    assert paths >= 2000, f"expected >= 2000 path ops, got {paths}"

    # Should trigger vector text detection: paths >= 1000 && paths > text * 200
    assert paths >= 1000 and paths > text * 200


def test_normal_paths_not_vector_text():
    # Normal page: text with some decorative paths (charts, borders)
    content = b"BT (Some text content here) Tj ET\n" * 20
    content += b"100 200 m 150 250 l 200 200 c h f\n" * 10

    text, _, paths, _, _ = scan(content)
    assert text == 20
    assert paths >= 40, f"expected >= 40 path ops, got {paths}"

    # Should NOT trigger: paths < 1000
    assert not (paths >= 1000 and paths > text * 200)


def test_distribute_pages_includes_first_and_last():
    assert distribute_pages(0, 10) == []
    assert distribute_pages(20, 5) == [1, 2, 3, 4, 5]
    sampled = distribute_pages(4, 100)
    assert sampled[0] == 1
    assert sampled[-1] == 100
    assert len(sampled) == 4
