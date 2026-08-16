"""Classification behaviour on upstream's fixture corpus.

The fixtures are named after the behaviour they exist to pin down, so each of
these asserts the signal the name declares rather than a whole-document
snapshot — a snapshot would also lock in incidental details that upstream is
free to change.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest
from pypdf.errors import DependencyError

from pdf_inspector import classify_pdf, detect_pdf
from pdf_inspector import pdfdoc
from pdf_inspector.errors import NotAPdfError, PdfEncryptedError
from pdf_inspector.ocr_reasons import (
    OCR_REASON_SUSPECTED_GARBLED_TEXT,
    OCR_REASON_VECTOR_TEXT,
)
from pdf_inspector.pdfdoc import Document

FIXTURES = Path(__file__).parent / "fixtures"

# Fixtures needing a password, or that are deliberately huge.
ENCRYPTED = {"encrypted-secret123.pdf"}
SLOW = {"bits_pilani_feedback.pdf", "multiline_indent_cell_rect_grid.pdf"}


@pytest.fixture(autouse=True)
def quiet_pypdf():
    """pypdf logs a warning per repaired xref entry; the fixtures include
    deliberately damaged files, so the noise is expected."""
    logging.getLogger("pypdf").setLevel(logging.ERROR)


def fixture_names() -> list[str]:
    return sorted(p.name for p in FIXTURES.glob("*.pdf"))


@pytest.mark.parametrize("name", [n for n in fixture_names() if n not in ENCRYPTED])
def test_every_fixture_classifies(name: str):
    """No fixture may crash the detector, including the malformed ones."""
    if name in SLOW:
        pytest.skip("covered by test_large_document_classifies")
    result = classify_pdf(FIXTURES / name)
    assert result.pdf_type in ("text_based", "scanned", "image_based", "mixed")
    assert result.page_count >= 1
    assert 0.0 <= result.confidence <= 1.0


def test_scan_with_native_header_text_is_image_backed():
    """A scanned page with a native header must still route to OCR.

    The header's text operators must not be enough to call the page text-based.
    """
    result = detect_pdf(FIXTURES / "scan_with_native_header_text.pdf")
    assert result.pdf_type in ("scanned", "image_based")
    assert result.pages_needing_ocr == [1]


def test_watermark_image_does_not_force_ocr():
    """The mirror case: a text page carrying a background image stays text."""
    result = detect_pdf(FIXTURES / "text_page_with_watermark_image.pdf")
    assert result.pdf_type == "text_based"
    assert result.pages_needing_ocr == []


def test_vector_outlined_text_needs_ocr_with_reason():
    """Glyphs drawn as paths cannot be extracted as text at all."""
    result = detect_pdf(FIXTURES / "vector_outlined_text_with_caption.pdf")
    assert result.pages_needing_ocr == [1]
    reasons = {r for entry in result.ocr_reasons_by_page for r in entry.reasons}
    assert OCR_REASON_VECTOR_TEXT in reasons


def test_identity_h_without_tounicode_is_flagged():
    """Identity-H with no ToUnicode decodes to garbage, so the page needs OCR
    even though it has a real text layer."""
    result = detect_pdf(FIXTURES / "shinagawa_identity_h.pdf")
    assert result.pages_needing_ocr == [1]
    reasons = {r for entry in result.ocr_reasons_by_page for r in entry.reasons}
    assert OCR_REASON_SUSPECTED_GARBLED_TEXT in reasons


def test_broken_startxref_still_parses():
    """A damaged cross-reference table must not stop classification."""
    result = classify_pdf(FIXTURES / "broken_startxref_pointer.pdf")
    assert result.page_count >= 1


def test_large_document_classifies():
    """A 430-page document exercises the sampling strategy end to end."""
    result = classify_pdf(FIXTURES / "bits_pilani_feedback.pdf")
    assert result.pdf_type == "text_based"
    assert result.page_count == 430


def test_pages_needing_ocr_is_zero_indexed_in_classify():
    """classify_pdf reports 0-indexed pages; detect_pdf reports 1-indexed."""
    classification = classify_pdf(FIXTURES / "scan_with_native_header_text.pdf")
    detection = detect_pdf(FIXTURES / "scan_with_native_header_text.pdf")
    assert classification.pages_needing_ocr == [0]
    assert detection.pages_needing_ocr == [1]


has_cryptography = importlib.util.find_spec("cryptography") is not None


@pytest.mark.skipif(
    not has_cryptography,
    reason="AES decryption needs the optional 'crypto' extra",
)
class TestEncryptedPdf:
    PATH = FIXTURES / "encrypted-secret123.pdf"

    def test_no_password_is_rejected(self):
        with pytest.raises(PdfEncryptedError):
            classify_pdf(self.PATH)

    def test_wrong_password_is_rejected(self):
        with pytest.raises(PdfEncryptedError):
            Document.from_path(self.PATH, "wrong")

    def test_correct_password_opens(self):
        assert Document.from_path(self.PATH, "secret123").page_count == 8


def test_missing_cryptography_says_what_to_install(monkeypatch):
    """Without the extra, an AES-encrypted PDF must name the fix.

    pypdf raises DependencyError here; surfacing that as a generic parse error
    would leave the caller with no idea that one `pip install` fixes it.
    """

    def raise_dependency_error(*args, **kwargs):
        raise DependencyError("cryptography>=3.1 is required for AES algorithm")

    monkeypatch.setattr(pdfdoc, "PdfReader", raise_dependency_error)

    with pytest.raises(PdfEncryptedError) as excinfo:
        Document.from_path(FIXTURES / "encrypted-secret123.pdf")

    assert "pdf-inspector-py[crypto]" in str(excinfo.value)


def test_non_pdf_bytes_rejected():
    with pytest.raises(NotAPdfError):
        Document.from_bytes(b"<html><body>not a pdf</body></html>")
