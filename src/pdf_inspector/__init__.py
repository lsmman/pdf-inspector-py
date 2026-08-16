"""Pure-Python PDF classification and Markdown extraction.

A port of `firecrawl/pdf-inspector <https://github.com/firecrawl/pdf-inspector>`_.

The public API mirrors upstream's Python bindings. Functions that depend on the
extraction pipeline are not available yet — see the port status table in the
README for what has landed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .detector import (
    DetectionConfig,
    PdfType,
    PdfTypeResult,
    ScanStrategy,
    detect_from_document,
)
from .errors import (
    NotAPdfError,
    PdfEncryptedError,
    PdfError,
    PdfInvalidStructureError,
    PdfIoError,
    PdfParseError,
)
from .pdfdoc import Document
from .process_mode import ProcessMode

__version__ = "0.1.0"

__all__ = [
    "PageOcrReasons",
    "PdfClassification",
    "PdfResult",
    "PdfType",
    "ProcessMode",
    "DetectionConfig",
    "ScanStrategy",
    "PdfError",
    "PdfIoError",
    "PdfParseError",
    "PdfEncryptedError",
    "PdfInvalidStructureError",
    "NotAPdfError",
    "classify_pdf",
    "classify_pdf_bytes",
    "detect_pdf",
    "detect_pdf_bytes",
]


@dataclass
class PageOcrReasons:
    """OCR reasons for a single 1-indexed page."""

    page: int
    reasons: list[str]


@dataclass
class PdfClassification:
    """Lightweight PDF classification result."""

    #: 'text_based', 'scanned', 'image_based', or 'mixed'.
    pdf_type: str
    page_count: int
    #: 0-indexed page numbers that need OCR.
    pages_needing_ocr: list[int]
    confidence: float


@dataclass
class PdfResult:
    """Result of processing a PDF file."""

    #: 'text_based', 'scanned', 'image_based', or 'mixed'.
    pdf_type: str
    markdown: str | None
    page_count: int
    processing_time_ms: int
    #: 1-indexed page numbers that need OCR.
    pages_needing_ocr: list[int]
    #: Machine-readable OCR reasons by 1-indexed page.
    ocr_reasons_by_page: list[PageOcrReasons]
    title: str | None
    confidence: float
    is_complex_layout: bool = False
    pages_with_tables: list[int] = field(default_factory=list)
    pages_with_columns: list[int] = field(default_factory=list)
    has_encoding_issues: bool = False


def classify_pdf(path: str | Path, password: str | None = None) -> PdfClassification:
    """Lightweight classification — type, page count, and OCR pages (0-indexed)."""
    return classify_pdf_bytes(Path(path).read_bytes(), password)


def classify_pdf_bytes(data: bytes, password: str | None = None) -> PdfClassification:
    """Lightweight classification from bytes.

    Pages in ``pages_needing_ocr`` are 0-indexed, matching upstream.
    """
    detection = _detect(data, password)
    return PdfClassification(
        pdf_type=detection.pdf_type.value,
        page_count=detection.page_count,
        # Convert from 1-indexed to 0-indexed for caller convenience
        pages_needing_ocr=[p - 1 for p in detection.pages_needing_ocr],
        confidence=detection.confidence,
    )


def detect_pdf(path: str | Path, password: str | None = None) -> PdfResult:
    """Fast detection only — no text extraction."""
    return detect_pdf_bytes(Path(path).read_bytes(), password)


def detect_pdf_bytes(data: bytes, password: str | None = None) -> PdfResult:
    """Fast detection from bytes.

    ``markdown`` is always ``None`` and the layout-complexity fields stay at
    their defaults: those come from the extraction pipeline, which this mode
    deliberately skips.
    """
    started = time.perf_counter()
    detection = _detect(data, password)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return PdfResult(
        pdf_type=detection.pdf_type.value,
        markdown=None,
        page_count=detection.page_count,
        processing_time_ms=elapsed_ms,
        pages_needing_ocr=list(detection.pages_needing_ocr),
        ocr_reasons_by_page=[
            PageOcrReasons(page=page, reasons=list(reasons))
            for page, reasons in sorted(detection.ocr_reasons_by_page.items())
        ],
        title=detection.title,
        confidence=detection.confidence,
    )


def _detect(
    data: bytes, password: str | None, config: DetectionConfig | None = None
) -> PdfTypeResult:
    doc = Document.from_bytes(data, password)
    return detect_from_document(doc, doc.page_count, config or DetectionConfig())
