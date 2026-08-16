"""Smart PDF type detection without a full document load.

Detects whether a PDF is text-based, scanned, or image-based by sampling
content streams for text operators (Tj/TJ) rather than decoding every object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

from .ocr_reasons import (
    OCR_REASON_NO_TEXT,
    OCR_REASON_SCANNED,
    OCR_REASON_SUSPECTED_GARBLED_TEXT,
    OCR_REASON_VECTOR_TEXT,
)
from .pdfdoc import Document, ObjectId, PageRef, stream_bytes
from .tounicode import cid_values_look_like_unicode
from .truetype import font_has_unicode_cmap

logger = logging.getLogger(__name__)

_PDF_WHITESPACE = frozenset(b"\x00\t\n\x0c\r ")
#: Rust's ``u8::is_ascii_whitespace`` — note it excludes NUL, unlike the PDF
#: whitespace set above.
_ASCII_WHITESPACE = frozenset(b"\t\n\x0c\r ")
_PDF_NAME_DELIMITERS = _PDF_WHITESPACE | frozenset(b"()<>[]{}/%")
_ASCII_ALNUM_BYTES = frozenset(
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


class PdfType(Enum):
    """PDF type classification."""

    #: PDF has extractable text (Tj/TJ operators found).
    TEXT_BASED = "text_based"
    #: PDF appears to be scanned (images only, no text operators).
    SCANNED = "scanned"
    #: PDF contains mostly images with minimal/no text.
    IMAGE_BASED = "image_based"
    #: PDF has a mix of text and image-heavy pages.
    MIXED = "mixed"


@dataclass(frozen=True)
class ScanStrategy:
    """Strategy for which pages to scan during detection.

    Construct one through the class methods rather than directly:
    :meth:`early_exit`, :meth:`full`, :meth:`sample`, :meth:`pages`.
    """

    kind: str
    max_pages: int = 0
    page_numbers: tuple[int, ...] = ()

    @classmethod
    def early_exit(cls) -> ScanStrategy:
        """Scan all pages, stop on the first non-text page.

        Best for pipelines that route TextBased PDFs to fast extraction.
        """
        return cls("early_exit")

    @classmethod
    def full(cls) -> ScanStrategy:
        """Scan all pages, no early exit.

        Best when accurate Mixed vs Scanned classification is needed.
        """
        return cls("full")

    @classmethod
    def sample(cls, max_pages: int) -> ScanStrategy:
        """Sample up to ``max_pages`` evenly distributed pages.

        Best for very large PDFs where speed matters more than precision.
        """
        return cls("sample", max_pages=max_pages)

    @classmethod
    def pages(cls, page_numbers: Iterable[int]) -> ScanStrategy:
        """Only scan these specific 1-indexed page numbers."""
        return cls("pages", page_numbers=tuple(page_numbers))


@dataclass
class DetectionConfig:
    """Configuration for PDF type detection."""

    #: Strategy for which pages to scan. EarlyExit is too aggressive for PDFs
    #: with an image-only cover followed by text-heavy pages (annual reports),
    #: so the default samples instead.
    strategy: ScanStrategy = field(default_factory=lambda: ScanStrategy.sample(8))
    #: Minimum text operator count per page to consider as text-based.
    min_text_ops_per_page: int = 3
    #: Threshold ratio of text pages to total pages for classification.
    text_page_ratio_threshold: float = 0.6


@dataclass
class PdfTypeResult:
    """Result of PDF type detection."""

    pdf_type: PdfType
    page_count: int
    pages_sampled: int
    pages_with_text: int
    confidence: float
    title: str | None
    #: Whether OCR is recommended for better extraction. True when images
    #: provide essential context (e.g. template-based PDFs).
    ocr_recommended: bool
    #: 1-indexed page numbers that need OCR (image-only or insufficient text).
    #: Empty for TextBased. All pages for Scanned/ImageBased. Specific pages for
    #: Mixed.
    pages_needing_ocr: list[int]
    #: Per-page explanation for :attr:`pages_needing_ocr`: 1-indexed page ->
    #: reason codes. Only contains pages that need OCR.
    ocr_reasons_by_page: dict[int, list[str]]


# ── entry points ─────────────────────────────────────────────────────


def detect_pdf_type(
    path: str | Path, config: DetectionConfig | None = None
) -> PdfTypeResult:
    """Detect PDF type from a file path."""
    doc = Document.from_path(path)
    return detect_from_document(doc, doc.page_count, config or DetectionConfig())


def detect_pdf_type_mem(
    buffer: bytes, config: DetectionConfig | None = None
) -> PdfTypeResult:
    """Detect PDF type from a memory buffer."""
    doc = Document.from_bytes(buffer)
    return detect_from_document(doc, doc.page_count, config or DetectionConfig())


def estimate_page_count_from_bytes(buffer: bytes) -> int:
    """Heuristic page-count fallback for malformed PDFs that cannot be parsed.

    Scans raw bytes for page dictionaries (``/Type /Page``) while excluding the
    page tree node (``/Type /Pages``). Intended as a low-confidence hint for
    diagnostics; parsed page-tree counts remain authoritative.
    """
    count = 0
    pos = 0
    needle = b"/Type"

    while True:
        idx = buffer.find(needle, pos)
        if idx == -1:
            break

        value_pos = _skip_pdf_whitespace(buffer, idx + len(needle))

        if value_pos < len(buffer) and buffer[value_pos] == 0x2F:  # '/'
            name_start = value_pos + 1
            name_end = name_start + 4  # len("Page")
            if (
                name_end <= len(buffer)
                and buffer[name_start:name_end] == b"Page"
                and (
                    name_end >= len(buffer)
                    or buffer[name_end] in _PDF_NAME_DELIMITERS
                )
            ):
                count += 1

        pos = idx + len(needle)

    return count


def _skip_pdf_whitespace(buffer: bytes, pos: int) -> int:
    while pos < len(buffer) and buffer[pos] in _PDF_WHITESPACE:
        pos += 1
    return pos


# ── core detection ───────────────────────────────────────────────────


def detect_from_document(
    doc: Document, page_count: int, config: DetectionConfig
) -> PdfTypeResult:
    """Detection logic on a pre-loaded document."""
    pages = doc.get_pages()
    total_pages = len(pages)

    sample_indices, allow_early_exit = _select_pages(config.strategy, total_pages)

    pages_with_text = 0
    pages_with_images = 0
    pages_with_template_images = 0
    pages_with_vector_text = 0
    total_text_ops = 0
    # Cache Phase 1 results to avoid re-analyzing sampled pages in Phase 2
    analysis_cache: dict[int, PageAnalysis] = {}
    pages_actually_sampled = 0

    for page_num in sample_indices:
        page = pages.get(page_num)
        if page is None:
            continue
        analysis = analyze_page_content(doc, page)
        pages_actually_sampled += 1
        logger.debug("page %s: %s", page_num, analysis)

        is_image_dominated = (
            analysis.image_count > 10
            and analysis.image_count > analysis.text_operator_count * 3
        )
        if analysis.has_images or analysis.image_count > 0:
            effective_min_ops = max(config.min_text_ops_per_page, 10)
        else:
            effective_min_ops = config.min_text_ops_per_page

        if (
            analysis.text_operator_count >= effective_min_ops
            and not is_image_dominated
            and analysis.unique_text_chars >= 5
            and not analysis.has_vector_text
            and not analysis.has_only_type3_fonts
        ):
            pages_with_text += 1
        if analysis.has_images:
            pages_with_images += 1

        # Only count as a template-image page if it looks like a scan (single
        # full-page image) rather than a text page with figures. Scanned-with-OCR
        # PDFs have 1 large image per page + an OCR text overlay; text PDFs with
        # figures have multiple smaller images alongside real text.
        #
        # Exception: CID-encoded fonts with ToUnicode produce low
        # unique_alphanum_chars in raw bytes but are fully decodable. When a page
        # has decodable fonts and enough text ops, treat it as having real text
        # regardless of raw byte diversity.
        alphanum_ok = analysis.unique_alphanum_chars < 10 and not (
            analysis.has_decodable_text_fonts and analysis.text_operator_count >= 10
        )
        if analysis.has_template_image and (
            analysis.image_count <= 1
            and analysis.text_operator_count < 50
            and alphanum_ok
        ):
            pages_with_template_images += 1
        if analysis.has_vector_text:
            pages_with_vector_text += 1
        total_text_ops += analysis.text_operator_count
        analysis_cache[page_num] = analysis

        # Early exit: if this page is non-text (insufficient meaningful text but
        # has images), this PDF won't be purely TextBased.
        if (
            allow_early_exit
            and (
                analysis.text_operator_count < config.min_text_ops_per_page
                or is_image_dominated
                or analysis.unique_text_chars < 5
            )
            and (analysis.has_images or analysis.has_template_image)
        ):
            break

    pages_sampled = pages_actually_sampled
    text_ratio = pages_with_text / pages_sampled if pages_sampled > 0 else 0.0

    # Check if this is a template-based PDF (images provide essential context).
    # Template PDFs have text AND large background images on most pages.
    has_template_images = pages_with_template_images > 0
    template_ratio = (
        pages_with_template_images / pages_sampled if pages_sampled > 0 else 0.0
    )

    # OCR is recommended when template images are present (text alone is
    # insufficient), or the PDF is scanned/image-based.
    if has_template_images and pages_with_text > 0:
        ocr_recommended = True
        # Template-based PDF: has text but images provide essential context
        pdf_type = PdfType.MIXED
        confidence = 0.5 + (0.3 * (1.0 - template_ratio))
    elif text_ratio >= config.text_page_ratio_threshold:
        ocr_recommended = False
        pdf_type, confidence = PdfType.TEXT_BASED, text_ratio
    elif pages_with_text == 0 and (pages_with_images > 0 or pages_with_vector_text > 0):
        # No extractable text but has images or vector-outlined text
        ocr_recommended = True
        if total_text_ops == 0 and pages_with_vector_text == 0:
            pdf_type, confidence = PdfType.SCANNED, 0.95
        else:
            pdf_type, confidence = PdfType.IMAGE_BASED, 0.8
    elif pages_with_text > 0 and (pages_with_images > 0 or pages_with_vector_text > 0):
        ocr_recommended = True
        pdf_type, confidence = PdfType.MIXED, 0.7
    elif total_text_ops == 0:
        ocr_recommended = True
        pdf_type, confidence = PdfType.SCANNED, 0.9
    else:
        ocr_recommended = False
        pdf_type, confidence = PdfType.TEXT_BASED, max(text_ratio, 0.5)

    ocr_recommended = _apply_newspaper_heuristic(
        pdf_type, pages_sampled, analysis_cache, ocr_recommended
    )

    pages_needing_ocr = _build_ocr_page_list(
        doc, pages, pdf_type, total_pages, config, analysis_cache
    )

    # Explain each OCR-flagged page. Pages that were analyzed get a
    # signal-derived reason; pages flagged only by whole-document classification
    # (unsampled pages of a Scanned/ImageBased doc) default to `scanned`.
    ocr_reasons_by_page: dict[int, list[str]] = {}
    for page_num in pages_needing_ocr:
        analysis = analysis_cache.get(page_num)
        reasons = page_ocr_reasons(analysis) if analysis else [OCR_REASON_SCANNED]
        ocr_reasons_by_page[page_num] = list(reasons)

    return PdfTypeResult(
        pdf_type=pdf_type,
        page_count=page_count,
        pages_sampled=pages_sampled,
        pages_with_text=pages_with_text,
        confidence=confidence,
        title=get_document_title(doc),
        ocr_recommended=ocr_recommended,
        pages_needing_ocr=pages_needing_ocr,
        ocr_reasons_by_page=ocr_reasons_by_page,
    )


def _select_pages(strategy: ScanStrategy, total_pages: int) -> tuple[list[int], bool]:
    """Page numbers to scan, and whether early exit is permitted."""
    if strategy.kind == "early_exit":
        return list(range(1, total_pages + 1)), True
    if strategy.kind == "full":
        return list(range(1, total_pages + 1)), False
    if strategy.kind == "sample":
        n = min(strategy.max_pages, total_pages)
        return distribute_pages(n, total_pages), False
    # "pages"
    valid = sorted({p for p in strategy.page_numbers if 1 <= p <= total_pages})
    return valid, False


def _apply_newspaper_heuristic(
    pdf_type: PdfType,
    pages_sampled: int,
    analysis_cache: dict[int, PageAnalysis],
    ocr_recommended: bool,
) -> bool:
    """Phase 1b: newspaper-style layout detection.

    Dense multi-column newspapers (WSJ, NYT) have extractable text but produce
    poor output due to complex interleaved article layouts. Detected via
    consistently high text density combined with moderate font switches and a
    low Tf/Tj ratio.

    The Tf/Tj ratio distinguishes newspapers from styled legal/business
    documents: newspapers land at 0.02-0.06 (dense prose with occasional font
    switches), while rich-styled docs (DPAs, contracts) land at 0.25-0.35
    (per-character styling).

    Upstream calibrated the thresholds against a WSJ 50-page newspaper
    (text_ops 1500-3800, font_changes 50-194, ratio 0.02-0.06), DPAs/contracts
    (text_ops 1300-2260, font_changes 327-630, ratio 0.25-0.32), SEC filings
    (only 1-2 dense pages), and normal docs (text_ops < 700, font_changes < 55).
    """
    if pdf_type is not PdfType.TEXT_BASED or pages_sampled < 3:
        return ocr_recommended

    newspaper_pages = 0
    for analysis in analysis_cache.values():
        if analysis.text_operator_count > 0:
            ratio = analysis.font_change_count / analysis.text_operator_count
        else:
            ratio = 1.0
        if (
            analysis.text_operator_count >= 1500
            and analysis.font_change_count >= 50
            and ratio < 0.15
        ):
            newspaper_pages += 1

    if newspaper_pages / pages_sampled >= 0.5:
        logger.debug(
            "newspaper layout detected: %s/%s pages with high text_ops + "
            "font_changes -> OCR recommended",
            newspaper_pages,
            pages_sampled,
        )
        return True
    return ocr_recommended


def _build_ocr_page_list(
    doc: Document,
    pages: dict[int, PageRef],
    pdf_type: PdfType,
    total_pages: int,
    config: DetectionConfig,
    analysis_cache: dict[int, PageAnalysis],
) -> list[int]:
    """Phases 2 and 3: which pages need OCR."""
    pages_needing_ocr: list[int] = []

    if pdf_type is PdfType.TEXT_BASED:
        pages_needing_ocr = []
    elif pdf_type in (PdfType.SCANNED, PdfType.IMAGE_BASED):
        pages_needing_ocr = list(range(1, total_pages + 1))
    else:  # Mixed
        for page_num in range(1, total_pages + 1):
            analysis = analysis_cache.get(page_num)
            if analysis is None:
                page = pages.get(page_num)
                if page is None:
                    continue
                # Cache the fresh analysis so the reason-classification pass
                # below sees the real signals (vector_text, etc.) instead of
                # defaulting to "scanned".
                analysis = analyze_page_content(doc, page)
                analysis_cache[page_num] = analysis

            # Template images only need OCR when the page looks like a scan
            # (single full-page image) rather than figures alongside text.
            # CID-encoded fonts with ToUnicode produce low unique_alphanum_chars
            # in raw bytes but are fully decodable — don't treat as a scan.
            alphanum_low = analysis.unique_alphanum_chars < 10 and not (
                analysis.has_decodable_text_fonts and analysis.text_operator_count >= 10
            )
            looks_like_scan = (
                analysis.image_count <= 1
                and analysis.text_operator_count < 50
                and alphanum_low
            )
            if (
                (analysis.has_template_image and looks_like_scan)
                or analysis.has_vector_text
                or (
                    analysis.text_operator_count < config.min_text_ops_per_page
                    and analysis.has_images
                )
            ):
                pages_needing_ocr.append(page_num)

    # Phase 3: flag pages with undecodable fonts for OCR.
    # - Identity-H/V without ToUnicode: raw CID values can't map to Unicode
    # - Type3-only without ToUnicode: glyph bitmaps can't map to Unicode
    flagged = set(pages_needing_ocr)
    for page_num in sorted(analysis_cache):
        analysis = analysis_cache[page_num]
        if (
            analysis.has_identity_h_no_tounicode or analysis.has_only_type3_fonts
        ) and page_num not in flagged:
            pages_needing_ocr.append(page_num)
            flagged.add(page_num)

    # Check uncached pages too (when not all pages were sampled), using
    # analyze_page_content to get the usage-based font checks.
    if len(pages_needing_ocr) < total_pages:
        for page_num in range(1, total_pages + 1):
            if page_num in analysis_cache or page_num in flagged:
                continue
            page = pages.get(page_num)
            if page is None:
                continue
            analysis = analyze_page_content(doc, page)
            if analysis.has_identity_h_no_tounicode or analysis.has_only_type3_fonts:
                pages_needing_ocr.append(page_num)
                flagged.add(page_num)
                # Cache so the reason pass reports suspected_garbled_text rather
                # than defaulting to "scanned".
                analysis_cache[page_num] = analysis

    return sorted(set(pages_needing_ocr))


def distribute_pages(n: int, total: int) -> list[int]:
    """Distribute ``n`` page indices evenly across ``total`` pages (1-indexed).

    Always includes the first and last page, with remaining pages spaced evenly
    in between.
    """
    if n == 0:
        return []
    if n >= total:
        return list(range(1, total + 1))

    indices = [1]
    if n > 1:
        indices.append(total)

    remaining = max(n - 2, 0)
    if remaining > 0 and total > 2:
        step = (total - 2) // (remaining + 1)
        for i in range(1, remaining + 1):
            idx = 1 + (step * i)
            if 1 < idx < total and idx not in indices:
                indices.append(idx)

    return sorted(set(indices))


@dataclass
class PageAnalysis:
    """Page content analysis result."""

    text_operator_count: int = 0
    has_images: bool = False
    #: Whether the page has a large background/template image (>50% coverage).
    has_template_image: bool = False
    #: Total image area in pixels.
    total_image_area: int = 0
    #: Number of Do (XObject invocation) operators in content streams.
    image_count: int = 0
    #: Number of unique non-whitespace text bytes found in string operands.
    unique_text_chars: int = 0
    #: Number of unique ASCII alphanumeric bytes (letters + digits) in operands.
    unique_alphanum_chars: int = 0
    #: Number of path construction/painting ops (m, l, c, h, f, re, etc.).
    path_op_count: int = 0
    #: Whether the page has vector-outlined text (many path ops, few text ops).
    has_vector_text: bool = False
    #: Whether the page has Type0 fonts with Identity-H/V encoding but no
    #: ToUnicode CMap. These produce garbage text because CID values can't be
    #: mapped to Unicode.
    has_identity_h_no_tounicode: bool = False
    #: Whether the page uses only Type3 fonts (no normal text fonts). Type3
    #: fonts render each glyph as a custom drawing/bitmap — without a ToUnicode
    #: CMap, the character codes can't be mapped to Unicode.
    has_only_type3_fonts: bool = False
    #: Number of Tf (set font) operators — a high count indicates many font
    #: switches.
    font_change_count: int = 0
    #: Whether the page has fonts that can produce decodable text (ToUnicode,
    #: standard encoding, Type1/TrueType with known encoding). CID-encoded text
    #: with ToUnicode produces low unique_alphanum_chars in raw bytes but is
    #: fully decodable — this flag prevents misclassifying it as a scan.
    has_decodable_text_fonts: bool = False


def page_ocr_reasons(a: PageAnalysis) -> list[str]:
    """Explain *why* a page needs OCR, from its content analysis.

    Priority: undecodable fonts (``suspected_garbled_text``) and
    vector-outlined text (``vector_text``) come first because they persist even
    when a text layer is present; otherwise a page with no extractable text is
    ``scanned`` when an image backs it, or ``no_text`` when nothing does.
    """
    reasons: list[str] = []
    if a.has_identity_h_no_tounicode or a.has_only_type3_fonts:
        reasons.append(OCR_REASON_SUSPECTED_GARBLED_TEXT)
    if a.has_vector_text:
        reasons.append(OCR_REASON_VECTOR_TEXT)
    if not reasons:
        has_extractable_text = a.text_operator_count > 0 and a.unique_text_chars > 0
        if not has_extractable_text and not a.has_images and not a.has_template_image:
            reasons.append(OCR_REASON_NO_TEXT)
        else:
            # Image-backed with no usable text, or too little text to trust.
            reasons.append(OCR_REASON_SCANNED)
    return reasons


@dataclass
class FontInfo:
    """Font properties needed for the decodability / Identity-H checks.

    Stored without holding a reference to the document, matching upstream.
    """

    subtype: str | None
    encoding: str | None
    has_tounicode: bool
    #: The raw font dictionary, needed for fallback checks (DescendantFonts ->
    #: W array, embedded cmap).
    dictionary: Any


def collect_fonts_from_resource_dict(
    doc: Document, resources: Any, font_map: dict[ObjectId, FontInfo]
) -> None:
    """Collect font entries from a Resources/Font dictionary into the font map.

    Each entry maps font ObjectId -> :class:`FontInfo`. Using the ObjectId as
    the key avoids name collisions: different resource dictionaries can legally
    define ``/F1`` pointing to different font objects, and the ObjectId
    identifies the underlying font regardless of the name used to reach it.

    Inline font dictionaries (rare — fonts are almost always indirect refs) are
    skipped because they have no ObjectId.
    """
    font_dict = doc.get_dictionary(resources.get("/Font")) if resources else None
    if font_dict is None:
        return

    for name in list(font_dict.keys()):
        value = font_dict.raw_get(name)
        font_obj_id = Document.object_id(value)
        # Only indirect references have a stable ObjectId.
        if font_obj_id is None:
            continue
        if font_obj_id in font_map:
            continue
        resolved = doc.get_dictionary(value)
        if resolved is None:
            continue
        font_map[font_obj_id] = FontInfo(
            subtype=Document.name_of(doc.resolve(resolved.get("/Subtype"))),
            encoding=Document.name_of(doc.resolve(resolved.get("/Encoding"))),
            has_tounicode="/ToUnicode" in resolved,
            dictionary=resolved,
        )


def resolve_font_names_to_ids(
    doc: Document,
    resources: Any,
    font_names: set[str],
    used_font_ids: set[ObjectId],
) -> None:
    """Resolve font names collected from a content stream to ObjectIds.

    This is how font-name resolution is scoped correctly: each content stream
    (page-level or Form XObject) resolves ``/FontName`` against its own
    Resources/Font dictionary, yielding the correct underlying font object.
    """
    font_dict = doc.get_dictionary(resources.get("/Font")) if resources else None
    if font_dict is None:
        return
    for name in font_names:
        key = "/" + name
        if key not in font_dict:
            continue
        object_id = Document.object_id(font_dict.raw_get(key))
        if object_id is not None:
            used_font_ids.add(object_id)


def lookup_font_id(doc: Document, resources: Any, font_name: str) -> ObjectId | None:
    """Look up a single font name in a resource dictionary."""
    font_dict = doc.get_dictionary(resources.get("/Font")) if resources else None
    if font_dict is None:
        return None
    key = "/" + font_name
    if key not in font_dict:
        return None
    return Document.object_id(font_dict.raw_get(key))


def resolve_with_shadowing(
    doc: Document,
    own_resources: Any,
    ancestor_resource_ids: Sequence[ObjectId],
    names: set[str],
    used_font_ids: set[ObjectId],
) -> None:
    """Resolve page-level font names with PDF resource inheritance shadowing.

    PDF 32000-1 7.7.3.4: a page inherits ``/Resources`` from its parent
    ``/Pages`` nodes, but a definition in a more-specific scope shadows the same
    name from an ancestor. Resource dictionaries arrive most-specific-first, so
    the first dictionary that defines a given font name wins.
    """
    for name in names:
        # Check the page's own inline /Resources first (most specific scope)
        if own_resources is not None:
            object_id = lookup_font_id(doc, own_resources, name)
            if object_id is not None:
                used_font_ids.add(object_id)
                continue
        # Walk inherited resource dicts (most-specific to root); first hit wins
        found = False
        for ancestor_id in ancestor_resource_ids:
            resources = doc.dictionary_for_id(ancestor_id)
            if resources is None:
                continue
            object_id = lookup_font_id(doc, resources, name)
            if object_id is not None:
                used_font_ids.add(object_id)
                found = True
                break
        if found:
            continue


def analyze_page_content(doc: Document, page: PageRef) -> PageAnalysis:
    """Analyze a page's content stream for text operators and images."""
    text_ops = 0
    has_images = False
    image_count = 0
    path_ops = 0
    font_changes = 0
    all_unique_chars: set[int] = set()
    # Collect font ObjectIds (not names) to avoid cross-scope name collisions.
    # Each content stream resolves its Tf font names against its own resource
    # dictionary, producing the correct underlying font ObjectId.
    used_font_ids: set[ObjectId] = set()

    # Font map keyed by ObjectId, covering page-level Resources plus Form
    # XObject Resources.
    font_map: dict[ObjectId, FontInfo] = {}

    own_resources, ancestor_resource_ids = doc.get_page_resources(page)

    for content in doc.get_page_contents(page):
        page_font_names: set[str] = set()
        ops, imgs, paths, fonts = scan_content_for_text_operators(
            content, all_unique_chars, page_font_names
        )
        text_ops += ops
        image_count += imgs
        path_ops += paths
        font_changes += fonts
        has_images = has_images or imgs > 0

        # Resolve font names against the page's resource dictionaries,
        # respecting PDF resource inheritance shadowing.
        resolve_with_shadowing(
            doc, own_resources, ancestor_resource_ids, page_font_names, used_font_ids
        )

    # Scan XObject Form contents for text operators, collect their fonts, and
    # resolve font names per-XObject scope.
    visited: set[ObjectId] = set()
    for resources in _page_resource_dicts(doc, own_resources, ancestor_resource_ids):
        collect_fonts_from_resource_dict(doc, resources, font_map)
        ops, imgs, paths, fonts = scan_xobjects_in_resources(
            doc, resources, visited, all_unique_chars, used_font_ids, font_map
        )
        text_ops += ops
        image_count += imgs
        path_ops += paths
        font_changes += fonts
        has_images = has_images or imgs > 0

    found_images, total_image_area, has_template_image = analyze_page_images(doc, page)
    if found_images:
        has_images = True

    unique_alphanum_chars = sum(1 for b in all_unique_chars if b in _ASCII_ALNUM_BYTES)

    # Vector-outlined text: many path ops with minimal text ops. Each outlined
    # glyph needs ~10-30 path commands, so a page of outlined text produces
    # thousands of path ops.
    #
    # Few unique alphanumeric bytes is also required: real outlined-text pages
    # have very few because each glyph is a path, not a Tj/TJ text op. Pages with
    # real selectable text plus decorative paths (column borders, dividers) have
    # many — those are NOT vector-outlined text.
    has_vector_text = (
        path_ops >= 1000 and path_ops > text_ops * 200 and unique_alphanum_chars < 30
    )

    # Only fonts actually used by Tf operators are considered, and fonts from
    # Form XObject Resources are included.
    has_identity_h_no_tounicode = text_ops > 0 and used_fonts_have_identity_h_no_tounicode(
        used_font_ids, font_map, doc
    )
    has_only_type3_fonts = text_ops > 0 and used_fonts_are_only_type3(
        used_font_ids, font_map
    )
    has_decodable_text_fonts = text_ops > 0 and used_fonts_have_decodable_text(
        used_font_ids, font_map, doc
    )

    return PageAnalysis(
        text_operator_count=text_ops,
        has_images=has_images,
        has_template_image=has_template_image,
        total_image_area=total_image_area,
        image_count=image_count,
        unique_text_chars=len(all_unique_chars),
        unique_alphanum_chars=unique_alphanum_chars,
        path_op_count=path_ops,
        has_vector_text=has_vector_text,
        has_identity_h_no_tounicode=has_identity_h_no_tounicode,
        has_only_type3_fonts=has_only_type3_fonts,
        font_change_count=font_changes,
        has_decodable_text_fonts=has_decodable_text_fonts,
    )


def _page_resource_dicts(
    doc: Document, own_resources: Any, ancestor_resource_ids: Sequence[ObjectId]
) -> list[Any]:
    dicts: list[Any] = []
    if own_resources is not None:
        dicts.append(own_resources)
    for object_id in ancestor_resource_ids:
        resources = doc.dictionary_for_id(object_id)
        if resources is not None:
            dicts.append(resources)
    return dicts


def identity_h_font_has_fallback(doc: Document, font_dict: Any) -> bool:
    """Whether an Identity-H font without ToUnicode can still be decoded.

    Checks the two fallback paths the extraction pipeline provides.
    """
    desc_fonts = doc.resolve(font_dict.get("/DescendantFonts"))
    if not isinstance(desc_fonts, (list, tuple)) or not desc_fonts:
        return False
    cid_font_dict = doc.get_dictionary(desc_fonts[0])
    if cid_font_dict is None:
        return False

    # Fallback 1: W array CIDs look like Unicode codepoints, so passthrough
    # works. Many PDF generators (Chromium, wkhtmltopdf) use Identity-H where
    # CID == Unicode.
    if cid_values_look_like_unicode(cid_font_dict, doc):
        return True

    # Fallback 2: the embedded TrueType/OpenType font has a usable cmap table.
    font_descriptor = doc.get_dictionary(cid_font_dict.get("/FontDescriptor"))
    if font_descriptor is not None:
        for key in ("/FontFile2", "/FontFile3"):
            if key not in font_descriptor:
                continue
            stream = doc.get_stream(font_descriptor.raw_get(key))
            if stream is None:
                continue
            if font_has_unicode_cmap(stream_bytes(stream)):
                return True

    return False


def used_fonts_have_identity_h_no_tounicode(
    used_font_ids: set[ObjectId],
    font_map: dict[ObjectId, FontInfo],
    doc: Document,
) -> bool:
    """Do the used fonts include an undecodable Identity-H/V font with no other
    decodable font to compensate?

    Only fonts actually referenced by Tf operators are considered, including
    those from Form XObject Resources.
    """
    has_undecodable_identity_h = False
    has_other_decodable_font = False

    for object_id in used_font_ids:
        info = font_map.get(object_id)
        if info is None:
            continue
        if info.subtype == "Type0":
            if info.encoding not in ("Identity-H", "Identity-V"):
                # Type0 with a non-Identity encoding (e.g. a named CMap) is
                # decodable.
                has_other_decodable_font = True
                continue
            if info.has_tounicode:
                has_other_decodable_font = True
                continue
            if identity_h_font_has_fallback(doc, info.dictionary):
                has_other_decodable_font = True
                continue
            has_undecodable_identity_h = True
        elif info.subtype == "Type3":
            # Handled separately by used_fonts_are_only_type3
            continue
        else:
            # Type1, TrueType, MMType1, etc. — generally decodable
            has_other_decodable_font = True

    return has_undecodable_identity_h and not has_other_decodable_font


def used_fonts_are_only_type3(
    used_font_ids: set[ObjectId], font_map: dict[ObjectId, FontInfo]
) -> bool:
    """Are ALL used fonts Type3 without ToUnicode?"""
    if not used_font_ids:
        return False
    has_type3 = False
    for object_id in used_font_ids:
        info = font_map.get(object_id)
        if info is None:
            continue
        if info.subtype == "Type3":
            # Type3 with a ToUnicode CMap can still produce usable text
            if info.has_tounicode:
                return False
            has_type3 = True
        else:
            # Has a non-Type3 font — the page has real text fonts
            return False
    return has_type3


def used_fonts_have_decodable_text(
    used_font_ids: set[ObjectId],
    font_map: dict[ObjectId, FontInfo],
    doc: Document,
) -> bool:
    """Do the used fonts include at least one that can produce decodable text?"""
    for object_id in used_font_ids:
        info = font_map.get(object_id)
        if info is None:
            continue
        if info.has_tounicode:
            return True
        if info.subtype in ("Type1", "TrueType", "MMType1"):
            # Decodable via the Adobe Glyph List or encoding vectors.
            return True
        if info.subtype == "Type0":
            if identity_h_font_has_fallback(doc, info.dictionary):
                return True
    return False


def scan_xobjects_in_resources(
    doc: Document,
    resources: Any,
    visited: set[ObjectId],
    unique_chars: set[int],
    used_font_ids: set[ObjectId],
    font_map: dict[ObjectId, FontInfo],
) -> tuple[int, int, int, int]:
    """Recurse through Form XObjects, accumulating the same content signals."""
    text_ops = 0
    image_count = 0
    path_ops = 0
    font_changes = 0

    xobj_dict = doc.get_dictionary(resources.get("/XObject")) if resources else None
    if xobj_dict is None:
        return text_ops, image_count, path_ops, font_changes

    for name in list(xobj_dict.keys()):
        raw = xobj_dict.raw_get(name)
        obj_id = Document.object_id(raw)
        if obj_id is None:
            continue
        if obj_id in visited:
            continue
        visited.add(obj_id)

        stream = doc.get_stream(raw)
        if stream is None:
            continue
        subtype = Document.name_of(doc.resolve(stream.get("/Subtype")))

        if subtype == "Form":
            content = stream_bytes(stream)
            xobj_font_names: set[str] = set()
            ops, imgs, paths, fonts = scan_content_for_text_operators(
                content, unique_chars, xobj_font_names
            )
            text_ops += ops
            image_count += imgs
            path_ops += paths
            font_changes += fonts

            xobj_res = doc.get_dictionary(stream.get("/Resources"))
            if xobj_res is not None:
                # Resolve font names against the XObject's own resource dict
                resolve_font_names_to_ids(doc, xobj_res, xobj_font_names, used_font_ids)
                # Collect font definitions from this scope
                collect_fonts_from_resource_dict(doc, xobj_res, font_map)
                # Recurse into nested XObjects
                ops2, imgs2, paths2, fonts2 = scan_xobjects_in_resources(
                    doc, xobj_res, visited, unique_chars, used_font_ids, font_map
                )
                text_ops += ops2
                image_count += imgs2
                path_ops += paths2
                font_changes += fonts2
        elif subtype == "Image":
            image_count += 1

    return text_ops, image_count, path_ops, font_changes


def scan_content_for_text_operators(
    content: bytes,
    unique_chars: set[int],
    used_font_names: set[str],
) -> tuple[int, int, int, int]:
    """Fast scan of content stream bytes for text operators.

    Looks for ``Tj`` (show text), ``TJ`` (show text with positioning), and
    ``Tf`` (set font), plus path construction/painting operators.

    Returns ``(text_op_count, image_count, path_op_count, font_change_count)``.
    Unique non-whitespace text bytes are collected into ``unique_chars``.

    ``image_count`` is always 0 here: ``Do`` invokes any XObject — including
    Form XObjects that contain text — so image detection is left to
    :func:`scan_xobjects_in_resources` (which checks Subtype) and
    :func:`analyze_page_images` (which measures pixel area).
    """
    text_ops = 0
    image_count = 0
    path_ops = 0
    font_changes = 0
    n = len(content)

    def is_word_start(pos: int) -> bool:
        return pos == 0 or content[pos - 1] in _ASCII_WHITESPACE

    def is_word_end(pos: int) -> bool:
        return pos + 1 >= n or content[pos + 1] in _ASCII_WHITESPACE

    # Each Tj/TJ/Tf lookback stops at the previous text/font operator, so a
    # malformed `] TJ` (no `[`) cannot rescan the entire prefix — that was
    # quadratic in the number of operators. Tj/TJ are only counted when the
    # preceding token closes a string or array (')', '>', ']'), so `Tj` inside
    # `(Hello Tj World)` cannot pin the floor.
    operand_floor = 0
    i = 0
    while i < n:
        b = content[i]

        # Look for 'T' followed by 'j', 'J', or 'f'
        if b == 0x54 and i + 1 < n:  # 'T'
            nxt = content[i + 1]
            if nxt in (0x6A, 0x4A):  # 'j', 'J'
                # Verify it's an operator (followed by whitespace or newline)
                if (
                    i + 2 >= n or content[i + 2] in _ASCII_WHITESPACE
                ) and preceding_operand_closer(content, i, operand_floor):
                    text_ops += 1
                    collect_text_chars_before(content, i, unique_chars, operand_floor)
                    operand_floor = i
            elif nxt == 0x66:  # 'f'
                # Some PDFs concatenate Tf with the next operator without
                # whitespace (e.g. "25 Tf[<01>..." or "25 Tf(<text>..."), so
                # '[', '(', '<' and '/' are accepted as valid followers too.
                if i + 2 >= n or content[i + 2] in _ASCII_WHITESPACE or content[
                    i + 2
                ] in (0x5B, 0x28, 0x3C, 0x2F):
                    name = extract_font_name_before_tf(content, i, operand_floor)
                    if name is not None:
                        used_font_names.add(name)
                        font_changes += 1
                        operand_floor = i

        # Count path construction/painting operators.
        # Single-byte: m (moveto), l (lineto), c (curveto), h (closepath),
        #              f (fill), S (stroke), s (close+stroke), B (fill+stroke),
        #              F (fill, variant). These are the high-volume operators in
        #              vector-outlined text.
        if b in (0x6D, 0x6C, 0x63, 0x68, 0x66, 0x53, 0x73, 0x42, 0x46):
            if is_word_start(i) and is_word_end(i):
                path_ops += 1
        # Two-byte: re (rect)
        if (
            b == 0x72
            and i + 1 < n
            and content[i + 1] == 0x65
            and is_word_start(i)
            and (i + 2 >= n or content[i + 2] in _ASCII_WHITESPACE)
        ):
            path_ops += 1
        # Two-byte: f* (fill even-odd)
        if (
            b == 0x66
            and i + 1 < n
            and content[i + 1] == 0x2A
            and is_word_start(i)
            and (i + 2 >= n or content[i + 2] in _ASCII_WHITESPACE)
        ):
            path_ops += 1

        i += 1

    return text_ops, image_count, path_ops, font_changes


def preceding_operand_closer(content: bytes, op_pos: int, floor: int) -> bool:
    """True when the token before ``op_pos`` is a string/array closer.

    Whitespace is skipped and the scan does not cross ``floor``. Used so ``Tj``
    inside ``(Hello Tj World)`` is not treated as an operator.
    """
    j = op_pos
    while j > floor:
        j -= 1
        if content[j] not in _ASCII_WHITESPACE:
            return content[j] in (0x29, 0x3E, 0x5D)  # ')', '>', ']'
    return False


def extract_font_name_before_tf(
    content: bytes, tf_pos: int, floor: int
) -> str | None:
    """Extract the font name operand preceding a ``Tf`` operator.

    The operator syntax is ``/FontName size Tf``. The scan walks backward from
    the 'T' of 'Tf' past the size number and whitespace to find the ``/Name``
    token, and returns the name without its leading slash. ``floor`` is the
    start of the previous text/font operator; the lookback must not cross it.
    """
    # Scan backward past whitespace before "Tf"
    j = tf_pos
    while j > floor and content[j - 1] in _ASCII_WHITESPACE:
        j -= 1
    # Scan backward past the size number (digits, '.', '-')
    while j > floor and (
        0x30 <= content[j - 1] <= 0x39 or content[j - 1] in (0x2E, 0x2D)
    ):
        j -= 1
    # Scan backward past whitespace between font name and size
    while j > floor and content[j - 1] in _ASCII_WHITESPACE:
        j -= 1

    # j now points just after the font name. Scan backward to find '/'.
    name_end = j
    while j > floor and content[j - 1] != 0x2F:  # '/'
        # Font names consist of regular characters (not whitespace, not delimiters)
        if content[j - 1] in _ASCII_WHITESPACE or content[j - 1] in (0x28, 0x29):
            return None
        j -= 1
    if j <= floor or content[j - 1] != 0x2F:
        return None
    if j < name_end:
        return content[j:name_end].decode("latin-1")
    return None


def collect_text_chars_before(
    content: bytes, op_pos: int, unique_chars: set[int], floor: int
) -> None:
    """Collect unique non-whitespace bytes from the string operand before a
    Tj/TJ operator.

    Handles literal strings ``(...)``, hex strings ``<...>``, and TJ arrays.
    ``floor`` is the start of the previous text/font operator; the lookback must
    not cross it, or a missing ``[`` before ``TJ`` rescans the whole prefix.
    """
    # Walk backward past whitespace to find the closing delimiter
    j = op_pos
    while j > floor:
        j -= 1
        if content[j] not in _ASCII_WHITESPACE:
            break
    # All whitespace, or we landed on the previous operator token.
    if j == floor:
        return

    closing = content[j]

    if closing == 0x29:  # ')'
        # Literal string: scan backward for the matching '('
        depth = 1
        k = j
        while k > floor and depth > 0:
            k -= 1
            if content[k] == 0x29 and (k == 0 or content[k - 1] != 0x5C):
                depth += 1
            elif content[k] == 0x28 and (k == 0 or content[k - 1] != 0x5C):
                depth -= 1
        # k now points at '('; collect bytes between (k+1..j)
        if depth == 0 and k + 1 < j:
            for ch in content[k + 1 : j]:
                if ch not in _ASCII_WHITESPACE:
                    unique_chars.add(ch)

    elif closing == 0x3E:  # '>'
        # Hex string: scan backward for '<'
        k = j
        while k > floor:
            k -= 1
            if content[k] == 0x3C:
                break
        if content[k] == 0x3C and k + 1 < j:
            _collect_hex_bytes(content[k + 1 : j], unique_chars)

    elif closing == 0x5D:  # ']'
        # TJ array: scan backward for '[' and collect from all strings inside
        k = j
        while k > floor:
            k -= 1
            if content[k] == 0x5B:
                break
        if content[k] == 0x5B:
            _collect_from_tj_array(content, k + 1, j, unique_chars)


def _collect_from_tj_array(
    content: bytes, start: int, end: int, unique_chars: set[int]
) -> None:
    m = start
    while m < end:
        if content[m] == 0x28:  # '('
            str_start = m + 1
            depth = 1
            m += 1
            while m < end and depth > 0:
                if content[m] == 0x29 and content[m - 1] != 0x5C:
                    depth -= 1
                elif content[m] == 0x28 and content[m - 1] != 0x5C:
                    depth += 1
                if depth > 0:
                    m += 1
            for ch in content[str_start:m]:
                if ch not in _ASCII_WHITESPACE:
                    unique_chars.add(ch)
        elif content[m] == 0x3C:  # '<'
            hex_start = m + 1
            m += 1
            while m < end and content[m] != 0x3E:
                m += 1
            _collect_hex_bytes(content[hex_start:m], unique_chars)
        m += 1


def _collect_hex_bytes(hex_slice: bytes, unique_chars: set[int]) -> None:
    """Decode hex pairs and collect unique meaningful bytes."""
    hex_clean = bytes(b for b in hex_slice if b not in _ASCII_WHITESPACE)
    for index in range(0, len(hex_clean) - 1, 2):
        high = _hex_val(hex_clean[index])
        low = _hex_val(hex_clean[index + 1])
        if high is None or low is None:
            continue
        byte = (high << 4) | low
        if byte not in (0, 0x20, 0x09, 0x0A):
            unique_chars.add(byte)


def _hex_val(b: int) -> int | None:
    """Convert a hex ASCII byte to its numeric value (0-15)."""
    if 0x30 <= b <= 0x39:
        return b - 0x30
    if 0x61 <= b <= 0x66:
        return b - 0x61 + 10
    if 0x41 <= b <= 0x46:
        return b - 0x41 + 10
    return None


#: An image covering roughly half a page at 150+ DPI. 612*792/2 * (150/72)^2 is
#: about 1M pixels; upstream is deliberately conservative here.
TEMPLATE_IMAGE_THRESHOLD = 500_000


def analyze_page_images(doc: Document, page: PageRef) -> tuple[bool, int, bool]:
    """Analyze page images, returning ``(has_images, total_area, has_template_image)``.

    ``has_template_image`` means a single large (>50% page coverage) background
    image — the signal classification uses to route a page to OCR regardless of
    any incidental native text drawn over it.
    """
    has_images = False
    total_area = 0
    has_template_image = False
    visited: set[ObjectId] = set()

    resources = doc.get_dictionary(page.dictionary.get("/Resources"))

    if resources is not None:
        found, area, template = collect_images_from_resources(
            doc, resources, TEMPLATE_IMAGE_THRESHOLD, visited
        )
        has_images = has_images or found
        total_area += area
        has_template_image = has_template_image or template

        # Also check Pattern resources: tiling patterns can contain XObject
        # images (e.g. screenshots pasted into PDFs via Chrome "Save as PDF").
        pattern_dict = doc.get_dictionary(resources.get("/Pattern"))
        if pattern_dict is not None:
            for name in list(pattern_dict.keys()):
                raw = pattern_dict.raw_get(name)
                pat_ref = Document.object_id(raw)
                if pat_ref is None or pat_ref in visited:
                    continue
                visited.add(pat_ref)
                stream = doc.get_stream(raw)
                if stream is None:
                    continue
                pat_res = doc.get_dictionary(stream.get("/Resources"))
                if pat_res is None:
                    continue
                found, area, template = collect_images_from_resources(
                    doc, pat_res, TEMPLATE_IMAGE_THRESHOLD, visited
                )
                has_images = has_images or found
                total_area += area
                has_template_image = has_template_image or template

    # Tiled scans: many small image tiles (e.g. JBIG2 strips) that together
    # cover the full page. No individual tile trips the template threshold, but
    # the aggregate area clearly indicates a scanned/image-backed page.
    if not has_template_image and total_area >= TEMPLATE_IMAGE_THRESHOLD * 4:
        has_template_image = True

    return has_images, total_area, has_template_image


def collect_images_from_resources(
    doc: Document,
    resources: Any,
    threshold: int,
    visited: set[ObjectId],
) -> tuple[bool, int, bool]:
    """Recursively collect image dimensions from XObject resources.

    Includes images nested inside Form XObjects.
    """
    has_images = False
    total_area = 0
    has_template_image = False

    xobject_dict = doc.get_dictionary(resources.get("/XObject")) if resources else None
    if xobject_dict is None:
        return has_images, total_area, has_template_image

    for name in list(xobject_dict.keys()):
        raw = xobject_dict.raw_get(name)
        xobj_ref = Document.object_id(raw)
        if xobj_ref is None or xobj_ref in visited:
            continue
        visited.add(xobj_ref)

        stream = doc.get_stream(raw)
        if stream is None:
            continue
        subtype = Document.name_of(doc.resolve(stream.get("/Subtype")))

        if subtype == "Image":
            has_images = True
            width = _as_int(doc.resolve(stream.get("/Width")))
            height = _as_int(doc.resolve(stream.get("/Height")))
            area = width * height
            total_area += area
            if area >= threshold:
                has_template_image = True
        elif subtype == "Form":
            form_res = doc.get_dictionary(stream.get("/Resources"))
            if form_res is not None:
                found, area, template = collect_images_from_resources(
                    doc, form_res, threshold, visited
                )
                has_images = has_images or found
                total_area += area
                has_template_image = has_template_image or template

    return has_images, total_area, has_template_image


def page_ocr_signals(doc: Document, page: PageRef) -> tuple[bool, bool]:
    """``(needs_ocr_for_template_image, has_vector_text)`` from one analysis pass.

    ``analyze_page_content`` decompresses and scans every content stream plus
    image coverage, so callers that need both signals must not invoke it twice
    per page.

    ``needs_ocr_for_template_image`` is true when a page's template image should
    be treated as a scan needing OCR — a single full-page background image with
    little or no real text — rather than a text page carrying a watermark,
    letterhead, or figure.

    The text-volume gate deliberately uses ``min_text_ops_per_page`` (3) rather
    than the higher ``effective_min_ops`` floor that whole-document
    ImageBased/Scanned classification applies: that floor is a cross-page
    aggregate decision this per-page function cannot replicate, and the lower
    threshold is the one a single page's own signals can agree with.
    """
    analysis = analyze_page_content(doc, page)

    if not analysis.has_template_image:
        needs_ocr_for_template_image = False
    else:
        alphanum_low = analysis.unique_alphanum_chars < 10 and not (
            analysis.has_decodable_text_fonts and analysis.text_operator_count >= 10
        )
        looks_like_scan = (
            analysis.image_count <= 1
            and analysis.text_operator_count < 50
            and alphanum_low
        )
        insufficient_text = (
            analysis.text_operator_count < DetectionConfig().min_text_ops_per_page
        )
        needs_ocr_for_template_image = looks_like_scan or insufficient_text

    return needs_ocr_for_template_image, analysis.has_vector_text


def get_document_title(doc: Document) -> str | None:
    """Get the document title from the Info dictionary."""
    try:
        info = doc.get_dictionary(doc.trailer.get("/Info"))
    except Exception:
        return None
    if info is None:
        return None
    title = doc.resolve(info.get("/Title"))
    if title is None:
        return None

    raw = getattr(title, "get_original_bytes", None)
    if callable(raw):
        try:
            data = raw()
        except Exception:
            data = None
        if isinstance(data, bytes):
            # Handle UTF-16BE encoding (BOM: 0xFE 0xFF)
            if len(data) >= 2 and data[0] == 0xFE and data[1] == 0xFF:
                body = data[2:]
                body = body[: len(body) - (len(body) % 2)]
                return body.decode("utf-16-be", errors="replace")
            return data.decode("utf-8", errors="replace")

    return str(title) if isinstance(title, str) else None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
