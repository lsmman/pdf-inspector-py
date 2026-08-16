"""Shared types used across the extraction and markdown pipelines.

Centralises :class:`TextItem`, :class:`TextLine`, :class:`PdfRect`, font-width /
encoding type aliases, and the :class:`ItemType` enum so that every module can
import them from one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .text_utils import should_join_items


def _ratio(numerator: float, denominator: float) -> float:
    """IEEE-style division: inf or NaN on a zero denominator, never an exception.

    Upstream computes these ratios in ``f32``, where dividing by zero yields
    inf (or NaN for ``0.0 / 0.0``). Both compare False against every threshold,
    so reproducing them keeps the comparisons faithful.
    """
    if denominator == 0.0:
        if numerator == 0.0:
            return math.nan
        return math.inf if numerator > 0.0 else -math.inf
    return numerator / denominator

# ── Font types (package-internal) ────────────────────────────────────

#: Font encoding map: maps byte codes to Unicode characters.
FontEncodingMap = dict[int, str]

#: All font encodings for a page, keyed by font resource name.
PageFontEncodings = dict[str, FontEncodingMap]


@dataclass
class FontWidthInfo:
    """Font width information extracted from PDF font dictionaries."""

    #: Glyph widths: maps character code to width in font units.
    widths: dict[int, int] = field(default_factory=dict)
    #: Default width for glyphs not in the widths table.
    default_width: int = 0
    #: Width of the space character (code 32) if known.
    space_width: int = 0
    #: Whether this is a CID font (2-byte character codes).
    is_cid: bool = False
    #: Scale factor to convert font units to text space units.
    #: For Type1/TrueType: 0.001 (widths in 1000ths of em).
    #: For Type3: FontMatrix[0] (e.g. 0.00048828125 for a 2048-unit grid).
    units_scale: float = 0.001
    #: Writing mode: 0 = horizontal (default), 1 = vertical.
    wmode: int = 0


#: All font width info for a page, keyed by font resource name.
PageFontWidths = dict[str, FontWidthInfo]


# ── Public types ─────────────────────────────────────────────────────


class ItemType(Enum):
    """Type of extracted item.

    ``LINK`` carries its URL in :attr:`TextItem.link_url`; upstream models this
    as a Rust enum variant with a payload, which Python's ``Enum`` cannot do.
    """

    TEXT = "text"
    IMAGE = "image"
    LINK = "link"
    FORM_FIELD = "form_field"


@dataclass
class LayoutComplexity:
    """Layout complexity analysis result.

    Callers can use this to decide whether the extracted markdown is reliable or
    whether the PDF should be routed to an OCR pipeline instead.
    """

    #: True if any page has tables or multi-column text.
    is_complex: bool = False
    #: 1-indexed pages where table borders were detected (rect count > 6).
    pages_with_tables: list[int] = field(default_factory=list)
    #: 1-indexed pages where 2+ text columns were detected.
    pages_with_columns: list[int] = field(default_factory=list)


@dataclass
class PdfLine:
    """A line segment from PDF path operators (``m``/``l``/``S``)."""

    x1: float
    y1: float
    x2: float
    y2: float
    page: int


@dataclass
class PdfRect:
    """A rectangle from a PDF ``re`` operator (cell boundary, border, etc.)."""

    x: float
    y: float
    width: float
    height: float
    page: int


@dataclass
class TextItem:
    """A text item with position information."""

    #: The text content.
    text: str
    #: X position on page.
    x: float
    #: Y position on page (PDF coordinates, origin at bottom-left).
    y: float
    #: Width of text.
    width: float
    #: Height (approximated from font size).
    height: float
    #: Font name.
    font: str
    #: Font size.
    font_size: float
    #: Page number (1-indexed).
    page: int
    #: Whether the font is bold.
    is_bold: bool = False
    #: Whether the font is italic.
    is_italic: bool = False
    #: Whether the text is underlined (drawn rule/thin rect under the baseline —
    #: PDFs have no underline font flag, so this is detected geometrically after
    #: extraction).
    is_underline: bool = False
    #: Whether the text is struck out (drawn rule/thin rect crossing the glyphs
    #: at mid x-height). Same geometric detection as underline, different
    #: vertical window.
    is_strikeout: bool = False
    #: Type of item (text, image, link).
    item_type: ItemType = ItemType.TEXT
    #: Target URL when :attr:`item_type` is :attr:`ItemType.LINK`.
    link_url: str | None = None
    #: Marked Content ID from the content stream's BDC/BMC operator. Used to
    #: link this item to the PDF structure tree for tagged PDFs.
    mcid: int | None = None


#: Result tuple returned by page-level text extraction: text items, rectangles,
#: and line segments.
PageExtraction = tuple[list[TextItem], list[PdfRect], list[PdfLine]]


@dataclass
class TextLine:
    """A line of text (grouped text items)."""

    items: list[TextItem]
    y: float
    page: int
    #: Adaptive join threshold from page-level letter-spacing detection.
    #: Default 0.10 for normal PDFs; higher for Canva-style PDFs.
    adaptive_threshold: float = 0.10

    def text(self) -> str:
        return self.text_with_formatting(False, False, False)

    def text_with_formatting(
        self,
        format_bold: bool,
        format_italic: bool,
        format_decorations: bool,
    ) -> str:
        """Get text with optional bold/italic/decorative markdown formatting.

        ``format_decorations`` enables both geometrically detected source
        decorations: underline (``<u>``) and strikeout (``<s>``).
        """
        if not format_bold and not format_italic and not format_decorations:
            return self._text_plain()

        single_char_threshold = self.adaptive_threshold

        result: list[str] = []
        current_bold = False
        current_italic = False
        current_underline = False
        current_strikeout = False

        for i, item in enumerate(self.items):
            text = item.text
            text_trimmed = text.strip()

            # Skip empty items
            if not text_trimmed:
                continue

            joined = "".join(result)

            # Determine spacing
            if i == 0 or not joined:
                needs_space = False
            else:
                prev_item = self.items[i - 1]
                needs_space = self._needs_space_between(
                    prev_item, item, joined, single_char_threshold
                )

            # Preserve leading whitespace from the item text. Items like
            # " means any person" have a leading space that indicates a word
            # boundary. _needs_space_between returns False for these (because
            # space_already_exists), but the space still has to be emitted since
            # text_trimmed is pushed below (which strips it).
            has_leading_space = text.startswith(" ")

            # Check for style changes. Source decorations are exclusive:
            # <u>/<s> content stays free of **/* markers — consumers (and the
            # eval harnesses this feeds) match tag content literally, and mixed
            # nesting breaks that. A struck-and-underlined item is emitted as
            # struck text because deletion is the stronger semantic distinction
            # in redline documents.
            item_strikeout = format_decorations and item.is_strikeout
            item_underline = (
                format_decorations and item.is_underline and not item_strikeout
            )
            item_bold = (
                format_bold and item.is_bold and not item_underline and not item_strikeout
            )
            item_italic = (
                format_italic
                and item.is_italic
                and not item_underline
                and not item_strikeout
            )

            # Close previous styles if they change
            if current_italic and not item_italic:
                result.append("*")
                current_italic = False
            if current_bold and not item_bold:
                result.append("**")
                current_bold = False
            if current_underline and not item_underline:
                result.append("</u>")
                current_underline = False
            if current_strikeout and not item_strikeout:
                result.append("</s>")
                current_strikeout = False

            # Add space: either from spacing logic or preserved from item text
            joined = "".join(result)
            if needs_space or (
                has_leading_space and joined and not joined.endswith(" ")
            ):
                result.append(" ")

            # Open new styles
            if item_underline and not current_underline:
                result.append("<u>")
                current_underline = True
            if item_strikeout and not current_strikeout:
                result.append("<s>")
                current_strikeout = True
            if item_bold and not current_bold:
                result.append("**")
                current_bold = True
            if item_italic and not current_italic:
                result.append("*")
                current_italic = True

            result.append(text_trimmed)

        # Close any remaining open styles
        if current_italic:
            result.append("*")
        if current_bold:
            result.append("**")
        if current_underline:
            result.append("</u>")
        if current_strikeout:
            result.append("</s>")

        return "".join(result)

    def _text_plain(self) -> str:
        """Get plain text without formatting."""
        single_char_threshold = self.adaptive_threshold

        result: list[str] = []
        for i, item in enumerate(self.items):
            text = item.text
            if i == 0:
                result.append(text)
            else:
                prev_item = self.items[i - 1]
                if self._needs_space_between(
                    prev_item, item, "".join(result), single_char_threshold
                ):
                    result.append(" ")
                result.append(text)
        return "".join(result)

    def _needs_space_between(
        self,
        prev_item: TextItem,
        item: TextItem,
        result: str,
        single_char_threshold: float,
    ) -> bool:
        """Determine if a space is needed between two items."""
        text = item.text

        # Don't add space before/after hyphens for hyphenated words
        prev_ends_with_hyphen = result.endswith("-")
        curr_is_hyphen = text.strip() == "-"
        curr_starts_with_hyphen = text.startswith("-")

        # Detect subscript/superscript: smaller font size and/or Y offset.
        # A zero font size divides to inf/NaN in Rust, and both compare False
        # against the 0.85 threshold — _ratio reproduces that rather than
        # raising or falling back to 0.0, which would flip the result.
        font_ratio = _ratio(item.font_size, prev_item.font_size)
        reverse_font_ratio = _ratio(prev_item.font_size, item.font_size)
        y_diff = abs(item.y - prev_item.y)

        is_sub_super = font_ratio < 0.85 and y_diff > 1.0
        was_sub_super = reverse_font_ratio < 0.85 and y_diff > 1.0

        # Use position-based spacing detection
        should_join = should_join_items(prev_item, item, single_char_threshold)

        # Check if space already exists
        prev_ends_with_space = result.endswith(" ")
        curr_starts_with_space = text.startswith(" ")
        space_already_exists = prev_ends_with_space or curr_starts_with_space

        # Add space unless one of these conditions applies
        return not (
            prev_ends_with_hyphen
            or curr_is_hyphen
            or curr_starts_with_hyphen
            or is_sub_super
            or was_sub_super
            or should_join
            or space_already_exists
        )
