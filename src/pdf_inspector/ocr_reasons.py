"""Machine-readable OCR reason identifiers and the helper that records them."""

from __future__ import annotations

#: Text was extracted but looks garbled (broken encoding, substitution cipher).
OCR_REASON_SUSPECTED_GARBLED_TEXT = "suspected_garbled_text"
#: The page is a scanned image.
OCR_REASON_SCANNED = "scanned"
#: The page has no extractable text layer.
OCR_REASON_NO_TEXT = "no_text"
#: The page draws its text as vector paths rather than as text operators.
OCR_REASON_VECTOR_TEXT = "vector_text"


def add_ocr_reason(reasons_by_page: dict[int, list[str]], page: int, reason: str) -> None:
    """Record ``reason`` for ``page``, keeping the list free of duplicates."""
    reasons = reasons_by_page.setdefault(page, [])
    if reason not in reasons:
        reasons.append(reason)


def merge_ocr_reasons(
    reasons_by_page: dict[int, list[str]],
    extra_reasons_by_page: dict[int, list[str]],
) -> None:
    """Merge ``extra_reasons_by_page`` into ``reasons_by_page``."""
    for page, reasons in extra_reasons_by_page.items():
        for reason in reasons:
            add_ocr_reason(reasons_by_page, page, reason)


def sorted_pages(reasons_by_page: dict[int, list[str]]) -> list[int]:
    """Page numbers in ascending order.

    Upstream uses a ``BTreeMap``, which iterates in key order. Python dicts
    iterate in insertion order, so every place that relied on the Rust
    ordering sorts explicitly through this helper.
    """
    return sorted(reasons_by_page)
