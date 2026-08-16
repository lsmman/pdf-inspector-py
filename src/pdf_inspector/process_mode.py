"""How far the PDF processing pipeline runs."""

from __future__ import annotations

from enum import Enum


class ProcessMode(Enum):
    """Controls how far the PDF processing pipeline runs."""

    #: Only detect PDF type. Very fast — no text extraction.
    DETECT_ONLY = "detect_only"
    #: Detect type + extract text + compute layout complexity. Skips markdown.
    ANALYZE = "analyze"
    #: Full pipeline: detect, extract, convert to markdown (default).
    FULL = "full"
