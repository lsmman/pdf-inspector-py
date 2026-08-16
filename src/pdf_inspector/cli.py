"""Command-line entry points, mirroring upstream's ``pdf2md`` and ``detect-pdf``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import __version__, detect_pdf
from .errors import PdfError


def _quiet_pypdf() -> None:
    """Silence pypdf's recovery chatter.

    pypdf logs a warning for every damaged cross-reference entry it repairs.
    That is normal for the malformed files this tool is meant to survive, and it
    would otherwise drown the actual output.
    """
    logging.getLogger("pypdf").setLevel(logging.ERROR)


def detect_pdf_main(argv: list[str] | None = None) -> int:
    """Classify a PDF and print the result."""
    parser = argparse.ArgumentParser(
        prog="detect-pdf",
        description="Classify a PDF as text-based, scanned, image-based, or mixed.",
    )
    parser.add_argument("path", help="path to the PDF file")
    parser.add_argument(
        "--password", help="password for an encrypted PDF", default=None
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full result as JSON"
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)

    _quiet_pypdf()

    try:
        result = detect_pdf(args.path, args.password)
    except PdfError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "pdf_type": result.pdf_type,
                    "page_count": result.page_count,
                    "confidence": round(result.confidence, 4),
                    "title": result.title,
                    "pages_needing_ocr": result.pages_needing_ocr,
                    "ocr_reasons_by_page": {
                        str(entry.page): entry.reasons
                        for entry in result.ocr_reasons_by_page
                    },
                    "processing_time_ms": result.processing_time_ms,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"type:       {result.pdf_type}")
    print(f"pages:      {result.page_count}")
    print(f"confidence: {result.confidence:.2f}")
    if result.title:
        print(f"title:      {result.title}")
    if result.pages_needing_ocr:
        print(f"needs OCR:  {result.pages_needing_ocr}")
        for entry in result.ocr_reasons_by_page:
            print(f"  page {entry.page}: {', '.join(entry.reasons)}")
    else:
        print("needs OCR:  none")
    print(f"took:       {result.processing_time_ms} ms")
    return 0


def pdf2md_main(argv: list[str] | None = None) -> int:
    """Convert a PDF to Markdown.

    Not available yet: the extraction pipeline is still being ported. The
    command exists so the entry point is stable, and it exits non-zero with a
    pointer to what is missing rather than pretending to succeed.
    """
    parser = argparse.ArgumentParser(
        prog="pdf2md", description="Convert a PDF to Markdown."
    )
    parser.add_argument("path", help="path to the PDF file")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args(argv)

    print(
        "pdf2md is not available yet: the extraction and Markdown pipeline "
        "(upstream src/lib.rs) has not been ported. See the port status table "
        "in the README. Use `detect-pdf` for classification in the meantime.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(detect_pdf_main())
