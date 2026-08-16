"""Error types, mirroring upstream's ``PdfError`` enum."""

from __future__ import annotations


class PdfError(Exception):
    """Base class for every error this package raises."""


class PdfIoError(PdfError):
    """The file could not be read."""


class PdfParseError(PdfError):
    """The PDF could not be parsed."""


class PdfEncryptedError(PdfError):
    """The PDF is encrypted and no usable password was supplied."""

    def __init__(self, message: str = "PDF is encrypted") -> None:
        super().__init__(message)


class PdfInvalidStructureError(PdfError):
    """The PDF parsed but its structure is unusable."""


class NotAPdfError(PdfError):
    """The bytes are not a PDF."""
