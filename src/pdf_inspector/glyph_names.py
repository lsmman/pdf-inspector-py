"""Glyph name to Unicode mapping.

Maps Adobe Glyph List and related names to Unicode characters.
"""

from __future__ import annotations

from ._glyph_names_data import GLYPH_TO_UNICODE as _BASE

# Local overrides for non-standard glyph names seen in PDFs.
GLYPH_TO_UNICODE: dict[str, str] = dict(_BASE)
GLYPH_TO_UNICODE["C21"] = "≥"  # >=
GLYPH_TO_UNICODE["C25"] = "≈"  # ~=
# Some Type1 fonts use custom glyph names for ASCII tilde.
GLYPH_TO_UNICODE["C19"] = "~"
GLYPH_TO_UNICODE["C24"] = "~"


def _from_u32(code: int) -> str | None:
    """Return the character for a codepoint, or None when it is not a scalar value."""
    if code > 0x10FFFF or 0xD800 <= code <= 0xDFFF:
        return None
    return chr(code)


def glyph_to_char(name: str) -> str | None:
    """Convert a glyph name to its Unicode character."""
    # First check our mapping with the full name
    char = GLYPH_TO_UNICODE.get(name)
    if char is not None:
        return char

    # Per Adobe Glyph List spec, strip the suffix after '.' to get the base glyph
    # name. E.g., "zero.tf" -> "zero", "a.ss01" -> "a", "hyphen.case" -> "hyphen"
    dot_pos = name.find(".")
    if dot_pos != -1:
        char = GLYPH_TO_UNICODE.get(name[:dot_pos])
        if char is not None:
            return char

    # Try to parse uniXXXX format. The name may contain non-ASCII characters
    # (e.g. U+FFFD from a lossy decode of an attacker-controlled /Differences
    # name), so the hex parse has to reject them rather than assume ASCII.
    if name.startswith("uni") and len(name) >= 7:
        hex_part = name[3:7]
        if _is_ascii_hex(hex_part):
            code = int(hex_part, 16)
            # Strip PUA F000 offset: uniF0XX -> U+00XX (Windows Symbol convention)
            if 0xF000 <= code <= 0xF0FF:
                code -= 0xF000
            return _from_u32(code)

    # Try to parse uXXXX or uXXXXX format
    if name.startswith("u") and len(name) >= 5:
        hex_part = name[1:]
        if _is_ascii_hex(hex_part):
            return _from_u32(int(hex_part, 16))

    return None


def _is_ascii_hex(text: str) -> bool:
    """True when every character is an ASCII hex digit.

    ``int(text, 16)`` accepts Unicode digits and surrounding whitespace, which
    would let names like ``"u٠٠٠٠"`` parse. Upstream's ``from_str_radix`` does
    not, so the input is screened first.
    """
    return bool(text) and all(c in "0123456789abcdefABCDEF" for c in text)
