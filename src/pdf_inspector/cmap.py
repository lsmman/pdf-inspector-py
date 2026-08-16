"""ToUnicode CMap parsing.

Parses the CMap programs PDFs embed to convert CID-encoded text to Unicode,
plus the helpers that read and normalise their hex operands.

The font-level machinery that decides *which* CMap a given font should use —
including the fallbacks for fonts that ship a broken one or none at all — lives
in :mod:`pdf_inspector.tounicode`.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Range expansion stops after this many CID visits, so a font that declares a
#: full-width range cannot make the scan quadratic.
MAX_CID_W_EXPANSION = 65_536

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass
class ToUnicodeCMap:
    """A parsed ToUnicode CMap mapping CIDs to Unicode strings."""

    #: Direct character mappings (CID -> Unicode codepoint(s)).
    char_map: dict[int, str] = field(default_factory=dict)
    #: Range mappings: (start_cid, end_cid, base_unicode), sorted by start.
    ranges: list[tuple[int, int, int]] = field(default_factory=list)
    #: Byte width of source codes (1 or 2), from the codespace or the entries.
    code_byte_length: int = 0
    #: When true, unmapped CIDs are interpreted as Unicode codepoints directly.
    #: Used as a last resort for Identity-H fonts with no ToUnicode/cmap/glyph
    #: names.
    cid_passthrough: bool = False

    # ── parsing ──────────────────────────────────────────────────────

    @classmethod
    def parse(cls, content: bytes) -> ToUnicodeCMap | None:
        """Parse a ToUnicode CMap from its decompressed content."""
        text = content.decode("utf-8", errors="replace")
        cmap = cls()
        src_hex_lengths: list[int] = []

        codespace_byte_len = _parse_codespace_byte_len(text)
        use_cmap_name = find_usecmap_name(text)

        for section in _sections(text, "beginbfchar", "endbfchar"):
            cmap._parse_bfchar_section(section, src_hex_lengths)

        for section in _sections(text, "beginbfrange", "endbfrange"):
            cmap._parse_bfrange_section(section, src_hex_lengths)

        if not cmap.char_map and not cmap.ranges:
            return None

        cmap.code_byte_length = _resolve_code_byte_length(
            codespace_byte_len, src_hex_lengths
        )

        # Sorted by start CID so lookup() can binary-search.
        cmap.ranges.sort(key=lambda r: r[0])

        if use_cmap_name is not None:
            from .tounicode import load_builtin_cmap_by_name

            base = load_builtin_cmap_by_name(use_cmap_name)
            if base is not None:
                cmap = merge_cmaps(base, cmap)
            else:
                logger.warning("usecmap=%s could not be loaded", use_cmap_name)

        return cmap

    def _parse_bfchar_section(
        self, section: str, src_hex_lengths: list[int]
    ) -> None:
        """Parse a bfchar section: ``<src> <dst>`` pairs."""
        scanner = _Scanner(section)
        while True:
            scanner.skip_whitespace()
            src_hex = scanner.read_hex_token()
            if src_hex is None:
                break

            trimmed = src_hex.strip()
            if trimmed:
                src_hex_lengths.append(len(trimmed))

            scanner.skip_whitespace()
            dst_hex = scanner.read_hex_token()
            if dst_hex is None:
                # Upstream continues rather than breaking here: a malformed pair
                # should not discard the entries that follow it.
                continue

            src = parse_hex_u16(src_hex)
            dst = hex_to_unicode_string(dst_hex)
            if src is not None and dst is not None:
                self.char_map[src] = dst

    def _parse_bfrange_section(
        self, section: str, src_hex_lengths: list[int]
    ) -> None:
        """Parse a bfrange section.

        Entries are ``<start> <end> <base>`` or
        ``<start> <end> [<u1> <u2> ...]``.
        """
        scanner = _Scanner(section)
        while True:
            scanner.skip_whitespace()
            start_hex = scanner.read_hex_token()
            if start_hex is None:
                break

            trimmed = start_hex.strip()
            if trimmed:
                src_hex_lengths.append(len(trimmed))

            scanner.skip_whitespace()
            end_hex = scanner.read_hex_token()
            if end_hex is None:
                continue

            scanner.skip_whitespace()

            if scanner.peek() == "<":
                base_hex = scanner.read_hex_token()
                start = parse_hex_u16(start_hex)
                end = parse_hex_u16(end_hex)
                base = hex_to_unicode_scalar(base_hex) if base_hex is not None else None
                if start is not None and end is not None and base is not None:
                    self.ranges.append((start, end, base))
            elif scanner.peek() == "[":
                scanner.advance()
                start = parse_hex_u16(start_hex)
                end = parse_hex_u16(end_hex)
                if start is None or end is None:
                    scanner.skip_to_close_bracket()
                    continue
                self._parse_bfrange_array(scanner, start, end)

    def _parse_bfrange_array(self, scanner: _Scanner, start: int, end: int) -> None:
        """Consume ``[<u1> <u2> ...]``, mapping each entry to ``start + index``."""
        cid = start
        while True:
            scanner.skip_whitespace()
            if scanner.peek() == "]":
                scanner.advance()
                return
            hex_token = scanner.read_hex_token()
            if hex_token is None:
                return
            unicode_str = hex_to_unicode_string(hex_token)
            if unicode_str is not None:
                self.char_map[cid] = unicode_str
            if cid >= end:
                # More entries than the range declares; drop the rest.
                scanner.skip_to_close_bracket()
                return
            cid = min(cid + 1, 0xFFFF)

    # ── lookup ───────────────────────────────────────────────────────

    def lookup(self, cid: int) -> str | None:
        """Look up a CID and return the Unicode string."""
        direct = self.char_map.get(cid)
        if direct is not None:
            return direct

        if not self.ranges:
            return None

        # Upstream binary-searches on the range start, then checks the hit and
        # the range before it — a CID can fall inside a range that starts
        # earlier.
        idx = bisect_left([r[0] for r in self.ranges], cid)

        for candidate in (idx, idx - 1):
            if 0 <= candidate < len(self.ranges):
                start, end, base = self.ranges[candidate]
                if start <= cid <= end:
                    char = _from_u32(base + (cid - start))
                    if char is not None:
                        return char

        return None

    def lookup_bytes(self, data: bytes) -> list[tuple[int, str | None]]:
        """Per-byte CMap lookup without the Latin-1 fallback.

        Returns ``(raw_byte, cmap_result_or_None)`` for each byte. Only
        meaningful for single-byte (``code_byte_length == 1``) CMaps.
        """
        result: list[tuple[int, str | None]] = []
        for b in data:
            mapped = self.lookup(b)
            if mapped is not None and "�" in mapped:
                mapped = None
            result.append((b, mapped))
        return result

    def decode_cids(self, data: bytes) -> str:
        """Decode a byte string, respecting the CMap's code byte width."""
        out: list[str] = []
        unmapped_count = 0

        if self.code_byte_length == 1:
            # Single-byte codes: each byte is a code.
            for b in data:
                mapped = self.lookup(b)
                if mapped is not None and "�" not in mapped:
                    out.append(mapped)
                else:
                    # The byte IS the character code in most legacy encodings.
                    if b >= 0x20:
                        out.append(chr(b))
                    unmapped_count += 1
            total = len(data)
        else:
            # Two-byte codes: CIDs are 2 bytes each, big-endian.
            for i in range(0, len(data) - 1, 2):
                cid = (data[i] << 8) | data[i + 1]
                mapped = self.lookup(cid)
                if mapped is not None and "�" not in mapped:
                    out.append(mapped)
                elif self.cid_passthrough:
                    # Last resort: treat the CID as a Unicode codepoint. Valid
                    # for Identity-H fonts where the generator used Unicode
                    # values as CIDs but stripped the cmap.
                    char = _from_u32(cid)
                    if char is not None and (
                        not _is_control(char) or char in ("\t", "\n")
                    ):
                        out.append(char)
                    else:
                        unmapped_count += 1
                else:
                    # CIDs are font-internal indices, not Unicode values.
                    # Unmapped 2-byte CIDs are skipped to avoid CJK garbage.
                    unmapped_count += 1
            total = len(data) // 2

        # Too many unmapped codes: signal failure with an empty string so the
        # caller falls through to other decoding methods.
        if total > 0 and unmapped_count > total // 2:
            return ""

        return "".join(out)

    # ── remapping ────────────────────────────────────────────────────

    def min_source_cid(self) -> int | None:
        """The smallest source CID across char_map and ranges."""
        candidates = list(self.char_map) + [r[0] for r in self.ranges]
        return min(candidates) if candidates else None

    def max_source_cid(self) -> int | None:
        """The largest source CID across char_map and ranges."""
        candidates = list(self.char_map) + [r[1] for r in self.ranges]
        return max(candidates) if candidates else None

    def remap_to_sequential(self) -> ToUnicodeCMap:
        """Remap a CMap that references pre-subsetting GIDs to sequential ones.

        Collects all source CIDs, sorts them, and reassigns to 1, 2, 3, ...
        Range expansion stops after :data:`MAX_CID_W_EXPANSION` CID visits,
        counting overwrites, so repeated full-width ``bfrange`` entries cannot
        re-expand the 16-bit domain.
        """
        cid_to_unicode: dict[int, str] = {}
        expand_bfranges_for_remap(self.ranges, cid_to_unicode, MAX_CID_W_EXPANSION)

        # char_map entries override range entries.
        cid_to_unicode.update(self.char_map)

        new_cmap = ToUnicodeCMap()
        # GID 0 is .notdef, so content CIDs start at 1.
        for index, old_cid in enumerate(sorted(cid_to_unicode), start=1):
            new_cmap.char_map[index] = cid_to_unicode[old_cid]
        new_cmap.code_byte_length = self.code_byte_length
        return new_cmap

    def entry_count(self) -> int:
        """Number of mappings, counting each range as one entry (as upstream does)."""
        return len(self.char_map) + len(self.ranges)


# ── scanning helpers ─────────────────────────────────────────────────


class _Scanner:
    """A cursor over a CMap section, reading ``<...>`` tokens."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0

    def peek(self) -> str | None:
        return self._text[self._pos] if self._pos < len(self._text) else None

    def advance(self) -> None:
        self._pos += 1

    def skip_whitespace(self) -> None:
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def read_hex_token(self) -> str | None:
        """Read a ``<...>`` token, returning its body, or None if not at one."""
        if self.peek() != "<":
            return None
        self._pos += 1
        start = self._pos
        while self._pos < len(self._text) and self._text[self._pos] != ">":
            self._pos += 1
        body = self._text[start : self._pos]
        if self._pos < len(self._text):
            self._pos += 1  # consume '>'
        return body

    def skip_to_close_bracket(self) -> None:
        while self._pos < len(self._text) and self._text[self._pos] != "]":
            self._pos += 1
        if self._pos < len(self._text):
            self._pos += 1


def _sections(text: str, begin: str, end: str) -> list[str]:
    """Every ``begin``…``end`` section body, in order."""
    out: list[str] = []
    pos = 0
    while True:
        start = text.find(begin, pos)
        if start == -1:
            break
        section_start = start + len(begin)
        stop = text.find(end, section_start)
        if stop == -1:
            break
        out.append(text[section_start:stop])
        pos = stop
    return out


def _parse_codespace_byte_len(text: str) -> int | None:
    """Byte width declared by ``begincodespacerange``, if present."""
    start = text.find("begincodespacerange")
    if start == -1:
        return None
    section_start = start + len("begincodespacerange")
    stop = text.find("endcodespacerange", section_start)
    if stop == -1:
        return None

    byte_len: int | None = None
    in_hex = False
    hex_len = 0
    for ch in text[section_start:stop]:
        if ch == "<":
            in_hex = True
            hex_len = 0
        elif ch == ">":
            if in_hex and hex_len > 0:
                byte_len = (hex_len + 1) // 2  # 2 hex digits = 1 byte
            in_hex = False
        elif in_hex and ch in _HEX_DIGITS:
            hex_len += 1
    return byte_len


def _resolve_code_byte_length(
    codespace_byte_len: int | None, src_hex_lengths: list[int]
) -> int:
    """Decide the source code width from the codespace and the entries."""
    if codespace_byte_len is not None:
        # A codespace of <0000><FFFF> paired with entries that are all <20>,
        # <41>, … really means one byte; trusting the codespace there would read
        # every pair of single-byte codes as one two-byte code.
        if (
            codespace_byte_len == 2
            and src_hex_lengths
            and all(length <= 2 for length in src_hex_lengths)
        ):
            return 1
        return codespace_byte_len

    if src_hex_lengths:
        return 1 if max(src_hex_lengths) <= 2 else 2

    return 2


def expand_bfranges_for_remap(
    ranges: list[tuple[int, int, int]],
    cid_to_unicode: dict[int, str],
    max_assignments: int,
) -> int:
    """Expand ``bfrange`` entries into individual CID->Unicode inserts.

    Returns how many CIDs were visited. Overwrites count, so a repeated
    full-width range cannot keep working after ``max_assignments``.
    """
    assigned = 0
    for start, end, base in ranges:
        if start > end:
            continue
        for cid in range(start, end + 1):
            if assigned >= max_assignments:
                return assigned
            assigned += 1
            char = _from_u32(base + (cid - start))
            if char is not None:
                cid_to_unicode[cid] = char
    return assigned


def merge_cmaps(base: ToUnicodeCMap, overlay: ToUnicodeCMap) -> ToUnicodeCMap:
    """Overlay one CMap onto another, with the overlay winning."""
    merged = ToUnicodeCMap(
        char_map=dict(base.char_map),
        ranges=list(base.ranges),
        code_byte_length=base.code_byte_length,
        cid_passthrough=base.cid_passthrough,
    )
    merged.char_map.update(overlay.char_map)
    merged.ranges.extend(overlay.ranges)
    merged.ranges.sort(key=lambda r: r[0])
    if overlay.code_byte_length:
        merged.code_byte_length = overlay.code_byte_length
    return merged


# ── hex helpers ──────────────────────────────────────────────────────


def parse_hex_u16(hex_text: str) -> int | None:
    """Parse a hex string to a 16-bit value."""
    text = hex_text.strip()
    if not text or not all(c in _HEX_DIGITS for c in text):
        return None
    value = int(text, 16)
    return value if 0 <= value <= 0xFFFF else None


def hex_to_unicode_string(hex_text: str) -> str | None:
    """Convert a ToUnicode destination hex string to Unicode.

    PDF ToUnicode destinations are UTF-16BE strings. Supplementary-plane
    characters are encoded as surrogate pairs, so treating each 4-hex chunk as a
    scalar would drop emoji like ``D83CDF1F``.
    """
    hex_clean = "".join(ch for ch in hex_text if not _is_ascii_space(ch))
    if not hex_clean or len(hex_clean) % 2 != 0:
        return None
    if not all(c in _HEX_DIGITS for c in hex_clean):
        return None

    data = bytes.fromhex(hex_clean)

    if len(data) % 2 == 0:
        try:
            result = data.decode("utf-16-be")
        except UnicodeDecodeError:
            result = None
        if result:
            return normalize_tounicode_destination(result)

    # Be permissive for non-standard one-byte destinations.
    if len(data) == 1:
        char = chr(data[0])
        if not _is_control(char) or char in ("\t", "\n"):
            return char

    return None


def normalize_tounicode_destination(text: str) -> str:
    """Collapse the malformed multi-codepoint destinations some producers emit."""
    is_multi_char = len(text) > 1

    # Some malformed producer CMaps put a list of alternative whitespace or
    # hyphen codepoints into one destination. Ordinary multi-character mappings
    # stay intact unless that signature is present.
    if (
        is_multi_char
        and all(ch.isspace() for ch in text)
        and any(ch in "\t\n\r" for ch in text)
    ):
        return "\t" if "\t" in text else " "

    if (
        is_multi_char
        and "­" in text
        and all(ch in "-­‐‑‒–−" for ch in text)
    ):
        return "-"

    return text


def hex_to_unicode_scalar(hex_text: str) -> int | None:
    """A destination hex string as a single codepoint, or None if it is longer."""
    text = hex_to_unicode_string(hex_text)
    if text is None or len(text) != 1:
        return None
    return ord(text)


def find_usecmap_name(text: str) -> str | None:
    """The name operand of a ``usecmap`` directive, without its leading slash."""
    for line in text.splitlines():
        if "usecmap" not in line:
            continue
        parts = line.split()
        for i, part in enumerate(parts):
            if part == "usecmap" and i > 0:
                name = parts[i - 1].strip()
                if name.startswith("/"):
                    return name[1:]
    return None


def _from_u32(code: int) -> str | None:
    """The character for a codepoint, or None when it is not a scalar value."""
    if code < 0 or code > 0x10FFFF or 0xD800 <= code <= 0xDFFF:
        return None
    return chr(code)


def _is_control(char: str) -> bool:
    """Rust's ``char::is_control`` — the Unicode Cc category."""
    code = ord(char)
    return code < 0x20 or 0x7F <= code <= 0x9F


def _is_ascii_space(char: str) -> bool:
    return char in " \t\n\r\x0b\x0c"
