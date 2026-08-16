"""Character classification and text utility functions.

Pure helpers that operate on characters, strings, or ``TextItem`` sequences.
No PDF parsing happens here — these are shared across the extraction and
markdown pipelines.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Iterable, MutableSequence, Sequence

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .types import TextItem

_ASCII_DIGITS = frozenset("0123456789")
_ASCII_ALNUM = frozenset(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
_ASCII_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

#: Upstream's default join threshold for non-letterspaced pages.
DEFAULT_JOIN_THRESHOLD = 0.10


def _ascii_lower(text: str) -> str:
    """Lowercase only ASCII A-Z, matching Rust's ``to_ascii_lowercase``.

    Python's ``str.lower()`` also folds non-ASCII (e.g. ``İ`` -> ``i̇``), which
    would change how these page-number probes behave on non-Latin text.
    """
    return text.translate(_ASCII_LOWER_TABLE)


_ASCII_LOWER_TABLE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"
)


def _byte_len(text: str) -> int:
    """UTF-8 byte length, matching Rust's ``str::len``."""
    return len(text.encode("utf-8"))


def _is_number(value: str) -> bool:
    return bool(value) and all(ch in _ASCII_DIGITS for ch in value)


def is_explicit_page_number_expression(text: str) -> bool:
    """Return whether text is an explicit page-number expression.

    This strict form is suitable before layout, where removing one numeric item
    from substantive text such as ``Page 42 explains the result`` would lose
    data.
    """
    trimmed = text.strip()
    if not trimmed:
        return False

    if _byte_len(trimmed) <= 4 and _is_number(trimmed):
        return True

    if _byte_len(trimmed) >= 3 and trimmed.startswith("-") and trimmed.endswith("-"):
        inner = trimmed[1:-1].strip()
        if _is_number(inner):
            return True

    lowercase = _ascii_lower(trimmed)
    if lowercase.startswith("page"):
        rest = lowercase[len("page") :]
        words = rest.split()
        if len(words) >= 3 and _is_number(words[0]) and words[1] == "of" and _is_number(words[2]):
            return True
        if len(words) >= 2 and words[0] == "of" and _is_number(words[1]):
            return True
        if not words or words == ["of"]:
            return True
        if len(words) == 1:
            return _is_number(words[0])
        if len(words) == 2 and words[0] == "of":
            return _is_number(words[1])
        if len(words) == 3 and words[1] == "of":
            return _is_number(words[0]) and _is_number(words[2])
        return False

    words = lowercase.split()
    if len(words) == 3 and words[1] == "of":
        return _is_number(words[0]) and _is_number(words[2])
    return False


def is_page_number_line(text: str) -> bool:
    """Return whether a completed Markdown line looks like a page number or a
    labeled running header.

    At this stage the complete line and surrounding breaks are available, so a
    leading ``Page N`` remains compatible with the existing header cleanup even
    when the PDF appends a chapter or document title.
    """
    if is_explicit_page_number_expression(text):
        return True

    lowercase = _ascii_lower(text.strip())
    if not lowercase.startswith("page"):
        return False

    rest = lowercase[len("page") :].lstrip()
    index = 0
    has_page_number = False
    while index < len(rest) and rest[index] in _ASCII_DIGITS:
        has_page_number = True
        index += 1

    if not has_page_number:
        return False
    # Trailing content is fine as long as the digits end at a word boundary.
    return index >= len(rest) or rest[index].isspace()


def is_cjk_char(c: str) -> bool:
    """Check if a character is CJK (Chinese, Japanese, Korean).

    CJK languages don't use spaces between words, so word-boundary heuristics
    should not apply when CJK characters are involved.
    """
    code = ord(c)
    return (
        0x1100 <= code <= 0x11FF  # Hangul Jamo
        or 0x3000 <= code <= 0x303F  # CJK Symbols and Punctuation
        or 0x3040 <= code <= 0x309F  # Hiragana
        or 0x30A0 <= code <= 0x30FF  # Katakana
        or 0x3130 <= code <= 0x318F  # Hangul Compatibility Jamo
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
        or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
        or 0xFF00 <= code <= 0xFFEF  # Halfwidth and Fullwidth Forms
    )


def is_rtl_char(c: str) -> bool:
    code = ord(c)
    return (
        0x0590 <= code <= 0x05FF  # Hebrew
        or 0x0600 <= code <= 0x06FF  # Arabic
        or 0x0700 <= code <= 0x074F  # Syriac
        or 0x0750 <= code <= 0x077F  # Arabic Supplement
        or 0x0780 <= code <= 0x07BF  # Thaana
        or 0x07C0 <= code <= 0x07FF  # NKo
        or 0x0800 <= code <= 0x083F  # Samaritan
        or 0x0840 <= code <= 0x085F  # Mandaic
        or 0x08A0 <= code <= 0x08FF  # Arabic Extended-A
        or 0xFB1D <= code <= 0xFB4F  # Hebrew Presentation Forms
        or 0xFB50 <= code <= 0xFDFF  # Arabic Presentation Forms-A
        or 0xFE70 <= code <= 0xFEFF  # Arabic Presentation Forms-B
    )


def is_arabic_presentation_form(c: str) -> bool:
    # U+FEFF is BOM/ZWNJ, not an Arabic presentation form despite falling in the
    # Presentation Forms-B codepoint range.
    code = ord(c)
    return 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFE


def is_rtl_text(texts: Iterable[str]) -> bool:
    rtl = 0
    ltr = 0
    for text in texts:
        for c in text:
            if is_rtl_char(c):
                rtl += 1
            elif c.isalpha() and not is_cjk_char(c):
                ltr += 1
    return rtl > 0 and rtl > ltr


def sort_line_items(items: MutableSequence[TextItem]) -> None:
    """Sort a line's items into reading order, in place."""
    rtl = is_rtl_text(item.text for item in items)
    ordered = sorted(items, key=lambda item: item.x, reverse=rtl)
    items[:] = ordered


def is_bold_font(font_name: str) -> bool:
    """Detect if a font name indicates bold style.

    Common patterns: "Bold", "Bd", "Black", "Heavy", "Demi", "Semi" (semi-bold).
    """
    lower = font_name.lower()

    # Note: need to be careful with "Oblique" not matching "Obl" + false positive
    # for bold.
    return (
        "bold" in lower
        or "-bd" in lower
        or "_bd" in lower
        or "black" in lower
        or "heavy" in lower
        or "demibold" in lower
        or "semibold" in lower
        or "demi-bold" in lower
        or "semi-bold" in lower
        or "extrabold" in lower
        or "ultrabold" in lower
        # Some fonts use Medium for semi-bold
        or ("medium" in lower and "mediumitalic" not in lower)
        # URW Type 1 fonts abbreviate Medium as "Medi" (e.g. NimbusRomNo9L-Medi,
        # the Times-Bold substitute in LaTeX documents; -MediItal is bold italic).
        or ("-medi" in lower and "mediumital" not in lower)
    )


def is_italic_font(font_name: str) -> bool:
    """Detect if a font name indicates italic/oblique style.

    Common patterns: "Italic", "It", "Oblique", "Obl", "Slant", "Inclined".
    """
    lower = font_name.lower()

    return (
        "italic" in lower
        or "oblique" in lower
        or "-it" in lower
        or "_it" in lower
        or "slant" in lower
        or "inclined" in lower
        or "kursiv" in lower  # German for italic
    )


def expand_ligatures(text: str) -> str:
    """Expand Unicode ligature characters to their component characters.

    This makes extracted text more searchable and semantically correct. Also
    applies NFKC normalization (converts Arabic presentation forms to base
    characters, decomposes Latin ligatures, etc.) and reverses visual-order
    Arabic text back to logical order when presentation forms are detected.
    """
    # Strip null bytes and other control characters (except newline/tab)
    if any(ch < " " and ch not in "\n\r\t" for ch in text):
        text = "".join(ch for ch in text if ch >= " " or ch in "\n\r\t")

    # Detect Arabic presentation forms before normalization — their presence
    # signals visual-order storage that needs reversal after NFKC.
    had_presentation_forms = any(is_arabic_presentation_form(ch) for ch in text)

    # Apply NFKC normalization only when Arabic presentation forms are present.
    # This converts forms (U+FB50-FDFF, U+FE70-FEFF) back to base Arabic
    # (U+0600-06FF). Broad NFKC on all non-ASCII text is avoided because it
    # would convert NBSP (U+00A0) to a regular space, breaking downstream logic.
    # Latin ligatures are already handled by the explicit branches below.
    if had_presentation_forms:
        text = unicodedata.normalize("NFKC", text)

    result: list[str] = []
    for ch in text:
        # Keep explicit ligature expansion as fallback for fonts that bypass
        # NFKC (e.g. custom ToUnicode mappings to PUA codepoints)
        if ch == "ﬀ":
            result.append("ff")
        elif ch == "ﬁ":
            result.append("fi")
        elif ch == "ﬂ":
            result.append("fl")
        elif ch == "ﬃ":
            result.append("ffi")
        elif ch == "ﬄ":
            result.append("ffl")
        elif ch in ("ﬅ", "ﬆ"):
            result.append("st")
        # Strip invisible Unicode characters that pollute markdown output
        elif ch == "­":  # soft hyphen
            pass
        elif ch == "​":  # zero-width space
            pass
        elif ch == "﻿":  # BOM / zero-width no-break space
            pass
        elif ch in ("‌", "‍"):  # ZWNJ / ZWJ
            pass
        elif ch == "⁠":  # word joiner
            pass
        # Normalize typographic spaces to ASCII space so downstream spacing
        # logic (should_join_items) can detect word boundaries. Excludes NBSP
        # (U+00A0), which is common in PDFs and handled correctly by the
        # existing coordinate-based spacing.
        elif " " <= ch <= " ":  # en/em/thin/hair spaces etc.
            result.append(" ")
        else:
            result.append(ch)

    output = "".join(result)

    # If the original text had Arabic presentation forms, the characters are in
    # visual (LTR screen) order. After NFKC normalization, reverse to restore
    # logical reading order.
    if had_presentation_forms:
        output = reverse_visual_arabic(output)

    return output


def reverse_visual_arabic(text: str) -> str:
    """Reverse visual-order Arabic text to logical order.

    Pure RTL text (no ASCII alphanumerics) gets a simple character reversal.
    Mixed content (embedded numbers or Latin words) splits into LTR and non-LTR
    runs: run order is reversed, and only non-LTR runs are reversed internally.
    """
    # Check if there are any LTR runs (ASCII letters or digits)
    has_ltr = any(ch in _ASCII_ALNUM for ch in text)

    if not has_ltr:
        # Pure RTL: simple reversal
        return text[::-1]

    # Mixed content: split into runs of LTR (ASCII alphanumeric + adjacent
    # punctuation like '.', ',', '/', '-') vs non-LTR (Arabic + spaces + other).
    chars = list(text)
    runs: list[tuple[bool, str]] = []

    i = 0
    while i < len(chars):
        is_ltr = chars[i] in _ASCII_ALNUM or (
            chars[i] in _ASCII_PUNCTUATION and _is_adjacent_to_ascii_alnum(chars, i)
        )

        run: list[str] = []
        while i < len(chars):
            c = chars[i]
            c_is_ltr = c in _ASCII_ALNUM or (
                c in _ASCII_PUNCTUATION and _is_adjacent_to_ascii_alnum(chars, i)
            )
            if c_is_ltr != is_ltr:
                break
            run.append(c)
            i += 1
        runs.append((is_ltr, "".join(run)))

    # Reverse run order and reverse non-LTR runs internally
    runs.reverse()
    result: list[str] = []
    for is_ltr, content in runs:
        result.append(content if is_ltr else content[::-1])
    return "".join(result)


def _is_adjacent_to_ascii_alnum(chars: Sequence[str], idx: int) -> bool:
    """Check if the character at ``idx`` is adjacent to an ASCII alphanumeric."""
    return (idx > 0 and chars[idx - 1] in _ASCII_ALNUM) or (
        idx + 1 < len(chars) and chars[idx + 1] in _ASCII_ALNUM
    )


def decode_text_string(data: bytes) -> str:
    """Decode a PDF text string (ActualText, etc.).

    Handles UTF-16BE (BOM ``\\xFE\\xFF``) and PDFDocEncoding (a Latin-1
    superset).
    """
    if len(data) >= 2 and data[0] == 0xFE and data[1] == 0xFF:
        # UTF-16BE with BOM. Trailing odd byte is dropped, matching
        # chunks_exact(2); unpaired surrogates become U+FFFD.
        body = data[2:]
        body = body[: len(body) - (len(body) % 2)]
        return body.decode("utf-16-be", errors="replace")
    # PDFDocEncoding — identical to Latin-1 for the byte range we care about
    return data.decode("latin-1")


def effective_font_size(base_size: float, text_matrix: Sequence[float]) -> float:
    """Compute effective font size from base size and text matrix.

    Text matrix is ``[a, b, c, d, tx, ty]`` where ``a``/``d`` are scale factors.
    """
    # The scale factor is typically the magnitude of the transformation. For most
    # PDFs, text_matrix[0] (a) is the horizontal scale and text_matrix[3] (d) is
    # the vertical scale.
    scale_x = (text_matrix[0] ** 2 + text_matrix[1] ** 2) ** 0.5
    scale_y = (text_matrix[2] ** 2 + text_matrix[3] ** 2) ** 0.5
    # Use the larger of the two scales (usually equal for non-rotated text)
    return base_size * max(scale_x, scale_y)


def effective_width(item: TextItem) -> float:
    """Estimate the width of a text item.

    Falls back to a character-count heuristic when the width is 0.
    """
    if item.width > 0.0:
        return item.width
    return len(item.text) * item.font_size * 0.5


def is_cid_font(font: str) -> bool:
    return font.startswith("C2_") or font.startswith("C0_")


def _is_letterspaced(text: str) -> bool:
    """Check if the item text matches an ``x y z`` pattern.

    That is, single characters separated by single spaces.
    """
    trimmed = text.strip()
    chars = list(trimmed)
    # Need at least 3 chars: "a b" = ['a', ' ', 'b']
    if len(chars) < 3:
        return False
    # Pattern: non-space, space, non-space, space, ...
    return all((c != " ") if i % 2 == 0 else (c == " ") for i, c in enumerate(chars))


def fix_letterspaced_items(items: MutableSequence[TextItem]) -> float:
    """Detect and fix Canva-style letter-spacing within text items.

    Canva-generated PDFs render text character-by-character with CSS-style
    letter-spacing. The TJ handler inserts spaces between each character,
    producing items like ``"a r i b"`` instead of ``"arib"``. This function
    detects such items by checking if the text follows a strict pattern of
    alternating single characters and spaces, then removes the spurious spaces.

    Only activates when >=50% of items on the page are letter-spaced, to avoid
    false positives on normal PDFs with short items like ``"a b"``.

    Returns the adaptive join threshold for this page: 0.10 for normal pages, or
    a higher median-derived threshold for Canva-style pages.
    """
    if not items:
        return DEFAULT_JOIN_THRESHOLD

    # Count how many items are letter-spaced vs total non-trivial items
    letterspaced_count = 0
    total_text_items = 0
    for item in items:
        trimmed = item.text.strip()
        # Upstream compares UTF-8 byte length here, so a single CJK character
        # (3 bytes) counts as substantial while a 2-letter Latin word does not.
        if not trimmed or _byte_len(trimmed) < 3:
            continue
        total_text_items += 1
        if _is_letterspaced(item.text):
            letterspaced_count += 1

    # Only fix if >=50% of substantial items are letter-spaced
    if total_text_items < 4 or letterspaced_count * 2 < total_text_items:
        # Second detection path: per-character rendering without embedded
        # spaces. Canva sometimes emits each character as a separate TextItem
        # (no "a b c" pattern within items). Detect by checking if >50% of items
        # are single chars.
        single_char_count = sum(1 for item in items if len(item.text.strip()) == 1)
        if len(items) >= 10 and single_char_count * 2 >= len(items):
            threshold = compute_canva_join_threshold(items)
            if threshold > 0.40:
                return threshold
        return DEFAULT_JOIN_THRESHOLD

    # Compute threshold BEFORE removing spaces. Since this is a confirmed
    # Canva-style page (>=50% letterspaced), the ungated variant that includes
    # all pairs is used — the char-count guard in the normal function would
    # filter out long letterspaced items like "i s s i o n" (11 chars).
    threshold = compute_canva_join_threshold(items)

    # Remove spaces from letter-spaced items
    for item in items:
        if _is_letterspaced(item.text):
            item.text = item.text.replace(" ", "")

    return threshold


def compute_canva_join_threshold(items: Sequence[TextItem]) -> float:
    """Compute the join threshold for a confirmed Canva-style page.

    Uses ``median * 1.55`` on the gap/font_size ratio distribution. The
    page-level threshold is used for multi-char item pairs; single-char pairs
    use character-width-based joining in :func:`should_join_items` instead.
    """
    min_samples = 8

    ratios = collect_gap_ratios(items)
    if len(ratios) < min_samples:
        return DEFAULT_JOIN_THRESHOLD

    ordered = sorted(ratios)

    if ordered[-1] < 0.40 or ordered[0] < 0.40:
        return DEFAULT_JOIN_THRESHOLD

    median = ordered[len(ordered) // 2]
    return min(max(median * 1.55, 0.50), 2.0)


def collect_gap_ratios(items: Sequence[TextItem]) -> list[float]:
    """Collect positive gap/font_size ratios from adjacent item pairs.

    Filters out CJK, zero-width, and out-of-range values.
    """
    ratios: list[float] = []
    for prev, curr in zip(items, items[1:]):
        prev_trimmed = prev.text.strip()
        curr_trimmed = curr.text.strip()
        prev_c = prev_trimmed[-1] if prev_trimmed else None
        curr_c = curr_trimmed[0] if curr_trimmed else None
        if (prev_c is not None and is_cjk_char(prev_c)) or (
            curr_c is not None and is_cjk_char(curr_c)
        ):
            continue

        if prev.width <= 0.0 or prev.font_size <= 0.0:
            continue

        if prev.x <= curr.x:
            gap = curr.x - (prev.x + prev.width)
        else:
            gap = prev.x - (curr.x + curr.width)

        ratio = gap / prev.font_size

        if 0.0 <= ratio <= 3.0:
            ratios.append(ratio)
    return ratios


def should_join_items(
    prev_item: TextItem,
    curr_item: TextItem,
    single_char_threshold: float,
) -> bool:
    """Determine if two adjacent text items should be joined without a space.

    Based on their physical positions on the page and character case. Uses a
    hybrid approach: position-based with case-aware thresholds. CID fonts emit
    one word per text operator with gaps ~0 between words. Non-CID
    (Type1/TrueType) fonts emit phrases or fragments.
    """
    # If either text explicitly has leading/trailing spaces, respect them
    if prev_item.text.endswith(" ") or curr_item.text.startswith(" "):
        return False

    # Get the last character of previous and first character of current
    prev_stripped_end = prev_item.text.rstrip()
    curr_stripped_start = curr_item.text.lstrip()
    prev_last = prev_stripped_end[-1] if prev_stripped_end else None
    curr_first = curr_stripped_start[0] if curr_stripped_start else None

    # Always join if current starts with punctuation that typically follows
    # without space, e.g. "www" + ".com" -> "www.com", not "www .com"
    if curr_first is not None and curr_first in ".,;!?)]}'":
        return True

    # After colons, add space if followed by alphanumeric (typical label:value
    # pattern), e.g. "Clave:" + "T9N2I6" -> "Clave: T9N2I6"
    if prev_last == ":" and curr_first is not None and curr_first.isalnum():
        return False

    prev_trimmed = prev_item.text.strip()
    curr_trimmed = curr_item.text.strip()

    # When we have accurate width from font metrics, use a tight threshold
    if prev_item.width > 0.0:
        if prev_item.x <= curr_item.x:
            # LTR: prev is left of curr
            gap = curr_item.x - (prev_item.x + prev_item.width)
        else:
            # RTL: prev is right of curr
            gap = prev_item.x - (curr_item.x + curr_item.width)
        font_size = prev_item.font_size

        # Never join across column-scale gaps or large overlaps. Large negative
        # gaps arise when Tc/Tw inflate item widths past where adjacent items
        # actually start.
        if gap > font_size * 3.0 or gap < -font_size:
            return False

        # CID fonts (C2_*, C0_*) emit one word per text operator with gaps ~0
        # between words. Detect these and add spaces. Only applies to CID fonts —
        # non-CID fonts (Type1/TrueType) emit phrases or fragments with small
        # gaps from positioning imprecision and should NOT trigger this. Skipped
        # for CJK text, which doesn't use spaces between words.
        prev_chars = len(prev_trimmed)
        curr_chars = len(curr_trimmed)
        prev_last_char = prev_trimmed[-1] if prev_trimmed else None
        curr_first_char = curr_trimmed[0] if curr_trimmed else None
        is_cjk = (prev_last_char is not None and is_cjk_char(prev_last_char)) or (
            curr_first_char is not None and is_cjk_char(curr_first_char)
        )

        if (
            not is_cjk
            and gap >= 0.0
            and gap < font_size * 0.01
            and is_cid_font(prev_item.font)
        ):
            prev_word_count = len(prev_item.text.split())

            if prev_word_count >= 3:
                # Multi-word phrase from a line-level CID operator — likely a
                # mid-word boundary
                return gap < font_size * 0.15

            # CID font: each text operator is a separate word. Always add space.
            return False

        # Numeric continuity: digits, commas, periods, and percent signs that are
        # positioned close together are almost always a single number, e.g.
        # "34,20" + "8" -> "34,208", "+13." + "0" + "%" -> "+13.0%". Uses a
        # generous threshold since word spaces in numbers are rare. The lower
        # bound (-font_size) rejects large overlaps caused by Tc/Tw-inflated item
        # widths that make adjacent items appear to occupy the same space.
        if prev_last is not None and curr_first is not None:
            prev_is_numeric = prev_last in _ASCII_DIGITS or prev_last in ",."
            curr_is_numeric = curr_first in _ASCII_DIGITS or curr_first in "%."
            if prev_is_numeric and curr_is_numeric:
                return -font_size < gap < font_size * 0.3
            # Sign characters (+/-) followed by digits
            if prev_last in "+-" and curr_first in _ASCII_DIGITS:
                return -font_size < gap < font_size * 0.3

        # When the adaptive threshold indicates Canva-style letter-spacing (all
        # gaps wide), use character-width-based joining.
        #
        # Canva renders text character-by-character with CSS-style
        # letter-spacing. For single-char prev items, gap/char_width gives a
        # clean separation (~0.9-1.05 for letter gaps, ~1.5+ for word gaps). For
        # multi-char prev, avg_char_width normalizes for character mix.
        # Multi->multi pairs use the page-level threshold (gap/font_size).
        if single_char_threshold > 0.20:
            if prev_chars == 1:
                # Single-char prev: its rendered width is an accurate reference
                return gap < prev_item.width * 1.25
            if curr_chars == 1:
                # Multi->single: avg char width of prev normalises for
                # wide/narrow character mix (e.g. "ilw" includes i, l, w)
                avg_char_width = prev_item.width / prev_chars
                return gap < avg_char_width * 1.25
            # Both multi-char: use page-level threshold
            return gap < font_size * single_char_threshold

        # Single-character fragment joined to a multi-character item: use a
        # moderately generous threshold to rejoin split words like "b" + "illion"
        # or "C" + "ultural". Gap near 0 = same word; gap ~0.2+ = different words.
        if (prev_chars == 1) != (curr_chars == 1):
            return gap < font_size * 0.20

        # Both single-char: per-glyph positioning (character-by-character
        # rendering). Intra-word gaps are ~0, word boundaries are ~0.15x
        # font_size. For numeric chars (digits within "100,000"), use a generous
        # threshold. For alphabetic, use a tight threshold (0.10) to reliably
        # detect word boundaries in per-character PDFs like SEC filings.
        if prev_chars == 1 and curr_chars == 1:
            if prev_last is not None and curr_first is not None:
                p_numeric = prev_last in _ASCII_DIGITS or prev_last in ",.%+-"
                c_numeric = curr_first in _ASCII_DIGITS or curr_first in ",.%"
                if p_numeric and c_numeric:
                    return gap < font_size * 0.25
            return gap < font_size * single_char_threshold

        # With accurate widths, a gap < 15% of font size means glyphs are
        # adjacent (same word). Anything larger is a deliberate space. For
        # multi-char items with a lowercase->lowercase junction, use a slightly
        # wider threshold (0.18) to avoid mid-word space injection with imprecise
        # CID font metrics (e.g. "enterta"+"inment"). All-caps or mixed-case
        # junctions keep the tighter 0.15 threshold to preserve word boundaries
        # (e.g. "LCOE"+"WITH").
        if len(prev_trimmed) >= 2 and len(curr_trimmed) >= 2:
            prev_ends_lower = bool(prev_trimmed) and prev_trimmed[-1].islower()
            curr_starts_lower = bool(curr_trimmed) and curr_trimmed[0].islower()
            if prev_ends_lower and curr_starts_lower:
                return gap < font_size * 0.18
        return gap < font_size * 0.15

    # Fallback: estimate width from font size heuristics
    char_width = prev_item.font_size * 0.45

    estimated_prev_width = len(prev_item.text) * char_width

    # Calculate expected end position of previous item
    prev_end_x = prev_item.x + estimated_prev_width

    # Calculate gap between items
    gap = curr_item.x - prev_end_x

    # Never join across column-scale gaps (fallback path)
    if gap > char_width * 6.0:
        return False

    # CJK text: always join adjacent items — CJK languages don't use spaces
    # between words. The Latin case-based heuristics below would incorrectly
    # insert spaces within CJK words.
    is_cjk = (prev_last is not None and is_cjk_char(prev_last)) or (
        curr_first is not None and is_cjk_char(curr_first)
    )
    if is_cjk:
        return gap < char_width * 0.8

    # Use different thresholds based on character case. Same-case sequences (ALL
    # CAPS or all lowercase) are more likely to be word fragments that got split.
    # Mixed case suggests word boundaries.
    if (
        prev_last is not None
        and curr_first is not None
        and prev_last.isalpha()
        and curr_first.isalpha()
    ):
        same_case = (prev_last.isupper() and curr_first.isupper()) or (
            prev_last.islower() and curr_first.islower()
        )
        if same_case:
            # Same case: use generous threshold (likely same word fragment),
            # e.g. "CONST" + "ANCIA" -> "CONSTANCIA"
            return gap < char_width * 0.8
        if prev_last.islower() and curr_first.isupper():
            # Lowercase to uppercase transition (e.g. "presente" -> "CONSTANCIA")
            # is typically a word boundary. In Spanish/English, words don't
            # transition from lowercase to uppercase mid-word. Always add a
            # space for this case, regardless of position.
            return False
        # Uppercase to lowercase (e.g. "REGISTRO" -> "para"): use a stricter
        # threshold (likely word boundary)
        return gap < char_width * 0.3

    # Non-alphabetic: use moderate threshold
    return gap < char_width * 0.5
