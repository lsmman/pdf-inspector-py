"""Ported from the ``#[cfg(test)]`` block in upstream src/text_utils.rs."""

from __future__ import annotations

import pytest

from pdf_inspector.text_utils import (
    expand_ligatures,
    is_arabic_presentation_form,
    is_bold_font,
    reverse_visual_arabic,
)
from pdf_inspector.types import ItemType, TextItem


def make_char_item(ch: str, x: float, width: float, font_size: float) -> TextItem:
    """Helper to create a single-char TextItem at a given x position with width."""
    return TextItem(
        text=ch,
        x=x,
        y=100.0,
        width=width,
        height=font_size,
        font="TestFont",
        font_size=font_size,
        page=1,
        item_type=ItemType.TEXT,
    )


def test_bold_font_urw_medi_abbreviation():
    # URW Type 1 fonts (LaTeX default Times) abbreviate Medium as "Medi"
    assert is_bold_font("NROFIU+NimbusRomNo9L-Medi")
    assert is_bold_font("NimbusRomNo9L-MediItal")
    assert not is_bold_font("DSSZWN+NimbusRomNo9L-Regu")
    assert not is_bold_font("NimbusRomNo9L-ReguItal")
    # Medium-Italic exclusion still holds
    assert not is_bold_font("Foo-MediumItalic")


def test_strip_soft_hyphen():
    assert expand_ligatures("con­tent") == "content"


def test_strip_zero_width_space():
    assert expand_ligatures("hello​world") == "helloworld"


def test_strip_bom():
    assert expand_ligatures("﻿text") == "text"


def test_strip_zwnj_zwj_word_joiner():
    assert expand_ligatures("a‌b‍c⁠d") == "abcd"


def test_ligature_plus_invisible_chars():
    assert expand_ligatures("ﬁrst­ly") == "firstly"


def test_ligatures_still_expand():
    assert expand_ligatures("ﬀﬁﬂ") == "fffifl"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("• text", "• text"),  # EM SPACE
        ("a b", "a b"),  # EN SPACE
        ("x y", "x y"),  # THIN SPACE
    ],
)
def test_normalize_typographic_spaces(raw: str, expected: str):
    assert expand_ligatures(raw) == expected


def test_nbsp_preserved():
    # NBSP (U+00A0) should NOT be normalized
    assert expand_ligatures("a b") == "a b"


def test_nfkc_arabic_presentation_forms():
    # Arabic Presentation Form-B: FEE1 = MEEM medial, FEF3 = YEH initial.
    # NFKC maps these to base Arabic + reversal restores logical order.
    result = expand_ligatures("ﻡﻳ")
    assert not any(is_arabic_presentation_form(c) for c in result), (
        f"presentation forms should be normalized: {result!r}"
    )
    assert any("؀" <= c <= "ۿ" for c in result), (
        f"should contain base Arabic characters: {result!r}"
    )


def test_no_reversal_for_base_arabic():
    # Base Arabic already in logical order — no presentation forms, no reversal
    text = "مرحبا"  #
    assert expand_ligatures(text) == text


def test_latin_text_unaffected():
    assert expand_ligatures("Hello World") == "Hello World"


def test_reverse_visual_arabic_pure_rtl():
    # Pure RTL: simple reversal
    assert reverse_visual_arabic("با") == "اب"


def test_reverse_visual_arabic_with_ltr_run():
    # Mixed: Arabic + embedded number "123" + Arabic.
    # Visual order runs: [alef], [123], [beh] -> reversed: [beh], [123], [alef]
    assert reverse_visual_arabic("أ123ب") == "ب123أ"


def test_arabic_presentation_form_detection():
    # Presentation Forms-A range
    assert is_arabic_presentation_form("ﭐ")
    assert is_arabic_presentation_form("﷿")
    # Presentation Forms-B range (excludes U+FEFF which is BOM)
    assert is_arabic_presentation_form("ﹰ")
    assert is_arabic_presentation_form("﻾")
    assert not is_arabic_presentation_form("﻿")
    # Base Arabic — NOT a presentation form
    assert not is_arabic_presentation_form("م")
    # Latin
    assert not is_arabic_presentation_form("A")
