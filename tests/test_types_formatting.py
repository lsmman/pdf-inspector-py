"""Ported from the ``formatting_tests`` module in upstream src/types.rs."""

from __future__ import annotations

from pdf_inspector.types import ItemType, TextItem, TextLine


def item(text: str, x: float, width: float, strikeout: bool) -> TextItem:
    return TextItem(
        text=text,
        x=x,
        y=100.0,
        width=width,
        height=12.0,
        font="F1",
        font_size=12.0,
        page=1,
        is_strikeout=strikeout,
        item_type=ItemType.TEXT,
    )


def line(items: list[TextItem]) -> TextLine:
    return TextLine(items=items, y=100.0, page=1, adaptive_threshold=0.1)


def test_formatting_emits_semantic_strikeout():
    subject = line([item("deleted", 10.0, 42.0, True)])
    assert subject.text_with_formatting(True, True, True) == "<s>deleted</s>"


def test_formatting_closes_strikeout_before_live_text():
    subject = line(
        [
            item("keep", 10.0, 24.0, False),
            item("remove", 40.0, 42.0, True),
            item("keep", 88.0, 24.0, False),
        ]
    )
    assert subject.text_with_formatting(True, True, True) == "keep <s>remove</s> keep"


def test_formatting_coalesces_adjacent_struck_items():
    subject = line(
        [
            item("deleted", 10.0, 42.0, True),
            item("words", 58.0, 30.0, True),
        ]
    )
    assert subject.text_with_formatting(True, True, True) == "<s>deleted words</s>"


def test_strikeout_takes_precedence_over_other_styles():
    decorated = item("deleted", 10.0, 42.0, True)
    decorated.is_bold = True
    decorated.is_italic = True
    decorated.is_underline = True
    subject = line([decorated])

    assert subject.text_with_formatting(True, True, True) == "<s>deleted</s>"
    assert subject.text() == "deleted"
