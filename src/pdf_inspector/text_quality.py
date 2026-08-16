"""Text-quality detection: deciding when an extracted text layer is too broken
to serve and a page should fall back to OCR.

Extraction can produce plausible-looking bytes that are actually garbage —
failed CID->Unicode mappings, broken ToUnicode CMaps, mojibake. These
detectors catch that and let callers set ``needs_ocr``. They come in two
layers, sharing the same primitives:

- **Markdown-level** (:func:`detect_encoding_issues`, :func:`is_garbage_text`,
  :func:`is_cid_garbage`) run on a page's final markdown string. Used as a
  backstop on the region-extraction and whole-document paths.
- **Item/span-level** (:func:`analyze_text_quality`,
  :func:`region_items_have_decoding_issue`) run on individual ``TextItem``s and
  accumulate per-page evidence, so localized garbled spans on an otherwise
  clean page are caught without a single span having to condemn the page.

Detection classes, roughly by signal:

- **Replacement runs**: U+FFFD clusters (:func:`has_replacement_text_run`).
- **Private-use / C1-control runs**: CID passthrough landing in PUA or the C1
  block (:func:`has_private_use_text_run`, :func:`has_cid_control_token`).
- **Dollar-as-space**: ``Word$Word$Word`` from broken CMaps
  (:func:`has_dollar_as_space_pattern`).
- **Non-alphanumeric dominance**: symbol soup (:func:`is_garbage_text`).
- **Substitution-cipher letter statistics**: pure-ASCII output whose letter
  distribution is a permutation of natural language (:class:`CipherGarbleStats`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from .ocr_reasons import (
    OCR_REASON_SUSPECTED_GARBLED_TEXT,
    add_ocr_reason,
    sorted_pages,
)
from .types import ItemType, TextItem

_ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"
_ASCII_UPPER = _ASCII_LOWER.upper()
_ASCII_LETTERS = _ASCII_LOWER + _ASCII_UPPER
_ASCII_VOWELS = frozenset("aeiou")


def detect_encoding_issues(markdown: str) -> bool:
    """Detect broken font encodings in extracted markdown text.

    Three heuristics:

    1. **U+FFFD**: any replacement character indicates decode failures.
    2. **Dollar-as-space**: pattern like ``Word$Word$Word`` where ``$`` is used
       as a word separator due to broken ToUnicode CMaps. Triggers when either
       more than 50% of ``$`` are between letters (clear substitution pattern),
       or more than 20 letter-dollar-letter occurrences (even if some ``$`` are
       also used as trailing/leading separators, 20+ is far beyond normal
       financial text).
    3. **Substitution-cipher letter statistics** (broken ToUnicode).
    """
    if "�" in markdown:
        return True

    if has_dollar_as_space_pattern(markdown):
        return True

    stats = CipherGarbleStats()
    stats.add_text(markdown)
    return stats.looks_garbled()


def has_dollar_as_space_pattern(markdown: str) -> bool:
    total_dollars = markdown.count("$")
    if total_dollars > 10:
        # Upstream scans UTF-8 bytes. Continuation bytes are never ASCII
        # alphabetic, so byte-wise and char-wise scanning agree; bytes are used
        # here to keep the indices identical to upstream.
        data = markdown.encode("utf-8")
        letter_dollar_letter = 0
        for i in range(1, max(len(data) - 1, 1)):
            if (
                data[i] == 0x24
                and _is_ascii_alpha_byte(data[i - 1])
                and _is_ascii_alpha_byte(data[i + 1])
            ):
                letter_dollar_letter += 1
        if letter_dollar_letter > 20 or letter_dollar_letter * 2 > total_dollars:
            return True

    return False


def _is_ascii_alpha_byte(b: int) -> bool:
    return 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A


#: English letter frequencies (percent, a-z). Used as a natural-language
#: reference: every Latin-script language in the upstream eval corpus (Swedish,
#: Finnish, Turkish, German, romaji) scores >= 0.80 cosine similarity against
#: it, while substitution-cipher text scores ~0.53.
ENGLISH_LETTER_FREQ: tuple[float, ...] = (
    8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.15, 0.8, 4.0, 2.4,
    6.7, 7.5, 1.9, 0.1, 6.0, 6.3, 9.1, 2.8, 1.0, 2.4, 0.15, 2.0, 0.07,
)

_ENGLISH_FREQ_NORM = math.sqrt(sum(f * f for f in ENGLISH_LETTER_FREQ))
_ENGLISH_FREQ_SORTED = tuple(sorted(ENGLISH_LETTER_FREQ, reverse=True))
_ENGLISH_FREQ_SORTED_NORM = math.sqrt(sum(f * f for f in _ENGLISH_FREQ_SORTED))


@dataclass
class CipherGarbleStats:
    """Letter statistics for detecting substitution-cipher garbling.

    Broken ToUnicode CMaps sometimes shift every character by a per-range
    constant (e.g. ``Certificate`` extracted as ``8VceZWZTReV``). Such text is
    100% printable ASCII with word-like token lengths, so it defeats
    :func:`is_garbage_text` and produces no replacement characters — it needs
    its own discriminator.
    """

    #: Case-folded ASCII letter histogram.
    letter_counts: list[int] = field(default_factory=lambda: [0] * 26)
    ascii_letters: int = 0
    ascii_vowels: int = 0
    #: Accented Latin letters (Latin-1 Supplement through Latin Extended-B,
    #: plus Latin Extended Additional). Count toward Latin dominance only.
    latin_ext_letters: int = 0
    non_latin_letters: int = 0
    #: Adjacent ASCII-letter pairs, and how many of them switch from lowercase
    #: straight to uppercase mid-word.
    letter_bigrams: int = 0
    case_shift_bigrams: int = 0

    def add_text(self, text: str) -> None:
        prev: str | None = None
        for ch in text:
            if ch in _ASCII_LETTERS:
                lower = ch.lower()
                self.letter_counts[ord(lower) - 0x61] += 1
                self.ascii_letters += 1
                if lower in _ASCII_VOWELS:
                    self.ascii_vowels += 1
                if prev is not None:
                    self.letter_bigrams += 1
                    if prev in _ASCII_LOWER and ch in _ASCII_UPPER:
                        self.case_shift_bigrams += 1
                prev = ch
            else:
                if ch.isalpha():
                    code = ord(ch)
                    if 0xC0 <= code <= 0x24F or 0x1E00 <= code <= 0x1EFF:
                        self.latin_ext_letters += 1
                    else:
                        self.non_latin_letters += 1
                prev = None

    def english_cosine(self) -> float:
        """Cosine similarity between the observed letter histogram and English
        letter frequencies.

        A shifted alphabet permutes the histogram, which destroys the
        similarity regardless of the shift amount.
        """
        if self.ascii_letters == 0:
            return 1.0
        n = float(self.ascii_letters)
        dot = 0.0
        norm_obs = 0.0
        for count, freq in zip(self.letter_counts, ENGLISH_LETTER_FREQ):
            p = count / n
            dot += p * freq
            norm_obs += p * p
        return dot / (math.sqrt(norm_obs) * _ENGLISH_FREQ_NORM)

    def english_shape_cosine(self) -> float:
        """Cosine similarity between the observed histogram and English
        frequencies after sorting BOTH descending.

        This compares the *shape* of the frequency profile, ignoring which
        letter sits where. A substitution cipher is a bijection, so it
        preserves this shape exactly (att10k 0.97, arbitrary shifts 0.99)
        regardless of case or offset. Non-linguistic ASCII has a different
        profile: a small alphabet is far steeper (random DNA 0.74, hex dumps
        0.81), so the shape diverges.
        """
        if self.ascii_letters == 0:
            return 1.0
        n = float(self.ascii_letters)
        obs = sorted((count / n for count in self.letter_counts), reverse=True)

        dot = sum(o * e for o, e in zip(obs, _ENGLISH_FREQ_SORTED))
        norm_obs = math.sqrt(sum(o * o for o in obs))
        return dot / (norm_obs * _ENGLISH_FREQ_SORTED_NORM)

    def looks_garbled(self) -> bool:
        """Whether the letter statistics look like a substitution cipher.

        Thresholds are upstream's, validated against the 380-document
        pdf-evals snapshot corpus (0 false positives) and the garbled
        ParseBench ``att10k`` page (vowel ratio 0.245, case-shift rate 0.225,
        cosine 0.532). Closest legitimate document on each axis: vowel ratio
        0.264 (circuit schematic), case-shift rate 0.021, cosine 0.801.
        """
        # Need a statistically meaningful, Latin-dominant sample.
        if (
            self.ascii_letters < 200
            or self.non_latin_letters > self.ascii_letters + self.latin_ext_letters
        ):
            return False

        # Real Latin-script text keeps vowels above ~30% of letters even in
        # acronym- and part-number-heavy documents; shifted text starves them.
        vowel_ratio = self.ascii_vowels / self.ascii_letters
        if vowel_ratio > 0.30:
            return False

        # Signal 1: lowercase->uppercase transitions inside words. A shifted
        # lowercase alphabet straddles the ASCII uppercase block ('i'->'Z',
        # 't'->'e'), so garbled words flip case constantly. Real documents stay
        # <= 0.02 even with camelCase identifiers.
        case_shifts = (
            self.letter_bigrams >= 100
            and self.case_shift_bigrams >= self.letter_bigrams * 0.10
        )

        # Signal 2: the histogram is a permutation of natural language — an
        # English-like frequency SHAPE (sorted cosine high) but with letters in
        # the wrong POSITIONS (unsorted cosine low). This is the signature of a
        # substitution cipher and is case-independent, so it catches
        # all-lowercase and all-uppercase shifts as well as case-straddling
        # ones. Genuinely non-linguistic ASCII that is merely "unlike English"
        # fails one of the two halves: DNA/hex dumps have too steep a profile
        # (shape cosine < 0.90), while protein sequences, ticker symbols and
        # base64 are not sufficiently unlike English in position (unsorted
        # cosine >= 0.60) — so none of them are routed to OCR.
        permuted_language = (
            self.english_cosine() < 0.60 and self.english_shape_cosine() >= 0.90
        )

        return case_shifts or permuted_language


@dataclass
class TextQualityReport:
    pages_needing_ocr: list[int] = field(default_factory=list)
    has_encoding_issues: bool = False
    reasons_by_page: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class _PageTextQualityEvidence:
    chars: int = 0
    replacement_chars: int = 0
    replacement_spans: int = 0
    longest_replacement_run: int = 0
    cipher_garble: CipherGarbleStats = field(default_factory=CipherGarbleStats)


class _TextSpanIssueKind(Enum):
    REPLACEMENT = "replacement"
    STRONG = "strong"


def analyze_text_quality(items: Sequence[TextItem]) -> TextQualityReport:
    """Accumulate per-page evidence of broken text and report pages needing OCR."""
    reasons_by_page: dict[int, list[str]] = {}
    evidence_by_page: dict[int, _PageTextQualityEvidence] = {}

    for item in items:
        if item.item_type is not ItemType.TEXT:
            continue

        evidence = evidence_by_page.setdefault(item.page, _PageTextQualityEvidence())
        evidence.chars += sum(1 for ch in item.text if not ch.isspace())
        evidence.cipher_garble.add_text(item.text)

        kind = _text_span_decoding_issue_kind(item.text)
        if kind is _TextSpanIssueKind.STRONG:
            add_ocr_reason(reasons_by_page, item.page, OCR_REASON_SUSPECTED_GARBLED_TEXT)
        elif kind is _TextSpanIssueKind.REPLACEMENT:
            replacement, longest_run = replacement_text_stats(item.text)
            evidence.replacement_chars += replacement
            evidence.replacement_spans += 1
            evidence.longest_replacement_run = max(
                evidence.longest_replacement_run, longest_run
            )

    for page in sorted(evidence_by_page):
        evidence = evidence_by_page[page]
        if page in reasons_by_page:
            continue
        if (
            _page_replacement_evidence_needs_ocr(evidence)
            or evidence.cipher_garble.looks_garbled()
        ):
            add_ocr_reason(reasons_by_page, page, OCR_REASON_SUSPECTED_GARBLED_TEXT)

    pages_needing_ocr = sorted_pages(reasons_by_page)
    return TextQualityReport(
        pages_needing_ocr=pages_needing_ocr,
        has_encoding_issues=bool(pages_needing_ocr),
        reasons_by_page=reasons_by_page,
    )


def region_items_have_decoding_issue(items: Iterable[TextItem]) -> bool:
    return any(
        item.item_type is ItemType.TEXT and _text_span_has_decoding_issue(item.text)
        for item in items
    )


def _text_span_has_decoding_issue(text: str) -> bool:
    return _text_span_decoding_issue_kind(text) is not None


def _text_span_decoding_issue_kind(text: str) -> _TextSpanIssueKind | None:
    text = text.strip()
    if not text:
        return None

    if (
        has_dollar_as_space_pattern(text)
        or has_private_use_text_run(text)
        or is_cid_garbage(text)
        or has_cid_control_token(text)
    ):
        return _TextSpanIssueKind.STRONG

    if has_replacement_text_run(text):
        return _TextSpanIssueKind.REPLACEMENT

    return None


def replacement_text_stats(text: str) -> tuple[int, int]:
    """Return (replacement char count, longest consecutive run)."""
    replacement = 0
    current_run = 0
    longest_run = 0

    for ch in text:
        if ch == "�":
            replacement += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    return replacement, longest_run


def _page_replacement_evidence_needs_ocr(evidence: _PageTextQualityEvidence) -> bool:
    if evidence.replacement_chars == 0 or evidence.chars == 0:
        return False

    # If the entire page is only a short broken text layer, even a short
    # replacement run is enough evidence. On otherwise text-heavy pages, require
    # density so math formulas do not force full-page OCR.
    if evidence.chars <= 80 and evidence.longest_replacement_run >= 2:
        return True

    replacement_density_bps = evidence.replacement_chars * 10_000 // evidence.chars
    enough_bad_text = evidence.replacement_chars >= 12 and replacement_density_bps >= 500
    repeated_bad_spans = evidence.replacement_spans >= 3 and replacement_density_bps >= 250
    long_bad_run = evidence.longest_replacement_run >= 8 and replacement_density_bps >= 250

    return enough_bad_text or repeated_bad_spans or long_bad_run


def has_replacement_text_run(text: str) -> bool:
    replacement, longest_run = replacement_text_stats(text)
    return longest_run >= 2 or replacement >= 3


def has_private_use_text_run(text: str) -> bool:
    total = 0
    private_use = 0
    current_run = 0
    longest_run = 0

    for ch in text:
        if ch.isspace():
            current_run = 0
            continue
        total += 1
        if is_private_use_char(ch):
            private_use += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    if private_use == 0:
        return False

    return longest_run >= 3 or (total >= 5 and private_use >= 2 and private_use * 2 >= total)


def has_cid_control_token(text: str) -> bool:
    return any(_token_has_cid_control(token) for token in text.split())


def _token_has_cid_control(token: str) -> bool:
    total = 0
    c1_control = 0

    for ch in token:
        total += 1
        if "" <= ch <= "":
            c1_control += 1

    return total >= 5 and c1_control >= 2 and c1_control * 20 >= total


def is_private_use_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0xE000 <= code <= 0xF8FF
        or 0xF0000 <= code <= 0xFFFFD
        or 0x100000 <= code <= 0x10FFFD
    )


def is_garbage_text(markdown: str) -> bool:
    """Check if extracted text is predominantly garbage (non-alphanumeric).

    Broken font encodings produce text like ``----1-.-.-.___  --.-. .._ I_---.``
    where most characters are punctuation/symbols. Real text in any language has
    >50% alphanumeric characters.
    """
    alphanum = 0
    non_alphanum = 0

    chars = markdown
    n = len(chars)
    i = 0
    while i < n:
        ch = chars[i]
        run_end = i + 1
        while run_end < n and chars[run_end] == ch:
            run_end += 1

        is_decorative_leader = ch in "._·" and run_end - i >= 3
        if not is_decorative_leader:
            for run_ch in chars[i:run_end]:
                if run_ch.isspace():
                    continue
                # Skip markdown syntax chars that we add (not from the PDF)
                if run_ch in "#*|-\n":
                    continue
                if run_ch.isalnum():
                    alphanum += 1
                else:
                    non_alphanum += 1
        i = run_end

    total = alphanum + non_alphanum
    return total >= 50 and alphanum * 2 < total


def is_cid_garbage(text: str) -> bool:
    """Detect garbage from failed CID-to-Unicode mapping on Identity-H fonts.

    When CID values don't correspond to Unicode codepoints, the raw bytes often
    produce characters in the C1 control range (U+0080-U+009F) or Private Use
    Area, mixed with random Latin Extended characters. Valid text in any
    language almost never contains C1 controls. Falls back to the general
    :func:`is_garbage_text` check for non-alphanumeric-heavy patterns.
    """
    if is_garbage_text(text):
        return True

    total = 0
    c1_control = 0
    high_latin = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        # C1 control characters (U+0080-U+009F) — almost never in real text
        if ch == "·":
            continue
        if "" <= ch <= "":
            c1_control += 1
        # High Latin-1 (U+00A0-U+00FF) — legitimate in Western European text but
        # when combined with ASCII in CID passthrough, indicates mojibake from
        # CID values being misinterpreted as Latin-1 characters.
        if " " <= ch <= "ÿ":
            high_latin += 1

    if total < 5:
        return False
    # If >= 5% of non-whitespace chars are C1 controls, it's garbage
    if c1_control >= 2 and c1_control * 20 >= total:
        return True
    # If >= 40% of non-whitespace chars are high Latin-1 AND the text has few
    # ASCII letters, it's likely CID-as-Latin-1 mojibake (Japanese/CJK PDFs
    # where CID values 0x80-0xFF become accented Latin characters). Keep a
    # minimum length so short math tokens like "2x()x" do not route a clean page
    # to OCR.
    ascii_letters = sum(1 for ch in text if ch in _ASCII_LETTERS)
    return total >= 20 and high_latin * 5 >= total * 2 and ascii_letters * 3 < total
