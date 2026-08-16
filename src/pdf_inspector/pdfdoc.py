"""A thin document layer over ``pypdf``.

Upstream is written against ``lopdf``, whose ``Document`` exposes a handful of
operations the rest of the code leans on: resolve an indirect reference, fetch a
page's content streams, and fetch a page's resource dictionaries in
most-specific-first order. This module provides those same operations on top of
``pypdf`` so the ported modules can read the way the Rust does.

Object identity follows ``lopdf``'s ``ObjectId``: a ``(number, generation)``
pair, used as a dictionary key so the same underlying object reached through two
different resource names compares equal.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from pypdf import PdfReader
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    IndirectObject,
    NameObject,
    StreamObject,
)

from .errors import NotAPdfError, PdfEncryptedError, PdfIoError, PdfParseError


def _decrypt(reader: PdfReader, password: str | None) -> None:
    """Decrypt an encrypted reader in place, or raise :class:`PdfEncryptedError`.

    Mirrors upstream: the supplied password is tried first, then the empty
    password as a fallback — owner-only encryption ("protected" but readable) is
    the common case, and it opens with an empty user password.
    """
    candidates = [password] if password else []
    candidates.append("")

    for candidate in candidates:
        try:
            if reader.decrypt(candidate):
                return
        except Exception:
            continue

    raise PdfEncryptedError()

#: ``(object number, generation)`` — the equivalent of lopdf's ``ObjectId``.
ObjectId = tuple[int, int]


@dataclass(frozen=True)
class PageRef:
    """A page's 1-indexed number together with its object id and dictionary."""

    number: int
    object_id: ObjectId
    dictionary: DictionaryObject


class Document:
    """The subset of ``lopdf::Document`` that the ported modules need."""

    def __init__(self, reader: PdfReader) -> None:
        self._reader = reader
        self._pages: dict[int, PageRef] | None = None

    # ── construction ─────────────────────────────────────────────────

    @classmethod
    def from_bytes(cls, data: bytes, password: str | None = None) -> Document:
        if not data.lstrip()[:5].startswith(b"%PDF-"):
            # Upstream's validate_pdf_bytes rejects non-PDF input up front so a
            # stray HTML error page does not surface as a parse error.
            raise NotAPdfError("invalid PDF file header")
        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
        except Exception as exc:  # pypdf raises a wide variety of types
            raise PdfParseError(str(exc)) from exc

        if reader.is_encrypted:
            _decrypt(reader, password)

        return cls(reader)

    @classmethod
    def from_path(cls, path: str | Path, password: str | None = None) -> Document:
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise PdfIoError(str(exc)) from exc
        return cls.from_bytes(data, password)

    # ── objects ──────────────────────────────────────────────────────

    def resolve(self, obj: Any) -> Any:
        """Follow indirect references until a direct object is reached."""
        seen = 0
        while isinstance(obj, IndirectObject):
            # A malformed file can point a reference at itself; bail rather than
            # spin. Upstream gets this for free from lopdf's resolution cache.
            if seen > 32:
                return None
            try:
                obj = obj.get_object()
            except Exception:
                return None
            seen += 1
        return obj

    def get_dictionary(self, obj: Any) -> DictionaryObject | None:
        """Resolve ``obj`` and return it if it is (or carries) a dictionary."""
        resolved = self.resolve(obj)
        if isinstance(resolved, StreamObject):
            return resolved
        if isinstance(resolved, DictionaryObject):
            return resolved
        return None

    def get_stream(self, obj: Any) -> StreamObject | None:
        resolved = self.resolve(obj)
        return resolved if isinstance(resolved, StreamObject) else None

    @staticmethod
    def object_id(obj: Any) -> ObjectId | None:
        """The ``(number, generation)`` id of an indirect reference, if it has one."""
        if isinstance(obj, IndirectObject):
            return (obj.idnum, obj.generation)
        ref = getattr(obj, "indirect_reference", None)
        if isinstance(ref, IndirectObject):
            return (ref.idnum, ref.generation)
        return None

    @staticmethod
    def name_of(obj: Any) -> str | None:
        """Return a ``/Name`` object's text without the leading slash."""
        if isinstance(obj, NameObject):
            return str(obj).lstrip("/")
        return None

    # ── pages ────────────────────────────────────────────────────────

    @property
    def page_count(self) -> int:
        return len(self.get_pages())

    def get_pages(self) -> dict[int, PageRef]:
        """1-indexed page number -> :class:`PageRef`, matching ``get_pages``."""
        if self._pages is None:
            pages: dict[int, PageRef] = {}
            try:
                raw_pages = self._reader.pages
            except Exception:
                raw_pages = []
            for index, page in enumerate(raw_pages, start=1):
                object_id = self.object_id(page)
                if object_id is None:
                    # Synthesise an id for pages that arrive without a reference
                    # so they still key the analysis caches distinctly.
                    object_id = (-index, 0)
                pages[index] = PageRef(index, object_id, page)
            self._pages = pages
        return self._pages

    def get_page_contents(self, page: PageRef) -> list[bytes]:
        """Decoded bytes of each content stream attached to a page."""
        contents = page.dictionary.get("/Contents")
        streams: list[bytes] = []
        resolved = self.resolve(contents)
        candidates: list[Any]
        if isinstance(resolved, ArrayObject):
            candidates = list(resolved)
        elif resolved is None:
            candidates = []
        else:
            candidates = [resolved]

        for candidate in candidates:
            stream = self.get_stream(candidate)
            if stream is None:
                continue
            streams.append(stream_bytes(stream))
        return streams

    def get_page_resources(
        self, page: PageRef
    ) -> tuple[DictionaryObject | None, list[ObjectId]]:
        """Resource dictionaries for a page, most-specific first.

        Mirrors ``lopdf``'s ``get_page_resources``: the first element is the
        page's own inline ``/Resources`` dictionary when it is written directly
        rather than as a reference, and the second is the object ids of every
        ``/Resources`` reference found on the page and then up the ``/Parent``
        chain. PDF 32000-1 7.7.3.4 makes a nearer definition shadow a further
        one, which is why the order matters.
        """
        own: DictionaryObject | None = None
        ancestors: list[ObjectId] = []
        seen_nodes: set[ObjectId] = set()

        node: Any = page.dictionary
        depth = 0
        while isinstance(node, DictionaryObject) and depth < 64:
            depth += 1
            resources = node.raw_get("/Resources") if "/Resources" in node else None
            if resources is not None:
                if isinstance(resources, IndirectObject):
                    object_id = (resources.idnum, resources.generation)
                    if object_id not in ancestors:
                        ancestors.append(object_id)
                elif isinstance(resources, DictionaryObject) and own is None:
                    own = resources

            parent = node.get("/Parent")
            parent_id = self.object_id(parent)
            if parent_id is not None:
                if parent_id in seen_nodes:
                    break
                seen_nodes.add(parent_id)
            node = self.resolve(parent)

        return own, ancestors

    def resource_dictionaries(self, page: PageRef) -> Iterator[DictionaryObject]:
        """Every resource dictionary for a page, most-specific first."""
        own, ancestor_ids = self.get_page_resources(page)
        if own is not None:
            yield own
        for object_id in ancestor_ids:
            dictionary = self.get_dictionary(self._indirect(object_id))
            if dictionary is not None:
                yield dictionary

    def _indirect(self, object_id: ObjectId) -> IndirectObject:
        return IndirectObject(object_id[0], object_id[1], self._reader)

    def dictionary_for_id(self, object_id: ObjectId) -> DictionaryObject | None:
        return self.get_dictionary(self._indirect(object_id))

    # ── trailer ──────────────────────────────────────────────────────

    @property
    def trailer(self) -> DictionaryObject:
        return self._reader.trailer

    @property
    def reader(self) -> PdfReader:
        return self._reader


def stream_bytes(stream: StreamObject) -> bytes:
    """Decoded stream contents, falling back to the raw bytes.

    Upstream does the same: ``decompressed_content()`` with the raw
    ``stream.content`` as the fallback, so a stream with an unsupported filter
    still contributes whatever plain text it happens to carry.
    """
    try:
        data = stream.get_data()
    except Exception:
        data = getattr(stream, "_data", b"")
    if isinstance(data, str):
        return data.encode("latin-1", errors="replace")
    return bytes(data or b"")
