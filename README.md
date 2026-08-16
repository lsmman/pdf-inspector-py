# pdf-inspector-py

Pure-Python PDF classification and Markdown extraction. A port of
[firecrawl/pdf-inspector](https://github.com/firecrawl/pdf-inspector) — the Rust
library that decides whether a PDF is text-based or scanned and, when it is
text-based, converts it to clean Markdown without OCR.

**Status: work in progress.** The port is being done module by module against
the upstream test corpus. See [Port status](#port-status) for what is finished.

## Why a Python port

The upstream project already ships Python bindings, but they are a compiled
Rust extension: you need a wheel for your platform, or a Rust toolchain to build
one. This port is pure Python, which buys three things:

- **No build step.** `pip install` works anywhere Python runs, including
  locked-down environments and ARM boxes without a prebuilt wheel.
- **It runs in the browser.** Pure Python means [Pyodide](https://pyodide.org)
  can run it client-side — that is what the demo page does, with no server.
- **It is readable.** The classification heuristics are the interesting part of
  the upstream project, and they are easier to study and adapt in Python.

The trade-off is speed. Upstream processes a text-based PDF in under 200 ms;
this port will not match that. If you need the throughput and can install a
compiled wheel, use upstream.

## Port status

| Upstream module | Lines | Python module | Status |
|---|---:|---|---|
| `glyph_names.rs` | 4627 | `glyph_names.py` + `_glyph_names_data.py` | done |
| `adobe_korea1.rs` | 17073 | `adobe_korea1.py` + `_adobe_korea1_data.py` | done |
| `types.rs` | 414 | `types.py` | done |
| `text_utils.rs` | 1257 | `text_utils.py` | done |
| `text_quality.rs` | 520 | `text_quality.py` | done |
| `process_mode.rs` | 11 | `process_mode.py` | done |
| `detector.rs` | 3783 | `detector.py` | done |
| `tounicode.rs` | 3483 | `cmap.py` + `tounicode.py` | done except bundled bcmaps |
| `structure_tree.rs` | 2015 | `structure_tree.py` | pending |
| `extractor/` (6 files) | 13409 | — | pending |
| `markdown/` (3 files) | 6233 | — | pending |
| `tables/` (8 files) | 18414 | — | pending |
| `lib.rs` | 7414 | `__init__.py` (public API) | classification half done |

Upstream is **85,497 lines** of Rust across 25 files. Roughly 22k of that is
generated lookup data, which this port machine-generates rather than retypes;
the rest is logic.

The one deliberate gap so far is upstream's bundled binary CMaps
(`external/bcmaps`), which resolve the Adobe-Japan1, Adobe-GB1 and Adobe-CNS1
CID collections. Adobe-Korea1 does not need them, so Korean CID fonts decode.

The two large data tables are mechanically generated from the Rust sources by
`tools/codegen_tables.py`, so they match upstream entry-for-entry (4532 glyph
names, 17056 Adobe-Korea1 CIDs) rather than being retyped.

## Verification

Correctness is measured against upstream's own corpus, copied verbatim into
`tests/`:

- `tests/fixtures/*.pdf` — 27 PDFs
- `tests/snapshots/*.md` — 7 expected Markdown outputs

Upstream's Rust unit tests are ported to pytest alongside the modules they
cover, so each module lands with its original assertions.

```bash
uv sync
uv run pytest
```

### A note on floating point

Upstream computes coordinates and thresholds in `f32`; Python floats are `f64`.
Comparisons that sit exactly on a threshold can therefore differ. Where this
matters the port keeps upstream's arithmetic shape (including its
divide-by-zero behaviour, which yields `inf`/`NaN` rather than raising) so the
comparisons resolve the same way.

## License

MIT, same as upstream. See [LICENSE](LICENSE) and [NOTICE](NOTICE) — the latter
records exactly which files are derived from upstream and at which commit.
