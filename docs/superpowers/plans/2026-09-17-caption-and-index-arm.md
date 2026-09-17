# DocLens Caption-and-Index Retrieval Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first multimodal retrieval arm: a Gemini vision model captions each page's
tables, diagrams, and figures at ingest time, and those captions are indexed as ordinary text
passages alongside the existing extracted-text chunks. This is the cheaper of the two planned
multimodal arms (no new embedding model, reuses the text-only pipeline's BM25/dense/rerank
machinery), and it gets compared against the text-only baseline on the same real benchmark
questions.

**Architecture:** A `CaptionCache` persists one JSON file per document (`corpus/<doc_id>.captions.json`)
mapping page number to caption text or `null` (page has no meaningful visual content), so the
(slow, rate-limited) Gemini calls only ever run once per page even across repeated index
rebuilds. A lazily-loaded `PageCaptioner` wraps the Gemini vision call; its sentinel-parsing
logic is a pure function, independently testable without any API call. `generate_caption_chunks`
captions a single document's uncached pages and returns `Chunk` objects (reusing the existing
`Chunk` model from Milestone 2, with a new `source_type` field distinguishing `"caption"` from
`"extracted_text"`); `load_all_cached_caption_chunks` is the read-only counterpart used at index-
build and query time, which never calls Gemini. `TextOnlyRetriever` (from Milestone 2) gains an
`include_captions` flag so the SAME retrieval code serves both arms, distinguished only by which
index directory it points at and whether captions are folded in.

**Tech Stack:** Python 3.11+ (existing), `google-generativeai` (existing, from Milestone 2),
`pillow` (new, to open page images for the vision API call).

## Global Constraints

- Never use the em dash character (`—`, U+2014) anywhere: code, comments, docstrings, commit
  messages, reports.
- All file paths in code are `pathlib.Path`, never raw strings (Windows dev environment).
- `PageCaptioner` must load its model lazily on first real use, not at construction time,
  matching the existing `OcrFallback`/`TextEmbedder`/`GeminiGenerator` pattern.
- Tests must never require network access, an API key, or a real model download to pass under
  `pytest -m "not slow"`. Any test that needs a real Gemini call is marked `@pytest.mark.slow`
  and skips cleanly (not fails) when `GEMINI_API_KEY` is unset, matching the existing
  `test_generator.py` convention.
- Captioning must be idempotent and resumable: a page already present in the cache (whether its
  cached value is a real caption or `null`) must never trigger another Gemini call.
- This plan does not modify Milestone 1/2's already-shipped behavior when `include_captions` is
  left at its default (`False`) or a corpus has no caption cache; the text-only arm must keep
  working exactly as it does today.

---

## Input scope for the real captioning run

`data/caption_scope.json` (already created) lists the 15 document `doc_id`s independently
confirmed during benchmark curation to contain genuine tables or diagrams (1,058 pages total
across those 15 documents, out of 2,634 pages in the full corpus). The real captioning run in
Task 5 targets this scope, not the full corpus, to keep the first real run inside Gemini's
free-tier rate/daily limits. Pages outside this scope simply have no caption cache entries; the
caption-and-index arm gracefully falls back to text-only behavior for those pages, which is
expected and not a bug.

---

## File Structure

```
DocLens/
  src/doclens/
    retrieval/
      models.py             # Task 1: Chunk gains source_type field
      pipeline.py            # Task 4: TextOnlyRetriever gains include_captions; build_caption_index()
    caption/
      __init__.py
      cache.py               # Task 1: CaptionCache
      captioner.py            # Task 2: PageCaptioner, _parse_caption_response
      chunker.py              # Task 3: generate_caption_chunks, load_all_cached_caption_chunks
    cli.py                   # Task 4: `doclens caption-corpus`, `doclens build-caption-index`
  tests/
    retrieval/
      test_models.py         # Task 1 (extended)
      test_pipeline.py       # Task 4 (extended)
    caption/
      test_cache.py           # Task 1
      test_captioner.py       # Task 2
      test_chunker.py         # Task 3
```

---

### Task 1: Chunk.source_type field + CaptionCache

**Files:**
- Modify: `src/doclens/retrieval/models.py`
- Create: `src/doclens/caption/__init__.py` (empty)
- Create: `src/doclens/caption/cache.py`
- Test: `tests/retrieval/test_models.py` (extend with one new test)
- Test: `tests/caption/test_cache.py`

**Interfaces:**
- Produces: `Chunk.source_type: Literal["extracted_text", "caption"] = "extracted_text"` added
  to the existing `Chunk` model (default preserves every existing caller's behavior unchanged);
  `CaptionCache(corpus_dir: Path)` in `doclens.caption.cache` with `.has(doc_id: str,
  page_number: int) -> bool` (True if this page has ever been processed, regardless of whether
  the cached value is a real caption or `None`), `.get(doc_id: str, page_number: int) -> str |
  None` (returns the cached caption, or `None` if the page was processed and found to have no
  visual content; behavior is undefined if `.has()` would return `False`, callers must check
  `.has()` first), and `.set(doc_id: str, page_number: int, caption: str | None) -> None`
  (persists immediately to `corpus_dir / f"{doc_id}.captions.json"`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/retrieval/test_models.py (append this test to the existing file)
from doclens.retrieval.models import Chunk


def test_chunk_source_type_defaults_to_extracted_text():
    chunk = Chunk(chunk_id="c1", doc_id="doc1", doc_title="Manual", page_number=1,
                  chunk_index=0, text="some text", source_url="https://example.com/a.pdf")
    assert chunk.source_type == "extracted_text"


def test_chunk_source_type_accepts_caption():
    chunk = Chunk(chunk_id="c1:caption", doc_id="doc1", doc_title="Manual", page_number=1,
                  chunk_index=0, text="a table showing X", source_url="https://example.com/a.pdf",
                  source_type="caption")
    assert chunk.source_type == "caption"
```

```python
# tests/caption/test_cache.py
from pathlib import Path

from doclens.caption.cache import CaptionCache


def test_has_returns_false_for_never_processed_page(tmp_path: Path):
    cache = CaptionCache(corpus_dir=tmp_path)
    assert cache.has("doc1", 5) is False


def test_set_and_get_a_real_caption(tmp_path: Path):
    cache = CaptionCache(corpus_dir=tmp_path)
    cache.set("doc1", 5, "A table showing torque values by bolt size.")

    assert cache.has("doc1", 5) is True
    assert cache.get("doc1", 5) == "A table showing torque values by bolt size."


def test_set_and_get_a_no_visual_content_page(tmp_path: Path):
    cache = CaptionCache(corpus_dir=tmp_path)
    cache.set("doc1", 3, None)

    assert cache.has("doc1", 3) is True
    assert cache.get("doc1", 3) is None


def test_cache_persists_across_instances(tmp_path: Path):
    cache_a = CaptionCache(corpus_dir=tmp_path)
    cache_a.set("doc1", 5, "A caption.")

    cache_b = CaptionCache(corpus_dir=tmp_path)
    assert cache_b.has("doc1", 5) is True
    assert cache_b.get("doc1", 5) == "A caption."


def test_cache_keeps_documents_separate(tmp_path: Path):
    cache = CaptionCache(corpus_dir=tmp_path)
    cache.set("doc1", 5, "Doc1 caption.")
    cache.set("doc2", 5, "Doc2 caption.")

    assert cache.get("doc1", 5) == "Doc1 caption."
    assert cache.get("doc2", 5) == "Doc2 caption."


def test_cache_accumulates_multiple_pages_for_same_document(tmp_path: Path):
    cache = CaptionCache(corpus_dir=tmp_path)
    cache.set("doc1", 5, "Page 5 caption.")
    cache.set("doc1", 8, "Page 8 caption.")

    assert cache.get("doc1", 5) == "Page 5 caption."
    assert cache.get("doc1", 8) == "Page 8 caption."
    assert cache.has("doc1", 3) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/retrieval/test_models.py tests/caption/test_cache.py -v`
Expected: FAIL. The two new `test_models.py` tests fail because `source_type` is not yet a valid
`Chunk` field; the `test_cache.py` tests fail with
`ModuleNotFoundError: No module named 'doclens.caption'`.

- [ ] **Step 3: Write minimal implementation**

Modify `src/doclens/retrieval/models.py`: add `from typing import Literal` to the imports if not
already present, and add the new field to `Chunk`:

```python
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    page_number: int
    chunk_index: int
    text: str
    source_url: str
    source_type: Literal["extracted_text", "caption"] = "extracted_text"
```

```python
# src/doclens/caption/cache.py
import json
from pathlib import Path


class CaptionCache:
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir

    def _cache_path(self, doc_id: str) -> Path:
        return self.corpus_dir / f"{doc_id}.captions.json"

    def _load(self, doc_id: str) -> dict:
        path = self._cache_path(doc_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def has(self, doc_id: str, page_number: int) -> bool:
        return str(page_number) in self._load(doc_id)

    def get(self, doc_id: str, page_number: int) -> str | None:
        return self._load(doc_id).get(str(page_number))

    def set(self, doc_id: str, page_number: int, caption: str | None) -> None:
        path = self._cache_path(doc_id)
        data = self._load(doc_id)
        data[str(page_number)] = caption
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/retrieval/test_models.py tests/caption/test_cache.py -v`
Expected: PASS (2 new model tests + 6 cache tests, plus all pre-existing `test_models.py` tests
still passing)

- [ ] **Step 5: Commit**

```bash
git add src/doclens/retrieval/models.py src/doclens/caption/__init__.py \
  src/doclens/caption/cache.py tests/retrieval/test_models.py tests/caption/test_cache.py
git commit -m "feat(caption): add Chunk source_type field and per-document caption cache"
```

---

### Task 2: Page captioner (lazy Gemini vision client)

**Files:**
- Create: `src/doclens/caption/captioner.py`
- Test: `tests/caption/test_captioner.py`

**Interfaces:**
- Produces: `NO_VISUAL_CONTENT_SENTINEL` (module-level constant string), `_parse_caption_response(text:
  str) -> str | None` (pure function: returns `None` if the stripped text equals the sentinel,
  otherwise returns the stripped text; this is what makes the sentinel-handling logic testable
  without any API call), `PageCaptioner(model_name: str = "gemini-2.0-flash", api_key: str |
  None = None)` in `doclens.caption.captioner` with `.caption_page(image_path: Path) -> str |
  None`. Constructing `PageCaptioner()` must NOT load the model or configure the API. Calling
  `.caption_page()` without an API key available raises `RuntimeError` (matches the
  `GeminiGenerator` pattern already in the codebase).

- [ ] **Step 1: Write the failing tests**

```python
# tests/caption/test_captioner.py
import os
from pathlib import Path

import pytest

from doclens.caption.captioner import (
    NO_VISUAL_CONTENT_SENTINEL,
    PageCaptioner,
    _parse_caption_response,
)


def test_parse_caption_response_returns_none_for_sentinel():
    assert _parse_caption_response(NO_VISUAL_CONTENT_SENTINEL) is None


def test_parse_caption_response_strips_whitespace_around_sentinel():
    assert _parse_caption_response(f"  {NO_VISUAL_CONTENT_SENTINEL}  \n") is None


def test_parse_caption_response_returns_stripped_text_for_real_caption():
    result = _parse_caption_response("  A table showing torque values.  ")
    assert result == "A table showing torque values."


def test_captioner_does_not_load_model_on_init():
    captioner = PageCaptioner(api_key="unused-for-this-test")
    assert captioner._model is None


def test_captioner_without_api_key_raises_on_caption_page(tmp_path: Path):
    image_path = tmp_path / "page-1.png"
    image_path.write_bytes(b"not a real png, never reached")
    captioner = PageCaptioner(api_key=None)
    with pytest.raises(RuntimeError):
        captioner.caption_page(image_path)


@pytest.mark.slow
def test_captioner_produces_a_real_caption_for_a_real_table_image():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set; skipping live Gemini call")

    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Table 1: Torque Values")
    page.insert_text((72, 100), "Bolt Size    Torque (Nm)")
    page.insert_text((72, 120), "M8           25")
    page.insert_text((72, 140), "M10          45")
    image_path = Path("test_caption_table.png")
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    pixmap.save(str(image_path))
    doc.close()

    try:
        captioner = PageCaptioner(api_key=api_key)
        result = captioner.caption_page(image_path)
        assert result is not None
        assert "torque" in result.lower() or "45" in result or "25" in result
    finally:
        image_path.unlink(missing_ok=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/caption/test_captioner.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.caption.captioner'`

- [ ] **Step 3: Write minimal implementation**

Install `pillow` with `pip install pillow` and add `pillow>=10.0` to `pyproject.toml`'s
dependencies list (check the existing list first, add to it, don't overwrite other entries).

```python
# src/doclens/caption/captioner.py
import os
from pathlib import Path

NO_VISUAL_CONTENT_SENTINEL = "NO_VISUAL_CONTENT"

CAPTION_PROMPT = (
    "Look at this page from a technical manual. If it contains a data table, "
    "diagram, wiring schematic, exploded parts view, or chart, describe it in "
    "detail: for tables, list the column headers and a representative sample "
    "of values; for diagrams, describe labeled parts, callouts, or "
    "connections. Be specific about numbers, labels, and units. If this page "
    "is plain text or prose with no meaningful table, diagram, or figure, "
    f"respond with exactly the single word {NO_VISUAL_CONTENT_SENTINEL} and "
    "nothing else."
)


def _parse_caption_response(text: str) -> str | None:
    stripped = text.strip()
    if stripped == NO_VISUAL_CONTENT_SENTINEL:
        return None
    return stripped


class PageCaptioner:
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)

    def caption_page(self, image_path: Path) -> str | None:
        self._ensure_loaded()
        import PIL.Image

        image = PIL.Image.open(image_path)
        response = self._model.generate_content([CAPTION_PROMPT, image])
        return _parse_caption_response(response.text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/caption/test_captioner.py -v -m "not slow"`
Expected: PASS (5 passed, 1 deselected)

Since a `GEMINI_API_KEY` is available in this environment, also run the slow test to confirm a
real caption is produced: `pytest tests/caption/test_captioner.py -v` (no marker filter). Report
the actual caption text returned in the task report; it should reference the table's content.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/doclens/caption/captioner.py tests/caption/test_captioner.py
git commit -m "feat(caption): add lazy Gemini vision page captioner"
```

---

### Task 3: Caption chunk generation

**Files:**
- Create: `src/doclens/caption/chunker.py`
- Test: `tests/caption/test_chunker.py`

**Interfaces:**
- Consumes: `doclens.ingest.models.Document`; `doclens.ingest.corpus.load_documents`;
  `doclens.caption.cache.CaptionCache`; `doclens.caption.captioner.PageCaptioner`;
  `doclens.retrieval.models.Chunk`
- Produces: `generate_caption_chunks(document: Document, cache: CaptionCache, captioner:
  PageCaptioner | None = None) -> list[Chunk]` in `doclens.caption.chunker`, which for every
  page in `document`: if the page is not yet in `cache` (`cache.has(...)` is `False`), calls
  `captioner.caption_page(page.image_path)` (constructing a default `PageCaptioner()` if none
  was injected) and stores the result via `cache.set(...)`; if the page IS already cached, reads
  the cached value directly without calling the captioner. Returns one `Chunk` (`source_type=
  "caption"`, `chunk_id=f"{doc_id[:16]}:{page_number}:caption"`) per page whose caption
  (fresh or cached) is not `None`; pages with no visual content produce no chunk. Also produces
  `load_all_cached_caption_chunks(corpus_dir: Path, cache: CaptionCache) -> list[Chunk]`, the
  read-only counterpart used at index-build and query time: iterates every document in
  `corpus_dir` via `load_documents`, and for every page already present in `cache` with a
  non-`None` cached caption, produces the same shape of `Chunk`. This function NEVER calls
  `PageCaptioner` (no API access, safe to call from a retriever's constructor on every query).

- [ ] **Step 1: Write the failing tests**

```python
# tests/caption/test_chunker.py
from datetime import datetime, timezone
from pathlib import Path

from doclens.caption.cache import CaptionCache
from doclens.caption.chunker import generate_caption_chunks, load_all_cached_caption_chunks
from doclens.ingest.models import Document, License, Page, SourceRecord


def _document(doc_id: str, title: str, page_count: int) -> Document:
    source = SourceRecord(
        source_url=f"https://example.com/{doc_id}.pdf",
        license=License.PUBLIC_DOMAIN,
        retrieved_at=datetime(2026, 9, 17, tzinfo=timezone.utc),
        sha256=doc_id,
    )
    pages = [
        Page(page_number=i + 1, text=f"page {i + 1}", text_source="extracted",
             image_path=Path(f"corpus/{doc_id}/page-{i + 1}.png"))
        for i in range(page_count)
    ]
    return Document(doc_id=doc_id, title=title, source=source, pages=pages)


class _StubCaptioner:
    def __init__(self, captions: dict[int, str | None]):
        self.captions = captions
        self.call_count = 0

    def caption_page(self, image_path: Path) -> str | None:
        self.call_count += 1
        # image_path looks like corpus/<doc_id>/page-<N>.png
        page_number = int(image_path.stem.split("-")[1])
        return self.captions[page_number]


def test_generate_caption_chunks_produces_a_chunk_only_for_pages_with_content(tmp_path: Path):
    document = _document("a" * 64, "Manual A", page_count=3)
    cache = CaptionCache(corpus_dir=tmp_path)
    captioner = _StubCaptioner({1: "A torque table.", 2: None, 3: "A wiring diagram."})

    chunks = generate_caption_chunks(document, cache, captioner)

    assert len(chunks) == 2
    assert {c.page_number for c in chunks} == {1, 3}
    assert all(c.source_type == "caption" for c in chunks)


def test_generate_caption_chunks_persists_to_cache(tmp_path: Path):
    document = _document("a" * 64, "Manual A", page_count=1)
    cache = CaptionCache(corpus_dir=tmp_path)
    captioner = _StubCaptioner({1: "A torque table."})

    generate_caption_chunks(document, cache, captioner)

    assert cache.has("a" * 64, 1) is True
    assert cache.get("a" * 64, 1) == "A torque table."


def test_generate_caption_chunks_skips_already_cached_pages(tmp_path: Path):
    document = _document("a" * 64, "Manual A", page_count=1)
    cache = CaptionCache(corpus_dir=tmp_path)
    cache.set("a" * 64, 1, "Already cached caption.")
    captioner = _StubCaptioner({})  # would KeyError if called; proves it wasn't

    chunks = generate_caption_chunks(document, cache, captioner)

    assert captioner.call_count == 0
    assert len(chunks) == 1
    assert chunks[0].text == "Already cached caption."


def test_load_all_cached_caption_chunks_reads_only_from_cache(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    document = _document("a" * 64, "Manual A", page_count=2)
    (corpus_dir / f"{document.doc_id}.json").write_text(
        document.model_dump_json(indent=2), encoding="utf-8"
    )
    cache = CaptionCache(corpus_dir=corpus_dir)
    cache.set("a" * 64, 1, "A torque table.")
    cache.set("a" * 64, 2, None)

    chunks = load_all_cached_caption_chunks(corpus_dir, cache)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].text == "A torque table."


def test_load_all_cached_caption_chunks_returns_empty_for_uncached_corpus(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    document = _document("a" * 64, "Manual A", page_count=2)
    (corpus_dir / f"{document.doc_id}.json").write_text(
        document.model_dump_json(indent=2), encoding="utf-8"
    )
    cache = CaptionCache(corpus_dir=corpus_dir)

    chunks = load_all_cached_caption_chunks(corpus_dir, cache)

    assert chunks == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/caption/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.caption.chunker'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/caption/chunker.py
from pathlib import Path

from doclens.caption.cache import CaptionCache
from doclens.caption.captioner import PageCaptioner
from doclens.ingest.corpus import load_documents
from doclens.ingest.models import Document
from doclens.retrieval.models import Chunk


def _caption_chunk(document: Document, page_number: int, caption: str) -> Chunk:
    return Chunk(
        chunk_id=f"{document.doc_id[:16]}:{page_number}:caption",
        doc_id=document.doc_id,
        doc_title=document.title,
        page_number=page_number,
        chunk_index=0,
        text=caption,
        source_url=document.source.source_url,
        source_type="caption",
    )


def generate_caption_chunks(
    document: Document, cache: CaptionCache, captioner: PageCaptioner | None = None
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in document.pages:
        if cache.has(document.doc_id, page.page_number):
            caption = cache.get(document.doc_id, page.page_number)
        else:
            resolved_captioner = captioner or PageCaptioner()
            caption = resolved_captioner.caption_page(page.image_path)
            cache.set(document.doc_id, page.page_number, caption)

        if caption:
            chunks.append(_caption_chunk(document, page.page_number, caption))
    return chunks


def load_all_cached_caption_chunks(corpus_dir: Path, cache: CaptionCache) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_documents(corpus_dir):
        for page in document.pages:
            if cache.has(document.doc_id, page.page_number):
                caption = cache.get(document.doc_id, page.page_number)
                if caption:
                    chunks.append(_caption_chunk(document, page.page_number, caption))
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/caption/test_chunker.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/doclens/caption/chunker.py tests/caption/test_chunker.py
git commit -m "feat(caption): generate and load caption chunks with cache-first resolution"
```

---

### Task 4: CLI commands + retriever/index integration

**Files:**
- Modify: `src/doclens/retrieval/pipeline.py` (add `include_captions` to `TextOnlyRetriever`;
  add `build_caption_index()`)
- Modify: `src/doclens/cli.py` (add `caption-corpus` and `build-caption-index` commands)
- Test: `tests/retrieval/test_pipeline.py` (extend with new tests)

**Interfaces:**
- Consumes: `doclens.caption.cache.CaptionCache`, `doclens.caption.chunker.generate_caption_chunks`,
  `doclens.caption.chunker.load_all_cached_caption_chunks`, `doclens.caption.captioner.PageCaptioner`
- Produces: `TextOnlyRetriever.__init__` gains `include_captions: bool = False`; when `True`,
  the retriever's chunk pool (used both for the in-memory BM25 index and for resolving
  `chunk_id` back to a full `Chunk` for reranking/citation) additionally includes every chunk
  from `load_all_cached_caption_chunks(corpus_dir, CaptionCache(corpus_dir))`, alongside the
  existing extracted-text chunks. `build_caption_index(corpus_dir: Path, index_dir: Path,
  embedder: TextEmbedder | None = None, vector_store: VectorStore | None = None) -> int` in
  `doclens.retrieval.pipeline`, mirroring `build_index()` but embedding and indexing BOTH text
  chunks and cached caption chunks together, returning the total chunk count indexed. CLI:
  `doclens caption-corpus --corpus-dir corpus --scope-file <path>` (scope-file is a JSON array
  of doc_ids to caption; every page of every listed document gets captioned, cache-first, so
  re-running after an interruption only processes the pages not yet done) and `doclens
  build-caption-index --corpus-dir corpus --index-dir <dir>` (builds the combined text+caption
  index). The existing `query` command gains a `--with-captions` flag that, when passed,
  constructs `TextOnlyRetriever(..., include_captions=True)` instead of the default.

- [ ] **Step 1: Write the failing tests**

```python
# tests/retrieval/test_pipeline.py (append these to the existing file)
from doclens.caption.cache import CaptionCache
from doclens.retrieval.pipeline import build_caption_index


def test_retriever_with_include_captions_returns_a_caption_chunk_from_retrieve(tmp_path: Path):
    # Page text is deliberately unrelated to torque, so a BM25/dense match on "torque" can only
    # come from the caption chunk, proving retrieve() actually surfaces it, not just loads it.
    corpus_dir = tmp_path / "corpus"
    _write_document(corpus_dir, "g" * 64, "Manual G",
                     ["This section covers general safety precautions and warnings."])

    cache = CaptionCache(corpus_dir=corpus_dir)
    cache.set("g" * 64, 1, "Table showing torque values: M8 is 25 Nm, M10 is 45 Nm.")

    index_dir = tmp_path / "index"
    build_caption_index(corpus_dir, index_dir, embedder=_StubEmbedder(),
                         vector_store=VectorStore(path=index_dir, vector_size=4))

    retriever = TextOnlyRetriever(
        corpus_dir=corpus_dir, index_dir=index_dir,
        embedder=_StubEmbedder(), vector_store=VectorStore(path=index_dir, vector_size=4),
        reranker=None, include_captions=True,
    )
    results = retriever.retrieve("what is the M8 torque value", top_k=5)

    assert any(chunk.source_type == "caption" for chunk, _score in results)


def test_retriever_without_include_captions_ignores_caption_cache(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _write_document(corpus_dir, "h" * 64, "Manual H", ["Some ordinary text."])

    cache = CaptionCache(corpus_dir=corpus_dir)
    cache.set("h" * 64, 1, "A caption that should be ignored by default.")

    index_dir = tmp_path / "index"
    build_index(corpus_dir, index_dir, embedder=_StubEmbedder(),
                vector_store=VectorStore(path=index_dir, vector_size=4))

    retriever = TextOnlyRetriever(
        corpus_dir=corpus_dir, index_dir=index_dir,
        embedder=_StubEmbedder(), vector_store=VectorStore(path=index_dir, vector_size=4),
        reranker=None,
    )

    assert all(chunk.source_type == "extracted_text" for chunk in retriever.chunks)


def test_build_caption_index_includes_both_text_and_caption_chunks(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _write_document(corpus_dir, "i" * 64, "Manual I", ["Some ordinary text about installation."])

    cache = CaptionCache(corpus_dir=corpus_dir)
    cache.set("i" * 64, 1, "A torque table caption.")

    index_dir = tmp_path / "index"
    count = build_caption_index(corpus_dir, index_dir, embedder=_StubEmbedder(),
                                 vector_store=VectorStore(path=index_dir, vector_size=4))

    assert count == 2  # one extracted_text chunk, one caption chunk
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/retrieval/test_pipeline.py -v`
Expected: FAIL. The `include_captions`/`build_caption_index` tests fail with `TypeError`
(unexpected keyword argument) or `ImportError`, since these do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Modify `src/doclens/retrieval/pipeline.py`: add the import and update `TextOnlyRetriever.__init__`
and add `build_caption_index`:

```python
from doclens.caption.cache import CaptionCache
from doclens.caption.chunker import load_all_cached_caption_chunks
```

```python
class TextOnlyRetriever:
    def __init__(
        self,
        corpus_dir: Path,
        index_dir: Path,
        embedder: TextEmbedder | None = None,
        vector_store: VectorStore | None = None,
        reranker: CrossEncoderReranker | None = _UNSET,
        include_captions: bool = False,
    ):
        self.chunks = _load_all_chunks(corpus_dir)
        if include_captions:
            cache = CaptionCache(corpus_dir=corpus_dir)
            self.chunks += load_all_cached_caption_chunks(corpus_dir, cache)
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

        self.bm25 = Bm25Index()
        self.bm25.build(self.chunks)

        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or VectorStore(path=index_dir)
        self.reranker = CrossEncoderReranker() if reranker is _UNSET else reranker

    # retrieve() is unchanged
```

```python
def build_caption_index(
    corpus_dir: Path,
    index_dir: Path,
    embedder: TextEmbedder | None = None,
    vector_store: VectorStore | None = None,
) -> int:
    text_chunks = _load_all_chunks(corpus_dir)
    cache = CaptionCache(corpus_dir=corpus_dir)
    caption_chunks = load_all_cached_caption_chunks(corpus_dir, cache)
    all_chunks = text_chunks + caption_chunks

    embedder = embedder or TextEmbedder()
    vector_store = vector_store or VectorStore(path=index_dir)

    embeddings = embedder.embed_passages([chunk.text for chunk in all_chunks])
    vector_store.build([chunk.chunk_id for chunk in all_chunks], embeddings)
    return len(all_chunks)
```

Append to `src/doclens/cli.py`:

```python
from doclens.caption.cache import CaptionCache
from doclens.caption.chunker import generate_caption_chunks
from doclens.ingest.corpus import load_documents
from doclens.retrieval.pipeline import build_caption_index


@app.command(name="caption-corpus")
def caption_corpus(
    corpus_dir: Path = typer.Option(Path("corpus"), help="Ingested corpus directory"),
    scope_file: Path = typer.Option(..., help="JSON array of doc_ids to caption"),
) -> None:
    scope_doc_ids = set(json.loads(scope_file.read_text(encoding="utf-8")))
    cache = CaptionCache(corpus_dir=corpus_dir)

    documents = [doc for doc in load_documents(corpus_dir) if doc.doc_id in scope_doc_ids]
    total_captioned = 0
    for document in documents:
        chunks = generate_caption_chunks(document, cache)
        typer.echo(f"{document.title}: {len(chunks)} pages with captions")
        total_captioned += len(chunks)

    typer.echo(f"Done. {total_captioned} caption chunks available across {len(documents)} documents.")


@app.command(name="build-caption-index")
def build_caption_index_command(
    corpus_dir: Path = typer.Option(Path("corpus"), help="Ingested corpus directory"),
    index_dir: Path = typer.Option(Path("index-caption"), help="Output vector index directory"),
) -> None:
    count = build_caption_index(corpus_dir=corpus_dir, index_dir=index_dir)
    typer.echo(f"Indexed {count} chunks (text + captions) from {corpus_dir} into {index_dir}")
```

Modify the existing `query` command to add the flag (find the existing `@app.command() def
query(...)` in `cli.py` and add this parameter alongside the existing ones, then use it when
constructing `TextOnlyRetriever`):

```python
    with_captions: bool = typer.Option(False, "--with-captions", help="Include indexed image/table/diagram captions"),
```

and change the retriever construction line from `TextOnlyRetriever(corpus_dir=corpus_dir,
index_dir=index_dir)` to `TextOnlyRetriever(corpus_dir=corpus_dir, index_dir=index_dir,
include_captions=with_captions)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/retrieval/test_pipeline.py -v`
Expected: PASS (all tests, including the 3 new ones)

Then run the full suite: `pytest tests/ -v -m "not slow"` Expected: all pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/doclens/retrieval/pipeline.py src/doclens/cli.py tests/retrieval/test_pipeline.py
git commit -m "feat(retrieval): wire caption chunks into the retriever and add caption-index CLI"
```

---

### Task 5: Real captioning run, real caption index, and a real comparison smoke test

**Files:** none created; this task runs the real pipeline against real data and records results.

This step validates the arm against real data, mirroring how Milestones 1 and 2 were smoke-tested
against the real corpus rather than only synthetic fixtures.

- [ ] **Step 1: Run the real captioning pass**

`GEMINI_API_KEY` is expected to already be set in the environment (confirm with a harmless check
like `echo $GEMINI_API_KEY | cut -c1-4` before starting, do not print the full key). Run:

```bash
python -m doclens.cli caption-corpus --corpus-dir corpus --scope-file data/caption_scope.json
```

This targets the 15 documents (1,058 pages) in `data/caption_scope.json`. Expect this to take
on the order of an hour or more given Gemini free-tier rate limits; run it and let it complete.
If it is interrupted for any reason, re-running the exact same command is safe and only
processes pages not yet in the cache (this is the entire point of the cache-first design in
Tasks 1 and 3, confirm this resumption behavior actually holds if an interruption happens to
occur). Record the exact command, its wall-clock duration, and its final summary line (total
caption chunks produced, across how many documents) in the task report.

- [ ] **Step 2: Build both indexes fresh**

```bash
python -m doclens.cli build-index --corpus-dir corpus --index-dir index
python -m doclens.cli build-caption-index --corpus-dir corpus --index-dir index-caption
```

Record both commands' chunk counts. The caption index's count should exceed the text-only
index's count by roughly the number of pages that received a real caption (some pages in the
15-document scope will have no visual content and contribute no caption chunk; this is expected,
not an error).

- [ ] **Step 3: Real comparison on real benchmark questions**

Pick 4 real `table_lookup` or `diagram_lookup` questions from `benchmarks/doclens-bench/questions.json`
whose `sources[0].doc_id` is one of the 15 documents in `data/caption_scope.json` (so a caption
was actually generated for that page). For each, run both:

```bash
python -m doclens.cli query "<question text>" --corpus-dir corpus --index-dir index --top-k 5
python -m doclens.cli query "<question text>" --corpus-dir corpus --index-dir index-caption --top-k 5 --with-captions
```

For each of the 4 questions, record in the task report: whether the text-only run's top result
actually contains the expected answer value, whether the caption-augmented run's top result
does, and whether a caption chunk (not just a text chunk) appears anywhere in the caption-
augmented run's top 5. Report this honestly: if the caption arm does not clearly outperform the
text-only arm on some or all of these 4 questions, say so plainly. A rigorous statistical
comparison across the full benchmark is a separate, later piece of work (the eventual
"comparison + failure taxonomy" report); this step is a real-data sanity check, not the final
verdict, and both outcomes (captions help here, captions don't help there) are valid, useful
findings to record honestly.

- [ ] **Step 4: Report**

Write a full report to the task's report file covering: the exact captioning run command and
duration, both index build commands and their chunk counts, the 4 comparison questions with
their real outputs from both arms, and an honest assessment of what was observed. This step has
no automated pass/fail gate; its job is to produce real evidence, not to make a test go green.

Then reply to the controller with: status (DONE / DONE_WITH_CONCERNS / BLOCKED), the captioning
run's total wall-clock time and chunk count, both index chunk counts, and a one-line honest
summary of the 4-question comparison (e.g. "caption arm won 3/4, text-only won 1/4" or whatever
was actually observed).

---

## Exit criteria

- [ ] `doclens caption-corpus` successfully captions the 15-document scope, resumably (verified
  by an interruption-and-resume test, not just a single clean run).
- [ ] `doclens build-caption-index` produces a working combined text+caption index.
- [ ] `doclens query --with-captions` retrieves caption chunks when they are more relevant than
  the available text chunks.
- [ ] A real, honestly-reported 4-question comparison exists between the text-only and caption-
  augmented arms.
- [ ] All tests in `tests/caption/` and the extended `tests/retrieval/` pass under `pytest -m
  "not slow"`, with no regressions to Milestones 1-2 or the benchmark infrastructure.

This closes the caption-and-index arm. The vision-embedding arm (SigLIP page-image embeddings)
gets its own plan next, followed by the full benchmark comparison across all three retrieval
systems.
