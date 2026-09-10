# DocLens Milestone 2: Text-Only Hybrid Retrieval Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the text-only hybrid retrieval baseline: chunk the ingested corpus, index it with
BM25 and dense multilingual embeddings, fuse the two with reciprocal rank fusion, rerank with a
cross-encoder, and generate a grounded, cited answer. This is "what everyone already builds" and
becomes the floor Milestone 3's multimodal retrieval arms are measured against.

**Architecture:** A `Chunk` model (page-scoped, never crosses page boundaries) backs two parallel
retrieval paths: `Bm25Index` (lexical, in-memory, rebuilt from the corpus each run) and
`VectorStore` (dense, persisted on disk via Qdrant's local mode, built once via `doclens
build-index`) fed by a lazily-loaded `TextEmbedder`. `reciprocal_rank_fusion()` combines both
paths' results; `CrossEncoderReranker` reranks the fused candidates; `GeminiGenerator` produces a
grounded answer with citations from the final top-k chunks. `TextOnlyRetriever` orchestrates
retrieval; a `doclens query` CLI command exposes it end to end.

**Tech Stack:** Python 3.11+ (existing), `rank-bm25`, `sentence-transformers` (embeddings +
cross-encoder reranking), `qdrant-client` (local on-disk mode, no server), `google-generativeai`
(Gemini free tier, dependency-injected so tests never need an API key).

## Global Constraints

- No Anthropic/Claude API calls anywhere in this pipeline (per DESIGN.md "Stack": $0 ongoing
  cost). Generation uses Gemini's free tier.
- Corpus is bilingual IT/EN; the embedding model and cross-encoder must both be multilingual.
- All file paths in code are `pathlib.Path`, never raw strings (Windows dev environment).
- Project requires Python >=3.11.
- Never use the em dash character (`—`, U+2014) anywhere: not in code, comments, docstrings,
  commit messages, or reports. Use a colon, comma, semicolon, or plain hyphen instead.
- Chunks never cross page boundaries (citations are per-page; a chunk spanning two pages would
  make "page N supports this answer" a lie).
- Every retrieval/generation component that loads a model (embedder, reranker, generator) must
  load it lazily on first real use, not at construction time, matching the existing
  `OcrFallback` pattern in `src/doclens/ingest/ocr_fallback.py` (constructing these objects must
  never trigger a network call or model download).
- Tests must never require network access, an API key, or a real model download to pass under
  `pytest -m "not slow"`. Any test that needs a real model/network call is marked
  `@pytest.mark.slow` and is optional (matches the existing `ocr_fallback` convention).

---

## File Structure

```
DocLens/
  src/doclens/
    ingest/
      corpus.py            # Task 1: load_documents() - shared corpus loader
      dataset_card.py       # Task 1: refactored to use load_documents()
    retrieval/
      __init__.py
      models.py             # Task 1: Chunk
      chunker.py            # Task 1: chunk_document()
      bm25_index.py         # Task 2: Bm25Index
      vector_store.py       # Task 4: VectorStore (Qdrant wrapper)
      hybrid.py             # Task 5: reciprocal_rank_fusion()
      reranker.py           # Task 6: CrossEncoderReranker
      pipeline.py           # Task 8: TextOnlyRetriever, build_index()
    embed/
      __init__.py
      text_embedder.py      # Task 3: TextEmbedder
    generation/
      __init__.py
      models.py             # Task 7: Citation, Answer
      generator.py          # Task 7: Generator protocol, GeminiGenerator
    cli.py                  # Task 8: `doclens build-index`, `doclens query` (extended)
  tests/
    ingest/
      test_corpus.py        # Task 1
    retrieval/
      test_chunker.py       # Task 1
      test_bm25_index.py    # Task 2
      test_vector_store.py  # Task 4
      test_hybrid.py        # Task 5
      test_reranker.py      # Task 6
      test_pipeline.py      # Task 8 (includes the smoke-test question set)
    embed/
      test_text_embedder.py # Task 3
    generation/
      test_generator.py     # Task 7
```

---

### Task 1: Corpus loader (shared, refactored) + Chunk model + chunker

**Files:**
- Create: `src/doclens/ingest/corpus.py`
- Modify: `src/doclens/ingest/dataset_card.py` (use the new shared loader instead of its own glob)
- Create: `src/doclens/retrieval/__init__.py` (empty)
- Create: `src/doclens/retrieval/models.py`
- Create: `src/doclens/retrieval/chunker.py`
- Test: `tests/ingest/test_corpus.py`
- Test: `tests/retrieval/test_chunker.py`

**Interfaces:**
- Consumes: `doclens.ingest.models.Document`, `Page` (already exist)
- Produces: `load_documents(corpus_dir: Path) -> list[Document]` in `doclens.ingest.corpus`
  (skips `registry.json` and any other file that fails `Document.model_validate_json`, printing
  a one-line warning for the latter case rather than crashing); `Chunk(chunk_id: str, doc_id:
  str, doc_title: str, page_number: int, chunk_index: int, text: str, source_url: str)` in
  `doclens.retrieval.models` (Pydantic `BaseModel`); `chunk_document(document: Document,
  chunk_size: int = 800, overlap: int = 100) -> list[Chunk]` in `doclens.retrieval.chunker`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ingest/test_corpus.py
import json
from pathlib import Path

from doclens.ingest.corpus import load_documents
from doclens.ingest.models import License
from doclens.ingest.pipeline import run_ingestion
from doclens.ingest.registry import SourceRegistry

import fitz  # PyMuPDF


def _make_pdf_with_text(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_load_documents_skips_registry_json(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    run_ingestion(
        pdf_bytes=_make_pdf_with_text("Torque: 45 Nm"),
        source_url="https://example.com/a.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Doc A",
        corpus_dir=corpus_dir,
        registry=registry,
    )

    documents = load_documents(corpus_dir)

    assert len(documents) == 1
    assert documents[0].title == "Doc A"


def test_load_documents_skips_unparseable_json_without_crashing(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    run_ingestion(
        pdf_bytes=_make_pdf_with_text("Voltage: 230V"),
        source_url="https://example.com/b.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Doc B",
        corpus_dir=corpus_dir,
        registry=registry,
    )
    # Simulate a stray, non-Document JSON file landing in corpus_dir.
    (corpus_dir / "notes.json").write_text(json.dumps({"unrelated": "data"}), encoding="utf-8")

    documents = load_documents(corpus_dir)

    assert len(documents) == 1
    assert documents[0].title == "Doc B"
```

```python
# tests/retrieval/test_chunker.py
from datetime import datetime, timezone
from pathlib import Path

import pytest

from doclens.ingest.models import Document, License, Page, SourceRecord
from doclens.retrieval.chunker import chunk_document


def _document(pages: list[Page]) -> Document:
    source = SourceRecord(
        source_url="https://example.com/manual.pdf",
        license=License.PUBLIC_DOMAIN,
        retrieved_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        sha256="a" * 64,
    )
    return Document(doc_id="a" * 64, title="Test Manual", source=source, pages=pages)


def test_chunk_document_produces_one_chunk_for_short_page():
    page = Page(page_number=1, text="Torque spec: 45 Nm", text_source="extracted",
                image_path=Path("corpus/x/page-1.png"))
    chunks = chunk_document(_document([page]), chunk_size=800, overlap=100)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert "Torque spec: 45 Nm" in chunks[0].text
    assert chunks[0].doc_title == "Test Manual"
    assert chunks[0].source_url == "https://example.com/manual.pdf"


def test_chunk_document_produces_overlapping_chunks_for_long_page():
    long_text = "word " * 500  # far longer than chunk_size
    page = Page(page_number=1, text=long_text, text_source="extracted",
                image_path=Path("corpus/x/page-1.png"))
    chunks = chunk_document(_document([page]), chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    # Overlap: the tail of chunk 0 should reappear at the head of chunk 1.
    assert chunks[0].text[-10:] in chunks[1].text


def test_chunk_document_skips_empty_text_pages():
    blank_page = Page(page_number=1, text="   ", text_source="extracted",
                       image_path=Path("corpus/x/page-1.png"))
    text_page = Page(page_number=2, text="Real content here", text_source="extracted",
                      image_path=Path("corpus/x/page-2.png"))
    chunks = chunk_document(_document([blank_page, text_page]))

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_chunk_document_never_merges_text_across_page_boundary():
    page1 = Page(page_number=1, text="End of page one.", text_source="extracted",
                 image_path=Path("corpus/x/page-1.png"))
    page2 = Page(page_number=2, text="Start of page two.", text_source="extracted",
                 image_path=Path("corpus/x/page-2.png"))
    chunks = chunk_document(_document([page1, page2]), chunk_size=800, overlap=100)

    assert len(chunks) == 2
    assert "End of page one" in chunks[0].text
    assert "Start of page two" not in chunks[0].text
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_chunk_document_rejects_overlap_not_smaller_than_chunk_size():
    page = Page(page_number=1, text="some text", text_source="extracted",
                image_path=Path("corpus/x/page-1.png"))
    with pytest.raises(ValueError):
        chunk_document(_document([page]), chunk_size=100, overlap=100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ingest/test_corpus.py tests/retrieval/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError` for `doclens.ingest.corpus` and `doclens.retrieval`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/ingest/corpus.py
from pathlib import Path

from pydantic import ValidationError

from doclens.ingest.models import Document


def load_documents(corpus_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for json_path in sorted(corpus_dir.glob("*.json")):
        if json_path.name == "registry.json":
            continue
        try:
            documents.append(Document.model_validate_json(json_path.read_text(encoding="utf-8")))
        except ValidationError:
            print(f"Skipping non-Document JSON file: {json_path.name}")
    return documents
```

Update `src/doclens/ingest/dataset_card.py` to use this instead of its own glob loop. Replace the
document-loading portion of `generate_dataset_card` (the loop that currently does
`for json_path in corpus_dir.glob("*.json"): if json_path.name == "registry.json": continue; ...`)
with:

```python
from doclens.ingest.corpus import load_documents

# inside generate_dataset_card, replace the manual glob loop with:
for document in load_documents(corpus_dir):
    total_pages += len(document.pages)
    ocr_pages += sum(1 for page in document.pages if page.text_source == "ocr")
```

```python
# src/doclens/retrieval/models.py
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    page_number: int
    chunk_index: int
    text: str
    source_url: str
```

```python
# src/doclens/retrieval/chunker.py
from doclens.ingest.models import Document
from doclens.retrieval.models import Chunk


def chunk_document(document: Document, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in document.pages:
        text = page.text.strip()
        if not text:
            continue

        start = 0
        index = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id[:16]}:{page.page_number}:{index}",
                        doc_id=document.doc_id,
                        doc_title=document.title,
                        page_number=page.page_number,
                        chunk_index=index,
                        text=chunk_text,
                        source_url=document.source.source_url,
                    )
                )
                index += 1
            if end >= len(text):
                break
            start = end - overlap

    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ingest/test_corpus.py tests/retrieval/test_chunker.py -v`
Expected: PASS (7 passed)

Also run the existing Milestone 1 dataset-card tests to confirm the refactor didn't break them:
`pytest tests/ingest/test_dataset_card.py -v` Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/doclens/ingest/corpus.py src/doclens/ingest/dataset_card.py \
  src/doclens/retrieval/__init__.py src/doclens/retrieval/models.py \
  src/doclens/retrieval/chunker.py tests/ingest/test_corpus.py tests/retrieval/test_chunker.py
git commit -m "feat(retrieval): add shared corpus loader, Chunk model, and page-scoped chunker"
```

---

### Task 2: BM25 lexical index

**Files:**
- Create: `src/doclens/retrieval/bm25_index.py`
- Test: `tests/retrieval/test_bm25_index.py`

**Interfaces:**
- Consumes: `doclens.retrieval.models.Chunk`
- Produces: `Bm25Index()` with `.build(chunks: list[Chunk]) -> None` and `.search(query: str,
  top_k: int = 10) -> list[tuple[str, float]]` (chunk_id, score pairs, descending by score,
  zero-score results excluded). Calling `.search()` before `.build()` raises `RuntimeError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_bm25_index.py
import pytest

from doclens.retrieval.bm25_index import Bm25Index
from doclens.retrieval.models import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc1", doc_title="Manual", page_number=1,
                 chunk_index=0, text=text, source_url="https://example.com/a.pdf")


def test_search_before_build_raises():
    index = Bm25Index()
    with pytest.raises(RuntimeError):
        index.search("torque")


def test_search_ranks_lexically_relevant_chunk_first():
    chunks = [
        _chunk("c1", "The pump requires a torque of 45 Nm on the flange bolts."),
        _chunk("c2", "Ambient temperature must remain between 5 and 40 degrees Celsius."),
        _chunk("c3", "Voltage tolerance is plus or minus ten percent of nominal."),
    ]
    index = Bm25Index()
    index.build(chunks)

    results = index.search("torque flange bolts", top_k=3)

    assert len(results) >= 1
    assert results[0][0] == "c1"


def test_search_returns_no_results_for_completely_unrelated_query():
    chunks = [_chunk("c1", "The pump requires a torque of 45 Nm.")]
    index = Bm25Index()
    index.build(chunks)

    results = index.search("xylophone giraffe umbrella")

    assert results == []


def test_search_respects_top_k():
    chunks = [_chunk(f"c{i}", "torque torque torque bolt") for i in range(5)]
    index = Bm25Index()
    index.build(chunks)

    results = index.search("torque bolt", top_k=2)

    assert len(results) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_bm25_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.retrieval.bm25_index'`

- [ ] **Step 3: Write minimal implementation**

Install `rank-bm25` and add `rank-bm25>=0.2` to `pyproject.toml`'s dependencies list.

```python
# src/doclens/retrieval/bm25_index.py
import re

from doclens.retrieval.models import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Bm25Index:
    def __init__(self):
        self._bm25 = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("Bm25Index.build() must be called before search()")

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [(chunk_id, float(score)) for chunk_id, score in ranked[:top_k] if score > 0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_bm25_index.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/doclens/retrieval/bm25_index.py tests/retrieval/test_bm25_index.py
git commit -m "feat(retrieval): add BM25 lexical index"
```

---

### Task 3: Text embedder (lazy, multilingual)

**Files:**
- Create: `src/doclens/embed/__init__.py` (empty)
- Create: `src/doclens/embed/text_embedder.py`
- Test: `tests/embed/test_text_embedder.py`

**Interfaces:**
- Produces: `TextEmbedder(model_name: str = "intfloat/multilingual-e5-base")` with
  `.embed_passages(texts: list[str]) -> list[list[float]]` and `.embed_query(text: str) ->
  list[float]`. Constructing `TextEmbedder()` must NOT load the underlying model (matches the
  `OcrFallback` lazy-load pattern). The e5 model family requires `"query: "` / `"passage: "`
  prefixes on inputs; this class applies them internally so callers never think about it.

- [ ] **Step 1: Write the failing test**

```python
# tests/embed/test_text_embedder.py
import pytest

from doclens.embed.text_embedder import TextEmbedder


def test_text_embedder_does_not_load_model_on_init():
    embedder = TextEmbedder()
    assert embedder._model is None


@pytest.mark.slow
def test_embed_query_and_passages_are_semantically_comparable():
    import numpy as np

    embedder = TextEmbedder()
    query_vector = np.array(embedder.embed_query("torque specification for flange bolts"))
    relevant = np.array(
        embedder.embed_passages(["The flange bolts require 45 Nm of torque."])[0]
    )
    irrelevant = np.array(
        embedder.embed_passages(["The office is closed on public holidays."])[0]
    )

    relevant_similarity = float(np.dot(query_vector, relevant))
    irrelevant_similarity = float(np.dot(query_vector, irrelevant))

    assert relevant_similarity > irrelevant_similarity
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/embed/test_text_embedder.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.embed.text_embedder'`

- [ ] **Step 3: Write minimal implementation**

Install `sentence-transformers` and add `sentence-transformers>=3.0` to `pyproject.toml`'s
dependencies list. (It depends on `torch`, already present from `easyocr` in Milestone 1, so this
should not require a large new download.)

```python
# src/doclens/embed/text_embedder.py
class TextEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        prefixed = [f"passage: {text}" for text in texts]
        return self._model.encode(prefixed, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        self._ensure_loaded()
        return self._model.encode(f"query: {text}", normalize_embeddings=True).tolist()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/embed/test_text_embedder.py -v -m "not slow"`
Expected: PASS (1 passed, 1 deselected)

As a final check, also run the slow test once to confirm the real model behaves sensibly:
`pytest tests/embed/test_text_embedder.py -v` (this downloads the model on first run and may take
a few minutes; report the result either way per the report contract below).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/doclens/embed/__init__.py src/doclens/embed/text_embedder.py \
  tests/embed/test_text_embedder.py
git commit -m "feat(embed): add lazy multilingual text embedder"
```

---

### Task 4: Vector store (Qdrant, local on-disk mode)

**Files:**
- Create: `src/doclens/retrieval/vector_store.py`
- Test: `tests/retrieval/test_vector_store.py`

**Interfaces:**
- Produces: `VectorStore(path: Path, collection_name: str = "doclens_chunks", vector_size: int =
  768)` with `.build(chunk_ids: list[str], embeddings: list[list[float]]) -> None` and
  `.search(query_embedding: list[float], top_k: int = 10) -> list[tuple[str, float]]`. Uses
  Qdrant's embedded/local mode (`QdrantClient(path=...)`), so no server process is required.
  Must NOT connect/initialize the client at construction time (lazy, same pattern as other
  components in this plan). A `VectorStore` reopened at the same `path` in a fresh instance must
  be able to `.search()` immediately without calling `.build()` again (the point of persisting to
  disk).

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_vector_store.py
from pathlib import Path

import pytest

from doclens.retrieval.vector_store import VectorStore


def test_vector_store_does_not_connect_on_init(tmp_path: Path):
    store = VectorStore(path=tmp_path / "index", vector_size=4)
    assert store._client is None


def test_build_and_search_returns_nearest_chunk_first(tmp_path: Path):
    store = VectorStore(path=tmp_path / "index", vector_size=4)
    chunk_ids = ["a", "b", "c"]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    store.build(chunk_ids, embeddings)

    results = store.search([0.9, 0.1, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0][0] == "a"


def test_vector_store_persists_across_instances(tmp_path: Path):
    index_path = tmp_path / "index"
    store_a = VectorStore(path=index_path, vector_size=4)
    store_a.build(["x"], [[1.0, 0.0, 0.0, 0.0]])

    store_b = VectorStore(path=index_path, vector_size=4)
    results = store_b.search([1.0, 0.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0][0] == "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_vector_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.retrieval.vector_store'`

- [ ] **Step 3: Write minimal implementation**

Install `qdrant-client` and add `qdrant-client>=1.9` to `pyproject.toml`'s dependencies list.

```python
# src/doclens/retrieval/vector_store.py
from pathlib import Path


class VectorStore:
    def __init__(self, path: Path, collection_name: str = "doclens_chunks", vector_size: int = 768):
        self.path = path
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self.path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.path))

    def build(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self._ensure_client()
        self._client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=index, vector=embedding, payload={"chunk_id": chunk_id})
            for index, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings))
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        self._ensure_client()
        results = self._client.search(
            collection_name=self.collection_name, query_vector=query_embedding, limit=top_k
        )
        return [(hit.payload["chunk_id"], hit.score) for hit in results]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_vector_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/doclens/retrieval/vector_store.py tests/retrieval/test_vector_store.py
git commit -m "feat(retrieval): add Qdrant local-mode vector store"
```

---

### Task 5: Reciprocal rank fusion

**Files:**
- Create: `src/doclens/retrieval/hybrid.py`
- Test: `tests/retrieval/test_hybrid.py`

**Interfaces:**
- Produces: `reciprocal_rank_fusion(result_lists: list[list[tuple[str, float]]], k: int = 60) ->
  list[tuple[str, float]]` (chunk_id, fused_score pairs, descending by fused_score). Input lists
  are already ranked (best first); the input scores themselves are ignored, only rank position
  within each list matters (this is what makes RRF work across two differently-scaled score
  systems like BM25 and cosine similarity, without needing score normalization).

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_hybrid.py
from doclens.retrieval.hybrid import reciprocal_rank_fusion


def test_chunk_ranked_first_in_both_lists_wins():
    bm25_results = [("a", 5.0), ("b", 3.0), ("c", 1.0)]
    dense_results = [("a", 0.9), ("c", 0.8), ("b", 0.7)]

    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=60)

    assert fused[0][0] == "a"


def test_chunk_appearing_in_only_one_list_still_included():
    bm25_results = [("a", 5.0), ("d", 2.0)]
    dense_results = [("b", 0.9), ("c", 0.8)]

    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=60)
    fused_ids = {chunk_id for chunk_id, _ in fused}

    assert fused_ids == {"a", "b", "c", "d"}


def test_fused_score_matches_hand_computed_rrf_formula():
    # rank 1 in list A, rank 2 in list B -> 1/(60+1) + 1/(60+2)
    bm25_results = [("a", 5.0), ("b", 3.0)]
    dense_results = [("b", 0.9), ("a", 0.8)]

    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=60)
    fused_by_id = dict(fused)

    expected_a = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    expected_b = 1.0 / (60 + 2) + 1.0 / (60 + 1)
    assert fused_by_id["a"] == pytest.approx(expected_a)
    assert fused_by_id["b"] == pytest.approx(expected_b)


def test_empty_result_lists_produce_empty_fusion():
    assert reciprocal_rank_fusion([[], []]) == []
```

Add `import pytest` at the top of the test file (needed for `pytest.approx`).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_hybrid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.retrieval.hybrid'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/retrieval/hybrid.py
def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, (chunk_id, _score) in enumerate(results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_hybrid.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/doclens/retrieval/hybrid.py tests/retrieval/test_hybrid.py
git commit -m "feat(retrieval): add reciprocal rank fusion for hybrid BM25/dense search"
```

---

### Task 6: Cross-encoder reranker

**Files:**
- Create: `src/doclens/retrieval/reranker.py`
- Test: `tests/retrieval/test_reranker.py`

**Interfaces:**
- Consumes: `doclens.retrieval.models.Chunk`
- Produces: `CrossEncoderReranker(model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")`
  with `.rerank(query: str, candidates: list[Chunk], top_k: int = 5) -> list[tuple[Chunk,
  float]]` (descending by relevance score). Constructing `CrossEncoderReranker()` must NOT load
  the model. Calling `.rerank()` with an empty `candidates` list must return `[]` WITHOUT loading
  the model either (avoids an unnecessary model download when there's nothing to rerank).

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_reranker.py
import pytest

from doclens.retrieval.models import Chunk
from doclens.retrieval.reranker import CrossEncoderReranker


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc1", doc_title="Manual", page_number=1,
                 chunk_index=0, text=text, source_url="https://example.com/a.pdf")


def test_reranker_does_not_load_model_on_init():
    reranker = CrossEncoderReranker()
    assert reranker._model is None


def test_rerank_with_empty_candidates_returns_empty_without_loading_model():
    reranker = CrossEncoderReranker()
    results = reranker.rerank("torque spec", candidates=[])

    assert results == []
    assert reranker._model is None


@pytest.mark.slow
def test_rerank_puts_relevant_candidate_first():
    candidates = [
        _chunk("c1", "Ambient temperature must remain between 5 and 40 degrees Celsius."),
        _chunk("c2", "The flange bolts require a torque of 45 Nm."),
        _chunk("c3", "Voltage tolerance is plus or minus ten percent of nominal."),
    ]
    reranker = CrossEncoderReranker()
    results = reranker.rerank("what torque should the flange bolts be tightened to", candidates, top_k=3)

    assert results[0][0].chunk_id == "c2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_reranker.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.retrieval.reranker'`

- [ ] **Step 3: Write minimal implementation**

`sentence-transformers` (installed in Task 3) also provides `CrossEncoder`; no new dependency
needed.

```python
# src/doclens/retrieval/reranker.py
from doclens.retrieval.models import Chunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: list[Chunk], top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []

        self._ensure_loaded()
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_reranker.py -v -m "not slow"`
Expected: PASS (2 passed, 1 deselected)

Also run the slow test once to confirm real reranking behavior:
`pytest tests/retrieval/test_reranker.py -v` (downloads the cross-encoder model on first run).

- [ ] **Step 5: Commit**

```bash
git add src/doclens/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "feat(retrieval): add multilingual cross-encoder reranker"
```

---

### Task 7: Answer generator (Gemini, dependency-injected)

**Files:**
- Create: `src/doclens/generation/__init__.py` (empty)
- Create: `src/doclens/generation/models.py`
- Create: `src/doclens/generation/generator.py`
- Test: `tests/generation/test_generator.py`

**Interfaces:**
- Consumes: `doclens.retrieval.models.Chunk`
- Produces: `Citation(doc_title: str, page_number: int, source_url: str)` and `Answer(text: str,
  citations: list[Citation], grounded: bool)` in `doclens.generation.models` (Pydantic
  `BaseModel`); `GeminiGenerator(model_name: str = "gemini-2.0-flash", api_key: str | None =
  None)` in `doclens.generation.generator` with `.generate(query: str, chunks: list[Chunk]) ->
  Answer`. `api_key` defaults to the `GEMINI_API_KEY` environment variable when not passed
  explicitly. Empty `chunks` returns an ungrounded "I don't have enough information" `Answer`
  with no citations and WITHOUT requiring an API key or network call (this is the one behavior
  testable without credentials). A non-empty `chunks` list with no API key available raises
  `RuntimeError` only when `.generate()` is actually called (lazy), not at construction time.

- [ ] **Step 1: Write the failing test**

```python
# tests/generation/test_generator.py
import os

import pytest

from doclens.generation.generator import GeminiGenerator
from doclens.retrieval.models import Chunk


def _chunk() -> Chunk:
    return Chunk(chunk_id="c1", doc_id="doc1", doc_title="Pump Manual", page_number=3,
                 chunk_index=0, text="The flange bolts require a torque of 45 Nm.",
                 source_url="https://example.com/manual.pdf")


def test_generator_does_not_configure_on_init():
    generator = GeminiGenerator(api_key="unused-for-this-test")
    assert generator._model is None


def test_generate_with_no_chunks_returns_ungrounded_answer_without_api_key():
    generator = GeminiGenerator(api_key=None)
    answer = generator.generate("What torque should be used?", chunks=[])

    assert answer.grounded is False
    assert answer.citations == []
    assert generator._model is None  # never attempted to load/configure


def test_generate_with_chunks_and_no_api_key_raises():
    generator = GeminiGenerator(api_key=None)
    with pytest.raises(RuntimeError):
        generator.generate("What torque should be used?", chunks=[_chunk()])


@pytest.mark.slow
def test_generate_with_real_api_key_produces_grounded_answer():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set; skipping live Gemini call")

    generator = GeminiGenerator(api_key=api_key)
    answer = generator.generate("What torque should the flange bolts be tightened to?", [_chunk()])

    assert answer.grounded is True
    assert len(answer.citations) == 1
    assert answer.citations[0].page_number == 3
    assert answer.text.strip() != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/generation/test_generator.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.generation'`

- [ ] **Step 3: Write minimal implementation**

Install `google-generativeai` and add `google-generativeai>=0.8` to `pyproject.toml`'s
dependencies list.

```python
# src/doclens/generation/models.py
from pydantic import BaseModel


class Citation(BaseModel):
    doc_title: str
    page_number: int
    source_url: str


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    grounded: bool
```

```python
# src/doclens/generation/generator.py
import os

from doclens.generation.models import Answer, Citation
from doclens.retrieval.models import Chunk

NO_INFO_TEXT = "I don't have enough information to answer this question."


class GeminiGenerator:
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

    def generate(self, query: str, chunks: list[Chunk]) -> Answer:
        if not chunks:
            return Answer(text=NO_INFO_TEXT, citations=[], grounded=False)

        self._ensure_loaded()
        context = "\n\n".join(
            f"[Source: {chunk.doc_title}, page {chunk.page_number}]\n{chunk.text}"
            for chunk in chunks
        )
        prompt = (
            "Answer the question using ONLY the provided context. "
            "If the context does not contain the answer, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        response = self._model.generate_content(prompt)
        citations = [
            Citation(doc_title=chunk.doc_title, page_number=chunk.page_number, source_url=chunk.source_url)
            for chunk in chunks
        ]
        return Answer(text=response.text.strip(), citations=citations, grounded=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/generation/test_generator.py -v -m "not slow"`
Expected: PASS (3 passed, 1 deselected)

If a `GEMINI_API_KEY` environment variable happens to be set in this environment, also run
`pytest tests/generation/test_generator.py -v` once to exercise the live call; if it is not set,
that single test will report `SKIPPED` and that is expected, not a failure.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/doclens/generation/__init__.py src/doclens/generation/models.py \
  src/doclens/generation/generator.py tests/generation/test_generator.py
git commit -m "feat(generation): add dependency-injected Gemini answer generator"
```

---

### Task 8: Retrieval pipeline orchestration, CLI, and smoke test

**Files:**
- Create: `src/doclens/retrieval/pipeline.py`
- Modify: `src/doclens/cli.py` (add `build-index` and `query` commands)
- Test: `tests/retrieval/test_pipeline.py`

**Interfaces:**
- Consumes: `load_documents` (`doclens.ingest.corpus`), `chunk_document`/`Chunk`
  (`doclens.retrieval.chunker`/`models`), `Bm25Index` (`doclens.retrieval.bm25_index`),
  `TextEmbedder` (`doclens.embed.text_embedder`), `VectorStore`
  (`doclens.retrieval.vector_store`), `reciprocal_rank_fusion` (`doclens.retrieval.hybrid`),
  `CrossEncoderReranker` (`doclens.retrieval.reranker`), `GeminiGenerator`/`Answer`
  (`doclens.generation.generator`/`models`)
- Produces: `build_index(corpus_dir: Path, index_dir: Path, embedder: TextEmbedder | None = None,
  vector_store: VectorStore | None = None) -> int` (returns chunk count indexed) and
  `TextOnlyRetriever(corpus_dir: Path, index_dir: Path, embedder: TextEmbedder | None = None,
  vector_store: VectorStore | None = None, reranker: CrossEncoderReranker | None = None)` with
  `.retrieve(query: str, top_k: int = 5, fusion_pool: int = 20) -> list[tuple[Chunk, float]]` in
  `doclens.retrieval.pipeline`. CLI commands `doclens build-index --corpus-dir corpus --index-dir
  index` and `doclens query "..." --corpus-dir corpus --index-dir index --top-k 5 [--answer]`
  (the `--answer` flag additionally calls `GeminiGenerator`; without it, `query` only prints
  retrieved chunks and their citations, so the CLI is usable end-to-end even without a
  `GEMINI_API_KEY`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/retrieval/test_pipeline.py
from datetime import datetime, timezone
from pathlib import Path

from doclens.ingest.models import Document, License, Page, SourceRecord
from doclens.retrieval.bm25_index import Bm25Index
from doclens.retrieval.models import Chunk
from doclens.retrieval.pipeline import TextOnlyRetriever, build_index
from doclens.retrieval.vector_store import VectorStore


def _write_document(corpus_dir: Path, doc_id: str, title: str, page_texts: list[str]) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    source = SourceRecord(
        source_url=f"https://example.com/{doc_id}.pdf",
        license=License.PUBLIC_DOMAIN,
        retrieved_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        sha256=doc_id,
    )
    pages = [
        Page(page_number=i + 1, text=text, text_source="extracted",
             image_path=Path(f"corpus/{doc_id}/page-{i + 1}.png"))
        for i, text in enumerate(page_texts)
    ]
    document = Document(doc_id=doc_id, title=title, source=source, pages=pages)
    (corpus_dir / f"{doc_id}.json").write_text(document.model_dump_json(indent=2), encoding="utf-8")


class _StubEmbedder:
    """Deterministic fake embedder: same text -> same vector, avoids a real model in tests."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        # Cheap deterministic "embedding": presence of key terms as one-hot-ish dims.
        return [
            1.0 if "torque" in text.lower() else 0.0,
            1.0 if "temperature" in text.lower() else 0.0,
            1.0 if "voltage" in text.lower() else 0.0,
            0.1,
        ]


def test_build_index_returns_chunk_count(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _write_document(corpus_dir, "d" * 64, "Manual One", ["Torque spec: 45 Nm on the flange."])

    count = build_index(corpus_dir, tmp_path / "index", embedder=_StubEmbedder(),
                         vector_store=VectorStore(path=tmp_path / "index", vector_size=4))

    assert count == 1


def test_retrieve_returns_chunk_with_citation_metadata(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _write_document(corpus_dir, "e" * 64, "Manual Two",
                     ["Ambient temperature must remain between 5 and 40 degrees.",
                      "Torque spec: 45 Nm on the flange bolts."])

    index_dir = tmp_path / "index"
    build_index(corpus_dir, index_dir, embedder=_StubEmbedder(),
                vector_store=VectorStore(path=index_dir, vector_size=4))

    retriever = TextOnlyRetriever(
        corpus_dir=corpus_dir, index_dir=index_dir,
        embedder=_StubEmbedder(), vector_store=VectorStore(path=index_dir, vector_size=4),
        reranker=None,
    )
    results = retriever.retrieve("what torque is required", top_k=2)

    assert len(results) >= 1
    top_chunk, _score = results[0]
    assert isinstance(top_chunk, Chunk)
    assert top_chunk.doc_title == "Manual Two"
    assert "torque" in top_chunk.text.lower()


def test_retrieve_with_no_reranker_falls_back_to_fused_order(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _write_document(corpus_dir, "f" * 64, "Manual Three", ["Voltage tolerance is ten percent."])

    index_dir = tmp_path / "index"
    build_index(corpus_dir, index_dir, embedder=_StubEmbedder(),
                vector_store=VectorStore(path=index_dir, vector_size=4))

    retriever = TextOnlyRetriever(
        corpus_dir=corpus_dir, index_dir=index_dir,
        embedder=_StubEmbedder(), vector_store=VectorStore(path=index_dir, vector_size=4),
        reranker=None,
    )
    results = retriever.retrieve("voltage tolerance", top_k=1)

    assert len(results) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/retrieval/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.retrieval.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/retrieval/pipeline.py
from pathlib import Path

from doclens.embed.text_embedder import TextEmbedder
from doclens.ingest.corpus import load_documents
from doclens.retrieval.bm25_index import Bm25Index
from doclens.retrieval.chunker import chunk_document
from doclens.retrieval.hybrid import reciprocal_rank_fusion
from doclens.retrieval.models import Chunk
from doclens.retrieval.reranker import CrossEncoderReranker
from doclens.retrieval.vector_store import VectorStore


def _load_all_chunks(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_documents(corpus_dir):
        chunks.extend(chunk_document(document))
    return chunks


def build_index(
    corpus_dir: Path,
    index_dir: Path,
    embedder: TextEmbedder | None = None,
    vector_store: VectorStore | None = None,
) -> int:
    chunks = _load_all_chunks(corpus_dir)
    embedder = embedder or TextEmbedder()
    vector_store = vector_store or VectorStore(path=index_dir)

    embeddings = embedder.embed_passages([chunk.text for chunk in chunks])
    vector_store.build([chunk.chunk_id for chunk in chunks], embeddings)
    return len(chunks)


class TextOnlyRetriever:
    def __init__(
        self,
        corpus_dir: Path,
        index_dir: Path,
        embedder: TextEmbedder | None = None,
        vector_store: VectorStore | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.chunks = _load_all_chunks(corpus_dir)
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

        self.bm25 = Bm25Index()
        self.bm25.build(self.chunks)

        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or VectorStore(path=index_dir)
        self.reranker = reranker if reranker is not None else CrossEncoderReranker()

    def retrieve(self, query: str, top_k: int = 5, fusion_pool: int = 20) -> list[tuple[Chunk, float]]:
        bm25_results = self.bm25.search(query, top_k=fusion_pool)
        dense_results = self.vector_store.search(self.embedder.embed_query(query), top_k=fusion_pool)

        fused = reciprocal_rank_fusion([bm25_results, dense_results])[:fusion_pool]
        # Pair each chunk with its fused score directly, rather than filtering two lists
        # independently and zipping afterward (that would misalign chunk<->score whenever a
        # fused chunk_id is missing from chunk_by_id).
        candidate_pairs = [
            (self.chunk_by_id[chunk_id], score) for chunk_id, score in fused if chunk_id in self.chunk_by_id
        ]

        if self.reranker is None or not candidate_pairs:
            return candidate_pairs[:top_k]

        candidates = [chunk for chunk, _score in candidate_pairs]
        return self.reranker.rerank(query, candidates, top_k=top_k)
```

Append to `src/doclens/cli.py`:

```python
from doclens.retrieval.pipeline import TextOnlyRetriever, build_index


@app.command(name="build-index")
def build_index_command(
    corpus_dir: Path = typer.Option(Path("corpus"), help="Ingested corpus directory"),
    index_dir: Path = typer.Option(Path("index"), help="Output vector index directory"),
) -> None:
    count = build_index(corpus_dir=corpus_dir, index_dir=index_dir)
    typer.echo(f"Indexed {count} chunks from {corpus_dir} into {index_dir}")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="The question to search for"),
    corpus_dir: Path = typer.Option(Path("corpus"), help="Ingested corpus directory"),
    index_dir: Path = typer.Option(Path("index"), help="Vector index directory (from build-index)"),
    top_k: int = typer.Option(5, help="Number of results to return"),
    answer: bool = typer.Option(False, "--answer", help="Also generate a grounded answer via Gemini"),
) -> None:
    retriever = TextOnlyRetriever(corpus_dir=corpus_dir, index_dir=index_dir)
    results = retriever.retrieve(query_text, top_k=top_k)

    for chunk, score in results:
        typer.echo(f"[{score:.4f}] {chunk.doc_title} (page {chunk.page_number}): {chunk.text[:200]}")

    if answer:
        from doclens.generation.generator import GeminiGenerator

        generator = GeminiGenerator()
        result = generator.generate(query_text, [chunk for chunk, _ in results])
        typer.echo(f"\nAnswer: {result.text}")
        for citation in result.citations:
            typer.echo(f"  - {citation.doc_title}, page {citation.page_number}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/retrieval/test_pipeline.py -v`
Expected: PASS (3 passed)

Then run the full Milestone 1 + Milestone 2 automated suite together:
`pytest tests/ -v -m "not slow"` Expected: all pass, no regressions in the Milestone 1 tests from
the Task 1 `dataset_card.py` refactor.

- [ ] **Step 5: Smoke test against the real seed corpus**

This step validates the CLI against real data, not synthetic fixtures, mirroring how the
Milestone 1 ingestion pipeline was smoke-tested against real manufacturer PDFs. It is a manual
verification step, not an automated pytest case (real model downloads and real corpus data make
it too slow/heavy for the default test run).

Run, from the repo root, using the seed corpus already ingested into `corpus/`:

```bash
python -m doclens.cli build-index --corpus-dir corpus --index-dir index
python -m doclens.cli query "what torque should be used on the flange bolts" --corpus-dir corpus --index-dir index --top-k 5
python -m doclens.cli query "cosa fare in caso di sovraccarico del motore" --corpus-dir corpus --index-dir index --top-k 5
```

Confirm: (a) `build-index` completes without error and reports a chunk count roughly proportional
to the corpus's page count; (b) each `query` call returns at least one result whose text is
plausibly related to the question; (c) the English and Italian queries both return sensible
results (validates the multilingual embedder/reranker choice against real bilingual content, not
just the synthetic English-only unit tests). Record the exact commands run and their output (or a
representative excerpt if very long) in the task report. If a query returns clearly irrelevant
results, that is worth noting as a concern in the report, not silently ignored, but it does not
block this task (retrieval quality tuning is Milestone 3/4's job, not this task's).

- [ ] **Step 6: Commit**

```bash
git add src/doclens/retrieval/pipeline.py src/doclens/cli.py tests/retrieval/test_pipeline.py
git commit -m "feat(retrieval): add end-to-end text-only retrieval pipeline and query CLI"
```

---

## Milestone 2 exit criteria (from DESIGN.md)

- [ ] `doclens build-index` runs end-to-end on the real seed corpus and produces a queryable
  on-disk vector index.
- [ ] `doclens query` returns hybrid (BM25 + dense, RRF-fused, cross-encoder-reranked) results
  with page-level citations, on both English and Italian real queries.
- [ ] `doclens query --answer` produces a grounded, cited answer via Gemini when a
  `GEMINI_API_KEY` is available (not required for the retrieval half to work).
- [ ] All automated tests in `tests/retrieval/`, `tests/embed/`, `tests/generation/`, and
  `tests/ingest/` pass under `pytest -m "not slow"`, with no regressions to Milestone 1.

This closes Milestone 2 from `DESIGN.md`. Milestone 3 (the multimodal retrieval arms: caption-
and-index, unified vision embedding, and the full 200-300 question benchmark comparing all
systems) gets its own plan once this baseline is working and smoke-tested.
