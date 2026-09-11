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
