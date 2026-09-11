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
