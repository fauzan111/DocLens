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
