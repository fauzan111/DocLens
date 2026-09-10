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
