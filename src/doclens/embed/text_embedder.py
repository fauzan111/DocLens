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
