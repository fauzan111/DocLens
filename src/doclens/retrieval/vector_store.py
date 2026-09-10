from pathlib import Path


class VectorStore:
    def __init__(self, path: Path, collection_name: str = "doclens_chunks", vector_size: int = 768):
        self.path = Path(path)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self.path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.path))

    def _close_client(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def build(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self._ensure_client()
        try:
            self._client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            points = [
                PointStruct(id=index, vector=embedding, payload={"chunk_id": chunk_id})
                for index, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings))
            ]
            self._client.upsert(collection_name=self.collection_name, points=points)
        finally:
            self._close_client()

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        self._ensure_client()
        results = self._client.search(
            collection_name=self.collection_name, query_vector=query_embedding, limit=top_k
        )
        return [(hit.payload["chunk_id"], hit.score) for hit in results]

    def __del__(self):
        self._close_client()
