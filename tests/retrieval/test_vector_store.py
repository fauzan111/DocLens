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


def test_repeated_search_on_same_instance(tmp_path: Path):
    store = VectorStore(path=tmp_path / "index", vector_size=4)
    chunk_ids = ["a", "b", "c"]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    store.build(chunk_ids, embeddings)

    first_results = store.search([0.9, 0.1, 0.0, 0.0], top_k=1)
    client_after_first_search = store._client
    second_results = store.search([0.0, 0.0, 0.9, 0.1], top_k=1)

    assert first_results[0][0] == "a"
    assert second_results[0][0] == "c"
    assert store._client is client_after_first_search


def test_search_on_nonexistent_index_raises_clear_error(tmp_path: Path):
    store = VectorStore(path=tmp_path / "nonexistent_index", vector_size=4)

    with pytest.raises(RuntimeError, match="No index found.*build-index"):
        store.search([1.0, 0.0, 0.0, 0.0], top_k=1)

    assert not (tmp_path / "nonexistent_index").exists()


def test_search_on_nonexistent_index_does_not_create_directory(tmp_path: Path):
    index_path = tmp_path / "index_not_yet_built"
    store = VectorStore(path=index_path, vector_size=4)

    try:
        store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    except RuntimeError:
        pass

    assert not index_path.exists()
