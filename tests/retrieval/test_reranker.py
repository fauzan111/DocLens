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
