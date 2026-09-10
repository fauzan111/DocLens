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
    chunks = [
        _chunk("c0", "The pump requires a torque of 45 Nm on the flange bolts."),
        _chunk("c1", "Tighten the mounting bolts to the specified torque rating."),
        _chunk("c2", "Ambient temperature must remain between 5 and 40 degrees Celsius."),
        _chunk("c3", "Voltage tolerance is plus or minus ten percent of nominal."),
        _chunk("c4", "Replace the filter cartridge every six months of operation."),
    ]
    index = Bm25Index()
    index.build(chunks)

    results = index.search("torque bolt", top_k=2)

    assert len(results) == 2
