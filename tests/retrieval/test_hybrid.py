import pytest

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
