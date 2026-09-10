def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using reciprocal rank fusion (RRF).

    Reciprocal rank fusion combines ranked lists by summing the reciprocals of
    ranks across all lists. This approach works across differently-scaled score
    systems (e.g., BM25 and cosine similarity) because it only considers rank
    position, not the actual scores.

    Args:
        result_lists: List of ranked result lists, where each result is a
            (chunk_id, score) tuple. Lists are expected to be ranked best-first.
            The scores themselves are ignored; only rank positions matter.
        k: Parameter for the RRF formula (default 60). Controls smoothing.
            Score for a chunk at rank r is 1/(k+r).

    Returns:
        List of (chunk_id, fused_score) tuples sorted by fused_score in
        descending order.
    """
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, (chunk_id, _score) in enumerate(results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
