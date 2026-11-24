from retrieval.models import RawHit


def rrf_merge(rankings: list[list[RawHit]], k: int = 60) -> list[RawHit]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank of d in that list).

    Uses ranks only, so lists with different score scales (cosine, BM25) can be combined.
    A chunk found by several queries is kept once, with the best-supported score.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first: dict[str, RawHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            best_rank[hit.id] = min(best_rank.get(hit.id, rank), rank)
            first.setdefault(hit.id, hit)
    order = sorted(scores, key=lambda i: (-scores[i], best_rank[i], i))
    return [first[i].model_copy(update={"score": scores[i]}) for i in order]
