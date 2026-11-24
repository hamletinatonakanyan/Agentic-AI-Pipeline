"""The metrics. Pure functions, no I/O, easy to check by hand.

Worked example: the relevant chunk is c7 and a run returned [c2, c7, c9].
    recall@1 = 0   recall@3 = 1   precision@3 = 1/3   hit_rate@3 = 1
    MRR = 1/2 (the first relevant result is at rank 2)
    nDCG@3 = (1 / log2(2 + 1)) / (1 / log2(1 + 1)) = 0.631
"""

import math
import statistics
from collections.abc import Sequence

Number = float | None


def _unique(ids: Sequence[str]) -> list[str]:
    """Keep the first occurrence of each id, in order. A chunk returned twice counts once."""
    seen: set[str] = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> Number:
    """Share of the relevant chunks found in the top k. None if nothing is relevant."""
    rel = set(relevant)
    if not rel:
        return None
    return len(set(_unique(retrieved)[:k]) & rel) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> Number:
    """Share of the top k that is relevant. The denominator is k even if fewer came back."""
    rel = set(relevant)
    if not rel:
        return None
    return len(set(_unique(retrieved)[:k]) & rel) / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> Number:
    """1 if at least one relevant chunk is in the top k, else 0."""
    rel = set(relevant)
    if not rel:
        return None
    return 1.0 if set(_unique(retrieved)[:k]) & rel else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> Number:
    """1 / rank of the first relevant chunk, 0 if none was found."""
    rel = set(relevant)
    if not rel:
        return None
    for rank, chunk in enumerate(_unique(retrieved), start=1):
        if chunk in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> Number:
    """Like MRR but rewards several relevant chunks and ranks near the top (binary relevance)."""
    rel = set(relevant)
    if not rel:
        return None
    top = _unique(retrieved)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(top) if c in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal


def source_accuracy(cited: Sequence[str], relevant: Sequence[str]) -> Number:
    """F1 of the cited chunk ids against the chunks that hold the evidence.

    Deterministic, no LLM. Citing the right chunks scores 1. Citing wrong chunks lowers precision,
    and missing the evidence lowers recall. No citations at all scores 0."""
    rel = set(relevant)
    if not rel:
        return None
    cites = set(cited)
    hits = len(cites & rel)
    if not hits:
        return 0.0
    precision, recall = hits / len(cites), hits / len(rel)
    return 2 * precision * recall / (precision + recall)


def retrieval_scores(
    retrieved: Sequence[str], relevant: Sequence[str], ks: Sequence[int]
) -> dict[str, Number]:
    scores: dict[str, Number] = {"mrr": reciprocal_rank(retrieved, relevant)}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        scores[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return scores


# ----- aggregation ----------------------------------------------------------------------------


def mean(values: Sequence[Number]) -> Number:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def stdev(values: Sequence[Number]) -> Number:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return statistics.stdev(present) if len(present) > 1 else 0.0


def percentile(values: Sequence[Number], p: float) -> Number:
    """Linear interpolation between the closest ranks (the same as numpy's default)."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return data[low] + (data[high] - data[low]) * (position - low)
