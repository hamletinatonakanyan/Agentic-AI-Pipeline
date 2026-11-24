import math

import pytest

from evals import metrics as m


def test_the_worked_example_from_the_docs():
    got, rel = ["c2", "c7", "c9"], ["c7"]
    assert m.recall_at_k(got, rel, 1) == 0 and m.recall_at_k(got, rel, 3) == 1
    assert m.precision_at_k(got, rel, 3) == pytest.approx(1 / 3)
    assert m.hit_rate_at_k(got, rel, 1) == 0 and m.hit_rate_at_k(got, rel, 3) == 1
    assert m.reciprocal_rank(got, rel) == 0.5
    assert m.ndcg_at_k(got, rel, 3) == pytest.approx(1 / math.log2(3))  # 0.631


def test_perfect_and_empty_results():
    assert m.ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)
    assert m.recall_at_k([], ["a"], 5) == 0 and m.reciprocal_rank([], ["a"]) == 0
    assert m.ndcg_at_k([], ["a"], 5) == 0


def test_no_relevant_chunks_means_not_computable_not_zero():
    for fn in (m.recall_at_k, m.precision_at_k, m.hit_rate_at_k, m.ndcg_at_k):
        assert fn(["a"], [], 3) is None
    assert m.reciprocal_rank(["a"], []) is None and m.source_accuracy(["a"], []) is None


def test_recall_with_several_relevant_chunks():
    assert m.recall_at_k(["a", "x", "b", "y"], ["a", "b", "c"], 3) == pytest.approx(2 / 3)
    assert m.recall_at_k(["a", "x", "b", "y"], ["a", "b", "c"], 1) == pytest.approx(1 / 3)


def test_precision_divides_by_k_even_when_fewer_results_came_back():
    assert m.precision_at_k(["a"], ["a"], 5) == pytest.approx(1 / 5)


def test_a_chunk_returned_twice_counts_once():
    assert m.recall_at_k(["a", "a", "b"], ["a", "b"], 2) == 1.0
    assert m.precision_at_k(["a", "a", "a"], ["a"], 3) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(["x", "x", "a"], ["a"]) == 0.5


def test_ndcg_rewards_ranking_the_relevant_chunk_higher():
    high = m.ndcg_at_k(["a", "x", "y"], ["a"], 3)
    low = m.ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert high == 1.0 and 0 < low < high
    assert low == pytest.approx(0.5)  # 1/log2(4) over 1/log2(2)


def test_ndcg_ideal_is_capped_by_k():
    # three relevant chunks but only k=2: the best possible top 2 is both relevant.
    assert m.ndcg_at_k(["a", "b"], ["a", "b", "c"], 2) == pytest.approx(1.0)


def test_source_accuracy_is_f1_of_citations():
    assert m.source_accuracy(["a"], ["a"]) == 1.0
    assert m.source_accuracy([], ["a"]) == 0.0
    assert m.source_accuracy(["x"], ["a"]) == 0.0
    assert m.source_accuracy(["a", "x"], ["a"]) == pytest.approx(2 / 3)  # precision 1/2, recall 1
    assert m.source_accuracy(["a"], ["a", "b"]) == pytest.approx(2 / 3)  # precision 1, recall 1/2


def test_retrieval_scores_has_a_key_per_metric_and_k():
    scores = m.retrieval_scores(["a", "b"], ["b"], [1, 3])
    assert set(scores) == {
        "mrr", "recall@1", "precision@1", "hit_rate@1", "ndcg@1",
        "recall@3", "precision@3", "hit_rate@3", "ndcg@3",
    }  # fmt: skip
    assert scores["recall@1"] == 0 and scores["recall@3"] == 1


def test_aggregation_skips_missing_values():
    assert m.mean([1.0, None, 3.0]) == 2.0 and m.mean([None]) is None and m.mean([]) is None
    assert m.stdev([2.0, 4.0]) == pytest.approx(math.sqrt(2)) and m.stdev([5.0]) == 0.0
    assert m.stdev([]) is None


def test_percentiles_interpolate():
    values = [10.0, 20.0, 30.0, 40.0, 100.0]
    assert m.percentile(values, 50) == 30.0
    assert m.percentile(values, 95) == pytest.approx(88.0)
    assert m.percentile([7.0], 95) == 7.0 and m.percentile([], 50) is None
    assert m.percentile([None, 4.0, 2.0], 50) == 3.0
