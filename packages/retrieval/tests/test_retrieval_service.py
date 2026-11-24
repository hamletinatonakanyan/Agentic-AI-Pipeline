import pytest
from qdrant_client import QdrantClient

from common import AppError
from observability import init_observability, shutdown_observability
from observability.exporters import InMemoryExporter
from observability.settings import ObsSettings
from retrieval.models import (
    Candidate,
    RerankRequest,
    RetrievalOptions,
    SparseVec,
)
from retrieval.service import RetrievalService
from retrieval.store import QdrantSearchStore
from retrieval.testing import DOCS, SPEC, FakeEncoder, FakeReranker, point_id, seed


def opts(**kw) -> RetrievalOptions:
    return RetrievalOptions(**kw)


def text(i: int) -> str:
    return DOCS[i - 1][3]


@pytest.fixture(autouse=True)
def _obs_reset():
    shutdown_observability()
    yield
    shutdown_observability()


# ----- search modes ---------------------------------------------------------------------------


def test_each_mode_encodes_only_what_it_needs(rig):
    rig.service.retrieve(text(3), opts(mode="dense"))
    assert (len(rig.encoder.dense_calls), len(rig.encoder.sparse_calls)) == (1, 0)
    rig.service.retrieve(text(3), opts(mode="sparse"))
    assert (len(rig.encoder.dense_calls), len(rig.encoder.sparse_calls)) == (1, 1)
    rig.service.retrieve(text(3), opts(mode="hybrid"))
    assert (len(rig.encoder.dense_calls), len(rig.encoder.sparse_calls)) == (2, 2)


def test_the_encoder_receives_the_collections_own_spec(rig):
    rig.service.retrieve("hello", opts())
    assert rig.encoder.dense_calls[0][1] == SPEC and rig.encoder.sparse_calls[0][1] == SPEC


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_the_exact_text_comes_back_first(rig, mode):
    result = rig.service.retrieve(text(3), opts(mode=mode))
    assert result.hits[0].chunk_id == point_id(3)
    assert result.hits[0].text == text(3)


def test_sparse_finds_a_rare_word(rig):
    result = rig.service.retrieve("zebrafinch", opts(mode="sparse"))
    assert [h.chunk_id for h in result.hits] == [point_id(6)]


def test_result_shape_and_echoed_options(rig):
    result = rig.service.retrieve("  daily walk  ", opts(mode="sparse", top_k=2))
    assert result.query == "daily walk" and result.collection == "docs"
    assert (result.mode, result.rewrite, result.rerank, result.top_k) == (
        "sparse",
        "none",
        False,
        2,
    )
    assert result.rewritten_queries == ["daily walk"] and result.rerank_model is None
    assert [h.rank for h in result.hits] == list(range(1, len(result.hits) + 1))
    hit = result.hits[0]
    assert "text" not in hit.payload and hit.payload["source_name"] == "dogs.txt"
    assert hit.rerank_score is None
    t = result.timings
    assert t.total_ms >= t.retrieve_ms > 0 and t.rewrite_ms == 0 and t.rerank_ms == 0


def test_defaults_come_from_settings(rig):
    result = rig.service.retrieve("water", opts())
    assert (result.top_k, result.candidates, result.collection) == (3, 8, "docs")
    assert len(result.hits) <= 3


# ----- filters ----------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_filters_narrow_every_mode(rig, mode):
    result = rig.service.retrieve(
        "water", opts(mode=mode, filters={"source_name": "fish.txt"}, top_k=5)
    )
    assert result.hits and {h.payload["source_name"] for h in result.hits} == {"fish.txt"}


def test_range_list_and_nested_filters(rig):
    r = rig.service.retrieve("animals", opts(mode="dense", top_k=8, filters={"page": {"gte": 3}}))
    assert {h.payload["page"] for h in r.hits} == {3, 4}
    r = rig.service.retrieve("animals", opts(mode="dense", top_k=8, filters={"page": [1, 2]}))
    assert {h.payload["page"] for h in r.hits} == {1, 2}
    r = rig.service.retrieve(
        "animals", opts(mode="dense", top_k=8, filters={"metadata.year": 2025, "page": 1})
    )
    assert {h.payload["source_name"] for h in r.hits} == {"birds.txt", "fish.txt"}


def test_a_bad_filter_is_a_422(rig):
    with pytest.raises(AppError) as err:
        rig.service.retrieve("x", opts(filters={"bad key": 1}))
    assert err.value.code == "invalid_filter"


# ----- reranking ----------------------------------------------------------------------------------


def test_rerank_fetches_candidates_then_keeps_the_best_top_k(rig):
    result = rig.service.retrieve(
        "dog needs daily walk water", opts(mode="dense", rerank=True, top_k=3, candidates=8)
    )
    (query, texts, model) = rig.reranker.calls[0]
    assert query == "dog needs daily walk water" and len(texts) == 8  # all candidates, not just 3
    assert model == "BAAI/bge-reranker-v2-m3-int8" and result.rerank_model == model
    assert len(result.hits) == 3
    assert result.hits[0].chunk_id == point_id(4)  # the dog/walk/water chunk, whatever dense said
    scores = [h.rerank_score for h in result.hits]
    assert all(s is not None for s in scores) and scores == sorted(scores, reverse=True)
    assert [h.rank for h in result.hits] == [1, 2, 3]
    assert result.timings.rerank_ms >= 0


def test_rerank_can_overturn_the_search_order(rig):
    rig.reranker = FakeReranker(lambda q, t: -float(len(t)))  # prefers the shortest text
    rig.service._reranker = rig.reranker
    result = rig.service.retrieve("animals", opts(mode="dense", rerank=True, top_k=8, candidates=8))
    lengths = [len(h.text) for h in result.hits]
    assert lengths == sorted(lengths)


def test_rerank_model_can_be_overridden_per_request(rig):
    result = rig.service.retrieve("x", opts(rerank=True, rerank_model="other-model"))
    assert rig.reranker.calls[0][2] == "other-model" and result.rerank_model == "other-model"


def test_no_rerank_means_the_reranker_is_never_called(rig):
    rig.service.retrieve("x", opts())
    assert rig.reranker.calls == []


def test_a_wrong_number_of_scores_is_an_error(rig):
    rig.service._reranker = type("Bad", (), {"score": lambda self, q, t, m=None: [1.0]})()
    with pytest.raises(AppError) as err:
        rig.service.retrieve("x", opts(rerank=True))
    assert err.value.code == "rerank_failed"


def test_reranking_nothing_is_fine(rig):
    result = rig.service.retrieve("x", opts(rerank=True, filters={"source_name": "nobody"}))
    assert result.hits == [] and rig.reranker.calls == []


# ----- rewriting ------------------------------------------------------------------------------------


def test_multi_query_searches_every_phrasing_and_merges(rig):
    result = rig.service.retrieve(text(1), opts(mode="dense", rewrite="multi_query", top_k=5))
    assert result.rewritten_queries == [text(1), "a second phrasing", "a third phrasing"]
    assert [q for q, _ in rig.encoder.dense_calls] == result.rewritten_queries
    assert rig.rewriter.calls == [(text(1), "multi_query")]
    ids = [h.chunk_id for h in result.hits]
    assert len(ids) == len(set(ids))  # a chunk found by several queries appears once
    assert ids[0] == point_id(1)  # found at the top by the original query
    assert result.timings.rewrite_ms >= 0


def test_the_reranker_sees_the_original_question_not_a_rewrite(rig):
    rig.service.retrieve("original question", opts(rewrite="hyde", rerank=True))
    assert rig.reranker.calls[0][0] == "original question"


def test_no_rewrite_means_the_rewriter_is_not_called(rig):
    rig.service.retrieve("x", opts(rewrite="none"))
    assert rig.rewriter.calls == []


def test_rewriting_without_a_rewriter_is_a_503(rig):
    service = RetrievalService(rig.settings, rig.store, rig.encoder, rig.reranker, rewriter=None)
    with pytest.raises(AppError) as err:
        service.retrieve("x", opts(rewrite="multi_query"))
    assert err.value.code == "rewriting_unavailable" and err.value.status_code == 503


def test_llm_errors_during_rewriting_pass_through(rig):
    class Failing:
        def rewrite(self, query, strategy):
            raise AppError("llm_not_configured", "OPENAI_API_KEY is not set.", 503)

    rig.service._rewriter = Failing()
    with pytest.raises(AppError) as err:
        rig.service.retrieve("x", opts(rewrite="hyde"))
    assert err.value.code == "llm_not_configured"


# ----- errors and edge cases -----------------------------------------------------------------------------


def test_unknown_collection_is_404_and_unmanaged_is_409(rig):
    with pytest.raises(AppError) as err:
        rig.service.retrieve("x", opts(collection="nope"))
    assert err.value.status_code == 404 and err.value.code == "collection_not_found"
    seed(rig.client, "foreign", spec=None)
    with pytest.raises(AppError) as err:
        rig.service.retrieve("x", opts(collection="foreign"))
    assert err.value.status_code == 409 and err.value.code == "collection_not_managed"


def test_bad_queries_and_options(rig):
    for query in ("", "   "):
        with pytest.raises(AppError) as err:
            rig.service.retrieve(query, opts())
        assert err.value.code == "invalid_query"
    with pytest.raises(AppError, match="longer than"):
        rig.service.retrieve("x" * 3000, opts())
    with pytest.raises(AppError) as err:  # top_k above the default candidates (8)
        rig.service.retrieve("x", opts(top_k=20))
    assert err.value.code == "invalid_options"


def test_a_query_with_no_keyword_terms_falls_back_to_dense_with_a_warning(rig):
    class NoKeywords(FakeEncoder):
        def encode_sparse(self, query, spec):
            return SparseVec(indices=[], values=[])

    rig.service._encoder = NoKeywords()
    hybrid = rig.service.retrieve(text(2), opts(mode="hybrid"))
    assert hybrid.hits[0].chunk_id == point_id(2) and "no keyword terms" in hybrid.warnings[0]
    sparse = rig.service.retrieve(text(2), opts(mode="sparse"))
    assert sparse.hits == [] and "no results" in sparse.warnings[0]


def test_qdrant_down_is_a_clear_503():
    down = QdrantSearchStore(
        QdrantClient(url="http://127.0.0.1:9", timeout=1, check_compatibility=False)
    )
    from retrieval.settings import RetrievalSettings

    service = RetrievalService(
        RetrievalSettings(_env_file=None), down, FakeEncoder(), FakeReranker()
    )
    with pytest.raises(AppError) as err:
        service.retrieve("x", opts())
    assert err.value.status_code == 503 and "make up" in err.value.message


# ----- tracing ---------------------------------------------------------------------------------------------


def test_one_retrieval_is_one_trace_with_a_span_per_stage(rig):
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    result = rig.service.retrieve("water", opts(rewrite="multi_query", rerank=True))
    spans = {}
    for s in exporter.spans:
        spans.setdefault(s.name, []).append(s)
    root = spans["retrieval.retrieve"][0]
    assert len(spans["retrieval.search"]) == 3  # one per phrasing
    assert len(spans["retrieval.rewrite"]) == 1 and len(spans["retrieval.rerank"]) == 1
    assert {s.parent_id for n in ("retrieval.search", "retrieval.rewrite", "retrieval.rerank")
            for s in spans[n]} == {root.span_id}  # fmt: skip
    assert len({s.trace_id for s in exporter.spans}) == 1 and result.trace_id == root.trace_id
    assert root.attributes["retrieval.mode"] == "hybrid" and root.attributes["retrieval.hits"] >= 1


def test_works_with_observability_off(rig):
    assert rig.service.retrieve("water", opts()).trace_id is None


# ----- the pieces on their own ---------------------------------------------------------------------------------


def test_rewrite_on_its_own(rig):
    out = rig.service.rewrite("why do cats purr", "step_back")
    assert out.rewritten_queries[0] == "why do cats purr" and len(out.rewritten_queries) == 3


def test_rerank_on_its_own_orders_and_trims(rig):
    out = rig.service.rerank(
        RerankRequest(
            query="cat purr",
            candidates=[
                Candidate(text="nothing relevant here", chunk_id="a"),
                Candidate(text="the cat will purr", chunk_id="b"),
                Candidate(text="a cat", chunk_id="c"),
            ],
            top_n=2,
        )
    )
    assert [(i.chunk_id, i.original_rank) for i in out.items] == [("b", 2), ("c", 3)]
    assert out.items[0].rerank_score > out.items[1].rerank_score and out.model


def test_strategies_lists_modes_models_and_collections(rig):
    s = rig.service.strategies()
    assert s.modes == ["dense", "sparse", "hybrid"]
    assert s.rewrite_strategies == ["none", "multi_query", "hyde", "step_back"]
    assert s.rerank_models[0].name == "fake-reranker"
    assert s.defaults["top_k"] == 3 and s.collections[0].name == "docs"
    assert s.collections[0].spec == SPEC
