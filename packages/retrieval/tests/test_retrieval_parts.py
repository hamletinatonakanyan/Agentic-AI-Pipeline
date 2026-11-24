import pytest
from qdrant_client import QdrantClient

from common import AppError
from llm.testing import FakeLLM
from retrieval.encoders import FastEmbedQueryEncoder, FastEmbedReranker
from retrieval.filters import build_filter
from retrieval.fusion import rrf_merge
from retrieval.models import RawHit, RetrievalOptions
from retrieval.rewriting import LLMRewriter
from retrieval.store import QdrantSearchStore
from retrieval.testing import DOCS, SPEC, point_id, seed

# ----- filters -------------------------------------------------------------------------------


def test_filters_build_must_conditions():
    flt = build_filter(
        {"source_name": "a.pdf", "page": [1, 2], "metadata.year": {"gte": 2024, "lt": 2026}}
    )
    keys = {c.key for c in flt.must}
    assert keys == {"source_name", "page", "metadata.year"}
    by_key = {c.key: c for c in flt.must}
    assert by_key["source_name"].match.value == "a.pdf"
    assert by_key["page"].match.any == [1, 2]
    assert (by_key["metadata.year"].range.gte, by_key["metadata.year"].range.lt) == (2024, 2026)


def test_no_filters_means_none():
    assert build_filter({}) is None


@pytest.mark.parametrize(
    "bad",
    [
        {"bad key": 1},
        {"../x": 1},
        {"a": None},
        {"a": []},
        {"a": [None]},
        {"a": {"between": 1}},
        {"a": {"gte": "x"}},
        {"a": {}},
        {"a": object()},
    ],
)
def test_bad_filters_are_422(bad):
    with pytest.raises(AppError) as err:
        build_filter(bad)
    assert err.value.code == "invalid_filter" and err.value.status_code == 422


# ----- rank fusion ----------------------------------------------------------------------------


def hit(i, score=0.0):
    return RawHit(id=f"c{i}", score=score, payload={"text": f"t{i}"})


def test_rrf_scores_follow_the_formula():
    merged = rrf_merge([[hit(1), hit(2)], [hit(2), hit(3)]], k=60)
    scores = {h.id: h.score for h in merged}
    assert scores["c2"] == pytest.approx(1 / 62 + 1 / 61)  # rank 2 in one list, rank 1 in the other
    assert scores["c1"] == pytest.approx(1 / 61)
    assert scores["c3"] == pytest.approx(1 / 62)
    assert [h.id for h in merged] == ["c2", "c1", "c3"]


def test_rrf_keeps_one_copy_and_breaks_ties_by_best_rank():
    merged = rrf_merge([[hit(1), hit(2)], [hit(2), hit(1)]])
    assert [h.id for h in merged] == [
        "c1",
        "c2",
    ]  # equal score: the one ranked first somewhere wins
    assert len(merged) == 2


def test_rrf_with_one_list_keeps_its_order_and_empty_input_is_empty():
    assert [h.id for h in rrf_merge([[hit(3), hit(1), hit(2)]])] == ["c3", "c1", "c2"]
    assert rrf_merge([]) == [] and rrf_merge([[], []]) == []


# ----- the Qdrant adapter ----------------------------------------------------------------------


@pytest.fixture
def store():
    client = QdrantClient(":memory:")
    seed(client)
    return QdrantSearchStore(client)


def test_get_collection_reads_the_spec_ingest_wrote(store):
    info = store.get_collection("docs")
    assert info.points == len(DOCS) and info.spec == SPEC
    assert store.get_collection("nope") is None
    assert [c.name for c in store.list_collections()] == ["docs"]


def test_a_collection_without_a_spec_is_reported_without_one():
    client = QdrantClient(":memory:")
    seed(client, "foreign", spec=None)
    assert QdrantSearchStore(client).get_collection("foreign").spec is None


def test_unknown_spec_fields_do_not_break_reading():
    client = QdrantClient(":memory:")
    seed(client, "x", spec=SPEC)
    info = client.get_collection("x")
    assert QdrantSearchStore(client).get_collection("x").spec.sparse_language == "english"
    assert info.config.metadata


def encoder_vectors(text):
    from retrieval.testing import FakeEncoder

    e = FakeEncoder()
    return e.encode_dense(text, SPEC), e.encode_sparse(text, SPEC)


def test_dense_finds_the_identical_text_first(store):
    dense, _ = encoder_vectors(DOCS[4][3])
    hits = store.search("docs", dense=dense, sparse=None, limit=3, filters={})
    assert hits[0].id == point_id(5) and hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].payload["text"] == DOCS[4][3]


def test_sparse_matches_on_words(store):
    _, sparse = encoder_vectors("zebrafinch")
    hits = store.search("docs", dense=None, sparse=sparse, limit=3, filters={})
    assert [h.id for h in hits][:1] == [point_id(6)]  # the only chunk with that word


def test_hybrid_ranks_a_chunk_found_by_both_first(store):
    dense, _ = encoder_vectors(DOCS[0][3])  # exact text of doc 1
    _, sparse = encoder_vectors("purr")  # a word only doc 1 has
    hits = store.search("docs", dense=dense, sparse=sparse, limit=3, filters={})
    assert hits[0].id == point_id(1)


def test_nothing_to_search_with_returns_nothing(store):
    assert store.search("docs", dense=None, sparse=None, limit=3, filters={}) == []


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_filters_apply_in_every_mode(store, mode):
    dense, sparse = encoder_vectors("water")
    kw = {
        "dense": dense if mode != "sparse" else None,
        "sparse": sparse if mode != "dense" else None,
    }
    hits = store.search("docs", limit=8, filters={"source_name": "dogs.txt"}, **kw)
    assert hits and {h.payload["source_name"] for h in hits} == {"dogs.txt"}
    nested = store.search("docs", limit=8, filters={"metadata.year": {"gte": 2025}}, **kw)
    assert nested and all(h.payload["metadata"]["year"] >= 2025 for h in nested)


# ----- query rewriting -----------------------------------------------------------------------------


def test_none_returns_just_the_query():
    llm = FakeLLM()
    assert LLMRewriter(llm).rewrite("q", "none") == ["q"] and llm.calls == []


def test_multi_query_keeps_the_original_first_and_drops_duplicates():
    llm = FakeLLM(['{"queries": ["other words", "Q", "another angle", "other words", "a fourth"]}'])
    out = LLMRewriter(llm, extra_queries=3).rewrite("q", "multi_query")
    assert out == ["q", "other words", "another angle", "a fourth"]  # original, then 3 distinct
    call = llm.calls[0]
    assert call.kind == "complete_json" and "Question: q" in call.prompt
    assert "same language" in call.system


def test_hyde_searches_with_the_original_and_a_written_passage():
    llm = FakeLLM("Cats purr because they are content.")
    assert LLMRewriter(llm).rewrite("why cats purr", "hyde") == [
        "why cats purr",
        "Cats purr because they are content.",
    ]
    assert llm.calls[0].kind == "complete"


def test_step_back_adds_a_more_general_question():
    llm = FakeLLM("How do domestic animals communicate?")
    assert LLMRewriter(llm).rewrite("why cats purr", "step_back")[1] == (
        "How do domestic animals communicate?"
    )


def test_an_empty_rewrite_leaves_just_the_original():
    assert LLMRewriter(FakeLLM("  ")).rewrite("q", "hyde") == ["q"]


# ----- encoders (stub models: no download) ----------------------------------------------------------------


class StubDense:
    def __init__(self):
        self.seen = []

    def embed(self, texts):
        self.seen += list(texts)
        for t in texts:
            yield __import__("numpy").array([float(len(t)), 0.0], dtype="float32")


class StubSparseEmb:
    def __init__(self):
        import numpy as np

        self.indices = np.array([5, 9])
        self.values = np.array([1, 1])


class StubSparse:
    def __init__(self):
        self.queries = []

    def query_embed(self, query):
        self.queries.append(query)
        yield StubSparseEmb()


def test_dense_query_gets_the_collections_query_prefix():
    enc = FastEmbedQueryEncoder()
    enc._dense["fake-dense"] = stub = StubDense()
    vec = enc.encode_dense("hello", SPEC)  # SPEC.query_prefix == "Q: "
    assert stub.seen == ["Q: hello"] and vec == [float(len("Q: hello")), 0.0]


def test_sparse_query_uses_the_collections_language_model():
    enc = FastEmbedQueryEncoder()
    german = SPEC.model_copy(update={"sparse_language": "german"})
    enc._sparse[("Qdrant/bm25", "german")] = stub = StubSparse()
    enc._sparse[("Qdrant/bm25", "english")] = other = StubSparse()
    out = enc.encode_sparse("hallo", german)
    assert stub.queries == ["hallo"] and other.queries == []
    assert out.indices == [5, 9] and out.values == [1.0, 1.0]


def test_reranker_scores_come_back_as_floats_and_the_default_model_is_used():
    class StubCross:
        def rerank(self, query, texts):
            return [len(t) for t in texts]

    rr = FastEmbedReranker("default-model")
    rr._models["default-model"] = StubCross()
    assert rr.score("q", ["a", "bbb"]) == [1.0, 3.0]


def test_the_rerank_model_list_comes_from_fastembeds_registry_without_downloading():
    models = FastEmbedReranker("x").models()
    names = {m.name for m in models}
    assert "BAAI/bge-reranker-v2-m3-int8" in names and all(m.size_gb for m in models)


def test_options_reject_candidates_below_top_k():
    with pytest.raises(ValueError):
        RetrievalOptions(top_k=10, candidates=5)


def test_hybrid_prefetches_wide_so_a_keyword_hit_still_gets_credit_from_the_dense_list(store):
    """Bug found by the e2e contract test: with prefetch == limit, a chunk that is #1 for keywords
    but not in the dense top-k lost the bonus from being in both lists."""
    dense, _ = encoder_vectors("something unrelated to anything")
    _, sparse = encoder_vectors("Jupiter")  # only the space chunk matches the keyword
    narrow = store.search("docs", dense=dense, sparse=sparse, limit=2, filters={})
    wide = store.search("docs", dense=dense, sparse=sparse, limit=2, filters={}, prefetch_limit=8)
    assert wide[0].id == point_id(8)  # found by both lists, so it wins
    assert wide[0].score > max(h.score for h in narrow if h.id != point_id(8))
