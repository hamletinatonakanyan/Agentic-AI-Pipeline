"""Live tests for retrieval: `make test-live`.

They download models on first use (the Gemma embedding model about 1.2 GB, the reranker about
0.6 GB). The rewriting test needs OPENAI_API_KEY in .env and costs a fraction of a cent.
"""

import os

import pytest

pytestmark = pytest.mark.live


def test_the_real_encoder_and_reranker_work_together():
    from retrieval.encoders import FastEmbedQueryEncoder, FastEmbedReranker
    from retrieval.models import CollectionSpec

    spec = CollectionSpec(
        embedding_model="google/embeddinggemma-300m", dense_dim=768, sparse_model="Qdrant/bm25",
        sparse_language="english", query_prefix="task: search result | query: ",
    )  # fmt: skip
    enc = FastEmbedQueryEncoder()
    vec = enc.encode_dense("Why do cats purr?", spec)
    assert len(vec) == 768
    sparse = enc.encode_sparse("Why do cats purr?", spec)
    assert sparse.indices and len(sparse.indices) == len(sparse.values)

    scores = FastEmbedReranker("BAAI/bge-reranker-v2-m3-int8").score(
        "Why do cats purr?",
        ["Cats purr when they are content or soothing themselves.", "Jupiter is a gas giant."],
    )
    assert scores[0] > scores[1]  # the relevant passage is ranked above the unrelated one


def test_query_rewriting_with_the_real_llm():
    from dotenv import dotenv_values

    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(".env").get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY is not set")
    from llm import LLMSettings, OpenAILLM
    from retrieval.rewriting import LLMRewriter

    rewriter = LLMRewriter(OpenAILLM(LLMSettings(_env_file=None, openai_api_key=key)))
    multi = rewriter.rewrite("Why do cats purr?", "multi_query")
    assert multi[0] == "Why do cats purr?" and len(multi) >= 2
    assert len(rewriter.rewrite("Why do cats purr?", "hyde")) == 2
