"""Live tests for ingestion: `make test-live`.

They need network access and LLAMA_CLOUD_API_KEY in .env (the LlamaParse test spends about one
credit), and they download the embedding models once (about 1.2 GB for Gemma). They check the
LlamaParse client against the real service.
"""

import os

import pytest

pytestmark = pytest.mark.live


def test_llamaparse_really_parses_a_page(make_pdf):
    from dotenv import dotenv_values

    key = os.environ.get("LLAMA_CLOUD_API_KEY") or dotenv_values(".env").get("LLAMA_CLOUD_API_KEY")
    if not key:
        pytest.skip("LLAMA_CLOUD_API_KEY is not set")

    from ingest.parsers.llamaparse import LlamaParseClient

    client = LlamaParseClient(api_key=key, tier="fast", timeout_s=180)
    pdf = make_pdf(["Quarterly revenue grew twelve percent in the northern region."])
    result = client.parse(pdf, [1])
    assert list(result) == [1]
    assert "revenue" in result[1].lower()


def test_gemma_and_bm25_really_embed():
    from ingest.embedding import FastEmbedDense, FastEmbedSparse, resolve_prefixes

    query_prefix, doc_prefix = resolve_prefixes("google/embeddinggemma-300m")
    dense = FastEmbedDense("google/embeddinggemma-300m", query_prefix, doc_prefix)
    vectors = dense.embed_documents(["The cat sat on the mat.", "Le chat est sur le tapis."])
    assert len(vectors) == 2 and len(vectors[0]) == dense.dim == 768
    assert vectors[0] != vectors[1]

    sparse = FastEmbedSparse()
    for language in ("english", "german", "none"):
        (vec,) = sparse.embed_documents(["Hello wonderful world"], language)
        assert vec.indices and len(vec.indices) == len(vec.values)
