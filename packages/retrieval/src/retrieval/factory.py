"""Builds the real thing from settings. Nothing connects or downloads at build time."""

from dataclasses import dataclass

from qdrant_client import QdrantClient

from llm import LLM, LLMSettings, OpenAILLM
from retrieval.encoders import FastEmbedQueryEncoder, FastEmbedReranker
from retrieval.protocols import QueryEncoder, QueryRewriter, Reranker
from retrieval.rewriting import LLMRewriter
from retrieval.service import RetrievalService
from retrieval.settings import RetrievalSettings
from retrieval.store import QdrantSearchStore


@dataclass
class Retrieval:
    service: RetrievalService

    def close(self) -> None:
        pass


def build_retrieval(
    settings: RetrievalSettings,
    *,
    client: QdrantClient | None = None,
    encoder: QueryEncoder | None = None,
    reranker: Reranker | None = None,
    rewriter: QueryRewriter | None = None,
    llm: LLM | None = None,
) -> Retrieval:
    """The real wiring. Tests inject an in-memory `client` and fakes."""
    client = client or QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=30,
        check_compatibility=False,
    )
    # The rewriter needs an LLM, but a missing OPENAI_API_KEY only matters if a rewrite is asked
    # for: the LLM raises a clear error at that point, so retrieval without rewriting still works.
    rewriter = rewriter or LLMRewriter(
        llm or OpenAILLM(LLMSettings()), extra_queries=settings.rewrite_queries
    )
    service = RetrievalService(
        settings=settings,
        store=QdrantSearchStore(client),
        encoder=encoder or FastEmbedQueryEncoder(),
        reranker=reranker or FastEmbedReranker(settings.rerank_model),
        rewriter=rewriter,
    )
    return Retrieval(service=service)
