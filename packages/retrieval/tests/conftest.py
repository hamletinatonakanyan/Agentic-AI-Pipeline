"""Retrieval test helpers. The collection is seeded directly, in the layout ingest writes."""

import pytest
from qdrant_client import QdrantClient

from retrieval.service import RetrievalService
from retrieval.settings import RetrievalSettings
from retrieval.store import QdrantSearchStore
from retrieval.testing import FakeEncoder, FakeReranker, FakeRewriter, seed


class Rig:
    def __init__(self, **settings) -> None:
        self.client = QdrantClient(":memory:")
        seed(self.client)
        self.settings = RetrievalSettings(
            _env_file=None, collection="docs", top_k=3, candidates=8, **settings
        )
        self.store = QdrantSearchStore(self.client)
        self.encoder = FakeEncoder()
        self.reranker = FakeReranker()
        self.rewriter = FakeRewriter()
        self.service = RetrievalService(
            self.settings, self.store, self.encoder, self.reranker, self.rewriter
        )


@pytest.fixture
def rig() -> Rig:
    return Rig()


@pytest.fixture
def rig_factory():
    return Rig
