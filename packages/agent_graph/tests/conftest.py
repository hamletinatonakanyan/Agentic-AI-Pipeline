"""A safeguard: no test may load the real embedding model (a ~0.3 GB download). Live tests, which
are run explicitly with `make test-live`, are exempt."""

import pytest


@pytest.fixture(autouse=True)
def never_load_the_real_embedding_model(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        return
    from agent_graph.memory import FastEmbedFactEmbedder

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to load the real embedding model: pass FakeEmbedder")

    monkeypatch.setattr(FastEmbedFactEmbedder, "embed", refuse)
    monkeypatch.setattr(FastEmbedFactEmbedder, "dim", property(refuse))
