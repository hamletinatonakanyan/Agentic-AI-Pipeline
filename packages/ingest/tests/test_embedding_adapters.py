"""The fastembed adapters, with a stub model: no download, no network."""

import numpy as np

from ingest.embedding import FastEmbedDense, FastEmbedSparse


class StubDenseModel:
    def __init__(self) -> None:
        self.seen: list[str] = []
        self.batch_size = None

    def embed(self, texts, batch_size=None):
        self.seen += list(texts)
        self.batch_size = batch_size
        for t in texts:
            yield np.array([float(len(t)), 1.0, 0.0], dtype=np.float32)


def test_dense_adds_the_document_prefix_and_returns_plain_lists():
    dense = FastEmbedDense("m", doc_prefix="title: none | text: ", batch_size=7)
    stub = StubDenseModel()
    dense._model = stub  # skip the download
    out = dense.embed_documents(["hello", "world!"])
    assert stub.seen == ["title: none | text: hello", "title: none | text: world!"]
    assert stub.batch_size == 7
    assert out[0][0] == len("title: none | text: hello")
    assert isinstance(out[0], list) and isinstance(out[0][0], float)


def test_dense_without_prefix_sends_text_unchanged():
    dense = FastEmbedDense("m")
    dense._model = stub = StubDenseModel()
    dense.embed_documents(["abc"])
    assert stub.seen == ["abc"]


def test_gemma_dimension_comes_from_the_registry_without_loading_the_model():
    dense = FastEmbedDense("google/embeddinggemma-300m")
    assert dense.dim == 768
    assert dense._model is None  # nothing was downloaded or loaded


class StubSparseEmbedding:
    def __init__(self, indices, values) -> None:
        self.indices = np.array(indices)
        self.values = np.array(values, dtype=np.float32)


class StubSparseModel:
    def embed(self, texts):
        for _ in texts:
            yield StubSparseEmbedding([3, 9], [1.5, 0.5])


def test_sparse_converts_embeddings_and_keeps_one_model_per_language():
    sparse = FastEmbedSparse()
    sparse._models["english"] = StubSparseModel()
    sparse._models["none"] = StubSparseModel()
    (vec,) = sparse.embed_documents(["x"], "english")
    assert vec.indices == [3, 9] and vec.values == [1.5, 0.5]
    assert sparse._load("english") is sparse._models["english"]
    assert sparse._load("none") is not sparse._load("english")
