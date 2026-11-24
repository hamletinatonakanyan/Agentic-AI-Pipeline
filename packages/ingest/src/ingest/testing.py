"""Fakes for tests in any package: no model download, no network."""

import hashlib

from ingest.models import SparseVec

PUNCT = ".,;:!?\"'()"  # identical in ingest.testing and retrieval.testing


class FakeDense:
    model_name = "fake-dense"
    doc_prefix = ""
    query_prefix = ""
    dim = 4

    def __init__(self) -> None:
        self.calls = 0
        self.texts_embedded = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:4]] for t in texts]


class FakeSparse:
    model_name = "Qdrant/bm25"

    def embed_documents(self, texts: list[str], language: str) -> list[SparseVec]:
        out = []
        for t in texts:
            words = sorted({w.strip(PUNCT).lower() for w in t.split()} - {""})
            out.append(
                SparseVec(
                    indices=[int(hashlib.md5(w.encode()).hexdigest()[:6], 16) for w in words],
                    values=[1.0] * len(words),
                )
            )
        return out
