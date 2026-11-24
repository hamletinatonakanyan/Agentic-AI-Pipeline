"""Local query encoding and reranking through fastembed. Models download on first real use."""

import threading
from typing import Any

from retrieval.models import CollectionSpec, RerankModelInfo, SparseVec


class FastEmbedQueryEncoder:
    """Encodes a query the way the collection's documents were encoded, using the collection's
    own spec: its embedding model, its query prefix, and its BM25 language."""

    def __init__(self) -> None:
        self._dense: dict[str, Any] = {}
        self._sparse: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()

    def _dense_model(self, name: str) -> Any:
        with self._lock:
            if name not in self._dense:
                from fastembed import TextEmbedding

                self._dense[name] = TextEmbedding(name)
            return self._dense[name]

    def _sparse_model(self, name: str, language: str) -> Any:
        with self._lock:
            if (name, language) not in self._sparse:
                from fastembed import SparseTextEmbedding

                if language == "none":
                    model = SparseTextEmbedding(name, disable_stemmer=True)
                else:
                    model = SparseTextEmbedding(name, language=language)
                self._sparse[(name, language)] = model
            return self._sparse[(name, language)]

    def encode_dense(self, query: str, spec: CollectionSpec) -> list[float]:
        model = self._dense_model(spec.embedding_model)
        (vector,) = list(model.embed([spec.query_prefix + query]))
        return [float(x) for x in vector]

    def encode_sparse(self, query: str, spec: CollectionSpec) -> SparseVec:
        model = self._sparse_model(spec.sparse_model, spec.sparse_language)
        (emb,) = list(model.query_embed(query))
        return SparseVec(indices=emb.indices.tolist(), values=[float(v) for v in emb.values])


class FastEmbedReranker:
    """A cross-encoder: reads the query and one chunk together and scores the pair."""

    def __init__(self, default_model: str) -> None:
        self._default = default_model
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _load(self, name: str) -> Any:
        with self._lock:
            if name not in self._models:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._models[name] = TextCrossEncoder(name)
            return self._models[name]

    def score(self, query: str, texts: list[str], model: str | None = None) -> list[float]:
        return [float(s) for s in self._load(model or self._default).rerank(query, texts)]

    def models(self) -> list[RerankModelInfo]:
        """The models fastembed supports. Reads its registry, downloads nothing."""
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return [
            RerankModelInfo(
                name=m["model"], size_gb=m.get("size_in_GB"), description=m.get("description", "")
            )
            for m in TextCrossEncoder.list_supported_models()
        ]
