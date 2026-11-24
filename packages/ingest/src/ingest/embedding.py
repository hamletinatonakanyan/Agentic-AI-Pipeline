"""Local embeddings through fastembed (ONNX, no API key, CPU).

Models download from Hugging Face the first time they are used, not when the app starts and not in
tests. Both adapters load lazily.
"""

import threading
from typing import Any

from ingest.models import SparseVec
from ingest.settings import NO_LANGUAGE

# (query prefix, document prefix). fastembed's `passage_embed` does NOT add these for the models
# below, so we add them ourselves, and store both prefixes on the collection so that retrieval
# encodes queries the same way.
MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: "),
}
_E5_PREFIXES = ("query: ", "passage: ")


def resolve_prefixes(
    model: str, query_override: str | None = None, doc_override: str | None = None
) -> tuple[str, str]:
    """(query_prefix, doc_prefix): the setting if given, else the built-in value for the model."""
    default_query, default_doc = MODEL_PREFIXES.get(model, ("", ""))
    if not default_query and "e5" in model.lower():
        default_query, default_doc = _E5_PREFIXES
    return (
        default_query if query_override is None else query_override,
        default_doc if doc_override is None else doc_override,
    )


class FastEmbedDense:
    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self._batch = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        # Reads the model's registry entry. Does not download or load the model.
        from fastembed import TextEmbedding

        return int(TextEmbedding.get_embedding_size(self.model_name))

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self.model_name)
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        prefixed = [self.doc_prefix + t for t in texts]
        return [v.tolist() for v in model.embed(prefixed, batch_size=self._batch)]


class FastEmbedSparse:
    """BM25 sparse vectors. One model instance per language, since the stemmer is per language."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self.model_name = model_name
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _load(self, language: str) -> Any:
        with self._lock:
            if language not in self._models:
                from fastembed import SparseTextEmbedding

                if language == NO_LANGUAGE:
                    # No stemming and no stop words: works for any language.
                    model = SparseTextEmbedding(self.model_name, disable_stemmer=True)
                else:
                    model = SparseTextEmbedding(self.model_name, language=language)
                self._models[language] = model
            return self._models[language]

    def embed_documents(self, texts: list[str], language: str) -> list[SparseVec]:
        model = self._load(language)
        return [
            SparseVec(indices=e.indices.tolist(), values=e.values.tolist())
            for e in model.embed(texts)
        ]
