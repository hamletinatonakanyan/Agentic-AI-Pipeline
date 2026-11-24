"""Long-term memory: facts about a user, kept across conversations, searched by meaning.

Built on LangGraph's Store. Facts live under the namespace ("users", user_id, "facts"), so one
user's facts are never returned for another user. Each fact is embedded, and a search returns the
facts closest in meaning to a question.

Gemma-style models need different prefixes for stored texts and for queries. The Store applies one
function to both, so the document prefix is applied when a fact is saved (in a separate field that
is the one embedded) and the query prefix when searching.
"""

import sqlite3
import threading
import time
import uuid
from pathlib import Path

from langgraph.store.sqlite import SqliteStore

from agent_graph.models import Fact
from agent_graph.protocols import FactEmbedder


class FactMemory:
    def __init__(
        self,
        path: str,
        embedder: FactEmbedder,
        doc_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._store = SqliteStore(
            self._conn,
            index={"embed": embedder.embed, "dims": embedder.dim, "fields": ["embed_text"]},
        )
        self._store.setup()
        self._doc, self._query = doc_prefix, query_prefix
        self._lock = threading.Lock()

    @property
    def store(self) -> SqliteStore:
        """The underlying LangGraph store, for compiling it into the graph."""
        return self._store

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _ns(user_id: str) -> tuple[str, str, str]:
        return ("users", user_id, "facts")

    def add(self, user_id: str, text: str) -> Fact:
        text = " ".join(text.split())
        with self._lock:
            for existing in self._all(user_id):
                if existing.text.lower() == text.lower():
                    return existing  # saying the same thing twice is not two facts
            fact_id = uuid.uuid4().hex[:12]
            now = time.time()
            self._store.put(
                self._ns(user_id),
                fact_id,
                {"text": text, "embed_text": self._doc + text, "created_at": now},
            )
        return Fact(id=fact_id, text=text, created_at=now)

    def _all(self, user_id: str) -> list[Fact]:
        items = self._store.search(self._ns(user_id), limit=1000)
        facts = [
            Fact(id=i.key, text=i.value["text"], created_at=i.value.get("created_at"))
            for i in items
        ]
        return sorted(facts, key=lambda f: f.created_at or 0)

    def all_facts(self, user_id: str) -> list[Fact]:
        with self._lock:
            return self._all(user_id)

    def delete(self, user_id: str, fact_id: str) -> bool:
        with self._lock:
            if self._store.get(self._ns(user_id), fact_id) is None:
                return False
            self._store.delete(self._ns(user_id), fact_id)
            return True

    def search(self, user_id: str, query: str, k: int, min_score: float = 0.0) -> list[Fact]:
        """The facts closest in meaning to `query`, best first, above `min_score`."""
        with self._lock:
            hits = self._store.search(self._ns(user_id), query=self._query + query, limit=k)
        return [
            Fact(
                id=h.key,
                text=h.value["text"],
                created_at=h.value.get("created_at"),
                score=h.score,
            )
            for h in hits
            if (h.score or 0.0) >= min_score
        ]


class FastEmbedFactEmbedder:
    """Local embeddings through fastembed. The model downloads on first real use."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        from fastembed import TextEmbedding

        return int(TextEmbedding.get_embedding_size(self.model_name))

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self.model_name)
        return [[float(x) for x in v] for v in self._model.embed(texts)]
