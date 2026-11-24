from typing import Protocol

from pydantic import BaseModel

from evals.models import (
    AccuracyResult,
    CompletenessResult,
    FaithfulnessResult,
    TargetResponse,
)


class Target(Protocol):
    def ask(self, question: str) -> TargetResponse: ...


class Judge(Protocol):
    model: str
    prompt_version: str

    def score_faithfulness(self, answer: str, contexts: list[str]) -> FaithfulnessResult: ...

    def score_completeness(self, answer: str, reference: str) -> CompletenessResult: ...

    def score_accuracy(self, question: str, answer: str, reference: str) -> AccuracyResult: ...


class ChunkRef(BaseModel):
    chunk_id: str
    text: str
    source_name: str = ""
    page: int | None = None


class ChunkSource(Protocol):
    """Where question drafting reads chunks from. The default reads Qdrant directly, so this
    package does not import ingest."""

    def list_collections(self) -> list[str]: ...

    def sample(self, collection: str, n: int, seed: int, min_chars: int) -> list[ChunkRef]: ...
