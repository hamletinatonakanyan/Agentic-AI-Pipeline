"""Evals test helpers: a service wired to fakes. No network, no LLM, no Qdrant."""

import pytest

from common import AppError
from evals.generator import QuestionGenerator
from evals.models import (
    AccuracyResult,
    Claim,
    CompletenessResult,
    CoveragePoint,
    FaithfulnessResult,
    TargetResponse,
)
from evals.protocols import ChunkRef
from evals.service import EvalService, Preset
from evals.settings import EvalSettings
from evals.store import EvalStore
from evals.targets import CallableTarget, TargetError
from llm.testing import FakeLLM


class FakeJudge:
    model = "fake-judge"
    prompt_version = "t1"

    def __init__(self, faith=0.5, comp=0.75, acc=0.8) -> None:
        self.faith, self.comp, self.acc = faith, comp, acc
        self.calls: list[str] = []
        self.raise_on: dict[str, Exception] = {}

    def _maybe_raise(self, name: str) -> None:
        self.calls.append(name)
        if name in self.raise_on:
            raise self.raise_on[name]

    def score_faithfulness(self, answer, contexts):
        self._maybe_raise("faithfulness")
        return FaithfulnessResult(
            claims=[Claim(text="c", supported=True, evidence="e")], score=self.faith
        )

    def score_completeness(self, answer, reference):
        self._maybe_raise("completeness")
        return CompletenessResult(
            points=[CoveragePoint(point="p", covered=True, note="n")], score=self.comp
        )

    def score_accuracy(self, question, answer, reference):
        self._maybe_raise("accuracy")
        return AccuracyResult(score=self.acc, rationale="because")


class FakeChunks:
    def __init__(self, chunks: list[ChunkRef] | None = None) -> None:
        self.chunks = chunks if chunks is not None else [
            ChunkRef(chunk_id=f"c{i}", text=f"Chunk number {i} " * 20, source_name="a.txt", page=i)
            for i in range(1, 7)
        ]  # fmt: skip
        self.calls: list[tuple] = []

    def list_collections(self):
        return ["docs"]

    def sample(self, collection, n, seed, min_chars):
        self.calls.append((collection, n, seed, min_chars))
        pool = [c for c in self.chunks if len(c.text) >= min_chars]
        return pool[:n]


class Scripted:
    """A target whose replies are decided per question. Records what it was asked."""

    def __init__(self, replies: dict[str, TargetResponse | Exception], default=None) -> None:
        self.replies, self.default, self.asked = replies, default, []

    def __call__(self, question: str) -> TargetResponse:
        self.asked.append(question)
        reply = self.replies.get(question, self.default)
        if isinstance(reply, Exception):
            raise reply
        if reply is None:
            raise TargetError("no scripted reply")
        return reply


class Rig:
    def __init__(self, **settings) -> None:
        self.settings = EvalSettings(_env_file=None, db_path=":memory:", **settings)
        self.store = EvalStore(":memory:")
        self.judge = FakeJudge()
        self.llm = FakeLLM(
            lambda prompt: '{"question": "What is it about?", "answer": "It is about chunks."}'
        )
        self.chunks = FakeChunks()
        self.script = Scripted({})
        self.presets = {
            "scripted": Preset(lambda options: CallableTarget(self.script), returns_answer=True),
            "ids_only": Preset(lambda options: CallableTarget(self.script), returns_answer=False),
        }
        self.service = EvalService(
            self.settings, self.store, self.judge, QuestionGenerator(self.llm),
            self.chunks, self.presets, inline=True,
        )  # fmt: skip


def jsonl(*items: dict) -> bytes:
    import json

    return "\n".join(json.dumps(i) for i in items).encode()


@pytest.fixture
def rig() -> Rig:
    return Rig()


@pytest.fixture
def rig_factory():
    return Rig


@pytest.fixture
def make_jsonl():
    return jsonl


@pytest.fixture
def app_error():
    return AppError
