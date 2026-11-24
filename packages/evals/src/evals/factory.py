"""Builds the real thing from settings. Nothing connects at build time."""

from dataclasses import dataclass

from qdrant_client import QdrantClient

from evals.chunk_source import QdrantChunkSource
from evals.generator import QuestionGenerator
from evals.judge import LLMJudge
from evals.protocols import ChunkSource, Judge
from evals.service import EvalService, Preset
from evals.settings import EvalSettings
from evals.store import EvalStore
from evals.targets import build_retrieval_http_target
from llm import LLM, LLMSettings, OpenAILLM


@dataclass
class Evals:
    service: EvalService

    def close(self) -> None:
        self.service.close()


def build_evals(
    settings: EvalSettings,
    *,
    presets: dict[str, Preset] | None = None,
    client: QdrantClient | None = None,
    chunks: ChunkSource | None = None,
    llm: LLM | None = None,
    judge: Judge | None = None,
    inline: bool = False,
    http_client=None,
) -> Evals:
    """The real wiring. Tests inject fakes. The judge and the question drafter use the global LLM
    settings, unless EVAL_JUDGE_MODEL names another model."""
    llm_settings = LLMSettings()
    if settings.judge_model:
        llm_settings = llm_settings.model_copy(update={"model": settings.judge_model})
    llm = llm or OpenAILLM(llm_settings)
    judge = judge or LLMJudge(llm, llm_settings.model, settings.judge_prompt_version)
    client = client or QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=30,
        check_compatibility=False,
    )
    # Built-in: evaluate a running retrieval service over HTTP. e2e adds an in-process "retrieval".
    all_presets = {
        "retrieval_http": Preset(
            factory=lambda options: build_retrieval_http_target(
                options, settings, client=http_client
            ),
            returns_answer=False,
            description="A running retrieval service, over HTTP (retrieval metrics only).",
        ),
        **(presets or {}),
    }
    service = EvalService(
        settings=settings,
        store=EvalStore(settings.db_path),
        judge=judge,
        generator=QuestionGenerator(llm),
        chunks=chunks or QdrantChunkSource(client),
        presets=all_presets,
        http_client=http_client,
        inline=inline,
    )
    return Evals(service=service)
