"""Glue between packages. e2e is the one place allowed to know several of them."""

import asyncio
import concurrent.futures
import contextvars
from typing import Any

from agent_graph.models import (
    AgentOptions,
    MessageRequest,
    RetrievalChoice,
    ThreadCreate,
    ToolChoice,
)
from agent_graph.models import Passage as AgentPassage
from agent_graph.protocols import AgentTool, RetrievedContext, ToolOutcome
from agent_graph.runtime import Agent
from common import AppError
from evals.models import Context, TargetResponse
from evals.service import Preset  # noqa: I001
from evals.targets import CallableTarget
from ingest.service import IngestService
from mcp_tools.corpus import (
    CorpusChunk,
    CorpusChunks,
    CorpusError,
    CorpusSearchOutput,
    Passage,
    SourceItem,
    trim,
)
from mcp_tools.llm_tools import LlmToolSet
from mcp_tools.manager import MCPClientManager
from retrieval.models import RetrievalOptions
from retrieval.service import RetrievalService


def retrieval_preset(service: RetrievalService) -> Preset:
    """An eval target that calls the retrieval service in-process: no HTTP, no second key.

    Options are the retrieval options (`mode`, `top_k`, `rerank`, `rewrite`, `filters`, ...).
    Retrieval returns chunks and no answer, so only retrieval metrics are computed for it."""

    def factory(options: dict[str, Any]) -> CallableTarget:
        opts = RetrievalOptions.model_validate(options)  # a bad option is a 422 for the caller

        def ask(question: str) -> TargetResponse:
            result = service.retrieve(question, opts)
            return TargetResponse(
                contexts=[Context(chunk_id=h.chunk_id, text=h.text) for h in result.hits],
                retrieved_chunk_ids=[h.chunk_id for h in result.hits],
                timings={"retrieval_ms": result.timings.total_ms},
            )

        return CallableTarget(ask)

    return Preset(
        factory=factory,
        returns_answer=False,
        description="The retrieval service, in-process (retrieval metrics only).",
    )


class InProcessCorpus:
    """What `search_corpus` and `corpus://sources` use when our MCP server is mounted in e2e: the
    retrieval and ingest services directly, so the app does not call itself over HTTP."""

    def __init__(
        self,
        retrieval: RetrievalService | None,
        ingest: IngestService | None,
        default_collection: str = "docs",
        passage_chars: int = 1200,
        max_chunks: int = 5000,
    ) -> None:
        self._retrieval = retrieval
        self._ingest = ingest
        self._default = default_collection
        self._chars = passage_chars
        self._max_chunks = max_chunks

    async def chunks(self, collection: str | None, source_name: str | None) -> CorpusChunks:
        """Every stored chunk (of one source, or of the collection), for the table tools."""
        if self._retrieval is None:
            raise CorpusError("retrieval is turned off in this app")
        service, name, limit = self._retrieval, collection or self._default, self._max_chunks

        def read() -> CorpusChunks:
            found: list[CorpusChunk] = []
            after: str | None = None
            while True:
                page = service.chunks_page(name, source_name, after, 500)
                found += [
                    CorpusChunk(
                        chunk_id=c.chunk_id,
                        source_name=c.source_name,
                        page=c.page,
                        chunk_index=c.chunk_index,
                        text=c.text,
                    )
                    for c in page.chunks
                ]
                after = page.next
                if len(found) >= limit:
                    return CorpusChunks(chunks=found[:limit], truncated=bool(after))
                if not after:
                    return CorpusChunks(chunks=found)

        try:
            return await asyncio.to_thread(read)  # blocking Qdrant reads, off the event loop
        except AppError as exc:
            raise CorpusError(exc.message) from None

    async def search(self, query: str, top_k: int, collection: str | None) -> CorpusSearchOutput:
        if self._retrieval is None:
            raise CorpusError("retrieval is turned off in this app")
        name = collection or self._default
        options = RetrievalOptions(collection=name, top_k=top_k, mode="hybrid")
        try:
            # Retrieval is blocking code, so it runs in a thread (which keeps the trace context).
            result = await asyncio.to_thread(self._retrieval.retrieve, query, options)
        except AppError as exc:
            raise CorpusError(exc.message) from None
        return CorpusSearchOutput(
            query=query,
            collection=name,
            results=[
                Passage(
                    rank=h.rank,
                    chunk_id=h.chunk_id,
                    source_name=str(h.payload.get("source_name", "")),
                    page=h.payload.get("page"),
                    score=h.score,
                    text=trim(h.text, self._chars),
                )
                for h in result.hits
            ],
        )

    async def sources(self, collection: str | None) -> list[SourceItem]:
        if self._ingest is None:
            raise CorpusError("ingest is turned off in this app")
        items = await asyncio.to_thread(self._ingest.list_sources, collection or self._default)
        return [
            SourceItem(
                source_id=i.source_id,
                source_name=i.source_name,
                chunk_count=i.chunk_count,
                pages=i.pages,
            )
            for i in items
        ]


class InProcessRetriever:
    """The agent's `Retriever`, backed by the retrieval service directly: no HTTP, no second key.
    Called from a worker thread (the graph's sync nodes), which is where blocking code belongs."""

    def __init__(self, service: RetrievalService) -> None:
        self._service = service

    def version(self, choice: RetrievalChoice) -> str:
        """Changes whenever any collection is ingested into or deleted, so the agent's search
        cache never serves passages from data that is gone."""
        return "|".join(f"{c.name}:{c.points}" for c in self._service.list_collections())

    def retrieve(self, query: str, choice: RetrievalChoice, top_k: int) -> RetrievedContext:
        options = RetrievalOptions(
            collection=choice.collection, mode=choice.mode, rewrite=choice.rewrite,
            rerank=choice.rerank, top_k=top_k,
        )  # fmt: skip
        result = self._service.retrieve(query, options)
        return RetrievedContext(
            passages=[
                AgentPassage(
                    chunk_id=h.chunk_id,
                    text=h.text,
                    source_name=str(h.payload.get("source_name", "")),
                    page=h.payload.get("page"),
                    score=h.rerank_score if h.rerank_score is not None else h.score,
                )
                for h in result.hits
            ],
            rewritten_queries=result.rewritten_queries,
            retrieval_ms=result.timings.total_ms,
        )


class McpToolProvider:
    """The agent's `ToolProvider`, backed by the MCP client manager. The manager already applies
    each server's allow-list, both when listing and when calling."""

    def __init__(self, manager: MCPClientManager) -> None:
        self._manager = manager
        self._set: LlmToolSet | None = None

    async def tools(self) -> list[AgentTool]:
        self._set = await self._manager.llm_tools()
        return [
            AgentTool(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
                read_only=self._set.read_only.get(t["name"]),
            )
            for t in self._set.tools
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        if self._set is None:
            await self.tools()
        target = self._set.resolve(name) if self._set else None
        if target is None:
            return ToolOutcome(ok=False, content=f"Unknown tool '{name}'.", error_kind="not_found")
        result = await self._manager.call_tool(target[0], target[1], arguments)
        return ToolOutcome(
            ok=result.ok,
            content=result.content,
            error_kind=result.error_kind,
            truncated=result.truncated,
            duration_ms=result.duration_ms,
        )


# ----- the agent as an eval target ---------------------------------------------------------------

AGENT_TARGET_OPTIONS = {
    "collection", "mode", "top_k", "rerank", "rewrite", "tools", "check", "max_check_retries",
}  # fmt: skip
AGENT_RETRIEVAL_KEYS = ("collection", "mode", "top_k", "rerank", "rewrite")
AGENT_ASK_TIMEOUT_S = 600.0


def agent_preset(agent: Agent, loop_holder: dict[str, Any]) -> Preset:
    """Scores the whole pipeline: documents, tools, the answer and its citations.

    The agent lives on the app's event loop (its checkpointer is bound to it), but an eval job
    runs in a worker thread. So each question is sent to that loop and waited for. Every question
    gets a thread of its own, deleted afterwards, and always runs with every cache off, so one
    run's answers can never leak into another's scores."""

    def factory(options: dict[str, Any]) -> CallableTarget:
        unknown = set(options) - AGENT_TARGET_OPTIONS
        if unknown:
            raise AppError(
                "invalid_target", f"Unknown agent options: {', '.join(sorted(unknown))}.", 422
            )
        try:
            run_options = AgentOptions(
                retrieval=RetrievalChoice(
                    **{k: options[k] for k in AGENT_RETRIEVAL_KEYS if k in options}
                ),
                tools=ToolChoice(enabled=bool(options.get("tools", False))),  # off unless asked
                check=bool(options.get("check", True)),
                max_check_retries=options.get("max_check_retries"),
                use_cache=False,
                remember=False,
            )
        except ValueError as exc:  # pydantic's ValidationError: a bad option is a 422
            raise AppError("invalid_target", f"Invalid agent option: {exc}"[:300], 422) from exc

        async def ask_async(question: str) -> TargetResponse:
            service = agent.service
            if service is None:
                raise AppError("agent_not_ready", "The agent is not running.", 503)
            thread = service.create_thread(ThreadCreate(title="eval"))
            try:
                r = await service.run(
                    thread.thread_id, MessageRequest(message=question, options=run_options)
                )
            finally:
                await service.delete_thread(thread.thread_id)
            if r.status == "interrupted":
                raise AppError(
                    "approval_required",
                    "The agent asked for approval to run a tool, and an eval cannot give it.",
                    409,
                )
            return TargetResponse(
                answer=r.answer,
                contexts=[Context(chunk_id=p.chunk_id, text=p.text) for p in r.passages],
                retrieved_chunk_ids=[p.chunk_id for p in r.passages],
                cited_chunk_ids=[c.chunk_id for c in r.citations],
                timings={
                    k: v
                    for k, v in {
                        "retrieval_ms": r.timings.retrieval_ms,
                        "total_ms": r.timings.total_ms,
                    }.items()
                    if v is not None
                },
            )

        def ask(question: str) -> TargetResponse:
            loop = loop_holder.get("loop")
            if loop is None:
                raise AppError("agent_not_ready", "The app is not running yet.", 503)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:  # called on the loop's own thread: waiting here would freeze it
                raise AppError(
                    "invalid_target", "The agent target must run in a worker thread.", 500
                )
            # Hand the question to the app's loop, keeping this thread's trace context, so the
            # agent's spans become children of the eval question's span (one trace per run).
            context = contextvars.copy_context()
            done: concurrent.futures.Future[TargetResponse] = concurrent.futures.Future()
            tasks: list[asyncio.Task] = []

            def start() -> None:
                task = loop.create_task(ask_async(question), context=context)
                tasks.append(task)

                def finished(t: asyncio.Task) -> None:
                    if t.cancelled():
                        done.cancel()
                    elif t.exception() is not None:
                        done.set_exception(t.exception())
                    else:
                        done.set_result(t.result())

                task.add_done_callback(finished)

            loop.call_soon_threadsafe(start)
            try:
                return done.result(timeout=AGENT_ASK_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                for task in tasks:
                    loop.call_soon_threadsafe(task.cancel)
                raise AppError("run_timeout", "The agent did not answer in time.", 504) from None

        return CallableTarget(ask)

    return Preset(
        factory=factory,
        returns_answer=True,
        description=(
            "The whole agent, in-process: documents, optional tools, answer and citations. "
            "Caches are always off. Options: collection, mode, top_k, rerank, rewrite, tools, "
            "check."
        ),
    )
