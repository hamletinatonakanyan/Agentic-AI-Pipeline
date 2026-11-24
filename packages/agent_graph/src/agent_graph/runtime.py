"""Builds the agent, and opens and closes what needs a lifespan: the checkpointer.

    agent = build_agent(settings, llm=..., retriever=..., tools=...)
    async with agent.lifespan():
        await agent.service.run(...)

The checkpointer is an async SQLite connection, so it must be opened inside the running event
loop. That is why this is a lifespan (the apps enter it at startup) and not a constructor.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.cache.memory import InMemoryCache
from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent_graph.cache import CachingLLM, LlmCache
from agent_graph.graph import Deps, build_graph
from agent_graph.memory import FactMemory, FastEmbedFactEmbedder
from agent_graph.protocols import FactEmbedder, Retriever, ToolProvider
from agent_graph.service import AgentService
from agent_graph.settings import AgentSettings
from agent_graph.threads import ThreadStore
from llm import LLM

# Gemma wants a different prefix for stored texts and for queries.
_PREFIXES = {
    "google/embeddinggemma-300m": ("title: none | text: ", "task: search result | query: ")
}


class Agent:
    def __init__(
        self,
        settings: AgentSettings,
        deps: Deps,
        threads: ThreadStore,
        llm_cache: LlmCache,
        tools: ToolProvider | None,
    ) -> None:
        self.settings = settings
        self._deps = deps
        self._threads = threads
        self._llm_cache = llm_cache
        self._tools = tools
        self.service: AgentService | None = None

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator["Agent"]:
        s = self.settings
        if s.checkpoint_backend == "sqlite":
            async with AsyncSqliteSaver.from_conn_string(s.path("checkpoints.sqlite")) as saver:
                yield self._start(saver, SqliteCache(path=s.path("node_cache.sqlite")))
                self._stop()
        else:
            yield self._start(InMemorySaver(), InMemoryCache())
            self._stop()

    def _start(self, saver, node_cache) -> "Agent":
        graph = build_graph(self._deps).compile(
            checkpointer=saver,
            cache=node_cache,
            store=self._deps.memory.store if self._deps.memory else None,
        )
        self.service = AgentService(
            self.settings, graph, saver, self._threads, self._deps.memory, self._tools
        )
        return self

    def _stop(self) -> None:
        self.service = None
        if self._deps.memory:
            self._deps.memory.close()
        self._llm_cache.close()
        self._threads.close()


def build_agent(
    settings: AgentSettings,
    *,
    llm: LLM,
    retriever: Retriever | None = None,
    tools: ToolProvider | None = None,
    embedder: FactEmbedder | None = None,
) -> Agent:
    """Nothing here connects or downloads. The embedding model loads on first use."""
    memory_path = settings.path("memory.sqlite")
    doc_prefix, query_prefix = _PREFIXES.get(settings.memory_embed_model, ("", ""))
    memory = FactMemory(
        memory_path,
        embedder or FastEmbedFactEmbedder(settings.memory_embed_model),
        doc_prefix,
        query_prefix,
    )
    llm_cache = LlmCache(settings.path("agent.sqlite"), settings.cache_ttl_s)
    # Anything outside a request that changes the answer must change this, or a stale answer
    # could come back: the model, the reasoning setting, and the prompt version.
    deps = Deps(
        llm=llm,
        settings=settings,
        cached_llm=CachingLLM(llm, llm_cache, lambda: _namespace(llm)),
        retriever=retriever,
        tools=tools,
        memory=memory,
    )
    return Agent(settings, deps, ThreadStore(settings.path("agent.sqlite")), llm_cache, tools)


def _namespace(llm: LLM) -> str:
    from agent_graph.prompts import PROMPT_VERSION

    inner = getattr(llm, "_s", None)  # OpenAILLM keeps its settings here
    model = getattr(inner, "model", "unknown")
    effort = getattr(inner, "effort", "unknown")  # read each time: the UI can flip it
    return f"{model}|{effort}|prompts-{PROMPT_VERSION}"
