"""Agent test helpers: the graph wired to scripted fakes, with in-memory persistence.

(Not a conftest.py: another package has one, and two modules with the same name collide.)"""

import asyncio
import uuid

from langchain_core.messages import HumanMessage
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_graph.cache import CachingLLM, LlmCache
from agent_graph.graph import Deps, build_graph
from agent_graph.memory import FactMemory
from agent_graph.models import AgentOptions
from agent_graph.settings import AgentSettings
from agent_graph.testing import Brain, FakeEmbedder, FakeRetriever, FakeTools


def run(coro):
    return asyncio.run(coro)


class Rig:
    def __init__(
        self,
        brain: Brain | None = None,
        retriever: FakeRetriever | None | bool = True,
        tools: FakeTools | None | bool = None,
        memory: bool = False,
        cached: bool = False,
        **settings,
    ) -> None:
        self.settings = AgentSettings(_env_file=None, checkpoint_backend="memory", **settings)
        self.brain = brain or Brain()
        self.retriever = FakeRetriever() if retriever is True else (retriever or None)
        self.tools = FakeTools() if tools is True else (tools or None)
        self.memory = FactMemory(":memory:", FakeEmbedder()) if memory else None
        self.cache = LlmCache(":memory:", 600) if cached else None
        deps = Deps(
            llm=self.brain,
            settings=self.settings,
            cached_llm=CachingLLM(self.brain, self.cache, "test") if cached else None,
            retriever=self.retriever,
            tools=self.tools,
            memory=self.memory,
        )
        self.app = build_graph(deps).compile(checkpointer=InMemorySaver(), cache=InMemoryCache())

    @staticmethod
    def config(thread: str) -> dict:
        return {"configurable": {"thread_id": thread}, "recursion_limit": 100}

    def input(self, message: str, user_id=None, stream=False, **options) -> dict:
        return {
            "messages": [HumanMessage(message)],
            "run_id": uuid.uuid4().hex[:8],
            "user_id": user_id,
            "options": AgentOptions(**options).model_dump(),
            "stream": stream,
        }

    async def say(self, message: str, thread="t1", user_id=None, stream=False, **options) -> dict:
        return await self.app.ainvoke(
            self.input(message, user_id, stream, **options), self.config(thread)
        )

    async def resume(self, value, thread="t1") -> dict:
        return await self.app.ainvoke(Command(resume=value), self.config(thread))

    async def events(self, message: str, thread="t1", **options) -> list:
        """Everything streamed for one message: (namespace, mode, chunk)."""
        out = []
        async for item in self.app.astream(
            self.input(message, stream=True, **options),
            self.config(thread),
            stream_mode=["updates", "custom", "tasks"],
            subgraphs=True,
        ):
            out.append(item)
        return out

    async def state(self, thread="t1"):
        return await self.app.aget_state(self.config(thread))
