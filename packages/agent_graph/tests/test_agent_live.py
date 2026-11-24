"""Live tests for the agent: `make test-live`.

They need OPENAI_API_KEY in .env, and the first memory test downloads the embedding model
(google/embeddinggemma-300m, which may require accepting its licence on Hugging Face and an
HF_TOKEN). Nothing here touches Qdrant.
"""

import asyncio

import pytest

from agent_graph.models import MessageRequest, ResumeRequest, ThreadCreate
from agent_graph.protocols import AgentTool
from agent_graph.runtime import build_agent
from agent_graph.settings import AgentSettings
from agent_graph.testing import FakeRetriever, FakeTools
from llm import LLMSettings, OpenAILLM

pytestmark = pytest.mark.live


def agent(tmp_path, **kw):
    llm_settings = LLMSettings()
    if not llm_settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    settings = AgentSettings(data_dir=str(tmp_path))
    return build_agent(settings, llm=OpenAILLM(llm_settings), **kw)


def test_a_real_model_routes_retrieves_answers_with_citations_and_passes_its_own_check(tmp_path):
    async def go():
        retriever = FakeRetriever()  # the documents are fake, the model is real
        async with agent(tmp_path, retriever=retriever).lifespan() as a:
            t = a.service.create_thread(ThreadCreate())
            r = await a.service.run(t.thread_id, MessageRequest(message="Why do cats purr?"))
            assert r.status == "completed" and r.route == "retrieve" and retriever.calls
            assert r.answer and r.citations and r.check is not None
            assert r.timings.total_ms > 0

    asyncio.run(go())


def test_a_real_model_chats_directly_when_no_documents_are_needed(tmp_path):
    async def go():
        async with agent(tmp_path, retriever=FakeRetriever()).lifespan() as a:
            t = a.service.create_thread(ThreadCreate())
            r = await a.service.run(
                t.thread_id, MessageRequest(message="Say hello in one short sentence.")
            )
            assert r.route == "direct" and r.answer and r.citations == []

    asyncio.run(go())


def test_a_real_model_calls_a_tool_pauses_for_approval_and_continues_after_it(tmp_path):
    tools = FakeTools(
        tools=[AgentTool(name="files__delete", description="Delete the file at `path`.",
                         parameters={"type": "object", "properties": {"path": {"type": "string"}},
                                     "required": ["path"]})],
        behaviour={"files__delete": lambda a: f"deleted {a['path']}"},
    )  # fmt: skip

    async def go():
        async with agent(tmp_path, tools=tools).lifespan() as a:
            t = a.service.create_thread(ThreadCreate())
            r = await a.service.run(
                t.thread_id, MessageRequest(message="Please delete the file /tmp/demo.txt")
            )
            assert r.status == "interrupted" and r.interrupt.calls[0].tool == "files__delete"
            done = await a.service.resume(t.thread_id, ResumeRequest(approve=True))
            assert done.status == "completed" and tools.calls and done.answer

    asyncio.run(go())


def test_memory_recalls_a_fact_by_meaning_with_the_real_embedding_model(tmp_path):
    async def go():
        async with agent(tmp_path).lifespan() as a:
            await a.service.add_fact("ana", "Ana is a nurse who works night shifts")
            await a.service.add_fact("ana", "Ana's favourite colour is green")
            found = await a.service.search_facts("ana", "What is her job?")
            assert found and "nurse" in found[0].text

    asyncio.run(go())


def test_the_streamed_answer_arrives_as_tokens_before_the_end(tmp_path):
    async def go():
        async with agent(tmp_path, retriever=FakeRetriever()).lifespan() as a:
            t = a.service.create_thread(ThreadCreate())
            events = [
                e
                async for e in a.service.stream(
                    t.thread_id, MessageRequest(message="Why do cats purr?")
                )
            ]
            kinds = [e.type for e in events]
            assert kinds.count("token") > 1 and kinds[-1] == "done"
            assert events[-1].data["timings"]["ttft_ms"] is not None

    asyncio.run(go())
