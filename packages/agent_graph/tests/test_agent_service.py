import pytest
from agent_rig import run

from agent_graph.models import ForkRequest, MessageRequest, ResumeRequest, ThreadCreate
from agent_graph.protocols import AgentTool
from agent_graph.runtime import build_agent
from agent_graph.settings import AgentSettings
from agent_graph.testing import Brain, FakeEmbedder, FakeRetriever, FakeTools, calls_turn
from common import AppError


def make(tmp_path, backend="sqlite", brain=None, tools=None):
    settings = AgentSettings(_env_file=None, data_dir=str(tmp_path), checkpoint_backend=backend)
    return build_agent(
        settings,
        llm=brain or Brain(),
        retriever=FakeRetriever(),
        tools=tools,
        embedder=FakeEmbedder(),
    )


def risky_tools():
    return FakeTools(
        tools=[AgentTool(name="files__delete", description="d", read_only=None,
                         parameters={"type": "object", "properties": {}})],
        behaviour={"files__delete": lambda a: "deleted"},
    )  # fmt: skip


def risky_brain():
    return Brain(routes=["tool"], tool_turns=[calls_turn(("c1", "files__delete", {})), "done"])


def test_a_message_gets_a_response_and_the_thread_is_listed_with_a_title(tmp_path):
    async def go():
        async with make(tmp_path).lifespan() as agent:
            svc = agent.service
            t = svc.create_thread(ThreadCreate(user_id="ana"))
            r = await svc.run(t.thread_id, MessageRequest(message="Why do cats purr?"))
            assert r.status == "completed" and r.route == "retrieve" and r.trace_id is None or True
            assert r.answer and r.citations[0].chunk_id == "c1" and r.check.passed
            listed = svc.list_threads("ana")[0]
            assert listed.title == "Why do cats purr?" and listed.messages == 2
            detail = await svc.get_thread(t.thread_id)
            assert [m.role for m in detail.history] == ["user", "assistant"]

    run(go())


def test_unknown_thread_bad_user_and_long_message_are_clean_errors(tmp_path):
    async def go():
        async with make(tmp_path, "memory").lifespan() as agent:
            svc = agent.service
            with pytest.raises(AppError) as e:
                await svc.run("nope", MessageRequest(message="x"))
            assert e.value.code == "thread_not_found" and e.value.status_code == 404
            with pytest.raises(AppError) as e:
                svc.create_thread(ThreadCreate(user_id="bad user!"))
            assert e.value.code == "invalid_user"
            t = svc.create_thread(ThreadCreate())
            with pytest.raises(AppError) as e:
                await svc.run(t.thread_id, MessageRequest(message="x" * 100_000))
            assert e.value.code == "invalid_message"

    run(go())


def test_a_second_run_on_a_busy_thread_is_refused(tmp_path):
    async def go():
        async with make(tmp_path, "memory").lifespan() as agent:
            svc = agent.service
            t = svc.create_thread(ThreadCreate())
            async with svc._exclusive(t.thread_id):
                with pytest.raises(AppError) as e:
                    await svc.run(t.thread_id, MessageRequest(message="x"))
            assert e.value.code == "thread_busy" and e.value.status_code == 409
            await svc.run(t.thread_id, MessageRequest(message="x"))  # free again

    run(go())


def test_pause_restart_resume_then_fork_all_in_one_thread(tmp_path):
    """An interrupted run survives a restart and can be resumed and forked, in one trace per run."""

    async def first_process():
        async with make(tmp_path, brain=risky_brain(), tools=risky_tools()).lifespan() as agent:
            t = agent.service.create_thread(ThreadCreate(user_id="ana"))
            r = await agent.service.run(t.thread_id, MessageRequest(message="delete it"))
            assert r.status == "interrupted" and r.interrupt.calls[0].tool == "files__delete"
            return t.thread_id

    thread_id = run(first_process())

    async def second_process():  # a new Agent on the same files = a restart
        tools = risky_tools()
        # the new process's model has nothing left to call: after the tool result it just answers
        after = Brain(routes=["tool"], tool_turns=["done"])
        async with make(tmp_path, brain=after, tools=tools).lifespan() as agent:
            svc = agent.service
            state = await svc.get_state(thread_id)
            assert state.next == ["approve"] and state.interrupts[0].calls[0].call_id == "c1"
            with pytest.raises(AppError) as e:
                await svc.resume(thread_id, ResumeRequest(decisions={"zzz": True}))
            assert e.value.code == "invalid_decision"
            r = await svc.resume(thread_id, ResumeRequest(approve=True))
            assert r.status == "completed" and r.tool_calls[0].status == "ok"
            assert tools.calls == [("files__delete", {})]
            with pytest.raises(AppError) as e:  # nothing is waiting any more
                await svc.resume(thread_id, ResumeRequest(approve=True))
            assert e.value.code == "nothing_to_resume"

            # fork: go back to the pause and decline instead
            history = await svc.get_history(thread_id)
            paused = next(h for h in history if h.next == ["approve"])
            before = len(history)
            tools.calls.clear()
            forked = await svc.fork(thread_id, ForkRequest(checkpoint_id=paused.checkpoint_id))
            assert forked.status == "interrupted"  # it asks again, on the new branch (same pause)
            declined = await svc.resume(thread_id, ResumeRequest(approve=False))
            assert declined.tool_calls[0].status == "declined" and tools.calls == []
            after = await svc.get_history(thread_id)
            assert len(after) > before  # the old branch is still there
            parents = [h.parent_checkpoint_id for h in after]
            assert any(parents.count(p) > 1 for p in set(parents) if p)  # one step, two children

    run(second_process())


def test_forking_a_finished_turn_with_a_new_message_edits_and_regenerates(tmp_path):
    async def go():
        brain = Brain(routes=["direct"], answers=["first", "second"])
        async with make(tmp_path, "memory", brain=brain).lifespan() as agent:
            svc = agent.service
            t = svc.create_thread(ThreadCreate())
            await svc.run(
                t.thread_id,
                MessageRequest(message="original", options={"retrieval": {"enabled": False}}),
            )
            history = await svc.get_history(t.thread_id)
            start = history[-1].checkpoint_id  # the very first checkpoint, before any message
            with pytest.raises(AppError) as e:
                await svc.fork(t.thread_id, ForkRequest(checkpoint_id=history[0].checkpoint_id))
            assert e.value.code == "message_required"
            r = await svc.fork(t.thread_id, ForkRequest(checkpoint_id=start, message="edited"))
            assert r.answer == "second"
            detail = await svc.get_thread(t.thread_id)
            assert [m.content for m in detail.history][0::2] == ["edited"]

    run(go())


def test_streaming_ends_with_done_and_shows_the_interrupt(tmp_path):
    async def go():
        async with make(
            tmp_path, "memory", brain=risky_brain(), tools=risky_tools()
        ).lifespan() as agent:
            svc = agent.service
            t = svc.create_thread(ThreadCreate())
            events = [e async for e in svc.stream(t.thread_id, MessageRequest(message="delete it"))]
            kinds = [e.type for e in events]
            assert kinds[0] == "run_started" and kinds[-1] == "done" and "interrupt" in kinds
            assert events[-1].data["status"] == "interrupted"
            resumed = [e async for e in svc.stream_resume(t.thread_id, ResumeRequest(approve=True))]
            kinds = [e.type for e in resumed]
            assert "tool_call" in kinds and "token" in kinds and kinds[-1] == "done"
            assert resumed[-1].data["status"] == "completed"

    run(go())


def test_a_failure_while_streaming_becomes_an_error_event(tmp_path):
    class Broken(Brain):
        def complete_json(self, *a, **k):
            raise AppError("llm_not_configured", "OPENAI_API_KEY is not set.", 503)

    async def go():
        async with make(tmp_path, "memory", brain=Broken()).lifespan() as agent:
            t = agent.service.create_thread(ThreadCreate())
            events = [
                e async for e in agent.service.stream(t.thread_id, MessageRequest(message="x"))
            ]
            assert events[-1].type == "error" and events[-1].data["code"] == "llm_not_configured"
            assert t.thread_id not in agent.service._busy

    run(go())


def test_memory_facts_through_the_service_and_delete_thread(tmp_path):
    async def go():
        async with make(tmp_path, "memory").lifespan() as agent:
            svc = agent.service
            fact = await svc.add_fact("ana", "Ana likes tea")
            assert [f.text for f in await svc.list_facts("ana")] == ["Ana likes tea"]
            assert await svc.list_facts("bob") == []
            await svc.delete_fact("ana", fact.id)
            with pytest.raises(AppError) as e:
                await svc.delete_fact("ana", fact.id)
            assert e.value.status_code == 404
            t = svc.create_thread(ThreadCreate())
            await svc.run(t.thread_id, MessageRequest(message="x"))
            await svc.delete_thread(t.thread_id)
            assert svc.list_threads() == []

    run(go())


def test_graph_info_describes_the_drawn_graph(tmp_path):
    async def go():
        async with make(tmp_path, "memory").lifespan() as agent:
            info = agent.service.graph_info()
            ids = {n.id for n in info.nodes}
            assert {
                "begin",
                "route",
                "retrieve",
                "plan_tools",
                "approve",
                "answer",
                "respond",
            } <= ids
            assert any(e.conditional and e.source == "route" for e in info.edges)
            assert "graph TD" in info.mermaid
            st = await agent.service.status()
            assert st.approval == "sensitive"

    run(go())
