import sqlite3

from agent_rig import run

from agent_graph.models import MessageRequest, ThreadCreate
from agent_graph.protocols import AgentTool
from agent_graph.runtime import build_agent
from agent_graph.settings import AgentSettings
from agent_graph.testing import Brain, FakeEmbedder, FakeRetriever, FakeTools, calls_turn
from observability import init_observability, shutdown_observability
from observability.exporters import InMemoryExporter
from observability.settings import ObsSettings


def make(tmp_path=None, backend="memory", brain=None, tools=None, **settings):
    s = AgentSettings(
        _env_file=None,
        checkpoint_backend=backend,
        data_dir=str(tmp_path or "unused"),
        **settings,
    )
    return build_agent(
        s, llm=brain or Brain(), retriever=FakeRetriever(), tools=tools, embedder=FakeEmbedder()
    )


def traced(fn):
    exporter = InMemoryExporter()
    shutdown_observability()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    try:
        return run(fn()), exporter.spans
    finally:
        shutdown_observability()


def by_name(spans, name):
    return [s for s in spans if s.name == name]


# ----- what the trace shows ------------------------------------------------------------------


def test_the_run_span_carries_the_conversation_id_the_question_and_the_answer():
    async def go():
        async with make().lifespan() as agent:
            t = agent.service.create_thread(ThreadCreate())
            r = await agent.service.run(t.thread_id, MessageRequest(message="Why do cats purr?"))
            return t.thread_id, r

    (thread_id, response), spans = traced(go)
    (root,) = by_name(spans, "agent.run")
    assert root.attributes["session.id"] == thread_id
    assert root.attributes["input.value"] == "Why do cats purr?"
    assert root.attributes["output.value"] == response.answer
    assert root.attributes["agent.route"] == "retrieve" and root.attributes["agent.passages"] == 2
    assert {s.trace_id for s in spans if s.name.startswith("agent.")} == {root.trace_id}


def test_each_step_records_what_it_saw_and_decided():
    async def go():
        async with make().lifespan() as agent:
            t = agent.service.create_thread(ThreadCreate())
            await agent.service.run(
                t.thread_id,
                MessageRequest(
                    message="Why do cats purr?", options={"retrieval": {"mode": "dense"}}
                ),
            )

    _, spans = traced(go)
    route = by_name(spans, "agent.route")[0].attributes
    assert route["agent.route"] == "retrieve" and route["agent.reason"] == "because"
    assert route["agent.standalone_question"] == "Why do cats purr?"
    retrieve = by_name(spans, "agent.retrieve")[0].attributes
    assert retrieve["agent.passage_count"] == 2 and retrieve["agent.mode"] == "dense"
    first = retrieve["agent.passages"][0]
    assert first["n"] == 1 and first["chunk_id"] == "c1" and first["source"] == "cats.txt"
    assert "purr" in first["text"] and first["page"] == 1
    assert by_name(spans, "agent.generate")[0].attributes["agent.answer"].startswith("Cats purr")
    assert by_name(spans, "agent.check")[0].attributes["agent.passed"] is True
    assert by_name(spans, "agent.respond")[0].attributes["agent.citations"] == 1


def test_tool_steps_record_arguments_and_results():
    tools = FakeTools(
        tools=[AgentTool(name="own__calculate", read_only=True)],
        behaviour={"own__calculate": lambda a: "40.8"},
    )
    brain = Brain(
        routes=["tool"],
        tool_turns=[calls_turn(("c1", "own__calculate", {"expression": "340 * 0.12"})), "done"],
    )

    async def go():
        async with make(brain=brain, tools=tools).lifespan() as agent:
            t = agent.service.create_thread(ThreadCreate())
            await agent.service.run(t.thread_id, MessageRequest(message="12% of 340?"))

    _, spans = traced(go)
    plan = by_name(spans, "agent.plan_tools")[0].attributes
    assert plan["agent.call_names"] == ["own__calculate"]
    tool = by_name(spans, "agent.tool")[0].attributes
    assert tool["agent.tool_arguments"] == {"expression": "340 * 0.12"}
    assert tool["agent.tool_result"] == "40.8" and tool["agent.tool_status"] == "ok"


def test_with_content_capture_off_only_names_counts_and_timings_remain():
    async def go():
        async with make(trace_content=False).lifespan() as agent:
            t = agent.service.create_thread(ThreadCreate())
            await agent.service.run(t.thread_id, MessageRequest(message="secret question text"))

    _, spans = traced(go)
    (root,) = by_name(spans, "agent.run")
    assert "input.value" not in root.attributes and "output.value" not in root.attributes
    assert root.attributes["session.id"] and root.attributes["agent.passages"] == 2
    retrieve = by_name(spans, "agent.retrieve")[0].attributes
    assert "agent.passages" not in retrieve and retrieve["agent.passage_count"] == 2
    assert "secret question text" not in str([s.attributes for s in spans])


def test_long_text_is_clipped():
    brain = Brain(answers=["word " * 5000 + "[1]"])

    async def go():
        async with make(brain=brain, trace_text_chars=300).lifespan() as agent:
            t = agent.service.create_thread(ThreadCreate())
            await agent.service.run(t.thread_id, MessageRequest(message="x"))

    _, spans = traced(go)
    answer = by_name(spans, "agent.generate")[0].attributes["agent.answer"]
    assert len(answer) <= 300 and answer.endswith("…")


# ----- the chat history keeps the details ------------------------------------------------------


def test_reopening_a_conversation_returns_each_answers_sources_tools_and_trace(tmp_path):
    async def first_process():
        async with make(tmp_path, "sqlite").lifespan() as agent:
            t = agent.service.create_thread(ThreadCreate(user_id="ana"))
            await agent.service.run(
                t.thread_id,
                MessageRequest(
                    message="Why do cats purr?", options={"retrieval": {"collection": "beta"}}
                ),
            )
            return t.thread_id

    thread_id = run(first_process())

    async def after_restart():
        async with make(tmp_path, "sqlite").lifespan() as agent:
            detail = await agent.service.get_thread(thread_id)
            user, answer = detail.history
            assert user.run is None and answer.run is not None
            r = answer.run
            assert (
                r.question == "Why do cats purr?" and r.options["retrieval"]["collection"] == "beta"
            )
            assert [p.chunk_id for p in r.response.passages] == ["c1", "c2"]
            assert (
                r.response.citations[0].source_name == "cats.txt"
                and r.response.timings.total_ms > 0
            )
            assert detail.collection == "beta"  # the conversation remembers its domain
            assert agent.service.list_threads("ana")[0].collection == "beta"

    run(after_restart())


def test_every_answer_in_a_conversation_has_its_own_details():
    brain = Brain(routes=["retrieve"], answers=["First [1].", "Second [2]."])

    async def go():
        async with make(brain=brain).lifespan() as agent:
            t = agent.service.create_thread(ThreadCreate())
            await agent.service.run(t.thread_id, MessageRequest(message="one"))
            await agent.service.run(t.thread_id, MessageRequest(message="two"))
            return (await agent.service.get_thread(t.thread_id)).history

    history = run(go())
    runs = [m.run for m in history if m.run]
    assert [r.question for r in runs] == ["one", "two"]
    assert [r.response.citations[0].marker for r in runs] == [1, 2]
    assert runs[0].response.run_id != runs[1].response.run_id


def test_deleting_a_conversation_deletes_its_saved_details(tmp_path):
    async def go():
        async with make(tmp_path, "sqlite").lifespan() as agent:
            t = agent.service.create_thread(ThreadCreate())
            await agent.service.run(t.thread_id, MessageRequest(message="x"))
            assert agent.service._threads.runs(t.thread_id)
            await agent.service.delete_thread(t.thread_id)
            return agent.service._threads.runs(t.thread_id)

    assert run(go()) == {}


def test_a_threads_table_from_before_domains_is_upgraded_in_place(tmp_path):
    db = tmp_path / "agent.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE threads (thread_id TEXT PRIMARY KEY, user_id TEXT, title TEXT NOT NULL, "
        "created_at REAL NOT NULL, updated_at REAL NOT NULL, messages INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute("INSERT INTO threads VALUES ('old1', 'ana', 'An old chat', 1.0, 2.0, 4)")
    conn.commit()
    conn.close()

    async def go():
        async with make(tmp_path, "sqlite").lifespan() as agent:
            (info,) = agent.service.list_threads("ana")
            return info

    info = run(go())
    assert (info.thread_id, info.title, info.messages, info.collection) == (
        "old1",
        "An old chat",
        4,
        None,
    )
