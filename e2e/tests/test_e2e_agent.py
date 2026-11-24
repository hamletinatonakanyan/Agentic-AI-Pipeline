"""The agent inside the e2e app: real graph, real service, real in-process retrieval and MCP
tools, a scripted model, and no files, network, Docker or model downloads."""

import json

from qdrant_client import QdrantClient
from test_e2e import KEY, by_name, make_client, upload_to

from agent_graph.testing import Brain, calls_turn

H = {"X-API-Key": KEY}
SENTENCE = "Cats purr when they are content and relaxed."


def ingest_one(client, name="cats.txt", text=SENTENCE):
    job = upload_to(client, {name: text.encode()}).json()["job_id"]
    assert client.get(f"/api/ingest/jobs/{job}", headers=H).json()["state"] == "completed"


def new_thread(client):
    return client.post("/api/agent/threads", json={}, headers=H).json()["thread_id"]


def ask(client, thread, message, **options):
    return client.post(
        f"/api/agent/threads/{thread}/messages",
        json={"message": message, "options": options},
        headers=H,
    )


# ----- wiring -------------------------------------------------------------------------------


def test_the_agent_is_built_has_a_ui_tab_and_needs_the_key_like_everything_else():
    client = make_client(llm_key="sk-test")
    comps = by_name(client.get("/api/status", headers=H).json())
    assert comps["agent"]["configured"] is True and comps["agent"]["ui_path"] == "/ui/agent/"
    assert client.get("/api/agent/health").json() == {"ok": True}  # public
    for path in ("/api/agent/status", "/api/agent/threads", "/api/agent/graph"):
        assert client.get(path).status_code == 401, path
    page = client.get("/ui/agent/")
    assert (
        page.status_code == 200
        and KEY not in page.text
        and KEY not in client.get("/ui/agent/app.js").text
    )


def test_the_agent_can_be_switched_off():
    client = make_client(enable_agent=False)
    assert client.get("/api/agent/health").status_code == 404
    assert by_name(client.get("/api/status", headers=H).json())["agent"]["enabled"] is False


# ----- documents -----------------------------------------------------------------------------


def test_the_agent_answers_from_documents_ingested_through_the_real_pipeline():
    brain = Brain(routes=["retrieve"], answers=["Cats purr when content [1]."])
    with make_client(brain=brain, shared=QdrantClient(":memory:")) as client:
        ingest_one(client)
        r = ask(client, new_thread(client), SENTENCE, retrieval={"mode": "dense"}).json()
        assert r["status"] == "completed" and r["route"] == "retrieve"
        assert (
            r["passages"][0]["source_name"] == "cats.txt" and SENTENCE in r["passages"][0]["text"]
        )
        assert r["citations"][0]["source_name"] == "cats.txt"
        assert SENTENCE in brain.prompts_of("complete")[0]  # the model was shown the real chunk


def test_with_nothing_ingested_the_agent_says_so_to_the_model_instead_of_inventing_context():
    brain = Brain(routes=["retrieve"], answers=["I could not find that."])
    with make_client(brain=brain) as client:
        r = ask(client, new_thread(client), "anything at all", retrieval={"collection": "empty"})
        body = r.json()
        # A missing collection is reported, not passed off as "no results".
        assert r.status_code == 200 and body["passages"] == []
        assert "(nothing was retrieved)" in brain.prompts_of("complete")[0]
        assert body["warnings"] == [] or body["warnings"][0].startswith("search failed")


# ----- tools ---------------------------------------------------------------------------------


def test_the_agent_calls_our_own_mcp_calculator_through_the_real_manager():
    brain = Brain(
        routes=["tool"],
        tool_turns=[calls_turn(("c1", "own__calculate", {"expression": "340 * 0.12"})), "done"],
        answers=["It is 40.8."],
    )
    with make_client(brain=brain) as client:
        r = ask(client, new_thread(client), "what is 12% of 340?").json()
        assert r["status"] == "completed", r  # the calculator is read-only, so no approval
        (call,) = r["tool_calls"]
        assert (
            call["name"] == "own__calculate"
            and call["status"] == "ok"
            and "40.8" in call["content"]
        )
        assert call["approved"] is None
        plan = next(c for c in brain.calls if c.kind == "complete_with_tools")
        assert {"own__calculate", "own__search_corpus"} <= {t["name"] for t in plan.tools}


def test_the_agent_status_counts_the_tools_it_can_use():
    with make_client() as client:
        status = client.get("/api/agent/status", headers=H).json()
        assert status["tools_available"] == 4 and status["approval"] == "sensitive"


# ----- one trace per run ---------------------------------------------------------------------


def names(node):
    out = [node["name"]]
    for child in node.get("children", []):
        out += names(child)
    return out


def test_one_run_is_one_trace_with_every_step_inside_it():
    brain = Brain(routes=["retrieve"], answers=["Cats purr when content [1]."])
    with make_client(brain=brain, shared=QdrantClient(":memory:")) as client:
        ingest_one(client)
        r = ask(client, new_thread(client), SENTENCE, retrieval={"mode": "dense"}).json()
        traces = client.get("/api/obs/traces?name=agent", headers=H).json()
        assert [t["root_name"] for t in traces] == ["agent.run"]
        assert traces[0]["trace_id"] == r["trace_id"]
        tree = client.get(f"/api/obs/traces/{r['trace_id']}", headers=H).json()
        seen = names(tree["roots"][0])
        for step in (
            "agent.route",
            "agent.retrieve",
            "retrieval.retrieve",
            "agent.generate",
            "agent.check",
        ):
            assert step in seen, (step, seen)
        assert len(tree["roots"]) == 1  # nothing escaped into a separate trace


def test_a_tool_run_is_traced_down_to_the_mcp_call():
    brain = Brain(
        routes=["tool"],
        tool_turns=[calls_turn(("c1", "own__calculate", {"expression": "2 + 2"})), "done"],
    )
    with make_client(brain=brain) as client:
        r = ask(client, new_thread(client), "2+2?").json()
        tree = client.get(f"/api/obs/traces/{r['trace_id']}", headers=H).json()
        seen = names(tree["roots"][0])
        assert "agent.tool" in seen and "mcp.call_tool" in seen and len(tree["roots"]) == 1


# ----- streaming through the whole app --------------------------------------------------------


def test_streaming_works_through_the_full_app():
    brain = Brain(routes=["retrieve"], answers=["Cats purr when content [1]."])
    with make_client(brain=brain, shared=QdrantClient(":memory:")) as client:
        ingest_one(client)
        t = new_thread(client)
        r = client.post(
            f"/api/agent/threads/{t}/messages/stream",
            json={"message": SENTENCE, "options": {"retrieval": {"mode": "dense"}}},
            headers=H,
        )
        events = [json.loads(x[6:]) for x in r.text.split("\n\n") if x.startswith("data: ")]
        assert events[0]["type"] == "run_started" and events[-1]["type"] == "done"
        assert "".join(e["data"]["text"] for e in events if e["type"] == "token").startswith(
            "Cats purr"
        )
        assert events[0]["data"]["trace_id"] == events[-1]["data"]["trace_id"]
