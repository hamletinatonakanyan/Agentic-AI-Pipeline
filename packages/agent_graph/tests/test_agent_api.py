import json

import pytest
from fastapi.testclient import TestClient

from agent_graph.app import create_app
from agent_graph.protocols import AgentTool
from agent_graph.runtime import build_agent
from agent_graph.settings import AgentSettings
from agent_graph.testing import Brain, FakeEmbedder, FakeRetriever, FakeTools, calls_turn
from common import AppSettings, get_app_settings
from llm import reasoning_override, set_reasoning_override
from observability import shutdown_observability
from observability.settings import ObsSettings

KEY = "test-key-0123456789"
H = {"X-API-Key": KEY}


def client_for(tmp_path, brain=None, tools=None):
    shutdown_observability()
    settings = AgentSettings(_env_file=None, data_dir=str(tmp_path), checkpoint_backend="memory")
    agent = build_agent(
        settings,
        llm=brain or Brain(),
        retriever=FakeRetriever(),
        tools=tools,
        embedder=FakeEmbedder(),
    )
    app = create_app(
        agent=agent, obs_settings=ObsSettings(_env_file=None, local_db_path=":memory:")
    )
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=KEY)
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def client(tmp_path):
    with client_for(tmp_path) as c:
        yield c
    shutdown_observability()


def new_thread(c, **body):
    r = c.post("/api/agent/threads", json=body, headers=H)
    assert r.status_code == 201
    return r.json()["thread_id"]


def sse(response):
    return [
        json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")
    ]


# ----- access -------------------------------------------------------------------------------


def test_health_and_ui_are_public_everything_else_needs_the_key(client):
    assert client.get("/health").json() == {"ok": True}
    assert client.get("/ui/").status_code == 200
    for path in ("/status", "/api/agent/status", "/api/agent/threads", "/api/agent/graph",
                 "/api/agent/reasoning", "/api/agent/memory/ana"):  # fmt: skip
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401, path
    assert client.post("/api/agent/threads", json={}).status_code == 401
    assert client.put("/api/agent/reasoning", json={"use_reasoning": False}).status_code == 401
    assert client.get("/status", headers=H).status_code == 200


# ----- a conversation -----------------------------------------------------------------------


def test_send_a_message_and_read_the_thread_back(client):
    t = new_thread(client, user_id="ana")
    r = client.post(
        f"/api/agent/threads/{t}/messages", json={"message": "Why do cats purr?"}, headers=H
    )
    body = r.json()
    assert r.status_code == 200 and body["status"] == "completed" and body["route"] == "retrieve"
    assert body["citations"][0]["chunk_id"] == "c1" and body["check"]["passed"]
    assert body["timings"]["retrieval_ms"] == 12.5 and body["checkpoint_id"]
    thread = client.get(f"/api/agent/threads/{t}", headers=H).json()
    assert [m["role"] for m in thread["history"]] == ["user", "assistant"]
    assert client.get("/api/agent/threads?user_id=ana", headers=H).json()[0]["thread_id"] == t
    assert client.get("/api/agent/threads?user_id=bob", headers=H).json() == []
    assert client.delete(f"/api/agent/threads/{t}", headers=H).status_code == 204
    assert client.get(f"/api/agent/threads/{t}", headers=H).status_code == 404


def test_errors_have_the_standard_envelope_and_status(client):
    r = client.post("/api/agent/threads/nope/messages", json={"message": "x"}, headers=H)
    assert r.status_code == 404 and r.json()["error"]["code"] == "thread_not_found"
    t = new_thread(client)
    assert (
        client.post(f"/api/agent/threads/{t}/messages", json={"message": ""}, headers=H).status_code
        == 422
    )
    assert (
        client.post(f"/api/agent/threads/{t}/resume", json={"approve": True}, headers=H).json()[
            "error"
        ]["code"]
        == "nothing_to_resume"
    )
    assert client.post(f"/api/agent/threads/{t}/resume", json={}, headers=H).status_code == 422
    assert (
        client.post("/api/agent/threads", json={"user_id": "bad id"}, headers=H).status_code == 422
    )


def test_state_and_history_endpoints(client):
    t = new_thread(client)
    client.post(f"/api/agent/threads/{t}/messages", json={"message": "hello"}, headers=H)
    state = client.get(f"/api/agent/threads/{t}/state", headers=H).json()
    assert state["next"] == [] and state["values"]["route"] == "retrieve"
    assert state["values"]["messages"][0] == {"role": "user", "content": "hello"}
    history = client.get(f"/api/agent/threads/{t}/history", headers=H).json()
    assert len(history) > 5 and history[0]["checkpoint_id"] == state["checkpoint_id"]
    old = client.get(
        f"/api/agent/threads/{t}/state?checkpoint_id={history[-1]['checkpoint_id']}", headers=H
    ).json()
    assert old["values"].get("messages", []) == [] and old["next"] == ["__start__"]


def test_graph_endpoint_for_the_ui_drawing(client):
    g = client.get("/api/agent/graph", headers=H).json()
    assert {"begin", "route", "answer"} <= {n["id"] for n in g["nodes"]}
    assert any(e["source"] == "route" and e["conditional"] for e in g["edges"])


# ----- streaming ----------------------------------------------------------------------------


def test_the_stream_is_server_sent_events_ending_in_done(client):
    t = new_thread(client)
    r = client.post(
        f"/api/agent/threads/{t}/messages/stream", json={"message": "Why do cats purr?"}, headers=H
    )
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse(r)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "run_started" and kinds[-1] == "done" and "node_start" in kinds
    assert "".join(e["data"]["text"] for e in events if e["type"] == "token").startswith(
        "Cats purr"
    )
    assert events[-1]["data"]["status"] == "completed"


def test_a_refusal_before_the_stream_starts_is_a_normal_http_error(client):
    r = client.post("/api/agent/threads/nope/messages/stream", json={"message": "x"}, headers=H)
    assert r.status_code == 404 and r.json()["error"]["code"] == "thread_not_found"
    assert (
        client.post("/api/agent/threads/nope/messages/stream", json={"message": "x"}).status_code
        == 401
    )


# ----- approval over HTTP --------------------------------------------------------------------


def test_approval_flow_over_http_with_stream_and_fork(tmp_path):
    tools = FakeTools(
        tools=[
            AgentTool(
                name="files__delete",
                description="d",
                parameters={"type": "object", "properties": {}},
            )
        ],
        behaviour={"files__delete": lambda a: "deleted"},
    )
    brain = Brain(routes=["tool"], tool_turns=[calls_turn(("c1", "files__delete", {})), "done"])
    with client_for(tmp_path, brain, tools) as c:
        t = new_thread(c)
        r = c.post(
            f"/api/agent/threads/{t}/messages", json={"message": "delete it"}, headers=H
        ).json()
        assert r["status"] == "interrupted" and r["answer"] is None
        assert r["interrupt"]["calls"][0] == {
            "call_id": "c1", "tool": "files__delete", "arguments": {},
            "reason": "the tool's server did not say it only reads data",
        }  # fmt: skip
        assert tools.calls == []
        assert (
            c.get(f"/api/agent/threads/{t}/state", headers=H).json()["interrupts"][0]["calls"][0][
                "call_id"
            ]
            == "c1"
        )
        events = sse(
            c.post(
                f"/api/agent/threads/{t}/resume/stream", json={"decisions": {"c1": True}}, headers=H
            )
        )
        assert [e["type"] for e in events if e["type"] == "tool_call"] == ["tool_call"]
        assert events[-1]["type"] == "done" and events[-1]["data"]["status"] == "completed"
        assert tools.calls == [("files__delete", {})]
        assert (
            c.post(
                f"/api/agent/threads/{t}/fork", json={"checkpoint_id": "nope"}, headers=H
            ).status_code
            == 404
        )
    shutdown_observability()


# ----- memory --------------------------------------------------------------------------------


def test_memory_endpoints(client):
    r = client.post("/api/agent/memory/ana", json={"text": "Ana likes tea"}, headers=H)
    assert r.status_code == 201
    fid = r.json()["id"]
    assert [f["text"] for f in client.get("/api/agent/memory/ana", headers=H).json()] == [
        "Ana likes tea"
    ]
    assert client.get("/api/agent/memory/bob", headers=H).json() == []
    assert client.get("/api/agent/memory/ana?query=Ana likes tea", headers=H).json()[0]["id"] == fid
    assert client.delete(f"/api/agent/memory/ana/{fid}", headers=H).status_code == 204
    assert client.delete(f"/api/agent/memory/ana/{fid}", headers=H).status_code == 404
    assert client.post("/api/agent/memory/ana", json={"text": ""}, headers=H).status_code == 422


# ----- reasoning switch ----------------------------------------------------------------------


def test_the_reasoning_switch_is_global_and_reversible(client):
    try:
        base = client.get("/api/agent/reasoning", headers=H).json()
        assert base["overridden"] is False
        r = client.put(
            "/api/agent/reasoning", json={"use_reasoning": not base["use_reasoning"]}, headers=H
        ).json()
        assert r["overridden"] and r["use_reasoning"] != base["use_reasoning"]
        assert r["effort"] == ("low" if r["use_reasoning"] else "none")
        assert reasoning_override() == r["use_reasoning"]
    finally:
        set_reasoning_override(None)


def test_standalone_app_with_the_real_model_adapter_and_no_key_fails_cleanly_and_survives_restart(
    tmp_path, monkeypatch
):
    """The real `OpenAILLM` with no key refuses before any network call. Files go to tmp_path."""
    from agent_graph.factory import build_standalone
    from llm import LLMSettings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def standalone():
        shutdown_observability()
        agent = build_standalone(
            AgentSettings(_env_file=None, data_dir=str(tmp_path)),
            LLMSettings(_env_file=None, openai_api_key=""),
            embedder=FakeEmbedder(),  # never the real embedding model
        )
        app = create_app(
            agent=agent, obs_settings=ObsSettings(_env_file=None, local_db_path=":memory:")
        )
        app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=KEY)
        return TestClient(app)

    with standalone() as c:
        t = new_thread(c, user_id="ana")
        r = c.post(f"/api/agent/threads/{t}/messages", json={"message": "hi there"}, headers=H)
        assert r.status_code == 503 and r.json()["error"]["code"] == "llm_not_configured"
        # the failed run did not leave the thread stuck as busy
        assert (
            c.post(f"/api/agent/threads/{t}/messages", json={"message": "x"}, headers=H).status_code
            == 503
        )

    with standalone() as c:  # a second process on the same files
        assert [
            x["thread_id"] for x in c.get("/api/agent/threads?user_id=ana", headers=H).json()
        ] == [t]
        assert c.get(f"/api/agent/threads/{t}/state", headers=H).status_code == 200
    assert {p.name for p in tmp_path.iterdir()} >= {"checkpoints.sqlite", "agent.sqlite"}
    shutdown_observability()
