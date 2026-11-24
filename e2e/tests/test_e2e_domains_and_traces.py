"""Domains, conversation-level traces, deleting data and the shared UI, checked through the app."""

import json
import re
from pathlib import Path

from qdrant_client import QdrantClient
from test_e2e import DOC_TEXT, KEY, ingest_through_http, make_client

from agent_graph.testing import Brain, calls_turn

H = {"X-API-Key": KEY}
ROOT = Path(__file__).resolve().parents[2]
UI_FILES = [
    *ROOT.glob("packages/*/src/*/ui/*"),
    *ROOT.glob("e2e/src/e2e/ui/*"),
    *ROOT.glob("packages/common/src/common/ui_kit/*"),
]


def start(client):
    return client.post("/api/agent/threads", json={}, headers=H).json()["thread_id"]


def say(client, thread, message, **options):
    return client.post(
        f"/api/agent/threads/{thread}/messages",
        json={"message": message, "options": options},
        headers=H,
    ).json()


def names(node):
    return [node["name"]] + [n for c in node.get("children", []) for n in names(c)]


def find(node, name):
    if node["name"] == name:
        return node
    for child in node.get("children", []):
        hit = find(child, name)
        if hit:
            return hit
    return None


# ----- every page ------------------------------------------------------------------------------


def test_no_ui_file_writes_html_loads_from_the_web_or_holds_a_secret():
    assert len(UI_FILES) >= 15
    for path in UI_FILES:
        if path.suffix not in {".js", ".html", ".css"}:
            continue
        text = path.read_text()
        assert not re.search(r"\.innerHTML\s*=|insertAdjacentHTML|document\.write", text), path
        loads = r"""(src|href)\s*=\s*["']https?://|url\(\s*["']?https?://|@import|fetch\(\s*["']https?://"""
        assert not re.search(loads, text), f"{path} loads something from the web"
        assert not re.search(r"sk-[A-Za-z0-9]{10,}|APP_API_KEY=", text), path


def test_every_page_uses_the_shared_palette_and_defines_no_colors_of_its_own():
    kit = (ROOT / "packages/common/src/common/ui_kit/kit.css").read_text()
    for token in ("--accent", "--ok", "--warn", "--bad", "--c-route", "--c-search", "--c-write"):
        assert token in kit
    assert "prefers-color-scheme: dark" in kit  # light and dark
    for path in UI_FILES:
        if path.suffix == ".css" and path.name != "kit.css":
            assert not re.search(r"#[0-9a-fA-F]{3,8}\b", path.read_text()), (
                f"{path.name}: use the kit's tokens"
            )


def test_the_shell_has_a_domain_box_and_the_pages_follow_it():
    page = make_client().get("/")
    assert 'id="domain"' in page.text and "One domain = one collection" in page.text
    kit = (ROOT / "packages/common/src/common/ui_kit/kit.js").read_text()
    assert "getDomain" in kit and "setDomain" in kit and "onDomainChange" in kit
    for pkg in ("ingest", "retrieval", "evals"):
        assert "kit.getDomain" in (ROOT / f"packages/{pkg}/src/{pkg}/ui/app.js").read_text(), pkg
    assert "kit.getDomain" in (ROOT / "e2e/src/e2e/ui/chat.js").read_text()


def test_the_ingest_page_explains_its_options_and_can_delete_a_collection():
    html = (ROOT / "packages/ingest/src/ingest/ui/index.html").read_text()
    js = (ROOT / "packages/ingest/src/ingest/ui/app.js").read_text()
    assert 'id="c-delete-all"' in html and "Ingest path" in html and "key: value pairs" in html
    assert "?confirm=" in js and "Type the collection name to confirm" in js
    assert "never crosses a # heading" in js


def test_the_evals_page_has_a_metric_picker_driven_by_the_catalog():
    html = (ROOT / "packages/evals/src/evals/ui/index.html").read_text()
    js = (ROOT / "packages/evals/src/evals/ui/app.js").read_text()
    assert 'id="r-metrics"' in html and 'id="r-ret"' not in html  # the two checkboxes are gone
    assert '"/metrics"' in js and "chosenMetrics" in js and "relevant_chunk_ids" in js


# ----- a conversation and its traces -------------------------------------------------------------


def test_a_conversations_runs_are_found_by_its_id_and_each_trace_holds_the_whole_run():
    brain = Brain(routes=["retrieve", "tool"], answers=["Jupiter is the largest planet [1].", "It is 40.8."],
                  tool_turns=[calls_turn(("c1", "own__calculate", {"expression": "340 * 0.12"})), "done"])  # fmt: skip
    with make_client(shared=QdrantClient(":memory:"), brain=brain) as client:
        ingest_through_http(client)
        thread = start(client)
        question = DOC_TEXT.split(".")[0]
        first = say(client, thread, question, retrieval={"mode": "dense", "collection": "docs"})
        second = say(client, thread, "what is 12% of 340?")

        # find every run of the conversation by its id
        runs = client.get(f"/api/obs/traces?session_id={thread}", headers=H).json()
        assert [r["trace_id"] for r in runs] == [second["trace_id"], first["trace_id"]]
        assert [r["input_preview"] for r in runs] == ["what is 12% of 340?", question[:140]]
        assert {r["session_id"] for r in runs} == {thread} and all(
            r["llm_calls"] == 0 for r in runs
        )

        # open one: the question, every step with its details, and the answer
        tree = client.get(f"/api/obs/traces/{first['trace_id']}", headers=H).json()
        root = tree["roots"][0]
        assert root["attributes"]["session.id"] == thread
        assert root["attributes"]["input.value"] == question
        assert root["attributes"]["output.value"] == first["answer"]
        retrieve = find(root, "agent.retrieve")["attributes"]
        assert retrieve["agent.passage_count"] >= 1 and retrieve["agent.collection"] == "docs"
        top = retrieve["agent.passages"][0]
        assert (
            top["source"] == "animals.txt"
            and top["text"]
            and top["chunk_id"] == first["passages"][0]["chunk_id"]
        )
        assert find(root, "agent.route")["attributes"]["agent.route"] == "retrieve"
        assert find(root, "agent.generate")["attributes"]["agent.answer"] == first["answer"]

        # the tool run holds its arguments and result, and the MCP call is underneath it
        tool_root = client.get(f"/api/obs/traces/{second['trace_id']}", headers=H).json()["roots"][
            0
        ]
        tool = find(tool_root, "agent.tool")
        assert tool["attributes"]["agent.tool_arguments"] == {"expression": "340 * 0.12"}
        assert "40.8" in tool["attributes"]["agent.tool_result"] and "mcp.call_tool" in names(tool)

        assert client.get("/api/obs/traces?session_id=nobody", headers=H).json() == []


def test_reopening_a_conversation_over_http_returns_each_answers_details_and_its_domain():
    brain = Brain(routes=["retrieve"], answers=["Cats purr [1]."])
    with make_client(shared=QdrantClient(":memory:"), brain=brain) as client:
        ingest_through_http(client)
        thread = start(client)
        sent = say(
            client,
            thread,
            DOC_TEXT.split(".")[0],
            retrieval={"mode": "dense", "collection": "docs"},
        )
        detail = client.get(f"/api/agent/threads/{thread}", headers=H).json()
        assert detail["collection"] == "docs"
        user, answer = detail["history"]
        assert user["run"] is None
        run = answer["run"]
        assert run["question"] == DOC_TEXT.split(".")[0]
        assert run["response"]["trace_id"] == sent["trace_id"]
        assert run["response"]["passages"] and run["response"]["citations"]
        listed = client.get("/api/agent/threads", headers=H).json()
        assert listed[0]["collection"] == "docs"


def test_agent_traces_show_each_step_with_the_content_switch_on_by_default():
    brain = Brain(routes=["direct"], answers=["Hello."])
    with make_client(brain=brain) as client:
        thread = start(client)
        r = say(client, thread, "hi there", retrieval={"enabled": False}, tools={"enabled": False})
        root = client.get(f"/api/obs/traces/{r['trace_id']}", headers=H).json()["roots"][0]
        assert root["attributes"]["input.value"] == "hi there"
        assert root["attributes"]["output.value"] == "Hello."
        assert {"agent.route", "agent.generate", "agent.respond"} <= set(names(root))


# ----- deleting a domain, end to end ---------------------------------------------------------------


def test_two_domains_stay_apart_and_deleting_one_leaves_the_other():
    with make_client(shared=QdrantClient(":memory:")) as client:
        ingest_through_http(
            client,
            options={
                "collection": "beta",
                "chunking": {"strategy": "recursive", "chunk_size": 16, "overlap": 0},
            },
        )
        ingest_through_http(client, name="b.txt", text="Penguins live in the south. " * 6,
                            options={"collection": "acme", "chunking": {"strategy": "recursive", "chunk_size": 16, "overlap": 0}})  # fmt: skip
        cols = {
            c["name"]: c["points"] for c in client.get("/api/ingest/collections", headers=H).json()
        }
        assert set(cols) == {"beta", "acme"} and all(cols.values())

        found = client.post(
            "/api/retrieval/retrieve",
            json={"query": "Penguins live in the south.", "collection": "acme", "mode": "dense"},
            headers=H,
        ).json()
        assert found["hits"] and all(h["payload"]["source_name"] == "b.txt" for h in found["hits"])

        assert (
            client.delete("/api/ingest/collections/beta?confirm=beta", headers=H).status_code == 200
        )
        after = {
            c["name"]: c["points"] for c in client.get("/api/ingest/collections", headers=H).json()
        }
        assert after == {"acme": cols["acme"]}
        again = client.post(
            "/api/retrieval/retrieve",
            json={"query": "Penguins live in the south.", "collection": "acme", "mode": "dense"},
            headers=H,
        ).json()
        assert again["hits"]
        assert json.dumps(again) is not None
