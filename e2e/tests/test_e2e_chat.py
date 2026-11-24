"""The chat page, the agent as an eval target, and the whole journey in one app."""

import json
import re
from pathlib import Path

from qdrant_client import QdrantClient
from test_e2e import DOC_TEXT, KEY, by_name, ingest_through_http, make_client, wait_job

from agent_graph.testing import Brain, calls_turn
from evals.models import AccuracyResult, CompletenessResult, FaithfulnessResult

H = {"X-API-Key": KEY}
UI = Path(__file__).resolve().parents[1] / "src" / "e2e" / "ui"


class StubJudge:
    """Scores everything as perfect, and remembers what it was asked to judge."""

    model, prompt_version = "stub", "1"

    def __init__(self) -> None:
        self.answers: list[str] = []

    def score_faithfulness(self, answer, contexts):
        self.answers.append(answer)
        return FaithfulnessResult(claims=[], score=1.0)

    def score_completeness(self, answer, reference):
        return CompletenessResult(points=[], score=1.0)

    def score_accuracy(self, question, answer, reference):
        return AccuracyResult(score=1.0, rationale="stub")


# ----- the chat page --------------------------------------------------------------------------


def test_the_chat_page_and_script_are_public_and_hold_no_secret():
    client = make_client()
    page = client.get("/")
    for needle in ('id="c-send"', 'id="c-transcript"', 'id="c-approvals"', 'src="chat.js"'):
        assert needle in page.text
    for path in ("/chat.js", "/chat.css", "/app.js"):
        r = client.get(path)
        assert r.status_code == 200 and KEY not in r.text, path


def test_the_chat_script_never_writes_html_loads_nothing_from_the_web_and_sets_no_key():
    for name in ("chat.js", "chat.css", "index.html"):
        text = (UI / name).read_text()
        assert not re.search(r"\.innerHTML\s*=|insertAdjacentHTML|document\.write", text), name
        assert not re.search(r"https?://", text), f"{name} loads from the web"
        assert "APP_API_KEY=" not in text and "sk-" not in text, name
    assert "setKey" not in (UI / "chat.js").read_text()  # only the shell handles the key


def test_the_observability_page_and_the_chat_agree_on_the_link_format():
    root = Path(__file__).resolve().parents[2]
    obs = (root / "packages/observability/src/observability/ui/app.js").read_text()
    assert "(trace|conversation)" in obs and "hashchange" in obs  # #trace=<id>, #conversation=<id>
    chat = (UI / "chat.js").read_text()
    assert (
        '"#trace=" +' in chat
        and '"#conversation=" +' in chat
        and '"/ui/observability/" + hash' in chat
    )


def test_the_shell_tells_the_chat_what_the_status_says():
    assert 'dispatchEvent(new CustomEvent("status"' in (UI / "app.js").read_text()
    assert 'addEventListener("status"' in (UI / "chat.js").read_text()


# ----- the agent as an eval target -----------------------------------------------------------


def run_eval(client, dataset_id, options, config=None):
    body = {
        "dataset_id": dataset_id,
        "target": {"kind": "preset", "name": "agent", "options": options},
        "config": config or {"ks": [1, 3]},
    }
    r = client.post("/api/evals/runs", json=body, headers=H)
    assert r.status_code == 202, r.text
    return wait_job(client, r.json()["job_id"])["job_id"]


def reviewed_dataset(client):
    gen = client.post(
        "/api/evals/datasets/generate",
        json={"collection": "docs", "n": 3, "min_chars": 0},
        headers=H,
    )
    dataset_id = wait_job(client, gen.json()["job_id"])["result"]["dataset_id"]
    items = client.get(f"/api/evals/datasets/{dataset_id}", headers=H).json()["items"]
    for item in items:
        client.patch(
            f"/api/evals/datasets/{dataset_id}/items/{item['id']}",
            json={"status": "accepted"},
            headers=H,
        )
    return dataset_id


def test_the_agent_is_offered_as_an_eval_target_next_to_retrieval():
    client = make_client()
    status = client.get("/api/evals/status", headers=H).json()
    assert {"agent", "retrieval"} <= set(status["presets"])
    detail = by_name(client.get("/api/status", headers=H).json())["evals"]["detail"]
    assert "agent" in detail and "retrieval" in detail


def test_evals_score_the_whole_agent_answers_citations_and_retrieval_in_one_trace():
    judge = StubJudge()
    brain = Brain(routes=["retrieve"], answers=["The planet is Jupiter [1]."])
    with make_client(shared=QdrantClient(":memory:"), brain=brain, judge=judge) as client:
        ingest_through_http(client)
        dataset_id = reviewed_dataset(client)
        run_id = run_eval(client, dataset_id, {"mode": "dense", "top_k": 3})

        report = client.get(f"/api/evals/runs/{run_id}", headers=H).json()["report"]
        assert report["errors"] == 0
        assert report["metrics"]["recall@1"]["mean"] == 1.0  # the agent really retrieved
        assert "faithfulness" not in report["not_computed"]  # an answer came back, so it is judged
        assert judge.answers == ["The planet is Jupiter [1]."] * 3

        # Caches were off for every question: the model was asked all three times.
        assert brain.count["route"] == 3 and brain.count["answer"] == 3
        # Each question had a thread of its own, deleted afterwards.
        assert client.get("/api/agent/threads", headers=H).json() == []

        # One trace for the run, and the agent's steps sit inside each question.
        traces = client.get("/api/obs/traces?name=eval.run", headers=H).json()
        tree = client.get(f"/api/obs/traces/{traces[0]['trace_id']}", headers=H).json()
        assert len(tree["roots"]) == 1
        question = tree["roots"][0]["children"][0]
        assert question["name"] == "eval.question"

        def names(node):
            return [node["name"]] + [n for c in node.get("children", []) for n in names(c)]

        seen = names(question)
        assert "agent.run" in seen and "retrieval.retrieve" in seen and "agent.generate" in seen


def test_a_bad_agent_option_is_refused_before_the_run_starts():
    client = make_client(shared=QdrantClient(":memory:"))
    ds = client.post(
        "/api/evals/datasets", files={"file": ("g.jsonl", b'{"question": "Q?"}')}, headers=H
    ).json()
    for options, expected in (
        ({"colour": "red"}, "Unknown agent options"),
        ({"mode": "fuzzy"}, "Invalid agent option"),
    ):
        body = {
            "dataset_id": ds["id"],
            "target": {"kind": "preset", "name": "agent", "options": options},
        }
        r = client.post("/api/evals/runs", json=body, headers=H)
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_target"
        assert expected in r.json()["error"]["message"]


def test_an_eval_that_would_need_a_tool_approved_records_an_error_instead_of_hanging():
    from agent_graph.protocols import AgentTool  # noqa: F401

    # The last scripted turn repeats, so every question asks for the tool.
    brain = Brain(
        routes=["tool"],
        tool_turns=[calls_turn(("c1", "own__calculate", {"expression": "1 + 1"}))],
    )
    # `own__calculate` is read-only, so to force an approval the policy is "all".
    with make_client(shared=QdrantClient(":memory:"), brain=brain, judge=StubJudge()) as client:
        client.app.state.agent.settings.approval = "all"
        ingest_through_http(client)
        dataset_id = reviewed_dataset(client)
        run_id = run_eval(client, dataset_id, {"tools": True, "check": False})
        report = client.get(f"/api/evals/runs/{run_id}", headers=H).json()["report"]
        assert report["errors"] == 3  # recorded per question; the run itself completed
        assert (
            client.get("/api/agent/threads", headers=H).json() == []
        )  # and nothing was left behind


# ----- the whole journey --------------------------------------------------------------------


def test_done_when_documents_with_citations_a_tool_a_trace_per_message_and_a_truthful_status():
    brain = Brain(
        routes=["retrieve", "tool"],
        answers=["Jupiter is the largest planet [1].", "That is 40.8."],
        tool_turns=[calls_turn(("c1", "own__calculate", {"expression": "340 * 0.12"})), "done"],
    )
    with make_client(shared=QdrantClient(":memory:"), brain=brain, llm_key="sk-test") as client:
        ingest_through_http(client)
        status = by_name(client.get("/api/status", headers=H).json())
        assert all(
            status[n]["configured"] for n in ("agent", "llm", "ingest", "retrieval", "mcp", "evals")
        )
        assert status["agent"]["ui_path"] == "/ui/agent/"

        thread = client.post("/api/agent/threads", json={}, headers=H).json()["thread_id"]
        url = f"/api/agent/threads/{thread}/messages"

        first = client.post(
            url,
            json={"message": DOC_TEXT.split(".")[0], "options": {"retrieval": {"mode": "dense"}}},
            headers=H,
        ).json()
        assert first["status"] == "completed" and first["citations"] and first["passages"]
        assert first["citations"][0]["source_name"] == "animals.txt"

        second = client.post(url, json={"message": "what is 12% of 340?"}, headers=H).json()
        assert second["status"] == "completed" and second["route"] == "tool"
        assert (
            second["tool_calls"][0]["name"] == "own__calculate"
            and "40.8" in second["tool_calls"][0]["content"]
        )

        # Every message has its own trace, and the first one shows the pipeline's steps.
        assert first["trace_id"] != second["trace_id"]
        traces = client.get("/api/obs/traces?name=agent", headers=H).json()
        assert {t["trace_id"] for t in traces} == {first["trace_id"], second["trace_id"]}
        tree = client.get(f"/api/obs/traces/{first['trace_id']}", headers=H).json()
        assert tree["roots"][0]["name"] == "agent.run"

    # And the status panel tells the truth when something is missing.
    down = by_name(make_client(qdrant_up=False, llm_key="").get("/api/status", headers=H).json())
    assert down["agent"]["configured"] is False and down["agent"]["missing"] == ["OPENAI_API_KEY"]
    assert down["qdrant"]["reachable"] is False
    off = by_name(make_client(enable_agent=False).get("/api/status", headers=H).json())
    assert off["agent"]["enabled"] is False
    assert json.dumps(off).count(KEY) == 0


# ----- domains: deleting a collection through the whole app ------------------------------------


def test_after_a_collection_is_deleted_the_agent_cannot_answer_from_its_old_chunks():
    brain = Brain(routes=["retrieve"], answers=["Cats purr [1]."])
    with make_client(shared=QdrantClient(":memory:"), brain=brain) as client:
        ingest_through_http(client)
        thread = client.post("/api/agent/threads", json={}, headers=H).json()["thread_id"]
        url = f"/api/agent/threads/{thread}/messages"
        ask = {"message": DOC_TEXT.split(".")[0], "options": {"retrieval": {"mode": "dense"}}}

        before = client.post(url, json=ask, headers=H).json()
        assert before["passages"], "the document was found"
        again = client.post(url, json=ask, headers=H).json()  # cached search, same data
        assert again["passages"]

        gone = client.delete("/api/ingest/collections/docs?confirm=docs", headers=H)
        assert gone.status_code == 200 and gone.json()["chunks_deleted"] > 0

        after = client.post(url, json=ask, headers=H).json()
        assert after["passages"] == []  # NOT served from the cache of the deleted data
        assert after["status"] == "completed"
