"""Table questions through the whole app: ingest a csv, then count its rows exactly."""

import json

from qdrant_client import QdrantClient
from test_e2e import KEY, make_client

from agent_graph.testing import Brain, calls_turn

H = {"X-API-Key": KEY}
PLANS = ["Starter", "Team", "Business"]


def tickets_csv(rows=150, p1_every=7, prefix="HC"):
    """Deterministic rows: every `p1_every`-th ticket is P1, the plan cycles, hours are numbers."""
    lines = ["ticket_id,plan,priority,hours,summary"]
    for i in range(rows):
        priority = "P1" if i % p1_every == 0 else ["P2", "P3", "P4"][i % 3]
        lines.append(
            f"{prefix}-{1000 + i},{PLANS[i % 3]},{priority},{(i % 10) + 1},Issue number {i}"
        )
    return "\n".join(lines) + "\n"


def expected_p1(rows=150, p1_every=7):
    return len(range(0, rows, p1_every))


def ingest(client, name, text, collection):
    options = {
        "collection": collection,
        "chunking": {"strategy": "recursive", "chunk_size": 64, "overlap": 0},
    }
    parts = [("files", (name, text.encode()))]
    job = client.post(
        "/api/ingest/files", files=parts, data={"options": json.dumps(options)}, headers=H
    ).json()["job_id"]
    done = client.get(f"/api/ingest/jobs/{job}", headers=H).json()
    assert done["state"] == "completed", done
    return done["report"]


def tool(client, name, **arguments):
    r = client.post(
        f"/api/mcp/servers/own/tools/{name}/call", json={"arguments": arguments}, headers=H
    )
    assert r.status_code == 200, r.text
    return r.json()


# ----- exact answers ----------------------------------------------------------------------------------


def test_a_table_split_into_many_chunks_is_counted_exactly():
    with make_client(shared=QdrantClient(":memory:")) as client:
        ingest(client, "tickets.csv", tickets_csv(), "alpha")
        sources = client.get("/api/ingest/collections/alpha/sources", headers=H).json()
        assert sources[0]["chunk_count"] > 20  # the 150 rows really are spread over many chunks

        listing = tool(client, "list_tables", collection="alpha")
        (t,) = listing["structured"]["tables"]
        assert (t["name"], t["rows"]) == ("tickets.csv", 150)
        assert t["columns"] == ["ticket_id", "plan", "priority", "hours", "summary"]

        counted = tool(
            client, "query_table", table="tickets.csv", collection="alpha",
            filters=[{"column": "priority", "op": "=", "value": "P1"}],
        )  # fmt: skip
        assert counted["is_error"] is False
        assert counted["structured"]["result"] == expected_p1() == 22
        assert counted["structured"]["total_rows"] == 150

        per_plan = tool(
            client, "query_table", table="tickets.csv", collection="alpha", group_by=["plan"]
        )
        assert {g["plan"]: g["value"] for g in per_plan["structured"]["result"]} == {
            p: 50 for p in PLANS
        }

        total_hours = tool(
            client,
            "query_table",
            table="tickets.csv",
            collection="alpha",
            aggregate="sum",
            column="hours",
        )
        assert total_hours["structured"]["result"] == sum((i % 10) + 1 for i in range(150))


def test_a_wrong_name_gets_a_message_that_names_the_right_ones():
    with make_client(shared=QdrantClient(":memory:")) as client:
        ingest(client, "tickets.csv", tickets_csv(rows=10), "alpha")
        bad = tool(client, "query_table", table="ticket.csv", collection="alpha")
        assert bad["is_error"] and "Tables available: tickets.csv" in bad["content"]
        col = tool(
            client,
            "query_table",
            table="tickets.csv",
            collection="alpha",
            aggregate="sum",
            column="hour",
        )
        assert (
            col["is_error"]
            and "Columns: ticket_id, plan, priority, hours, summary" in col["content"]
        )
        missing = tool(client, "query_table", table="tickets.csv", collection="nowhere")
        assert missing["is_error"] and "No collection 'nowhere'" in missing["content"]


def test_two_domains_keep_their_own_tables_and_deleting_one_removes_its_table():
    with make_client(shared=QdrantClient(":memory:")) as client:
        ingest(client, "tickets.csv", tickets_csv(rows=150), "alpha")
        ingest(client, "tickets.csv", tickets_csv(rows=30, p1_every=3, prefix="LC"), "beta")
        p1 = {"table": "tickets.csv", "filters": [{"column": "priority", "op": "=", "value": "P1"}]}
        assert tool(client, "query_table", collection="alpha", **p1)["structured"]["result"] == 22
        assert tool(client, "query_table", collection="beta", **p1)["structured"]["result"] == 10

        assert (
            client.delete("/api/ingest/collections/alpha?confirm=alpha", headers=H).status_code
            == 200
        )
        gone = tool(client, "query_table", collection="alpha", **p1)
        assert gone["is_error"]  # the table is not there any more
        assert tool(client, "query_table", collection="beta", **p1)["structured"]["result"] == 10


def test_a_table_inside_a_markdown_document_can_be_queried_too():
    doc = "# Plans\n\nSome intro text.\n\n| plan | price |\n| --- | --- |\n| Starter | 29 |\n| Team | 99 |\n| Business | 349 |\n\nMore text.\n"
    with make_client(shared=QdrantClient(":memory:")) as client:
        ingest(client, "guide.md", doc, "alpha")
        (t,) = tool(client, "list_tables", collection="alpha")["structured"]["tables"]
        assert (t["name"], t["rows"]) == ("guide.md", 3)
        cheap = tool(client, "query_table", table="guide.md", collection="alpha", aggregate="rows",
                     filters=[{"column": "price", "op": "<", "value": 100}])  # fmt: skip
        assert [r["plan"] for r in cheap["structured"]["result"]] == ["Starter", "Team"]


# ----- the agent ----------------------------------------------------------------------------------------


def test_the_agent_counts_with_the_table_tool_and_stays_in_the_active_domain():
    brain = Brain(
        routes=["tool"],
        tool_turns=[
            # the model asks about ANOTHER collection: the agent must pin the active one
            calls_turn(("c1", "own__query_table", {
                "table": "tickets.csv", "collection": "beta",
                "filters": [{"column": "priority", "op": "=", "value": "P1"}],
            })),
            "done",
        ],
        answers=["There are 22 P1 tickets."],
    )  # fmt: skip
    with make_client(shared=QdrantClient(":memory:"), brain=brain) as client:
        ingest(client, "tickets.csv", tickets_csv(rows=150), "alpha")
        ingest(client, "tickets.csv", tickets_csv(rows=30, p1_every=3, prefix="LC"), "beta")
        thread = client.post("/api/agent/threads", json={}, headers=H).json()["thread_id"]
        r = client.post(
            f"/api/agent/threads/{thread}/messages",
            json={
                "message": "How many P1 tickets are there?",
                "options": {"retrieval": {"collection": "alpha"}},
            },
            headers=H,
        ).json()
        assert r["status"] == "completed" and r["route"] == "tool"  # read-only tool: no approval
        (call,) = r["tool_calls"]
        assert call["name"] == "own__query_table" and call["status"] == "ok"
        assert call["arguments"]["collection"] == "alpha"  # not "beta": pinned
        assert '"result":22' in call["content"].replace(" ", "")  # alpha's count, not beta's 10

        # the model was told the active collection, and the planner knows about table questions
        first = next(c for c in brain.calls if c.kind == "complete_with_tools")
        assert 'collection "alpha"' in first.items[0]["content"]
        assert "list_tables" in first.system and "query_table" in first.system
        tool_names = {t["name"] for t in first.tools}
        assert {"own__list_tables", "own__query_table"} <= tool_names

        # and the trace shows what was really sent
        tree = client.get(f"/api/obs/traces/{r['trace_id']}", headers=H).json()

        def find(node, name):
            if node["name"] == name:
                return node
            return next((hit for c in node["children"] if (hit := find(c, name))), None)

        sent = find(tree["roots"][0], "agent.tool")["attributes"]["agent.tool_arguments"]
        assert sent["collection"] == "alpha"


def test_the_router_is_told_that_counting_rows_needs_a_tool_not_a_search():
    brain = Brain(routes=["direct"], answers=["ok"])
    with make_client(brain=brain) as client:
        thread = client.post("/api/agent/threads", json={}, headers=H).json()["thread_id"]
        client.post(f"/api/agent/threads/{thread}/messages", json={"message": "hi"}, headers=H)
        route_prompt = next(c.prompt for c in brain.calls if c.kind == "complete_json")
        assert "counting, summing or averaging rows of a table" in route_prompt
        assert (
            "own__query_table: Count, sum, average" in route_prompt
        )  # the description starts with the job


def test_the_http_provider_the_stdio_server_uses_reads_every_row_from_a_live_app(live):
    """In the real app the agent's tool runs in a separate process that reaches retrieval over HTTP,
    sending the key. This is that path, against a real server on a real socket."""
    import asyncio

    import httpx

    from mcp_tools.corpus import HttpCorpusProvider
    from mcp_tools.settings import McpSettings
    from mcp_tools.tables import TableFilter, build_tables, find_table, run_query

    app = make_client(shared=QdrantClient(":memory:")).app
    with live(app) as server:
        options = {
            "collection": "alpha",
            "chunking": {"strategy": "recursive", "chunk_size": 64, "overlap": 0},
        }
        files = [("files", ("tickets.csv", tickets_csv().encode()))]
        sent = httpx.post(
            f"{server.url}/api/ingest/files",
            files=files,
            data={"options": json.dumps(options)},
            headers=H,
        )
        assert sent.status_code == 202

        provider = HttpCorpusProvider(
            McpSettings(_env_file=None, retrieval_url=f"{server.url}/api/retrieval"),
            env={"APP_API_KEY": KEY},
        )
        got = asyncio.run(provider.chunks("alpha", "tickets.csv"))
        assert len(got.chunks) > 20 and not got.truncated
        tables, _ = build_tables(got.chunks)
        table = find_table(tables, "tickets.csv")
        assert len(table.rows) == 150
        assert (
            run_query(table, filters=[TableFilter(column="priority", op="=", value="P1")]).result
            == 22
        )  # type: ignore[list-item]

        wrong_key = HttpCorpusProvider(
            McpSettings(_env_file=None, retrieval_url=f"{server.url}/api/retrieval"),
            env={"APP_API_KEY": "wrong"},
        )
        try:
            asyncio.run(wrong_key.chunks("alpha", None))
        except Exception as exc:  # the standard envelope: a clear message, no key in it
            assert "Missing or invalid X-API-Key" in str(exc)
            assert "wrong" not in str(exc)
        else:
            raise AssertionError("a wrong key must be refused")
