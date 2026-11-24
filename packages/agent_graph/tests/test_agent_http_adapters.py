import asyncio
import json

import httpx
import pytest

from agent_graph.http_adapters import HttpRetriever, HttpToolProvider
from agent_graph.models import RetrievalChoice
from common import AppError

KEY = "test-key-0123456789"


def retriever(handler):
    return HttpRetriever(
        "http://r:8002/", lambda: KEY, client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def tools(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpToolProvider("http://m:8004", lambda: KEY, client=client)


def test_retriever_sends_the_options_and_the_key_and_maps_hits_to_passages():
    seen = {}

    def handler(request):
        seen.update(
            url=str(request.url), key=request.headers["x-api-key"], body=json.loads(request.content)
        )
        return httpx.Response(200, json={
            "hits": [{"chunk_id": "c1", "text": "T", "score": 0.4, "rerank_score": 0.9,
                      "payload": {"source_name": "a.pdf", "page": 3}}],
            "rewritten_queries": ["q", "q2"], "timings": {"total_ms": 7.5},
        })  # fmt: skip

    out = retriever(handler).retrieve(
        "q", RetrievalChoice(mode="dense", rerank=True, collection="legal"), 4
    )
    assert seen["url"] == "http://r:8002/api/retrieval/retrieve" and seen["key"] == KEY
    assert seen["body"] == {
        "query": "q",
        "mode": "dense",
        "rewrite": "none",
        "rerank": True,
        "top_k": 4,
        "collection": "legal",
    }
    assert (out.passages[0].source_name, out.passages[0].page, out.passages[0].score) == (
        "a.pdf",
        3,
        0.9,
    )
    assert out.rewritten_queries == ["q", "q2"] and out.retrieval_ms == 7.5


def test_retriever_failures_become_clean_errors_without_the_key():
    down = retriever(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    with pytest.raises(AppError) as e:
        down.retrieve("q", RetrievalChoice(), 3)
    assert e.value.code == "retrieval_unreachable" and KEY not in e.value.message
    bad = retriever(
        lambda r: httpx.Response(503, json={"error": {"code": "x", "message": "Qdrant is down"}})
    )
    with pytest.raises(AppError) as e:
        bad.retrieve("q", RetrievalChoice(), 3)
    assert e.value.code == "retrieval_failed" and "Qdrant is down" in e.value.message


def test_tool_provider_lists_tools_with_read_only_and_calls_them_through_the_server_and_tool_names():
    calls = []

    def handler(request):
        if request.url.path == "/api/mcp/llm-tools":
            return httpx.Response(200, json={
                "tools": [{"type": "function", "name": "own__calculate", "description": "d", "parameters": {"type": "object"}}],
                "meta": {"own__calculate": {"server": "own", "tool": "calculate", "read_only": True}},
            })  # fmt: skip
        calls.append((request.url.path, json.loads(request.content), request.headers["x-api-key"]))
        return httpx.Response(
            200, json={"ok": True, "is_error": False, "content": "4", "duration_ms": 2.0}
        )

    async def go():
        provider = tools(handler)
        (tool,) = await provider.tools()
        assert (tool.name, tool.read_only) == ("own__calculate", True)
        out = await provider.call("own__calculate", {"expression": "2+2"})
        assert out.ok and out.content == "4" and out.duration_ms == 2.0
        assert calls == [
            ("/api/mcp/servers/own/tools/calculate/call", {"arguments": {"expression": "2+2"}}, KEY)
        ]
        unknown = await provider.call("nope", {})
        assert not unknown.ok and unknown.error_kind == "not_found"

    asyncio.run(go())


def test_tool_provider_reports_refusals_and_unreachable_services_as_failed_outcomes_never_raises():
    def handler(request):
        if request.url.path.endswith("llm-tools"):
            return httpx.Response(
                200,
                json={"tools": [], "meta": {"t": {"server": "s", "tool": "t", "read_only": None}}},
            )
        if request.url.host == "m":
            return httpx.Response(
                403,
                json={"error": {"code": "tool_not_allowed", "message": "not on the allow-list"}},
            )
        raise httpx.ConnectError("refused")

    async def go():
        p = tools(handler)
        await p.tools()
        refused = await p.call("t", {})
        assert (
            not refused.ok and refused.error_kind == "refused" and "allow-list" in refused.content
        )
        p._base = "http://elsewhere/api/mcp"
        down = await p.call("t", {})
        assert not down.ok and down.error_kind == "unreachable" and KEY not in down.content

    asyncio.run(go())
