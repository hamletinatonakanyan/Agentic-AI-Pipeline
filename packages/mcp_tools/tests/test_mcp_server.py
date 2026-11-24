import asyncio
import json
import sys

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from mcp_tools.server import build_server
from mcp_tools.settings import McpSettings
from mcp_tools.testing import FakeCorpus


def run(coro):
    return asyncio.run(coro)


def make(corpus=None):
    corpus = corpus or FakeCorpus()
    return build_server(McpSettings(_env_file=None), corpus), corpus


async def with_client(fn, corpus=None):
    server, fake = make(corpus)
    async with Client(server) as client:
        return await fn(client, fake)


# ----- what the server offers ---------------------------------------------------------------


def test_it_lists_its_tools_with_schemas_descriptions_and_hints():
    async def go(client, _):
        return (await client.list_tools()).tools

    tools = {t.name: t for t in run(with_client(go))}
    assert set(tools) == {"calculate", "search_corpus", "list_tables", "query_table"}

    calc = tools["calculate"]
    assert "arithmetic" in calc.description.lower()
    assert calc.input_schema["required"] == ["expression"]
    assert calc.input_schema["properties"]["expression"]["type"] == "string"
    assert calc.annotations.read_only_hint is True and calc.annotations.open_world_hint is False

    search = tools["search_corpus"]
    props = search.input_schema["properties"]
    assert search.input_schema["required"] == ["query"]  # top_k and collection are optional
    assert props["top_k"]["minimum"] == 1 and props["top_k"]["maximum"] == 20
    assert props["query"]["minLength"] == 1 and props["query"]["maxLength"] == 2000
    assert "natural language" in props["query"]["description"]
    assert search.output_schema["properties"]["results"]["type"] == "array"


def test_it_offers_a_resource_and_a_prompt():
    async def go(client, _):
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        prompts = await client.list_prompts()
        return resources, templates, prompts

    resources, templates, prompts = run(with_client(go))
    assert [str(r.uri) for r in resources.resources] == ["corpus://sources"]
    assert [t.uri_template for t in templates.resource_templates] == [
        "corpus://sources/{collection}"
    ]
    assert [p.name for p in prompts.prompts] == ["summarize_with_citations"]


# ----- calculate ------------------------------------------------------------------------------------


def test_calculate_returns_a_number_both_as_text_and_structured():
    async def go(client, _):
        return await client.call_tool("calculate", {"expression": "2 * (3 + 4)"})

    result = run(with_client(go))
    assert result.is_error is False
    assert result.structured_content == {"result": 14.0}
    assert result.content[0].text == "14.0"


def test_calculate_refuses_code_and_says_why_without_leaking_anything():
    async def go(client, _):
        bad = await client.call_tool("calculate", {"expression": "__import__('os').getcwd()"})
        zero = await client.call_tool("calculate", {"expression": "1/0"})
        return bad, zero

    bad, zero = run(with_client(go))
    assert bad.is_error and "not supported" in bad.content[0].text
    assert "Traceback" not in bad.content[0].text and "/Users" not in bad.content[0].text
    assert zero.is_error and "divide by zero" in zero.content[0].text


def test_wrong_argument_types_are_rejected_by_the_schema():
    async def go(client, _):
        return [
            await client.call_tool("calculate", {"expression": 5}),
            await client.call_tool("calculate", {}),
            await client.call_tool("search_corpus", {"query": "x", "top_k": 99}),
            await client.call_tool("search_corpus", {"query": ""}),
            await client.call_tool("search_corpus", {"query": "x" * 2001}),
        ]

    assert all(r.is_error for r in run(with_client(go)))


# ----- search_corpus ----------------------------------------------------------------------------------


def test_search_corpus_returns_passages_with_their_sources():
    async def go(client, corpus):
        result = await client.call_tool("search_corpus", {"query": "purring", "top_k": 2})
        return result, corpus

    result, corpus = run(with_client(go))
    assert result.is_error is False
    data = result.structured_content
    assert data["query"] == "purring" and data["collection"] == "docs"
    assert [p["rank"] for p in data["results"]] == [1, 2]
    assert data["results"][0]["source_name"] == "cats.txt" and data["results"][0]["page"] == 1
    assert corpus.searches == [
        ("purring", 2, None)
    ]  # the default top_k and collection pass through


def test_search_corpus_defaults_and_collection():
    async def go(client, corpus):
        await client.call_tool("search_corpus", {"query": "x"})
        await client.call_tool("search_corpus", {"query": "y", "collection": "legal"})
        return corpus.searches

    assert run(with_client(go)) == [("x", 5, None), ("y", 5, "legal")]


def test_a_failing_backend_becomes_a_clear_tool_error():
    async def go(client, _):
        return await client.call_tool("search_corpus", {"query": "x"})

    result = run(with_client(go, FakeCorpus(fail="cannot reach the retrieval service")))
    assert result.is_error and "cannot reach the retrieval service" in result.content[0].text


# ----- resource and prompt -----------------------------------------------------------------------------


def test_the_sources_resource_lists_ingested_documents():
    async def go(client, _):
        a = await client.read_resource("corpus://sources")
        b = await client.read_resource("corpus://sources/legal")
        return a, b

    a, b = run(with_client(go))
    items = json.loads(a.contents[0].text)
    assert items == [{"source_id": "s1", "source_name": "cats.txt", "chunk_count": 4, "pages": 2}]
    assert json.loads(b.contents[0].text) == items


def test_the_prompt_tells_the_model_to_search_and_cite():
    async def go(client, _):
        return await client.get_prompt("summarize_with_citations", {"topic": "cat sleep"})

    prompt = run(with_client(go))
    text = prompt.messages[0].content.text
    assert "cat sleep" in text and "search_corpus" in text and "cite" in text.lower()


# ----- over stdio, as a real subprocess ------------------------------------------------------------------


@pytest.fixture
def stdio_params(tmp_path):
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_tools.server"],
        # A dead address, so nothing real is ever contacted. cwd has no .env to read.
        env={"APP_API_KEY": "k" * 16, "MCP_RETRIEVAL_URL": "http://127.0.0.1:9/api/retrieval"},
        cwd=str(tmp_path),
    )


def test_the_real_server_works_over_stdio(stdio_params):
    async def go():
        async with Client(stdio_params) as client:
            names = sorted(t.name for t in (await client.list_tools()).tools)
            calc = await client.call_tool("calculate", {"expression": "6 * 7"})
            search = await client.call_tool("search_corpus", {"query": "x"})
            return client.server_info.name, names, calc, search

    name, names, calc, search = run(go())
    assert name == "agentic-blocks" and names == [
        "calculate",
        "list_tables",
        "query_table",
        "search_corpus",
    ]
    assert calc.structured_content == {"result": 42.0}
    assert search.is_error and "cannot reach the retrieval service" in search.content[0].text


def test_the_stdio_server_reports_a_missing_key_clearly(tmp_path):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_tools.server"], env={}, cwd=str(tmp_path)
    )

    async def go():
        async with Client(params) as client:
            return await client.call_tool("search_corpus", {"query": "x"})

    result = run(go())
    assert result.is_error and "APP_API_KEY is not set" in result.content[0].text
