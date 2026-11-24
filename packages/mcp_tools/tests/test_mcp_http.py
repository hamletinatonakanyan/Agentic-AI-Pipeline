import asyncio

import httpx
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from common import AppSettings, get_app_settings
from mcp_tools.app import create_app
from mcp_tools.factory import build_mcp
from mcp_tools.models import ServerSpec
from mcp_tools.settings import McpSettings
from mcp_tools.testing import FakeCorpus
from observability import shutdown_observability
from observability.settings import ObsSettings

KEY = "test-key-0123456789"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _obs_reset():
    shutdown_observability()
    yield
    shutdown_observability()


def make_app(key: str = KEY):
    settings = McpSettings(_env_file=None, db_path=":memory:")
    mcp = build_mcp(settings, corpus=FakeCorpus(), key_provider=lambda: key)
    app = create_app(mcp=mcp, obs_settings=ObsSettings(_env_file=None, local_db_path=":memory:"))
    # The management API reads the key through a dependency, /mcp through the provider above.
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=key)
    return app, mcp


def mcp_client(url: str, key: str | None) -> Client:
    headers = {"X-API-Key": key} if key else {}
    http = httpx2.AsyncClient(headers=headers)
    return Client(streamable_http_client(url + "/mcp", http_client=http))


# ----- the key ------------------------------------------------------------------------------------


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"x-api-key": ""}])
def test_the_mcp_endpoint_refuses_requests_without_the_right_key(live, headers):
    app, _ = make_app()
    with live(app) as server:
        r = httpx.post(
            server.url + "/mcp/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Accept": "application/json, text/event-stream", **headers},
        )
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


def test_with_no_key_configured_the_endpoint_answers_503(live):
    app, _ = make_app(key="")
    with live(app) as server:
        r = httpx.post(server.url + "/mcp/", json={}, headers={"X-API-Key": "anything"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "not_configured"


def test_an_mcp_client_with_the_key_can_list_and_call_our_tools(live):
    app, _ = make_app()

    async def go(url):
        async with mcp_client(url, KEY) as client:
            names = sorted(t.name for t in (await client.list_tools()).tools)
            calc = await client.call_tool("calculate", {"expression": "6 * 7"})
            search = await client.call_tool("search_corpus", {"query": "purring", "top_k": 2})
            return client.protocol_version, names, calc, search

    with live(app) as server:
        version, names, calc, search = run(go(server.url))
    assert version == "2026-07-28" and names == [
        "calculate",
        "list_tables",
        "query_table",
        "search_corpus",
    ]
    assert calc.structured_content == {"result": 42.0}
    assert [p["rank"] for p in search.structured_content["results"]] == [1, 2]


def test_an_mcp_client_without_the_key_cannot_connect(live):
    app, _ = make_app()

    async def go(url):
        async with mcp_client(url, None) as client:
            await client.list_tools()

    with live(app) as server, pytest.raises(BaseException):  # noqa: B017, PT011
        run(go(server.url))


def test_a_foreign_host_name_is_refused_even_with_the_key(live):
    """DNS rebinding: a web page on another domain must not be able to reach this server."""
    app, _ = make_app()
    with live(app) as server:
        r = httpx.post(
            server.url + "/mcp/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={
                "X-API-Key": KEY,
                "Host": "evil.example.com",
                "Accept": "application/json, text/event-stream",
            },
        )
    assert r.status_code in (400, 403, 421)


def test_the_management_api_and_the_mcp_endpoint_share_one_server(live):
    app, _ = make_app()
    with live(app) as server:
        assert httpx.get(server.url + "/health").json() == {"ok": True}
        assert httpx.get(server.url + "/api/mcp/servers").status_code == 401
        listing = httpx.get(server.url + "/api/mcp/servers", headers={"X-API-Key": KEY}).json()
        assert [s["spec"]["name"] for s in listing] == ["own"]


# ----- our own client talking to our own server over HTTP ----------------------------------------------


def http_spec(url: str, ref: str = "APP_API_KEY", **kw) -> ServerSpec:
    return ServerSpec(
        name="remote", transport="streamable-http", url=url + "/mcp",
        headers_refs={"X-API-Key": ref}, tool_allowlist=["calculate"], **kw,
    )  # fmt: skip


def test_the_manager_reaches_a_server_over_http_sending_the_key_from_the_environment(live):
    from mcp_tools.manager import MCPClientManager
    from mcp_tools.registry import ServerRegistry

    app, _ = make_app()
    with live(app) as server:
        manager = MCPClientManager(
            McpSettings(_env_file=None), ServerRegistry(":memory:"), env={"APP_API_KEY": KEY}
        )
        manager.register(http_spec(server.url))
        names = {t.name for t in run(manager.list_tools()).tools}
        result = run(manager.call_tool("remote", "calculate", {"expression": "2 ** 10"}))
        not_allowed = run(manager.call_tool("remote", "search_corpus", {"query": "x"}))
        assert run(manager.check("remote")).state == "ok"
    assert names == {"calculate", "search_corpus", "list_tables", "query_table"}
    assert result.ok and result.structured == {"result": 1024.0}
    assert not_allowed.error_kind == "not_allowed"  # the allow-list applies over HTTP too


def test_a_wrong_key_in_the_environment_is_a_clear_connection_failure(live):
    from mcp_tools.manager import MCPClientManager
    from mcp_tools.registry import ServerRegistry

    app, _ = make_app()
    with live(app) as server:
        manager = MCPClientManager(
            McpSettings(_env_file=None), ServerRegistry(":memory:"), env={"APP_API_KEY": "nope"}
        )
        manager.register(http_spec(server.url))
        result = run(manager.call_tool("remote", "calculate", {"expression": "1+1"}))
        status = run(manager.check("remote"))
    assert result.error_kind == "connection" and result.is_error
    assert status.state == "error" and "nope" not in status.error and "nope" not in result.content


def test_a_missing_environment_variable_is_reported_by_name_only():
    from mcp_tools.manager import MCPClientManager
    from mcp_tools.registry import ServerRegistry

    manager = MCPClientManager(McpSettings(_env_file=None), ServerRegistry(":memory:"), env={})
    manager.register(http_spec("http://localhost:9"))
    result = run(manager.call_tool("remote", "calculate", {"expression": "1+1"}))
    assert result.error_kind == "connection" and "APP_API_KEY is not set" in result.content


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
def test_both_forms_of_the_url_are_served_directly_without_a_redirect(live, path):
    """`/mcp` is the URL people give to clients. A plain Mount would only serve `/mcp/`."""
    app, _ = make_app()
    with live(app) as server:
        r = httpx.post(
            server.url + path,
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={"X-API-Key": KEY, "Accept": "application/json, text/event-stream"},
            follow_redirects=False,
        )
    assert r.status_code not in (301, 302, 307, 308, 404, 405)
