import pytest
from fastapi.testclient import TestClient
from mcp import Client

from common import AppSettings, get_app_settings
from mcp_tools.app import create_app
from mcp_tools.factory import build_mcp
from mcp_tools.settings import McpSettings
from mcp_tools.testing import FakeCorpus
from observability import shutdown_observability
from observability.settings import ObsSettings

KEY = "test-key-0123456789"
H = {"X-API-Key": KEY}


@pytest.fixture
def client(rig):
    """The real app. Servers are in-process: 'own' is our own server, 'probe' is a fake one."""
    shutdown_observability()
    holder = {}

    def factory(spec):
        if spec.name == "own":
            return Client(holder["mcp"].server)
        return rig.probes[spec.name].client()

    mcp = build_mcp(
        McpSettings(_env_file=None, db_path=":memory:"),
        corpus=FakeCorpus(),
        key_provider=lambda: KEY,
        client_factory=factory,
    )
    holder["mcp"] = mcp
    app = create_app(mcp=mcp, obs_settings=ObsSettings(_env_file=None, local_db_path=":memory:"))
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=KEY)
    yield TestClient(app, follow_redirects=False)
    shutdown_observability()


def server_json(name="probe", **kw):
    return {
        "name": name, "transport": "stdio", "command": "python",
        "args": ["-m", "mcp_tools.server"], "tool_allowlist": ["echo"], **kw,
    }  # fmt: skip


# ----- access ----------------------------------------------------------------------------------


def test_health_is_public_and_the_rest_needs_the_key(client):
    assert client.get("/health").status_code == 200
    protected = [
        ("GET", "/status"), ("GET", "/api/mcp/status"), ("GET", "/api/mcp/servers"),
        ("POST", "/api/mcp/servers"), ("DELETE", "/api/mcp/servers/own"),
        ("PATCH", "/api/mcp/servers/own/allowlist"), ("PATCH", "/api/mcp/servers/own/enabled"),
        ("POST", "/api/mcp/servers/own/check"), ("GET", "/api/mcp/servers/own/tools"),
        ("POST", "/api/mcp/servers/own/tools/calculate/call"), ("GET", "/api/mcp/llm-tools"),
        ("GET", "/api/mcp/exposed"),
    ]  # fmt: skip
    for method, path in protected:
        assert client.request(method, path).status_code == 401, path


def test_ui_is_public_and_has_no_secrets(client):
    assert client.get("/").headers["location"] == "/ui/"
    for path in ("/ui/", "/ui/app.js", "/ui/kit/kit.js"):
        r = client.get(path)
        assert r.status_code == 200 and KEY not in r.text
    assert "MCP" in client.get("/ui/").text


# ----- servers ---------------------------------------------------------------------------------------


def test_our_own_server_is_registered_from_the_start(client):
    servers = client.get("/api/mcp/servers", headers=H).json()
    own = next(s for s in servers if s["spec"]["name"] == "own")
    assert own["spec"]["tool_allowlist"] == [
        "calculate",
        "search_corpus",
        "list_tables",
        "query_table",
    ]
    assert own["state"] == "unknown"  # nothing has contacted it yet
    assert own["spec"]["env_refs"] == ["APP_API_KEY", "MCP_RETRIEVAL_URL", "MCP_INGEST_URL"]


def test_register_check_and_delete(client):
    created = client.post("/api/mcp/servers", json=server_json(), headers=H)
    assert created.status_code == 201 and created.json()["spec"]["name"] == "probe"
    assert client.post("/api/mcp/servers", json=server_json(), headers=H).status_code == 409
    checked = client.post("/api/mcp/servers/probe/check", headers=H).json()
    assert checked["state"] == "ok" and "echo" in checked["tool_names"]
    assert client.delete("/api/mcp/servers/probe", headers=H).json() == {"deleted": "probe"}
    gone = client.delete("/api/mcp/servers/probe", headers=H)
    assert gone.status_code == 404 and gone.json()["error"]["code"] == "server_not_found"


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (server_json(command="bash", args=["-c", "id"]), 403, "command_not_allowed"),
        (server_json(env_refs=["OPENAI_API_KEY"]), 403, "env_ref_not_allowed"),
        ({"name": "x", "transport": "streamable-http", "url": "https://evil.example.com/mcp"}, 403, "host_not_allowed"),
        ({"name": "x", "transport": "streamable-http", "url": "http://localhost/mcp",
          "headers_refs": {"X-Leak": "OPENAI_API_KEY"}}, 403, "header_ref_not_allowed"),
        ({"name": "Bad Name", "transport": "stdio", "command": "python"}, 422, "validation_error"),
        ({"name": "x", "transport": "stdio"}, 422, "validation_error"),
    ],
)  # fmt: skip
def test_registrations_that_break_the_rules_are_refused_with_the_envelope(
    client, body, status, code
):
    r = client.post("/api/mcp/servers", json=body, headers=H)
    assert r.status_code == status and r.json()["error"]["code"] == code
    names = [s["spec"]["name"] for s in client.get("/api/mcp/servers", headers=H).json()]
    assert names == ["own"]  # nothing was saved


def test_allowlist_and_enabled_can_be_changed(client):
    client.post("/api/mcp/servers", json=server_json(), headers=H)
    r = client.patch(
        "/api/mcp/servers/probe/allowlist",
        json={"tool_allowlist": ["fail", "echo", "echo"]},
        headers=H,
    )
    assert r.json()["tool_allowlist"] == ["echo", "fail"]  # sorted, no duplicates
    r = client.patch("/api/mcp/servers/probe/enabled", json={"enabled": False}, headers=H)
    assert r.json()["enabled"] is False
    states = {
        s["spec"]["name"]: s["state"] for s in client.get("/api/mcp/servers", headers=H).json()
    }
    assert states["probe"] == "disabled"
    assert (
        client.patch("/api/mcp/servers/none/enabled", json={"enabled": True}, headers=H).status_code
        == 404
    )


# ----- tools -------------------------------------------------------------------------------------------


def test_list_a_servers_tools_with_their_schemas(client):
    r = client.get("/api/mcp/servers/own/tools", headers=H).json()
    assert {t["name"] for t in r["tools"]} == {
        "calculate",
        "search_corpus",
        "list_tables",
        "query_table",
    } and r["errors"] == {}
    calc = next(t for t in r["tools"] if t["name"] == "calculate")
    assert calc["input_schema"]["required"] == ["expression"]
    unknown = client.get("/api/mcp/servers/none/tools", headers=H)
    assert unknown.status_code == 404


def call(client, server, tool, arguments):
    return client.post(
        f"/api/mcp/servers/{server}/tools/{tool}/call", json={"arguments": arguments}, headers=H
    )


def test_calling_a_tool(client):
    r = call(client, "own", "calculate", {"expression": "2 * (3 + 4)"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["structured"] == {"result": 14.0} and body["duration_ms"] > 0
    search = call(client, "own", "search_corpus", {"query": "purring", "top_k": 1}).json()
    assert search["ok"] and search["structured"]["results"][0]["source_name"] == "cats.txt"


def test_refused_calls_are_http_errors_but_a_failing_tool_is_a_normal_result(client):
    client.post("/api/mcp/servers", json=server_json(tool_allowlist=["echo"]), headers=H)
    cases = [
        (call(client, "own", "calculate", {}), 422, "tool_invalid_arguments"),
        (call(client, "own", "calculate", {"expression": 5}), 422, "tool_invalid_arguments"),
        (call(client, "probe", "secret", {}), 403, "tool_not_allowed"),
        (call(client, "nope", "echo", {}), 404, "tool_not_found"),
    ]
    for r, status, code in cases:
        assert (r.status_code, r.json()["error"]["code"]) == (status, code)

    # The tool ran and said no: for an agent that is information, so it is a 200.
    divided = call(client, "own", "calculate", {"expression": "1/0"})
    assert divided.status_code == 200
    assert divided.json()["is_error"] and divided.json()["error_kind"] == "tool_error"
    assert "divide by zero" in divided.json()["content"]

    client.patch("/api/mcp/servers/probe/enabled", json={"enabled": False}, headers=H)
    off = call(client, "probe", "echo", {"text": "a"})
    assert off.status_code == 409 and off.json()["error"]["code"] == "tool_disabled"


def test_changing_the_allowlist_changes_what_can_be_called(client):
    assert call(client, "own", "search_corpus", {"query": "x"}).status_code == 200
    client.patch(
        "/api/mcp/servers/own/allowlist", json={"tool_allowlist": ["calculate"]}, headers=H
    )
    assert call(client, "own", "search_corpus", {"query": "x"}).status_code == 403
    assert call(client, "own", "calculate", {"expression": "1+1"}).status_code == 200


# ----- for the model and for the UI ----------------------------------------------------------------------


def test_llm_tools_are_in_the_openai_function_format(client):
    tools = client.get("/api/mcp/llm-tools", headers=H).json()["tools"]
    assert [t["name"] for t in tools] == [
        "own__calculate",
        "own__search_corpus",
        "own__list_tables",
        "own__query_table",
    ]
    assert all(t["type"] == "function" and t["strict"] is False for t in tools)
    assert tools[0]["parameters"]["required"] == ["expression"]


def test_what_our_own_server_exposes(client):
    info = client.get("/api/mcp/exposed", headers=H).json()
    assert info["server_name"] == "agentic-blocks" and info["http_path"] == "/mcp"
    assert {t["name"] for t in info["tools"]} == {
        "calculate",
        "search_corpus",
        "list_tables",
        "query_table",
    }
    assert info["resources"] == ["corpus://sources"]
    assert info["resource_templates"] == ["corpus://sources/{collection}"]
    assert info["prompts"] == ["summarize_with_citations"]
    assert info["stdio_command"] == "python -m mcp_tools.server"


def test_status_lists_the_allow_lists_and_no_secrets(client):
    body = client.get("/api/mcp/status", headers=H).json()
    assert body == client.get("/status", headers=H).json()
    assert body["allowed_commands"] == ["python -m mcp_tools.server"]
    assert body["allowed_hosts"] == ["localhost", "127.0.0.1"] and body["servers"] == 1
    assert KEY not in str(body)
