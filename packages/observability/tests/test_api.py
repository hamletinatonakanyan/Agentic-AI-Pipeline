import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common import AppSettings, get_app_settings, install_error_handlers
from observability import init_observability, span
from observability.api import health_router, router, status_router
from observability.service import ObservabilityService
from observability.settings import ObsSettings

KEY = "test-key-0123456789"
H = {"X-API-Key": KEY}


def make_client(**settings) -> TestClient:
    handle = init_observability(
        ObsSettings(_env_file=None, local_db_path=":memory:", **settings), force=True
    )
    app = FastAPI()
    install_error_handlers(app)
    app.state.obs_service = ObservabilityService(handle)
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=KEY)
    app.include_router(health_router)
    app.include_router(status_router)
    app.include_router(router, prefix="/api/obs")
    return TestClient(app)


@pytest.fixture
def client():
    return make_client()


def test_health_is_public_and_others_need_key(client):
    assert client.get("/health").status_code == 200
    assert client.get("/status").status_code == 401
    assert client.get("/api/obs/traces").status_code == 401


def test_status(client):
    body = client.get("/status", headers=H).json()
    assert body["enabled"] is True
    assert body["store"] == "sqlite"
    assert body["exporters"] == ["local"]


def test_list_and_get_trace(client):
    with span("ingest.file"):
        with span("parse"):
            pass
    listing = client.get("/api/obs/traces", headers=H).json()
    assert [t["root_name"] for t in listing] == ["ingest.file"]
    trace_id = listing[0]["trace_id"]
    tree = client.get(f"/api/obs/traces/{trace_id}", headers=H).json()
    assert tree["roots"][0]["children"][0]["name"] == "parse"


def test_filters_through_query_string(client):
    with span("alpha"):
        pass
    with span("beta"):
        pass
    names = [t["root_name"] for t in client.get("/api/obs/traces?name=alp", headers=H).json()]
    assert names == ["alpha"]
    assert client.get("/api/obs/traces?status=error", headers=H).json() == []


def test_bad_query_values_are_422_envelope(client):
    r = client.get("/api/obs/traces?limit=0", headers=H)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    assert client.get("/api/obs/traces?status=weird", headers=H).status_code == 422


def test_missing_trace_is_404_envelope(client):
    r = client.get("/api/obs/traces/nope", headers=H)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "trace_not_found"


def test_disabled_returns_503_envelope():
    client = make_client(enabled=False)
    r = client.get("/api/obs/traces", headers=H)
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "observability_disabled"
    assert client.get("/status", headers=H).json()["enabled"] is False


def test_no_local_store_returns_503():
    client = make_client(exporters="console")
    r = client.get("/api/obs/traces", headers=H)
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "no_trace_store"


def test_traces_can_be_listed_by_conversation_through_the_api(client):
    for session, question in (("conv-1", "hello"), ("conv-2", "other"), ("conv-1", "again")):
        with span("agent.run", {"session.id": session, "input.value": question}):
            pass
    found = client.get("/api/obs/traces?session_id=conv-1", headers=H).json()
    assert [t["input_preview"] for t in found] == ["again", "hello"]
    assert {t["session_id"] for t in found} == {"conv-1"}
    assert client.get("/api/obs/traces?session_id=missing", headers=H).json() == []
    one = client.get(f"/api/obs/traces/{found[0]['trace_id']}", headers=H).json()
    assert one["summary"]["session_id"] == "conv-1" and one["summary"]["llm_calls"] == 0
