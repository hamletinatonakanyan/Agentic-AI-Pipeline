from fastapi.testclient import TestClient

from common import AppSettings, get_app_settings
from observability import span
from observability.app import create_app
from observability.settings import ObsSettings

KEY = "test-key-0123456789"


def make_client() -> TestClient:
    app = create_app(ObsSettings(_env_file=None, local_db_path=":memory:"))
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=KEY)
    return TestClient(app, follow_redirects=False)


def test_standalone_paths_and_auth():
    client = make_client()
    assert client.get("/health").status_code == 200
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"X-API-Key": KEY}).json()["store"] == "sqlite"
    assert client.get("/api/obs/status", headers={"X-API-Key": KEY}).json()["store"] == "sqlite"
    assert client.get("/api/obs/status").status_code == 401
    assert client.get("/api/obs/traces", headers={"X-API-Key": KEY}).json() == []


def test_root_redirects_to_ui_and_ui_is_public():
    client = make_client()
    assert client.get("/").headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200
    assert "Observability" in page.text
    for path in ("/ui/app.js", "/ui/kit/kit.js", "/ui/kit/kit.css"):
        assert client.get(path).status_code == 200


def test_spans_created_while_the_app_runs_are_listed():
    client = make_client()
    with span("standalone.demo"):
        pass
    names = [
        t["root_name"] for t in client.get("/api/obs/traces", headers={"X-API-Key": KEY}).json()
    ]
    assert names == ["standalone.demo"]
