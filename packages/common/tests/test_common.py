import json
import logging

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from common import (
    AppError,
    AppSettings,
    get_app_settings,
    install_error_handlers,
    mask_secret,
    mount_ui_kit,
    require_api_key,
    setup_logging,
)
from common.logging import JsonFormatter


def make_app(key: str) -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=key)

    @app.get("/health")
    def health():
        return {"ok": True}

    protected = APIRouter(dependencies=[Depends(require_api_key)])

    @protected.get("/secret")
    def secret():
        return {"data": 1}

    @protected.get("/boom")
    def boom():
        raise AppError("custom", "went wrong", status_code=418, details={"x": 1})

    @protected.get("/items/{n}")
    def item(n: int):
        return {"n": n}

    app.include_router(protected)
    mount_ui_kit(app)
    return app


def test_health_needs_no_key():
    client = TestClient(make_app("k" * 16))
    assert client.get("/health").status_code == 200


def test_missing_and_wrong_key_are_401_with_envelope():
    client = TestClient(make_app("k" * 16))
    for headers in ({}, {"X-API-Key": "wrong"}):
        r = client.get("/secret", headers=headers)
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "unauthorized"


def test_right_key_passes():
    client = TestClient(make_app("k" * 16))
    assert client.get("/secret", headers={"X-API-Key": "k" * 16}).json() == {"data": 1}


def test_unset_key_refuses_everything_with_503():
    client = TestClient(make_app(""))
    r = client.get("/secret", headers={"X-API-Key": ""})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "not_configured"


def test_app_error_uses_envelope():
    client = TestClient(make_app("k"))
    r = client.get("/boom", headers={"X-API-Key": "k"})
    assert r.status_code == 418
    assert r.json() == {"error": {"code": "custom", "message": "went wrong", "details": {"x": 1}}}


def test_validation_error_uses_envelope_and_hides_input():
    client = TestClient(make_app("k"))
    r = client.get("/items/not-a-number", headers={"X-API-Key": "k"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"
    assert "not-a-number" not in json.dumps(body)


def test_ui_kit_files_are_public():
    client = TestClient(make_app("k" * 16))
    assert client.get("/ui/kit/kit.js").status_code == 200
    assert client.get("/ui/kit/kit.css").status_code == 200


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "not set"),
        (None, "not set"),
        ("short", "set (hidden)"),
        ("a" * 20 + "WXYZ", "set (...WXYZ)"),
    ],
)
def test_mask_secret_never_returns_full_value(value, expected):
    assert mask_secret(value) == expected


def test_settings_read_prefix_from_env(monkeypatch):
    from pydantic_settings import SettingsConfigDict

    from common import BaseAppSettings

    class Demo(BaseAppSettings):
        model_config = SettingsConfigDict(env_prefix="DEMO_", env_file=None)
        threshold: int = 50

    monkeypatch.setenv("DEMO_THRESHOLD", "7")
    assert Demo().threshold == 7


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("x", logging.INFO, "f", 1, "hello %s", ("w",), None)
    record.trace_id = "abc"
    out = json.loads(JsonFormatter().format(record))
    assert out["message"] == "hello w"
    assert out["trace_id"] == "abc"


def test_setup_logging_is_idempotent():
    root = logging.getLogger()
    setup_logging()
    count = len(root.handlers)
    setup_logging()
    assert len(root.handlers) == count


def test_empty_env_value_means_use_the_default(monkeypatch):
    from pydantic_settings import SettingsConfigDict

    from common import BaseAppSettings

    class Demo(BaseAppSettings):
        model_config = SettingsConfigDict(env_prefix="DEMO_", env_file=None)
        size: int = 512

    monkeypatch.setenv("DEMO_SIZE", "")
    assert Demo().size == 512
