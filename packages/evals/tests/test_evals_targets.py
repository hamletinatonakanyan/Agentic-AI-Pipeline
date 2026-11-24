import json

import httpx
import pytest

from common import AppError
from evals.models import HttpTargetConfig
from evals.settings import EvalSettings
from evals.targets import (
    CallableTarget,
    TargetError,
    build_http_target,
    build_retrieval_http_target,
    describe,
)

KEY = "app-key-0123456789"


def settings(**kw) -> EvalSettings:
    return EvalSettings(_env_file=None, **kw)


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def http_config(url="http://localhost:9000/ask", **kw) -> HttpTargetConfig:
    return HttpTargetConfig(kind="http", url=url, **kw)


# ----- the allow-lists --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.com/ask",
        "http://169.254.169.254/latest",
        "http://localhost.evil.com/x",
    ],
)
def test_hosts_outside_the_allow_list_are_refused(url):
    with pytest.raises(AppError) as err:
        build_http_target(http_config(url), settings(), env={})
    assert err.value.code == "target_not_allowed" and err.value.status_code == 403
    assert "EVAL_ALLOWED_TARGET_HOSTS" in err.value.message


@pytest.mark.parametrize(
    "url", ["ftp://localhost/x", "file:///etc/passwd", "localhost:9000", "http://"]
)
def test_only_http_and_https_urls_are_accepted(url):
    with pytest.raises(AppError) as err:
        build_http_target(http_config(url), settings(), env={})
    assert err.value.code == "invalid_target"


def test_the_allow_list_can_be_extended_in_settings():
    s = settings(allowed_target_hosts="localhost, my-service.internal")
    assert s.allowed_target_hosts == ["localhost", "my-service.internal"]
    build_http_target(http_config("http://my-service.internal/ask"), s, env={})


def test_only_allow_listed_environment_variables_can_become_headers():
    cfg = http_config(headers_refs={"X-Leak": "OPENAI_API_KEY"})
    with pytest.raises(AppError) as err:
        build_http_target(cfg, settings(), env={"OPENAI_API_KEY": "sk-secret"})
    assert err.value.code == "target_not_allowed" and "sk-secret" not in err.value.message


def test_a_header_reference_to_an_unset_variable_is_an_error():
    with pytest.raises(AppError, match="APP_API_KEY is not set"):
        build_http_target(
            http_config(headers_refs={"X-API-Key": "APP_API_KEY"}), settings(), env={}
        )


def test_the_header_value_comes_from_the_environment_at_call_time():
    seen = {}

    def handler(request):
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"answer": "ok"})

    target = build_http_target(
        http_config(headers_refs={"X-API-Key": "APP_API_KEY"}),
        settings(), client=client(handler), env={"APP_API_KEY": KEY},
    )  # fmt: skip
    target.ask("q")
    assert seen["key"] == KEY


# ----- the HTTP contract ----------------------------------------------------------------------------


def test_an_http_target_posts_the_question_and_reads_the_contract_back():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "answer": "Forty-two.",
                "contexts": [{"chunk_id": "c1", "text": "the context"}],
                "retrieved_chunk_ids": ["c1", "c2"],
                "cited_chunk_ids": ["c1"],
                "timings": {"retrieval_ms": 12.5, "ttft_ms": 80, "total_ms": 300},
            },
        )

    target = build_http_target(http_config(), settings(), client=client(handler), env={})
    reply = target.ask("What is the answer?")
    assert seen["body"] == {"question": "What is the answer?"}
    assert reply.answer == "Forty-two." and reply.retrieved_chunk_ids == ["c1", "c2"]
    assert reply.contexts[0].text == "the context" and reply.timings["ttft_ms"] == 80


def test_missing_fields_stay_missing_so_those_metrics_are_not_computed():
    target = build_http_target(
        http_config(), settings(), client=client(lambda r: httpx.Response(200, json={})), env={}
    )
    reply = target.ask("q")
    assert reply.answer is None and reply.contexts is None and reply.retrieved_chunk_ids is None


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, text="not json"), "contract"),
        (httpx.Response(200, json={"contexts": "nope"}), "contract"),
        (httpx.ConnectError("refused"), "ConnectError"),
        (httpx.ReadTimeout("slow"), "did not answer"),
    ],
)
def test_target_failures_become_per_question_errors(response, message):
    def handler(request):
        if isinstance(response, Exception):
            raise response
        return response

    target = build_http_target(http_config(), settings(), client=client(handler), env={})
    with pytest.raises(TargetError, match=message):
        target.ask("q")


def test_callable_targets_turn_app_errors_into_target_errors():
    def fn(question):
        raise AppError("collection_not_found", "No collection 'x'.", 404)

    with pytest.raises(TargetError, match="collection_not_found"):
        CallableTarget(fn).ask("q")


def test_describe_never_includes_header_values():
    label = describe(
        http_config("http://localhost:9000/ask", headers_refs={"X-API-Key": "APP_API_KEY"})
    )
    assert label == "http localhost/ask" and "APP_API_KEY" not in label


# ----- the retrieval_http preset ------------------------------------------------------------------------


def retrieval_reply(request):
    return httpx.Response(
        200,
        json={
            "hits": [
                {"chunk_id": "c3", "text": "third", "rank": 1},
                {"chunk_id": "c1", "text": "first", "rank": 2},
            ],
            "timings": {"total_ms": 55.0, "retrieve_ms": 40.0},
        },
    )


def test_retrieval_http_maps_hits_to_the_target_contract_and_sends_the_key():
    seen = {}

    def handler(request):
        seen["body"], seen["key"] = json.loads(request.content), request.headers["x-api-key"]
        seen["url"] = str(request.url)
        return retrieval_reply(request)

    target = build_retrieval_http_target(
        {"mode": "dense", "top_k": 2, "rerank": True},
        settings(), client=client(handler), env={"APP_API_KEY": KEY},
    )  # fmt: skip
    reply = target.ask("why?")
    assert seen["url"] == "http://localhost:8002/api/retrieval/retrieve" and seen["key"] == KEY
    assert seen["body"] == {"query": "why?", "mode": "dense", "top_k": 2, "rerank": True}
    assert reply.retrieved_chunk_ids == ["c3", "c1"]
    assert [c.text for c in reply.contexts] == ["third", "first"]
    assert reply.timings == {"retrieval_ms": 55.0} and reply.answer is None


def test_retrieval_http_has_the_same_allow_list_and_rejects_unknown_options():
    with pytest.raises(AppError) as err:
        build_retrieval_http_target(
            {"url": "https://evil.example.com/x"}, settings(), env={"APP_API_KEY": KEY}
        )
    assert err.value.code == "target_not_allowed"
    with pytest.raises(AppError, match="Unknown retrieval options: temperature"):
        build_retrieval_http_target({"temperature": 1}, settings(), env={"APP_API_KEY": KEY})


def test_retrieval_http_reports_the_services_own_error_message():
    def handler(request):
        return httpx.Response(
            404, json={"error": {"code": "collection_not_found", "message": "No collection 'x'."}}
        )

    target = build_retrieval_http_target(
        {}, settings(), client=client(handler), env={"APP_API_KEY": KEY}
    )
    with pytest.raises(TargetError, match="No collection 'x'"):
        target.ask("q")


def test_retrieval_http_rejects_a_reply_that_is_not_retrieve_shaped():
    target = build_retrieval_http_target(
        {},
        settings(),
        client=client(lambda r: httpx.Response(200, json={"nope": 1})),
        env={"APP_API_KEY": KEY},
    )
    with pytest.raises(TargetError, match="does not look like"):
        target.ask("q")


# ----- settings ---------------------------------------------------------------------------------------------


def test_settings_parse_comma_separated_lists(monkeypatch):
    monkeypatch.setenv("EVAL_DEFAULT_KS", "1, 5,20")
    monkeypatch.setenv("EVAL_ALLOWED_HEADER_REFS", "APP_API_KEY,OTHER")
    s = EvalSettings(_env_file=None)
    assert s.default_ks == [1, 5, 20] and s.allowed_header_refs == ["APP_API_KEY", "OTHER"]
    assert s.allowed_target_hosts == ["localhost", "127.0.0.1"] and s.judge_model is None
