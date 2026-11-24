import json

import httpx
import pytest
from pydantic import BaseModel

from common import AppError
from llm import LLMSettings, OpenAILLM, build_request, llm_status, strict_schema
from llm.settings import FORBIDDEN_PARAMS
from llm.testing import FakeLLM
from observability import init_observability, shutdown_observability, span
from observability.exporters import InMemoryExporter
from observability.settings import ObsSettings

KEY = "sk-test-key-0123456789abcdef"


def settings(**kw) -> LLMSettings:
    return LLMSettings(_env_file=None, openai_api_key=KEY, **kw)


def message(text, phase="final_answer", **extra):
    return {
        "type": "message", "role": "assistant", "phase": phase, "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}], **extra,
    }  # fmt: skip


def reply(*items, usage=None, **top):
    body = {
        "status": "completed", "error": None, "incomplete_details": None, "output": list(items),
        "usage": usage or {"input_tokens": 11, "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 3}},
    }  # fmt: skip
    return {**body, **top}


class Server:
    """A fake OpenAI. `answers` are returned in order: a dict (200 JSON) or an httpx.Response."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, httpx.Response) else httpx.Response(200, json=answer)

    def body(self, i=0) -> dict:
        return json.loads(self.requests[i].content)


def llm_with(server: Server, sleeps=None, **kw) -> OpenAILLM:
    return OpenAILLM(
        settings(**kw),
        client=httpx.Client(transport=httpx.MockTransport(server)),
        sleep=(sleeps.append if sleeps is not None else (lambda _: None)),
    )


@pytest.fixture(autouse=True)
def _obs_reset():
    shutdown_observability()
    yield
    shutdown_observability()


# ----- the reasoning rule ----------------------------------------------------------


def test_reasoning_on_sends_low_and_off_sends_none():
    assert build_request(settings(use_reasoning=True), "q")["reasoning"] == {"effort": "low"}
    assert build_request(settings(use_reasoning=False), "q")["reasoning"] == {"effort": "none"}
    assert settings().use_reasoning is True  # default: on


@pytest.mark.parametrize("use_reasoning", [True, False])
def test_no_request_ever_contains_sampling_parameters(use_reasoning):
    class Out(BaseModel):
        answer: str

    s = settings(use_reasoning=use_reasoning, max_output_tokens=50)
    for body in (
        build_request(s, "q"),
        build_request(s, "q", system="sys"),
        build_request(s, "q", schema=Out),
        build_request(s, "q", stream=True),
    ):
        assert not set(body) & set(FORBIDDEN_PARAMS)
        assert "temperature" not in json.dumps(body)


def test_request_shape():
    body = build_request(settings(), "the question", system="be brief", max_output_tokens=99)
    assert body["model"] == "gpt-6-luna"
    assert body["input"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "the question"},
    ]
    assert body["store"] is False and body["max_output_tokens"] == 99
    assert "stream" not in body and "text" not in body


def test_what_goes_over_the_wire_matches(tmp_path):
    server = Server(reply(message("hi")))
    llm_with(server).complete("hello", system="s")
    request = server.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert server.body()["input"][-1] == {"role": "user", "content": "hello"}


# ----- strict structured output ------------------------------------------------------------------


class Claim(BaseModel):
    text: str
    supported: bool = True
    note: str | None = None


class Verdict(BaseModel):
    claims: list[Claim]
    score: float
    label: str = "ok"


def test_strict_schema_meets_openai_rules():
    schema = strict_schema(Verdict)

    def check(node):
        if isinstance(node, dict):
            assert "default" not in node
            if node.get("type") == "object" or "properties" in node:
                assert node["additionalProperties"] is False
                assert node["required"] == list(node["properties"])
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)

    check(schema)
    assert schema["required"] == ["claims", "score", "label"]
    assert schema["$defs"]["Claim"]["required"] == [
        "text",
        "supported",
        "note",
    ]  # optional stays listed


def test_a_property_called_title_survives():
    class Doc(BaseModel):
        title: str
        default: str

    schema = strict_schema(Doc)
    assert set(schema["properties"]) == {"title", "default"}
    assert schema["required"] == ["title", "default"]


def test_free_form_dicts_are_refused_with_a_clear_message():
    class Bad(BaseModel):
        data: dict[str, str]

    with pytest.raises(ValueError, match="free-form"):
        strict_schema(Bad)


def test_complete_json_sends_the_schema_and_returns_a_validated_object():
    payload = {
        "claims": [{"text": "a", "supported": True, "note": None}],
        "score": 0.5,
        "label": "x",
    }
    server = Server(reply(message(json.dumps(payload))))
    out = llm_with(server).complete_json("judge this", Verdict)
    assert isinstance(out, Verdict) and out.claims[0].text == "a" and out.score == 0.5
    fmt = server.body()["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"] == "Verdict"
    assert fmt["schema"]["additionalProperties"] is False


def test_the_extra_commentary_message_is_ignored():
    """gpt-6-luna sometimes returns working notes as a second message item."""
    good = json.dumps({"claims": [], "score": 1.0, "label": "ok"})
    server = Server(
        reply(message("We need output JSON. classify each...", phase="commentary"), message(good))
    )
    assert llm_with(server).complete_json("x", Verdict).score == 1.0
    # And the other order, and with no phase information at all (fall back to the last message).
    server = Server(reply(message(good), message("some notes", phase="commentary")))
    assert llm_with(server).complete_json("x", Verdict).score == 1.0
    server = Server(reply(message("notes", phase=None), message(good, phase=None)))
    assert llm_with(server).complete_json("x", Verdict).score == 1.0
    assert (
        llm_with(Server(reply(message("notes", phase="commentary"), message("answer")))).complete(
            "q"
        )
        == "answer"
    )


def test_unparseable_json_is_retried_once_then_fails_clearly():
    good = json.dumps({"claims": [], "score": 0.1, "label": "ok"})
    server = Server(reply(message("not json at all")), reply(message(good)))
    assert llm_with(server).complete_json("x", Verdict).score == 0.1
    assert len(server.requests) == 2

    server = Server(reply(message('{"score": "high"}')))
    with pytest.raises(AppError) as err:
        llm_with(server).complete_json("x", Verdict)
    assert err.value.code == "llm_invalid_output" and len(server.requests) == 2


# ----- reading answers -----------------------------------------------------------------------------


def test_text_comes_from_the_message_item():
    out = llm_with(
        Server(reply({"type": "reasoning", "summary": []}, message("The answer.")))
    ).complete("q")
    assert out == "The answer."


def test_a_refusal_an_incomplete_answer_and_an_empty_answer_are_distinct_errors():
    refusal = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": "I can't"}],
    }
    cases = [
        (reply(refusal), "llm_refused"),
        (reply(message("partial"), status="incomplete", incomplete_details={"reason": "max_output_tokens"}), "llm_incomplete"),
        (reply(), "llm_empty"),
        (reply(message("   ")), "llm_empty"),
        (reply(message("x"), status="failed", error={"message": "boom"}), "llm_failed"),
    ]  # fmt: skip
    for answer, code in cases:
        with pytest.raises(AppError) as err:
            llm_with(Server(answer)).complete("q")
        assert err.value.code == code, code


# ----- errors and retries ------------------------------------------------------------------------------


def test_a_missing_key_fails_before_any_request():
    server = Server(reply(message("x")))
    llm = OpenAILLM(
        LLMSettings(_env_file=None, openai_api_key=""),
        client=httpx.Client(transport=httpx.MockTransport(server)),
    )
    with pytest.raises(AppError) as err:
        llm.complete("q")
    assert err.value.code == "llm_not_configured" and err.value.status_code == 503
    assert server.requests == []


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "llm_auth"), (403, "llm_auth"), (404, "llm_model_not_found"), (400, "llm_bad_request")],
)
def test_client_errors_are_not_retried_and_have_clear_codes(status, code):
    server = Server(httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(AppError) as err:
        llm_with(server).complete("q")
    assert err.value.code == code and len(server.requests) == 1


def test_rate_limits_are_retried_honouring_retry_after():
    sleeps: list[float] = []
    server = Server(
        httpx.Response(429, headers={"retry-after": "7"}, json={}),
        httpx.Response(503, json={}),
        reply(message("finally")),
    )
    assert llm_with(server, sleeps).complete("q") == "finally"
    assert sleeps == [7.0, 2.0]  # Retry-After first, then exponential backoff


def test_retries_run_out():
    server = Server(httpx.Response(429, json={"error": {"message": "slow down"}}))
    with pytest.raises(AppError) as err:
        llm_with(server, max_retries=2).complete("q")
    assert err.value.code == "llm_rate_limited" and len(server.requests) == 3


def test_server_errors_and_network_failures_map_to_clear_errors():
    with pytest.raises(AppError) as err:
        llm_with(Server(httpx.Response(500, json={})), max_retries=0).complete("q")
    assert err.value.code == "llm_unavailable"
    with pytest.raises(AppError) as err:
        llm_with(Server(httpx.ConnectError("no route")), max_retries=0).complete("q")
    assert err.value.code == "llm_unavailable" and "ConnectError" in err.value.message
    with pytest.raises(AppError) as err:
        llm_with(Server(httpx.ReadTimeout("slow")), max_retries=0).complete("q")
    assert err.value.code == "llm_timeout" and err.value.status_code == 504


def test_a_network_blip_is_retried():
    server = Server(httpx.ConnectError("blip"), reply(message("ok")))
    assert llm_with(server).complete("q") == "ok"


def test_keys_echoed_by_the_api_are_never_passed_on():
    leak = "Incorrect API key provided: sk-proj-abcdef123456ZZZZ. You can find it..."
    server = Server(httpx.Response(400, json={"error": {"message": leak}}))
    with pytest.raises(AppError) as err:
        llm_with(server).complete("q")
    assert "sk-proj" not in err.value.message and "[REDACTED]" in err.value.message


# ----- tracing ---------------------------------------------------------------------------------------------


def test_each_call_is_a_span_with_token_counts():
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    with span("outer"):
        llm_with(Server(reply(message("x")))).complete("q")
    call = next(s for s in exporter.spans if s.name == "llm.complete")
    assert call.parent_id == next(s for s in exporter.spans if s.name == "outer").span_id
    assert call.attributes["gen_ai.request.model"] == "gpt-6-luna"
    assert call.attributes["gen_ai.usage.input_tokens"] == 11
    assert call.attributes["gen_ai.usage.output_tokens"] == 7
    assert call.attributes["llm.reasoning_tokens"] == 3
    assert call.attributes["llm.reasoning_effort"] == "low"
    assert KEY not in json.dumps(call.model_dump(), default=str)


def test_works_with_observability_off():
    assert llm_with(Server(reply(message("fine")))).complete("q") == "fine"


# ----- status ------------------------------------------------------------------------------------------------


def test_status_never_contains_the_key():
    st = llm_status(settings(use_reasoning=False))
    assert (st.model, st.reasoning_effort, st.configured) == ("gpt-6-luna", "none", True)
    assert KEY not in st.model_dump_json()
    assert llm_status(LLMSettings(_env_file=None, openai_api_key="")).configured is False
    assert KEY not in repr(settings())


def test_settings_read_the_shared_key_name(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-0000000000")
    monkeypatch.setenv("LLM_USE_REASONING", "false")
    s = LLMSettings(_env_file=None)
    assert s.openai_api_key == "sk-from-env-0000000000" and s.effort == "none"


# ----- the fake ---------------------------------------------------------------------------------------------------


def test_fake_llm_scripts_answers_and_records_calls():
    fake = FakeLLM(["one", {"claims": [], "score": 0.3, "label": "x"}])
    assert fake.complete("a", system="s") == "one"
    verdict = fake.complete_json("b", Verdict)
    assert verdict.score == 0.3
    assert [(c.kind, c.prompt, c.system) for c in fake.calls] == [
        ("complete", "a", "s"),
        ("complete_json", "b", None),
    ]
    with pytest.raises(AssertionError, match="ran out"):
        fake.complete("c")


def test_fake_llm_function_and_stream():
    fake = FakeLLM(lambda prompt: prompt.upper())
    assert fake.complete("hi") == "HI"
    stream = FakeLLM("one two three").stream("q")
    assert list(stream) == ["one", "two", "three"] and stream.ttft_ms and stream.total_ms


# ----- the cost estimate -------------------------------------------------------------------------------------


def test_cost_is_estimated_from_configured_prices_and_never_from_guesses():
    s = settings(price_input_per_m=2.0, price_output_per_m=10.0)
    assert s.estimate_cost(1_000_000, 100_000) == pytest.approx(3.0)
    assert settings(price_input_per_m=2.0).estimate_cost(500_000, 999) == pytest.approx(1.0)
    assert settings().estimate_cost(1000, 1000) is None  # no prices, no estimate


def test_a_call_records_its_estimated_cost_on_its_span_only_when_prices_are_set():
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    llm_with_prices = OpenAILLM(
        settings(price_input_per_m=1_000_000.0, price_output_per_m=2_000_000.0),
        client=httpx.Client(transport=httpx.MockTransport(Server(reply(message("x"))))),
    )
    llm_with_prices.complete("q")  # 11 input tokens, 7 output tokens
    llm_with(Server(reply(message("x")))).complete("q")
    calls = [s for s in exporter.spans if s.name == "llm.complete"]
    assert calls[0].attributes["llm.cost_usd"] == pytest.approx(11 + 14)
    assert "llm.cost_usd" not in calls[1].attributes
