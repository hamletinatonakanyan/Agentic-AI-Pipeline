import json

import httpx
import pytest

from common import AppError
from llm import LLMSettings, OpenAILLM
from observability import init_observability, shutdown_observability
from observability.exporters import InMemoryExporter
from observability.settings import ObsSettings

KEY = "sk-test-key-0123456789abcdef"


def sse(*events) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def stream_llm(body: bytes, status=200) -> OpenAILLM:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, content=body))
    return OpenAILLM(
        LLMSettings(_env_file=None, openai_api_key=KEY), client=httpx.Client(transport=transport)
    )


@pytest.fixture(autouse=True)
def _obs_reset():
    shutdown_observability()
    yield
    shutdown_observability()


def delta(text, item="msg_1"):
    return {"type": "response.output_text.delta", "item_id": item, "delta": text}


def test_text_pieces_arrive_in_order_with_timings_and_usage():
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    body = sse(
        {"type": "response.created", "response": {}},
        delta("Hel"), delta("lo "), delta("world"),
        {"type": "response.completed", "response": {"usage": {"input_tokens": 5, "output_tokens": 3}}},
    )  # fmt: skip
    handle = stream_llm(body).stream("q")
    assert handle.ttft_ms is None
    assert "".join(handle) == "Hello world"
    assert handle.ttft_ms is not None and handle.total_ms >= handle.ttft_ms
    span = next(s for s in exporter.spans if s.name == "llm.stream")
    assert span.attributes["gen_ai.usage.input_tokens"] == 5
    assert span.attributes["llm.ttft_ms"] == handle.ttft_ms


def test_the_stream_request_asks_for_streaming_and_sends_no_sampling_params():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(delta("x")))

    llm = OpenAILLM(
        LLMSettings(_env_file=None, openai_api_key=KEY),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    list(llm.stream("q"))
    assert seen["body"]["stream"] is True and "temperature" not in seen["body"]


def test_commentary_items_are_not_streamed():
    body = sse(
        {"type": "response.output_item.added", "item": {"id": "msg_n", "type": "message", "phase": "commentary"}},
        {"type": "response.output_item.added", "item": {"id": "msg_a", "type": "message", "phase": "final_answer"}},
        delta("working notes", "msg_n"), delta("The answer.", "msg_a"),
    )  # fmt: skip
    assert "".join(stream_llm(body).stream("q")) == "The answer."


def test_failure_events_raise_a_clear_error():
    body = sse(
        delta("part"), {"type": "response.failed", "response": {"error": {"message": "overloaded"}}}
    )
    with pytest.raises(AppError, match="overloaded") as err:
        list(stream_llm(body).stream("q"))
    assert err.value.code == "llm_failed"


def test_http_errors_before_the_stream_starts_are_mapped():
    with pytest.raises(AppError) as err:
        list(stream_llm(b'{"error": {"message": "bad key"}}', status=401).stream("q"))
    assert err.value.code == "llm_auth"


def test_stream_needs_a_key():
    llm = OpenAILLM(LLMSettings(_env_file=None, openai_api_key=""))
    with pytest.raises(AppError) as err:
        list(llm.stream("q"))
    assert err.value.code == "llm_not_configured"


def test_non_data_lines_and_done_markers_are_ignored():
    body = b": keep-alive\n\nevent: x\ndata: [DONE]\n\n" + sse(delta("ok"))
    assert "".join(stream_llm(body).stream("q")) == "ok"


def test_a_streamed_call_records_its_estimated_cost_too():
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    body = sse(
        delta("Hi"),
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 1000, "output_tokens": 500}},
        },
    )
    llm = OpenAILLM(
        LLMSettings(
            _env_file=None, openai_api_key=KEY, price_input_per_m=1000.0, price_output_per_m=2000.0
        ),
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))
        ),
    )
    list(llm.stream("q"))
    span = next(s for s in exporter.spans if s.name == "llm.stream")
    assert span.attributes["llm.cost_usd"] == pytest.approx(1.0 + 1.0)
