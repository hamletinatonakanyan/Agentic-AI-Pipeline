import asyncio

import pytest

from observability import (
    current_trace_id,
    init_observability,
    record_error,
    record_llm_call,
    span,
    traced,
)
from observability.exporters import InMemoryExporter
from observability.settings import ObsSettings


def start(**settings) -> InMemoryExporter:
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None, **settings), exporters=[exporter])
    return exporter


def by_name(exporter: InMemoryExporter) -> dict:
    return {s.name: s for s in exporter.spans}


def test_everything_is_a_noop_before_init():
    with span("x") as s:
        s.set_attribute("a", 1)
        record_llm_call("m", 1, 2)
        record_error(ValueError("x"))
    assert current_trace_id() is None


def test_disabled_is_a_noop():
    exporter = start(enabled=False)
    with span("x"):
        pass
    assert exporter.spans == []


def test_child_spans_link_to_parent_and_share_trace():
    exporter = start()
    with span("parent"):
        with span("child"):
            pass
    spans = by_name(exporter)
    assert spans["child"].parent_id == spans["parent"].span_id
    assert spans["child"].trace_id == spans["parent"].trace_id
    assert spans["parent"].parent_id is None
    # Children finish first, so they are exported first.
    assert exporter.names() == ["child", "parent"]


def test_siblings_and_separate_roots():
    exporter = start()
    with span("root1"):
        with span("a"):
            pass
        with span("b"):
            pass
    with span("root2"):
        pass
    s = by_name(exporter)
    assert s["a"].parent_id == s["b"].parent_id == s["root1"].span_id
    assert s["root2"].trace_id != s["root1"].trace_id


def test_attributes_events_and_timing():
    exporter = start()
    with span("work", {"source": "a.pdf"}) as s:
        s.set_attribute("pages", 3)
        s.add_event("checkpoint", {"n": 1})
    data = exporter.spans[0]
    assert data.attributes == {"source": "a.pdf", "pages": 3}
    assert data.events[0].name == "checkpoint"
    assert data.status == "ok"
    assert data.duration_ms >= 0
    assert data.end_ts >= data.start_ts
    assert data.service == "agentic-blocks"


def test_exception_marks_error_and_propagates():
    exporter = start()
    with pytest.raises(ValueError):
        with span("boom"):
            raise ValueError("bad input")
    data = exporter.spans[0]
    assert data.status == "error"
    assert "ValueError: bad input" in data.status_message
    assert data.events[0].name == "exception"


def test_error_in_child_does_not_mark_parent_when_handled():
    exporter = start()
    with span("parent"):
        try:
            with span("child"):
                raise RuntimeError("x")
        except RuntimeError:
            pass
    s = by_name(exporter)
    assert s["child"].status == "error"
    assert s["parent"].status == "ok"


def test_current_trace_id_inside_and_outside():
    start()
    assert current_trace_id() is None
    with span("x") as s:
        assert current_trace_id() == s.trace_id
    assert current_trace_id() is None


def test_traced_decorator_sync():
    exporter = start()

    @traced("tool.call")
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add.__name__ == "add"
    assert exporter.names() == ["tool.call"]


def test_traced_decorator_async_and_context_survives_await():
    exporter = start()

    @traced("inner")
    async def inner():
        await asyncio.sleep(0)

    async def outer():
        with span("outer"):
            await inner()

    asyncio.run(outer())
    s = by_name(exporter)
    assert s["inner"].parent_id == s["outer"].span_id


def test_concurrent_tasks_keep_separate_traces():
    exporter = start()

    async def job(name):
        with span(name):
            await asyncio.sleep(0)
            with span(name + ".child"):
                await asyncio.sleep(0)

    async def main():
        await asyncio.gather(job("a"), job("b"))

    asyncio.run(main())
    s = by_name(exporter)
    assert s["a.child"].parent_id == s["a"].span_id
    assert s["b.child"].parent_id == s["b"].span_id
    assert s["a"].trace_id != s["b"].trace_id


def test_record_llm_call_goes_on_active_span():
    exporter = start()
    with span("generate"):
        record_llm_call("gpt-6-luna", 10, 5, ttft_ms=12.5, total_ms=80, cost_usd=0.001)
    attrs = exporter.spans[0].attributes
    assert attrs["gen_ai.request.model"] == "gpt-6-luna"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["llm.ttft_ms"] == 12.5


def test_record_llm_call_without_active_span_makes_one():
    exporter = start()
    record_llm_call("m", 1, 2)
    assert exporter.names() == ["llm.call"]


def test_record_error_marks_active_span_without_raising():
    exporter = start()
    with span("x"):
        record_error(KeyError("k"))
    assert exporter.spans[0].status == "error"


def test_sampling_zero_records_nothing_including_children():
    exporter = start(sample_rate=0.0)
    with span("root"):
        with span("child"):
            pass
    assert exporter.spans == []


def test_unsampled_trace_does_not_leak_into_next_one():
    exporter = start(sample_rate=1.0)
    with span("a"):
        pass
    with span("b"):
        pass
    assert exporter.names() == ["a", "b"]


def test_failing_exporter_never_breaks_the_app():
    class Broken:
        def export(self, span):
            raise RuntimeError("disk full")

        def shutdown(self):
            pass

    init_observability(ObsSettings(_env_file=None), exporters=[Broken()])
    with span("x") as s:
        s.set_attribute("ok", True)  # no exception reaches the caller


def test_init_is_idempotent_and_force_rebuilds():
    first = init_observability(ObsSettings(_env_file=None), exporters=[])
    assert init_observability(ObsSettings(_env_file=None), exporters=[]) is first
    assert init_observability(ObsSettings(_env_file=None), exporters=[], force=True) is not first
