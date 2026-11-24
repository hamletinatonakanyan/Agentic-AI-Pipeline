import time

import pytest

from observability import init_observability, record_llm_call, span
from observability.models import TraceFilter
from observability.settings import ObsSettings
from observability.store import SqliteTraceStore


@pytest.fixture
def store():
    s = SqliteTraceStore(":memory:")
    init_observability(ObsSettings(_env_file=None), exporters=[s])
    return s


def make_trace(name="ingest.file", fail=False, tokens=None, sleep=0.0):
    try:
        with span(name):
            with span("parse"):
                time.sleep(sleep)
            with span("embed"):
                if tokens:
                    record_llm_call("m", *tokens)
                if fail:
                    raise RuntimeError("embed failed")
    except RuntimeError:
        pass


def test_list_traces_newest_first_with_summary(store):
    make_trace("first")
    time.sleep(0.01)
    make_trace("second", tokens=(10, 4))
    traces = store.list_traces(TraceFilter())
    assert [t.root_name for t in traces] == ["second", "first"]
    t = traces[0]
    assert t.span_count == 3
    assert (t.input_tokens, t.output_tokens) == (10, 4)
    assert t.status == "ok"


def test_filter_by_name_status_duration_and_time(store):
    make_trace("ingest.a")
    make_trace("retrieval.b", fail=True)
    make_trace("ingest.slow", sleep=0.05)

    assert {t.root_name for t in store.list_traces(TraceFilter(name="ingest"))} == {
        "ingest.a",
        "ingest.slow",
    }
    errors = store.list_traces(TraceFilter(status="error"))
    assert [t.root_name for t in errors] == ["retrieval.b"]
    assert len(store.list_traces(TraceFilter(status="ok"))) == 2
    slow = store.list_traces(TraceFilter(min_duration_ms=40))
    assert [t.root_name for t in slow] == ["ingest.slow"]
    assert store.list_traces(TraceFilter(since=time.time() + 100)) == []
    assert len(store.list_traces(TraceFilter(until=time.time() + 100))) == 3


def test_name_filter_treats_percent_and_underscore_literally(store):
    make_trace("a_b")
    make_trace("axb")
    assert [t.root_name for t in store.list_traces(TraceFilter(name="a_b"))] == ["a_b"]
    assert store.list_traces(TraceFilter(name="%")) == []


def test_limit_and_offset(store):
    for i in range(5):
        make_trace(f"t{i}")
        time.sleep(0.002)
    page = store.list_traces(TraceFilter(), limit=2, offset=2)
    assert [t.root_name for t in page] == ["t2", "t1"]


def test_get_trace_builds_the_tree(store):
    make_trace(tokens=(3, 2))
    trace_id = store.list_traces(TraceFilter())[0].trace_id
    tree = store.get_trace(trace_id)
    assert len(tree.roots) == 1
    root = tree.roots[0]
    assert root.name == "ingest.file"
    assert [c.name for c in root.children] == ["parse", "embed"]
    assert tree.summary.span_count == 3
    assert tree.summary.input_tokens == 3


def test_get_trace_marks_error_status(store):
    make_trace(fail=True)
    trace_id = store.list_traces(TraceFilter())[0].trace_id
    assert store.get_trace(trace_id).summary.status == "error"


def test_unknown_trace_is_none(store):
    assert store.get_trace("nope") is None


def test_span_with_missing_parent_becomes_a_root():
    s = SqliteTraceStore(":memory:")
    from observability.models import SpanData

    s.export(
        SpanData(
            trace_id="t",
            span_id="c",
            parent_id="gone",
            name="orphan",
            start_ts=1,
            end_ts=2,
            duration_ms=1000,
        )
    )
    assert [r.name for r in s.get_trace("t").roots] == ["orphan"]


def test_file_store_persists_across_reopen(tmp_path):
    path = str(tmp_path / "sub" / "obs.sqlite")
    first = SqliteTraceStore(path)
    init_observability(ObsSettings(_env_file=None), exporters=[first])
    make_trace("kept")
    first.shutdown()
    second = SqliteTraceStore(path)
    assert [t.root_name for t in second.list_traces(TraceFilter())] == ["kept"]


# ----- conversations, previews, calls and cost ---------------------------------------------------------


def run_in_conversation(session, question, cost_calls=()):
    with span("agent.run", {"session.id": session, "input.value": question}):
        for tokens, cost in cost_calls:
            with span("agent.generate"):
                record_llm_call("m", *tokens, cost_usd=cost)


def test_a_conversations_runs_can_be_found_by_its_id(store):
    run_in_conversation("conv-A", "first question")
    time.sleep(0.01)
    run_in_conversation("conv-B", "other conversation")
    time.sleep(0.01)
    run_in_conversation("conv-A", "second question")
    found = store.list_traces(TraceFilter(session_id="conv-A"))
    assert [t.input_preview for t in found] == ["second question", "first question"]
    assert {t.session_id for t in found} == {"conv-A"}
    assert store.list_traces(TraceFilter(session_id="nope")) == []
    assert len(store.list_traces(TraceFilter())) == 3


def test_summaries_count_model_calls_and_add_up_the_estimated_cost(store):
    run_in_conversation("c", "q", cost_calls=[((100, 20), 0.5), ((50, 10), 0.25)])
    (summary,) = store.list_traces(TraceFilter())
    assert (summary.llm_calls, summary.input_tokens, summary.output_tokens) == (2, 150, 30)
    assert summary.cost_usd == pytest.approx(0.75)
    tree = store.get_trace(summary.trace_id)
    assert tree.summary.cost_usd == pytest.approx(0.75) and tree.summary.llm_calls == 2
    assert tree.summary.session_id == "c" and tree.summary.input_preview == "q"


def test_no_prices_means_no_cost_not_zero_cost(store):
    run_in_conversation("c", "q", cost_calls=[((10, 2), None)])
    (summary,) = store.list_traces(TraceFilter())
    assert summary.cost_usd is None and summary.llm_calls == 1
    make_trace("plain")
    plain = next(t for t in store.list_traces(TraceFilter()) if t.root_name == "plain")
    assert plain.session_id is None and plain.input_preview == "" and plain.llm_calls == 0


def test_a_long_input_is_shortened_in_the_list_and_whitespace_is_collapsed(store):
    run_in_conversation("c", "word   " * 200)
    (summary,) = store.list_traces(TraceFilter())
    assert len(summary.input_preview) <= 140 and summary.input_preview.endswith("…")
    assert "  " not in summary.input_preview
