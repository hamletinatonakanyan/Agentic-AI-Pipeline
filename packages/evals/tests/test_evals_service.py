import json
import re

import pytest

from common import AppError
from evals.models import (
    Context,
    EvalConfig,
    GenerateRequest,
    ItemPatch,
    PresetTargetConfig,
    RunRequest,
    TargetResponse,
)
from evals.targets import TargetError
from observability import init_observability, shutdown_observability
from observability.exporters import InMemoryExporter
from observability.settings import ObsSettings


@pytest.fixture(autouse=True)
def _obs_reset():
    shutdown_observability()
    yield
    shutdown_observability()


def make_dataset(rig, make_jsonl, *items, name="ds"):
    data = make_jsonl(*items)
    return rig.service.upload_dataset(name, data)


def run_request(dataset_id, target="ids_only", options=None, **config) -> RunRequest:
    return RunRequest(
        dataset_id=dataset_id,
        target=PresetTargetConfig(kind="preset", name=target, options=options or {}),
        config=EvalConfig(**config),
    )


def answer(text="An answer.", retrieved=("c1",), cited=None, contexts=True, **timings):
    return TargetResponse(
        answer=text,
        retrieved_chunk_ids=list(retrieved),
        cited_chunk_ids=list(cited) if cited is not None else None,
        contexts=[Context(chunk_id=c, text=f"text of {c}") for c in retrieved]
        if contexts
        else None,
        timings=timings,
    )


def ids_only(*retrieved, **timings):
    return TargetResponse(retrieved_chunk_ids=list(retrieved), timings=timings)


# ----- datasets ----------------------------------------------------------------------------------


def test_upload_accepts_jsonl_and_marks_items_accepted(rig, make_jsonl):
    ds = make_dataset(
        rig,
        make_jsonl,
        {"id": "a", "question": "Q1?", "relevant_chunk_ids": ["c1"], "reference_answer": "A1"},
        {"question": "Q2?"},
    )
    assert ds.status == "ready" and ds.counts == {"draft": 0, "accepted": 2, "rejected": 0}
    full = rig.service.get_dataset(ds.id)
    assert [i.id for i in full.items][0] == "a" and len(full.items[1].id) == 10  # one was made up
    assert full.items[1].relevant_chunk_ids == [] and full.items[0].status == "accepted"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "no items"),
        (b"\n\n", "no items"),
        (b'{"question": "ok"}\nnot json\n{"question": ""}\n', "2 line(s)"),
        (b'["a list"]', "1 line(s)"),
        (b'{"id": "x", "question": "a"}\n{"id": "x", "question": "b"}', "same id"),
        (b"\xff\xfe\x00bad", "UTF-8"),
    ],
)
def test_bad_uploads_are_422(rig, data, message):
    with pytest.raises(AppError, match=re.escape(message)) as err:
        rig.service.upload_dataset("x", data)
    assert err.value.status_code == 422 and err.value.code == "invalid_dataset"


def test_upload_errors_name_the_bad_lines(rig):
    with pytest.raises(AppError) as err:
        rig.service.upload_dataset("x", b'{"question": "ok"}\nnot json\n{"question": ""}\n')
    assert set(err.value.details["lines"]) == {2, 3}


def test_review_flow_draft_to_ready(rig):
    job = rig.service.generate_dataset(GenerateRequest(collection="docs", n=3))
    dataset_id = job.result["dataset_id"]
    ds = rig.service.get_dataset(dataset_id)
    assert ds.status == "draft" and ds.counts["draft"] == 3

    first, second, third = ds.items
    rig.service.patch_item(dataset_id, first.id, ItemPatch(status="accepted"))
    rig.service.patch_item(dataset_id, second.id, ItemPatch(status="rejected"))
    edited = rig.service.patch_item(
        dataset_id, third.id,
        ItemPatch(status="accepted", question="A better question?", reference_answer="Better."),
    )  # fmt: skip
    assert edited.question == "A better question?" and edited.evidence  # evidence is kept
    after = rig.service.get_dataset(dataset_id)
    assert after.status == "ready" and after.counts == {"draft": 0, "accepted": 2, "rejected": 1}


def test_patching_unknown_things_is_404(rig, make_jsonl):
    ds = make_dataset(rig, make_jsonl, {"id": "a", "question": "Q?"})
    with pytest.raises(AppError) as err:
        rig.service.patch_item(ds.id, "nope", ItemPatch(status="rejected"))
    assert err.value.code == "item_not_found"
    with pytest.raises(AppError) as err:
        rig.service.get_dataset("nope")
    assert err.value.code == "dataset_not_found"


def test_delete_dataset(rig, make_jsonl):
    ds = make_dataset(rig, make_jsonl, {"question": "Q?"})
    rig.service.delete_dataset(ds.id)
    assert rig.service.list_datasets() == []
    with pytest.raises(AppError):
        rig.service.delete_dataset(ds.id)


# ----- question generation ------------------------------------------------------------------------------


def test_generation_is_a_job_that_drafts_one_question_per_chunk(rig):
    job = rig.service.generate_dataset(GenerateRequest(collection="docs", n=4, seed=7, name="mine"))
    assert job.state == "completed" and job.progress == 4
    assert job.result["drafted"] == 4 and job.result["failed"] == 0
    ds = rig.service.get_dataset(job.result["dataset_id"])
    assert ds.name == "mine" and ds.collection == "docs"
    assert [i.relevant_chunk_ids for i in ds.items] == [["c1"], ["c2"], ["c3"], ["c4"]]
    assert all(i.status == "draft" and i.evidence[0].text for i in ds.items)
    assert rig.chunks.calls == [("docs", 4, 7, 200)]


def test_generation_skips_failed_drafts_and_reports_them(rig):
    calls = iter(range(100))

    def flaky(prompt):
        if next(calls) % 2 == 0:  # every other chunk: the model gives unusable JSON twice
            raise AppError("llm_invalid_output", "bad json", 502)
        return '{"question": "Q?", "answer": "A."}'

    rig.llm._script = flaky
    job = rig.service.generate_dataset(GenerateRequest(collection="docs", n=4))
    assert job.state == "completed"
    assert (job.result["drafted"], job.result["failed"]) == (2, 2)
    assert len(rig.service.get_dataset(job.result["dataset_id"]).items) == 2


def test_generation_stops_when_the_llm_is_not_configured(rig):
    def boom(prompt):
        raise AppError("llm_not_configured", "OPENAI_API_KEY is not set.", 503)

    rig.service._gen._llm._script = boom
    job = rig.service.generate_dataset(GenerateRequest(collection="docs", n=3))
    assert job.state == "failed" and "llm_not_configured" in job.error
    assert rig.service.list_datasets() == []  # nothing half-made


def test_generation_limits(rig_factory):
    rig = rig_factory(max_generate=5)
    with pytest.raises(AppError, match="EVAL_MAX_GENERATE"):
        rig.service.generate_dataset(GenerateRequest(collection="docs", n=6))
    rig.chunks.chunks = []
    job = rig.service.generate_dataset(GenerateRequest(collection="docs", n=2))
    assert job.state == "failed" and "no_chunks" in job.error


# ----- runs: retrieval metrics ---------------------------------------------------------------------------------


def two_questions(rig, make_jsonl):
    ds = make_dataset(
        rig,
        make_jsonl,
        {"id": "q1", "question": "Q1?", "relevant_chunk_ids": ["c1"], "reference_answer": "R1"},
        {"id": "q2", "question": "Q2?", "relevant_chunk_ids": ["c2"], "reference_answer": "R2"},
    )
    rig.script.replies = {
        "Q1?": ids_only("c9", "c1", "c3", retrieval_ms=20.0),  # c1 is found at rank 2
        "Q2?": ids_only("c2", "c8", retrieval_ms=40.0),  # c2 is found at rank 1
    }
    return ds


def test_retrieval_metrics_are_computed_and_averaged_by_hand(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    job = rig.service.start_run(run_request(ds.id, ks=[1, 3]))
    assert job.state == "completed" and job.progress == job.total == 2
    run = rig.service.get_run(job.job_id)

    q1, q2 = run.results
    assert q1.metrics["recall@1"] == 0 and q1.metrics["recall@3"] == 1
    assert q1.metrics["mrr"] == 0.5 and q1.metrics["precision@3"] == pytest.approx(1 / 3)
    assert q2.metrics["recall@1"] == 1 and q2.metrics["mrr"] == 1.0

    m = run.report.metrics
    assert m["recall@1"].mean == 0.5 and m["recall@3"].mean == 1.0
    assert m["mrr"].mean == 0.75 and m["mrr"].n == 2
    assert m["precision@3"].mean == pytest.approx(1 / 3)
    assert m["mrr"].std == pytest.approx(0.35355, abs=1e-4)  # sample std of 0.5 and 1.0
    assert run.report.questions == 2 and run.report.errors == 0


def test_latency_percentiles_come_from_the_targets_timings(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    run = rig.service.get_run(rig.service.start_run(run_request(ds.id)).job_id)
    lat = run.report.latency["retrieval_ms"]
    assert (lat.p50, lat.n) == (30.0, 2) and lat.p95 == pytest.approx(39.0)
    assert run.report.latency["total_ms"].n == 2  # measured when the target gave none
    assert "ttft_ms" not in run.report.latency


def test_a_retrieval_only_target_reports_generation_as_not_computed_with_a_reason(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    job = rig.service.start_run(run_request(ds.id))
    report = rig.service.get_run(job.job_id).report
    assert report.not_computed == {
        "faithfulness": "the target returned no answer",
        "completeness": "the target returned no answer",
        "answer_accuracy": "the target returned no answer",
        "source_accuracy": "the target returned no answer",
    }
    assert rig.judge.calls == [] and report.judge_calls == 0  # the judge was never paid


def test_default_ks_come_from_settings_and_custom_ks_are_used(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    default = rig.service.get_run(rig.service.start_run(run_request(ds.id)).job_id)
    assert {"recall@1", "recall@3", "recall@5", "recall@10"} <= set(default.report.metrics)
    assert default.config["ks"] == [1, 3, 5, 10]
    custom = rig.service.get_run(rig.service.start_run(run_request(ds.id, ks=[2])).job_id)
    assert "recall@2" in custom.report.metrics and "recall@5" not in custom.report.metrics


def test_retrieval_can_be_switched_off(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    report = rig.service.get_run(
        rig.service.start_run(run_request(ds.id, retrieval_metrics=False)).job_id
    ).report
    assert report.metrics == {} and "retrieval" not in report.not_computed


def test_questions_without_relevant_chunks_are_not_scored_for_retrieval(rig, make_jsonl):
    ds = make_dataset(rig, make_jsonl, {"id": "n", "question": "Q?"})
    rig.script.default = ids_only("c1")
    run = rig.service.get_run(rig.service.start_run(run_request(ds.id)).job_id)
    assert run.results[0].metrics == {}
    assert run.report.not_computed["retrieval"] == "the question lists no relevant_chunk_ids"


# ----- runs: answer metrics -----------------------------------------------------------------------------------------


def test_answer_metrics_use_the_judge_and_source_accuracy_is_deterministic(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    rig.script.replies = {
        "Q1?": answer(retrieved=("c1", "c9"), cited=["c1"], ttft_ms=100.0),  # cites the right chunk
        "Q2?": answer(retrieved=("c2",), cited=["c2", "c7"]),  # right chunk plus a wrong one
    }
    job = rig.service.start_run(run_request(ds.id, target="scripted", ks=[1]))
    run = rig.service.get_run(job.job_id)

    r1, r2 = run.results
    assert r1.metrics["faithfulness"] == 0.5 and r1.metrics["completeness"] == 0.75
    assert r1.metrics["answer_accuracy"] == 0.8
    assert r1.metrics["source_accuracy"] == 1.0
    assert r2.metrics["source_accuracy"] == pytest.approx(2 / 3)  # precision 1/2, recall 1
    assert r1.faithfulness.claims and r1.accuracy.rationale == "because"
    assert run.report.judge_calls == 6 and rig.judge.calls.count("faithfulness") == 2
    assert run.report.metrics["source_accuracy"].mean == pytest.approx((1 + 2 / 3) / 2)
    assert run.report.metrics["faithfulness"].std == 0.0
    assert run.report.latency["ttft_ms"].n == 1
    assert run.report.not_computed == {}


def test_missing_target_fields_are_not_computed_for_those_metrics_only(rig, make_jsonl):
    ds = make_dataset(
        rig, make_jsonl,
        {"id": "q", "question": "Q?", "relevant_chunk_ids": ["c1"], "reference_answer": "R"},
    )  # fmt: skip
    rig.script.default = answer(retrieved=("c1",), cited=None, contexts=False)
    run = rig.service.get_run(rig.service.start_run(run_request(ds.id, target="scripted")).job_id)
    r = run.results[0]
    assert "faithfulness" not in r.metrics and "source_accuracy" not in r.metrics
    assert r.not_computed["faithfulness"] == "the target returned no contexts"
    assert r.not_computed["source_accuracy"] == "the target returned no cited_chunk_ids"
    assert r.metrics["answer_accuracy"] == 0.8 and r.metrics["recall@1"] == 1.0
    assert rig.judge.calls == ["completeness", "accuracy"]  # faithfulness was skipped, not paid


def test_no_reference_answer_skips_completeness_and_accuracy(rig, make_jsonl):
    ds = make_dataset(rig, make_jsonl, {"id": "q", "question": "Q?", "relevant_chunk_ids": ["c1"]})
    rig.script.default = answer(cited=["c1"])
    r = rig.service.get_run(
        rig.service.start_run(run_request(ds.id, target="scripted")).job_id
    ).results[0]
    assert "completeness" not in r.metrics and "answer_accuracy" not in r.metrics
    assert r.not_computed["completeness"] == "the question has no reference answer"
    assert rig.judge.calls == ["faithfulness"] and r.metrics["source_accuracy"] == 1.0


def test_a_blank_answer_is_not_judged(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    rig.script.default = answer(text="   ", retrieved=("c1",))
    run = rig.service.get_run(rig.service.start_run(run_request(ds.id, target="scripted")).job_id)
    assert rig.judge.calls == [] and run.results[0].not_computed["faithfulness"]


def test_generation_metrics_can_be_switched_off(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    rig.script.default = answer()
    run = rig.service.get_run(
        rig.service.start_run(
            run_request(ds.id, target="scripted", generation_metrics=False)
        ).job_id
    )
    assert rig.judge.calls == [] and "faithfulness" not in run.report.metrics


# ----- runs: things going wrong ----------------------------------------------------------------------------------------


def test_a_failing_target_is_recorded_per_question_and_the_run_carries_on(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    rig.script.replies["Q1?"] = TargetError("the target returned HTTP 500")
    run = rig.service.get_run(rig.service.start_run(run_request(ds.id)).job_id)
    assert run.state == "completed"
    assert run.results[0].error == "the target returned HTTP 500" and run.results[0].metrics == {}
    assert run.results[1].metrics["recall@1"] == 1.0
    assert run.report.errors == 1 and run.report.metrics["mrr"].n == 1


def test_an_unexpected_exception_in_a_target_does_not_stop_the_run(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    rig.script.replies["Q2?"] = RuntimeError("boom")
    run = rig.service.get_run(rig.service.start_run(run_request(ds.id)).job_id)
    assert run.state == "completed" and "RuntimeError: boom" in run.results[1].error


def test_a_judge_failure_is_noted_and_the_rest_still_scored(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    rig.script.replies = {}
    rig.script.default = answer(retrieved=("c1",))
    rig.judge.raise_on["completeness"] = AppError("llm_invalid_output", "bad json", 502)
    run = rig.service.get_run(rig.service.start_run(run_request(ds.id, target="scripted")).job_id)
    r = run.results[0]
    assert r.not_computed["completeness"] == "the judge failed (llm_invalid_output)"
    assert r.metrics["faithfulness"] == 0.5 and r.metrics["answer_accuracy"] == 0.8


def test_a_missing_llm_key_stops_the_run_instead_of_failing_every_question(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    rig.script.replies = {}
    rig.script.default = answer(retrieved=("c1",))
    rig.judge.raise_on["faithfulness"] = AppError(
        "llm_not_configured", "OPENAI_API_KEY is not set.", 503
    )
    job = rig.service.start_run(run_request(ds.id, target="scripted"))
    assert job.state == "failed" and "llm_not_configured" in job.error
    assert rig.judge.calls == ["faithfulness"]  # stopped at once, not once per question
    assert rig.service.get_run(job.job_id).report is None


# ----- starting runs ---------------------------------------------------------------------------------------------------------


def test_only_accepted_items_are_run(rig):
    job = rig.service.generate_dataset(GenerateRequest(collection="docs", n=3))
    dataset_id = job.result["dataset_id"]
    ds = rig.service.get_dataset(dataset_id)
    rig.service.patch_item(dataset_id, ds.items[0].id, ItemPatch(status="accepted"))
    rig.service.patch_item(dataset_id, ds.items[1].id, ItemPatch(status="rejected"))
    rig.script.default = ids_only("c1")
    run = rig.service.get_run(rig.service.start_run(run_request(dataset_id)).job_id)
    assert len(run.results) == 1 and rig.script.asked == [ds.items[0].question]


def test_a_dataset_with_nothing_accepted_cannot_run(rig):
    dataset_id = rig.service.generate_dataset(GenerateRequest(collection="docs", n=2)).result[
        "dataset_id"
    ]
    with pytest.raises(AppError) as err:
        rig.service.start_run(run_request(dataset_id))
    assert err.value.code == "no_accepted_items" and err.value.status_code == 422


def test_unknown_dataset_and_unknown_preset(rig, make_jsonl):
    with pytest.raises(AppError) as err:
        rig.service.start_run(run_request("nope"))
    assert err.value.status_code == 404
    ds = make_dataset(rig, make_jsonl, {"question": "Q?"})
    with pytest.raises(AppError, match="No preset 'magic'") as err:
        rig.service.start_run(run_request(ds.id, target="magic"))
    assert err.value.code == "unknown_target" and "scripted" in err.value.message


def test_the_same_idempotency_key_starts_one_run(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    first = rig.service.start_run(run_request(ds.id), idempotency_key="k")
    again = rig.service.start_run(run_request(ds.id), idempotency_key="k")
    assert first.job_id == again.job_id and len(rig.service.list_runs()) == 1
    assert rig.script.asked == ["Q1?", "Q2?"]  # asked once, not twice


def test_the_config_snapshot_records_what_is_needed_to_reproduce_a_run(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    run = rig.service.get_run(
        rig.service.start_run(run_request(ds.id, options={"mode": "dense"}, ks=[3, 1])).job_id
    )
    assert run.config["judge"] == {"model": "fake-judge", "prompt_version": "t1"}
    assert run.config["target"] == {
        "kind": "preset",
        "name": "ids_only",
        "options": {"mode": "dense"},
    }
    assert run.config["ks"] == [1, 3] and run.config["dataset"]["questions"] == 2
    assert run.target == "ids_only (mode=dense)" and run.dataset_name == "ds"
    assert run.started_at and run.finished_at and run.created_at <= run.started_at


def test_estimate_counts_judge_calls_before_anything_is_spent(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    est = rig.service.estimate(run_request(ds.id, target="scripted"))
    assert (est.questions, est.judge_calls_per_question, est.max_judge_calls) == (2, 3, 6)
    none = rig.service.estimate(run_request(ds.id, target="ids_only"))
    assert none.max_judge_calls == 0 and "no answers" in none.note
    off = rig.service.estimate(run_request(ds.id, target="scripted", generation_metrics=False))
    assert off.max_judge_calls == 0 and "off" in off.note
    assert rig.judge.calls == []  # an estimate costs nothing


def test_runs_are_listed_newest_first(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    a = rig.service.start_run(run_request(ds.id)).job_id
    b = rig.service.start_run(run_request(ds.id)).job_id
    assert [r.id for r in rig.service.list_runs()] == [b, a]
    with pytest.raises(AppError) as err:
        rig.service.get_run("nope")
    assert err.value.code == "run_not_found"


# ----- comparing -----------------------------------------------------------------------------------------------------------


def test_compare_shows_metric_deltas_and_the_questions_that_changed(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    a = rig.service.start_run(run_request(ds.id, ks=[1], options={"mode": "dense"})).job_id
    rig.script.replies["Q1?"] = ids_only("c1", "c9")  # now c1 is found at rank 1
    b = rig.service.start_run(run_request(ds.id, ks=[1], options={"mode": "hybrid"})).job_id

    cmp = rig.service.compare(a, b)
    deltas = {d.metric: d for d in cmp.metrics}
    assert (
        deltas["recall@1"].a == 0.5
        and deltas["recall@1"].b == 1.0
        and deltas["recall@1"].delta == 0.5
    )
    assert deltas["mrr"].delta == pytest.approx(0.25)  # 0.75 -> 1.0
    changed = {(c.item_id, c.metric): c for c in cmp.changed_questions}
    assert changed[("q1", "recall@1")].delta == 1.0 and ("q2", "recall@1") not in changed
    assert cmp.changed_questions[0].delta >= cmp.changed_questions[-1].delta  # biggest first
    assert (cmp.a.target, cmp.b.target) == ("ids_only (mode=dense)", "ids_only (mode=hybrid)")
    assert cmp.warnings == []


def test_compare_warns_when_the_datasets_differ(rig, make_jsonl):
    one = two_questions(rig, make_jsonl)
    two = make_dataset(
        rig, make_jsonl, {"id": "q1", "question": "Q1?", "relevant_chunk_ids": ["c1"]}, name="other"
    )
    a = rig.service.start_run(run_request(one.id)).job_id
    b = rig.service.start_run(run_request(two.id)).job_id
    assert "different datasets" in rig.service.compare(a, b).warnings[0]


def test_compare_needs_finished_runs(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    a = rig.service.start_run(run_request(ds.id)).job_id
    rig.judge.raise_on["faithfulness"] = AppError("llm_not_configured", "no key", 503)
    rig.script.replies = {}
    rig.script.default = answer()
    failed = rig.service.start_run(run_request(ds.id, target="scripted")).job_id
    with pytest.raises(AppError) as err:
        rig.service.compare(a, failed)
    assert err.value.code == "run_not_finished" and err.value.status_code == 409


# ----- tracing ---------------------------------------------------------------------------------------------------------------


def test_a_run_is_one_trace_with_a_span_per_question(rig, make_jsonl):
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    ds = two_questions(rig, make_jsonl)
    rig.script.replies["Q2?"] = TargetError("down")
    rig.service.start_run(run_request(ds.id))
    run = next(s for s in exporter.spans if s.name == "eval.run")
    questions = [s for s in exporter.spans if s.name == "eval.question"]
    assert len(questions) == 2 and {q.parent_id for q in questions} == {run.span_id}
    assert [q.status for q in questions] == ["ok", "error"]
    assert run.attributes["eval.errors"] == 1


def test_status_and_targets_info(rig):
    assert rig.service.status().judge_model == "fake-judge"
    info = rig.service.targets_info()
    assert "scripted" in info.presets and info.allowed_target_hosts == ["localhost", "127.0.0.1"]
    assert info.default_ks == [1, 3, 5, 10]
    assert json.dumps(info.model_dump())  # serialisable


# ----- choosing which metrics to compute ---------------------------------------------------------------------------


def answers_for_both_questions(rig):
    rig.script.replies = {
        "Q1?": answer(retrieved=("c9", "c1", "c3")),
        "Q2?": answer(retrieved=("c2", "c8")),
    }


def test_only_the_chosen_retrieval_metrics_are_computed_and_the_judge_is_never_called(
    rig, make_jsonl
):
    ds = two_questions(rig, make_jsonl)
    rig.script.default = answer()
    job = rig.service.start_run(
        run_request(ds.id, target="scripted", ks=[1, 3], metrics=["recall", "mrr"])
    )
    run = rig.service.get_run(job.job_id)
    assert set(run.report.metrics) == {"recall@1", "recall@3", "mrr"}
    assert rig.judge.calls == [] and run.report.judge_calls == 0
    assert run.report.not_computed == {}  # nothing unchosen is reported as "not computed"
    assert run.config["metrics"] == ["recall", "mrr"]


def test_choosing_one_judged_metric_makes_only_that_judge_call_per_question(rig, make_jsonl):
    ds = two_questions(rig, make_jsonl)
    answers_for_both_questions(rig)
    est = rig.service.estimate(run_request(ds.id, target="scripted", metrics=["faithfulness"]))
    assert (est.judge_calls_per_question, est.max_judge_calls) == (1, 2)
    run = rig.service.get_run(
        rig.service.start_run(
            run_request(ds.id, target="scripted", metrics=["faithfulness"])
        ).job_id
    )
    assert rig.judge.calls == ["faithfulness", "faithfulness"]
    assert set(run.report.metrics) == {"faithfulness"}  # no retrieval numbers, none asked for


def test_a_chosen_metric_the_dataset_cannot_support_is_explained_not_silently_dropped(
    rig, make_jsonl
):
    ds = make_dataset(
        rig, make_jsonl, {"id": "q1", "question": "Q1?", "relevant_chunk_ids": ["c1"]}
    )
    rig.script.replies = {"Q1?": answer()}
    run = rig.service.get_run(
        rig.service.start_run(
            run_request(
                ds.id, target="scripted", metrics=["completeness", "answer_accuracy", "recall"]
            )
        ).job_id
    )
    assert run.report.not_computed == {
        "completeness": "the question has no reference answer",
        "answer_accuracy": "the question has no reference answer",
    }
    assert "recall@1" in run.report.metrics and rig.judge.calls == []


def test_the_default_is_every_metric_and_bad_choices_are_refused():
    assert EvalConfig().wants("ndcg") and EvalConfig().retrieval_on and EvalConfig().generation_on
    only = EvalConfig(metrics=["source_accuracy"])
    assert only.generation_on and not only.retrieval_on and only.judge_metrics == []
    for bad in (["recall", "bogus"], []):
        with pytest.raises(ValueError):
            EvalConfig(metrics=bad)
