"""EvalService: datasets, runs, comparisons. Pure Python, no HTTP.

A run asks the target each accepted question, then scores the reply:
  retrieval metrics  (recall, precision, hit rate, MRR, nDCG)  from the retrieved chunk ids
  faithfulness       LLM judge: are the answer's claims supported by the retrieved passages?
  completeness       LLM judge: does the answer cover the reference's points?
  answer accuracy    LLM judge: is it correct against the reference?
  source accuracy    deterministic: do the cited chunks match the evidence chunks?
A metric the target cannot support is "not computed", with the reason, never a silent zero.
"""

import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from common import AppError
from evals import metrics as M
from evals.catalog import RETRIEVAL_METRICS
from evals.generator import QuestionGenerator
from evals.models import (
    Comparison,
    Dataset,
    DatasetItem,
    DatasetSummary,
    EstimateResponse,
    EvalConfig,
    EvalStatus,
    GenerateRequest,
    ItemPatch,
    JobStatus,
    LatencyStats,
    MetricDelta,
    MetricSummary,
    QuestionChange,
    QuestionResult,
    Run,
    RunReport,
    RunRequest,
    RunSummary,
    TargetConfig,
    TargetsInfo,
)
from evals.protocols import ChunkSource, Judge, Target
from evals.settings import EvalSettings
from evals.store import EvalStore
from evals.targets import TargetError, build_http_target, describe
from observability import span

_log = logging.getLogger("evals")

JUDGE_CALLS_PER_QUESTION = 3  # faithfulness, completeness, accuracy
MAX_DATASET_ITEMS = 5000
# Errors that mean every further judge call would fail too: stop the run instead of recording
# the same failure on every question.
_FATAL = {"llm_not_configured", "llm_auth", "llm_model_not_found"}

GENERATION_METRICS = ("faithfulness", "completeness", "answer_accuracy", "source_accuracy")


@dataclass(frozen=True)
class Preset:
    """An in-process target that e2e registers (for example retrieval)."""

    factory: Callable[[dict[str, Any]], Target]
    returns_answer: bool = False
    description: str = ""


class EvalService:
    def __init__(
        self,
        settings: EvalSettings,
        store: EvalStore,
        judge: Judge,
        generator: QuestionGenerator,
        chunks: ChunkSource,
        presets: dict[str, Preset] | None = None,
        http_client: httpx.Client | None = None,
        inline: bool = False,
    ) -> None:
        self._s = settings
        self._store = store
        self._judge = judge
        self._gen = generator
        self._chunks = chunks
        self._presets = presets or {}
        self._http = http_client
        self._pool = (
            None if inline else ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval")
        )
        self._store.fail_unfinished_jobs()

    def register_preset(self, name: str, preset: Preset) -> None:
        """Add an in-process target after construction (e2e does this for the agent, which is
        built after evals)."""
        self._presets[name] = preset

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        self._store.close()

    # ----- jobs -------------------------------------------------------------------------------

    def _submit(self, job_id: str, work: Callable[[], dict[str, Any] | None]) -> None:
        if self._pool is None:
            self._run_job(job_id, work)
        else:
            self._pool.submit(self._run_job, job_id, work)

    def _run_job(self, job_id: str, work: Callable[[], dict[str, Any] | None]) -> None:
        self._store.update_job(job_id, state="running")
        try:
            result = work() or {}
        except AppError as exc:
            self._store.update_job(job_id, state="failed", error=f"{exc.code}: {exc.message}")
        except Exception as exc:
            _log.exception("job %s failed", job_id)
            self._store.update_job(
                job_id, state="failed", error=f"{type(exc).__name__}: {exc}"[:500]
            )
        else:
            self._store.update_job(job_id, state="completed", result=result)

    def get_job(self, job_id: str) -> JobStatus:
        job = self._store.get_job(job_id)
        if job is None:
            raise AppError("job_not_found", f"No job {job_id}.", 404)
        return job

    # ----- datasets ---------------------------------------------------------------------------

    def upload_dataset(self, name: str, data: bytes) -> DatasetSummary:
        items = self._parse_jsonl(data)
        dataset_id = self._store.create_dataset(name, items)
        return self._summary(dataset_id)

    @staticmethod
    def _parse_jsonl(data: bytes) -> list[DatasetItem]:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise AppError("invalid_dataset", "The file is not UTF-8 text.", 422) from exc
        items: list[DatasetItem] = []
        problems: dict[int, str] = {}
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("each line must be a JSON object")
                items.append(
                    DatasetItem(
                        id=str(obj.get("id") or uuid.uuid4().hex[:10]),
                        question=str(obj.get("question") or ""),
                        relevant_chunk_ids=[str(c) for c in obj.get("relevant_chunk_ids") or []],
                        reference_answer=str(obj.get("reference_answer") or ""),
                        status="accepted",
                    )
                )
            except ValidationError as exc:
                problems[number] = exc.errors(include_input=False)[0]["msg"]
            except (ValueError, TypeError) as exc:
                problems[number] = str(exc)[:150]
        if problems:
            shown = dict(list(problems.items())[:10])
            raise AppError(
                "invalid_dataset",
                f"{len(problems)} line(s) are not valid (a question is required).",
                422,
                {"lines": shown},
            )
        if not items:
            raise AppError("invalid_dataset", "The file has no items.", 422)
        if len(items) > MAX_DATASET_ITEMS:
            raise AppError("invalid_dataset", f"At most {MAX_DATASET_ITEMS} items.", 422)
        ids = [i.id for i in items]
        if len(set(ids)) != len(ids):
            raise AppError("invalid_dataset", "Two items share the same id.", 422)
        return items

    def _summary(self, dataset_id: str) -> DatasetSummary:
        dataset = self.get_dataset(dataset_id)
        return DatasetSummary(**dataset.model_dump(exclude={"items"}))

    def get_dataset(self, dataset_id: str) -> Dataset:
        dataset = self._store.get_dataset(dataset_id)
        if dataset is None:
            raise AppError("dataset_not_found", f"No dataset {dataset_id}.", 404)
        return dataset

    def list_datasets(self) -> list[DatasetSummary]:
        return self._store.list_datasets()

    def patch_item(self, dataset_id: str, item_id: str, patch: ItemPatch) -> DatasetItem:
        item = self._store.patch_item(dataset_id, item_id, patch)
        if item is None:
            raise AppError("item_not_found", f"No item {item_id} in dataset {dataset_id}.", 404)
        return item

    def delete_dataset(self, dataset_id: str) -> None:
        if not self._store.delete_dataset(dataset_id):
            raise AppError("dataset_not_found", f"No dataset {dataset_id}.", 404)

    def list_collections(self) -> list[str]:
        return self._chunks.list_collections()

    def generate_dataset(
        self, req: GenerateRequest, idempotency_key: str | None = None
    ) -> JobStatus:
        if req.n > self._s.max_generate:
            raise AppError(
                "invalid_options", f"n is above EVAL_MAX_GENERATE ({self._s.max_generate}).", 422
            )
        job, created = self._store.create_job("generate", req.n, idempotency_key)
        if created:
            self._submit(job.job_id, lambda: self._generate(job.job_id, req))
        return self.get_job(job.job_id)

    def _generate(self, job_id: str, req: GenerateRequest) -> dict[str, Any]:
        chunks = self._chunks.sample(req.collection, req.n, req.seed, req.min_chars)
        if not chunks:
            raise AppError(
                "no_chunks",
                f"'{req.collection}' has no chunks of at least {req.min_chars} characters.",
                404,
            )
        items: list[DatasetItem] = []
        failed = 0
        with span("eval.generate", {"eval.collection": req.collection, "eval.requested": req.n}):
            for done, chunk in enumerate(chunks, start=1):
                try:
                    items.append(self._gen.draft(chunk))
                except AppError as exc:
                    if exc.code in _FATAL:
                        raise
                    failed += 1
                self._store.update_job(job_id, progress=done)
        if not items:
            raise AppError("generation_failed", "The model produced no usable question.", 502)
        name = req.name or f"generated from {req.collection}"
        dataset_id = self._store.create_dataset(name, items, req.collection)
        return {
            "dataset_id": dataset_id,
            "drafted": len(items),
            "failed": failed,
            "requested": req.n,
            "available": len(chunks),
        }

    # ----- runs -------------------------------------------------------------------------------

    def targets_info(self) -> TargetsInfo:
        return TargetsInfo(
            presets=sorted(self._presets),
            allowed_target_hosts=self._s.allowed_target_hosts,
            allowed_header_refs=self._s.allowed_header_refs,
            default_ks=self._s.default_ks,
            judge_model=self._judge.model,
            judge_prompt_version=self._judge.prompt_version,
        )

    def status(self) -> EvalStatus:
        return EvalStatus(
            db_path=self._s.db_path,
            judge_model=self._judge.model,
            judge_prompt_version=self._judge.prompt_version,
            presets=sorted(self._presets),
        )

    def _build_target(self, config: TargetConfig) -> Target:
        if config.kind == "http":
            return build_http_target(config, self._s, client=self._http)
        preset = self._presets.get(config.name)
        if preset is None:
            known = ", ".join(sorted(self._presets)) or "none registered"
            raise AppError("unknown_target", f"No preset '{config.name}'. Available: {known}.", 422)
        try:
            return preset.factory(config.options)
        except ValidationError as exc:
            msg = exc.errors(include_input=False)[0]["msg"]
            raise AppError("invalid_target", f"Bad target options: {msg}", 422) from exc

    def _accepted(self, dataset_id: str) -> tuple[Dataset, list[DatasetItem]]:
        dataset = self.get_dataset(dataset_id)
        accepted = [i for i in dataset.items if i.status == "accepted"]
        if not accepted:
            raise AppError(
                "no_accepted_items",
                "This dataset has no accepted items. Review the drafts and accept some first.",
                422,
            )
        return dataset, accepted

    def estimate(self, req: RunRequest) -> EstimateResponse:
        _, accepted = self._accepted(req.dataset_id)
        returns_answer = True  # an HTTP target may answer, so assume it does
        if req.target.kind == "preset":
            preset = self._presets.get(req.target.name)
            returns_answer = preset.returns_answer if preset else False
        judged = req.config.judge_metrics
        per = len(judged) if (req.config.generation_on and returns_answer) else 0
        if not req.config.generation_metrics:
            note = "Generation metrics are off, so the judge is never called."
        elif not req.config.generation_on or not judged:
            note = "No judged metric is selected, so the judge is never called."
        elif not returns_answer:
            note = "This target returns no answers, so no judge calls are made."
        else:
            note = (
                f"Up to {per} judge call(s) per question ({', '.join(judged)}), "
                "fewer if the target returns no answer or contexts."
            )
        return EstimateResponse(
            questions=len(accepted),
            judge_calls_per_question=per,
            max_judge_calls=per * len(accepted),
            note=note,
        )

    def start_run(self, req: RunRequest, idempotency_key: str | None = None) -> JobStatus:
        dataset, accepted = self._accepted(req.dataset_id)
        target = self._build_target(req.target)  # validated now, so errors are immediate
        ks = req.config.ks or self._s.default_ks
        job, created = self._store.create_job("run", len(accepted), idempotency_key)
        if not created:
            return job
        snapshot = {
            "target": req.target.model_dump(),
            "ks": ks,
            "retrieval_metrics": req.config.retrieval_metrics,
            "generation_metrics": req.config.generation_metrics,
            "metrics": req.config.metrics,
            "judge": {"model": self._judge.model, "prompt_version": self._judge.prompt_version},
            "dataset": {"id": dataset.id, "name": dataset.name, "questions": len(accepted)},
        }
        self._store.create_run(job.job_id, dataset.id, dataset.name, snapshot, describe(req.target))
        self._submit(
            job.job_id, lambda: self._execute(job.job_id, accepted, target, req.config, ks)
        )
        return self.get_job(job.job_id)

    def _execute(
        self, run_id: str, items: list[DatasetItem], target: Target, cfg: EvalConfig, ks: list[int]
    ) -> dict[str, Any]:
        self._store.mark_run_started(run_id)
        results: list[QuestionResult] = []
        judge_calls = 0
        with span("eval.run", {"eval.run_id": run_id, "eval.questions": len(items)}) as root:
            for position, item in enumerate(items):
                with span("eval.question", {"eval.item_id": item.id}) as sp:
                    result, calls = self._evaluate(item, target, cfg, ks)
                    if result.error:
                        sp.set_status("error", result.error)
                judge_calls += calls
                results.append(result)
                self._store.add_result(run_id, position, result)
                self._store.update_job(run_id, progress=position + 1)
            report = self._aggregate(results, ks, cfg, judge_calls)
            self._store.save_report(run_id, report)
            root.set_attributes({"eval.errors": report.errors, "eval.judge_calls": judge_calls})
        return {"questions": len(results), "errors": report.errors}

    def _evaluate(
        self, item: DatasetItem, target: Target, cfg: EvalConfig, ks: list[int]
    ) -> tuple[QuestionResult, int]:
        result = QuestionResult(
            item_id=item.id,
            question=item.question,
            reference_answer=item.reference_answer,
            relevant_chunk_ids=item.relevant_chunk_ids,
        )
        started = time.perf_counter()
        try:
            reply = target.ask(item.question)
        except TargetError as exc:
            result.error = str(exc)[:300]
            return result, 0
        except Exception as exc:  # a target must never stop the run
            _log.exception("target failed on item %s", item.id)
            result.error = f"{type(exc).__name__}: {exc}"[:300]
            return result, 0
        result.timings = dict(reply.timings)
        result.timings.setdefault("total_ms", (time.perf_counter() - started) * 1000)
        result.answer = reply.answer
        result.retrieved_chunk_ids = reply.retrieved_chunk_ids
        result.cited_chunk_ids = reply.cited_chunk_ids

        if cfg.retrieval_on:
            if reply.retrieved_chunk_ids is None:
                result.not_computed["retrieval"] = "the target returned no retrieved_chunk_ids"
            elif not item.relevant_chunk_ids:
                result.not_computed["retrieval"] = "the question lists no relevant_chunk_ids"
            else:
                scores = M.retrieval_scores(reply.retrieved_chunk_ids, item.relevant_chunk_ids, ks)
                # keys look like "recall@3" or "mrr": keep the ones whose metric was chosen
                result.metrics.update(
                    {k: v for k, v in scores.items() if cfg.wants(k.split("@")[0])}
                )

        calls = 0
        if cfg.generation_on:
            calls = self._generation_metrics(item, reply.answer, reply, result, cfg)
        return result, calls

    def _generation_metrics(
        self, item, answer, reply, result: QuestionResult, cfg: EvalConfig
    ) -> int:
        if not (answer or "").strip():
            for family in GENERATION_METRICS:
                if cfg.wants(family):
                    result.not_computed[family] = "the target returned no answer"
            return 0

        calls = 0

        def judged(family: str, fn: Callable[[], Any]) -> Any:
            nonlocal calls
            calls += 1
            try:
                return fn()
            except AppError as exc:
                if exc.code in _FATAL:
                    raise
                result.not_computed[family] = f"the judge failed ({exc.code})"
                return None

        if not cfg.wants("faithfulness"):
            pass
        elif reply.contexts is None:
            result.not_computed["faithfulness"] = "the target returned no contexts"
        else:
            texts = [c.text for c in reply.contexts]
            out = judged("faithfulness", lambda: self._judge.score_faithfulness(answer, texts))
            if out is not None:
                result.faithfulness = out
                result.metrics["faithfulness"] = out.score
                if out.score is None:
                    result.not_computed["faithfulness"] = "the answer has no checkable claims"

        needs_reference = [m for m in ("completeness", "answer_accuracy") if cfg.wants(m)]
        if needs_reference and not item.reference_answer.strip():
            for m in needs_reference:
                result.not_computed[m] = "the question has no reference answer"
        else:
            if cfg.wants("completeness"):
                out = judged(
                    "completeness",
                    lambda: self._judge.score_completeness(answer, item.reference_answer),
                )
                if out is not None:
                    result.completeness = out
                    result.metrics["completeness"] = out.score
            if cfg.wants("answer_accuracy"):
                acc = judged(
                    "answer_accuracy",
                    lambda: self._judge.score_accuracy(
                        item.question, answer, item.reference_answer
                    ),
                )
                if acc is not None:
                    result.accuracy = acc
                    result.metrics["answer_accuracy"] = acc.score

        if not cfg.wants("source_accuracy"):
            pass
        elif reply.cited_chunk_ids is None:
            result.not_computed["source_accuracy"] = "the target returned no cited_chunk_ids"
        elif not item.relevant_chunk_ids:
            result.not_computed["source_accuracy"] = "the question lists no relevant_chunk_ids"
        else:
            result.metrics["source_accuracy"] = M.source_accuracy(
                reply.cited_chunk_ids, item.relevant_chunk_ids
            )
        return calls

    @staticmethod
    def _aggregate(
        results: list[QuestionResult], ks: list[int], cfg: EvalConfig, judge_calls: int
    ) -> RunReport:
        keys: list[str] = []
        for r in results:
            for k in r.metrics:
                if k not in keys:
                    keys.append(k)
        summaries: dict[str, MetricSummary] = {}
        for key in keys:
            values = [r.metrics.get(key) for r in results if key in r.metrics]
            summaries[key] = MetricSummary(
                mean=M.mean(values),
                std=M.stdev(values),
                n=sum(v is not None for v in values),
            )

        wanted = (["retrieval"] if cfg.retrieval_on else []) + (
            [m for m in GENERATION_METRICS if cfg.wants(m)] if cfg.generation_on else []
        )
        retrieval_keys = [k for k in summaries if k.split("@")[0] in RETRIEVAL_METRICS]
        not_computed: dict[str, str] = {}
        for family in wanted:
            if family == "retrieval":
                done = any(summaries[k].n for k in retrieval_keys)
            else:
                summary = summaries.get(family)
                done = summary is not None and summary.n > 0
            if not done:
                reasons = [r.not_computed[family] for r in results if family in r.not_computed]
                not_computed[family] = (
                    Counter(reasons).most_common(1)[0][0] if reasons else "no question had data"
                )

        latency = {}
        for key in ("retrieval_ms", "ttft_ms", "total_ms"):
            values = [r.timings.get(key) for r in results if key in r.timings]
            if values:
                latency[key] = LatencyStats(
                    p50=M.percentile(values, 50), p95=M.percentile(values, 95), n=len(values)
                )
        return RunReport(
            questions=len(results),
            errors=sum(1 for r in results if r.error),
            metrics=summaries,
            latency=latency,
            not_computed=not_computed,
            judge_calls=judge_calls,
        )

    def get_run(self, run_id: str) -> Run:
        run = self._store.get_run(run_id)
        if run is None:
            raise AppError("run_not_found", f"No run {run_id}.", 404)
        return run

    def list_runs(self) -> list[RunSummary]:
        return self._store.list_runs()

    def compare(self, a_id: str, b_id: str, limit: int = 50) -> Comparison:
        a, b = self.get_run(a_id), self.get_run(b_id)
        warnings = []
        for run in (a, b):
            if run.report is None:
                raise AppError("run_not_finished", f"Run {run.id} has no report yet.", 409)
        if a.dataset_id != b.dataset_id:
            warnings.append(
                "The runs used different datasets, so the numbers are not like for like."
            )

        names = list(dict.fromkeys([*a.report.metrics, *b.report.metrics]))
        deltas = []
        for name in names:
            va = a.report.metrics[name].mean if name in a.report.metrics else None
            vb = b.report.metrics[name].mean if name in b.report.metrics else None
            deltas.append(
                MetricDelta(
                    metric=name, a=va, b=vb, delta=None if va is None or vb is None else vb - va
                )
            )

        by_item_a = {r.item_id: r for r in a.results}
        changes: list[QuestionChange] = []
        for rb in b.results:
            ra = by_item_a.get(rb.item_id)
            if ra is None:
                continue
            for name in names:
                va, vb = ra.metrics.get(name), rb.metrics.get(name)
                if va is not None and vb is not None and abs(vb - va) > 1e-9:
                    changes.append(
                        QuestionChange(
                            item_id=rb.item_id, question=rb.question, metric=name,
                            a=va, b=vb, delta=vb - va,
                        )
                    )  # fmt: skip
        changes.sort(key=lambda c: -abs(c.delta))
        return Comparison(
            a=RunSummary(**a.model_dump(include=set(RunSummary.model_fields))),
            b=RunSummary(**b.model_dump(include=set(RunSummary.model_fields))),
            metrics=deltas,
            changed_questions=changes[:limit],
            warnings=warnings,
        )
