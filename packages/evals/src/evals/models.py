"""The contract: datasets, targets, results, reports."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

from evals.catalog import ANSWER_SIDE_METRICS, JUDGED_METRICS, METRIC_IDS, RETRIEVAL_METRICS

ItemStatus = Literal["draft", "accepted", "rejected"]
RunState = Literal["queued", "running", "completed", "failed"]


# ----- datasets -------------------------------------------------------------------------------


class Evidence(BaseModel):
    """The chunk a question was drafted from, shown to the reviewer."""

    chunk_id: str
    text: str
    source_name: str = ""
    page: int | None = None


class DatasetItem(BaseModel):
    id: str
    question: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    status: ItemStatus = "accepted"
    evidence: list[Evidence] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    id: str
    name: str
    status: Literal["draft", "ready"]  # draft while any item still awaits review
    created_at: float
    collection: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)  # items by status


class Dataset(DatasetSummary):
    items: list[DatasetItem]


class ItemPatch(BaseModel):
    status: ItemStatus | None = None
    question: str | None = Field(default=None, min_length=1)
    reference_answer: str | None = None
    relevant_chunk_ids: list[str] | None = None


class GenerateRequest(BaseModel):
    collection: str = Field(min_length=1, max_length=100)
    n: int = Field(default=10, ge=1)
    name: str | None = None
    min_chars: int = Field(default=200, ge=0)  # skip chunks too short to ask about
    seed: int = 0


# ----- targets --------------------------------------------------------------------------------


class Context(BaseModel):
    chunk_id: str
    text: str


class TargetResponse(BaseModel):
    """What any system must return. Fields it cannot provide are left out, and the metrics that
    need them are reported as not computed."""

    answer: str | None = None
    contexts: list[Context] | None = None
    retrieved_chunk_ids: list[str] | None = None
    cited_chunk_ids: list[str] | None = None
    timings: dict[str, float] = Field(default_factory=dict)  # retrieval_ms, ttft_ms, total_ms


class HttpTargetConfig(BaseModel):
    kind: Literal["http"]
    url: str
    # header name -> NAME of an environment variable. Only the name is stored, never the value.
    headers_refs: dict[str, str] = Field(default_factory=dict)
    timeout_s: float | None = Field(default=None, gt=0)


class PresetTargetConfig(BaseModel):
    kind: Literal["preset"]
    name: str
    options: dict[str, Any] = Field(default_factory=dict)


TargetConfig = Annotated[HttpTargetConfig | PresetTargetConfig, Field(discriminator="kind")]


# ----- runs -----------------------------------------------------------------------------------


class EvalConfig(BaseModel):
    ks: list[int] | None = None  # default: EVAL_DEFAULT_KS
    retrieval_metrics: bool = True
    generation_metrics: bool = True
    # Which metrics to compute (ids from `GET /metrics`). None means every metric. Choosing fewer
    # also means fewer judge calls, which is what costs money.
    metrics: list[str] | None = None

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        unknown = sorted(set(v) - set(METRIC_IDS))
        if unknown:
            raise ValueError(f"unknown metrics: {', '.join(unknown)}")
        if not v:
            raise ValueError("choose at least one metric")
        return list(dict.fromkeys(v))

    def wants(self, metric: str) -> bool:
        return self.metrics is None or metric in self.metrics

    @property
    def retrieval_on(self) -> bool:
        return self.retrieval_metrics and any(self.wants(m) for m in RETRIEVAL_METRICS)

    @property
    def generation_on(self) -> bool:
        return self.generation_metrics and any(self.wants(m) for m in ANSWER_SIDE_METRICS)

    @property
    def judge_metrics(self) -> list[str]:
        return [m for m in JUDGED_METRICS if self.wants(m)]

    @field_validator("ks")
    @classmethod
    def _ks(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and (not v or any(k < 1 or k > 100 for k in v)):
            raise ValueError("ks must be a non-empty list of integers from 1 to 100")
        return sorted(set(v)) if v else v


class RunRequest(BaseModel):
    dataset_id: str
    target: TargetConfig
    config: EvalConfig = Field(default_factory=EvalConfig)


class EstimateResponse(BaseModel):
    questions: int
    judge_calls_per_question: int
    max_judge_calls: int
    note: str


class Claim(BaseModel):
    text: str
    supported: bool
    evidence: str


class FaithfulnessResult(BaseModel):
    claims: list[Claim]
    score: float | None  # supported / total, None if the answer has no claims


class CoveragePoint(BaseModel):
    point: str
    covered: bool
    note: str


class CompletenessResult(BaseModel):
    points: list[CoveragePoint]
    score: float | None


class AccuracyResult(BaseModel):
    score: float
    rationale: str


class QuestionResult(BaseModel):
    item_id: str
    question: str
    reference_answer: str = ""
    answer: str | None = None
    error: str = ""
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] | None = None
    cited_chunk_ids: list[str] | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)
    not_computed: dict[str, str] = Field(default_factory=dict)  # metric -> why
    faithfulness: FaithfulnessResult | None = None
    completeness: CompletenessResult | None = None
    accuracy: AccuracyResult | None = None
    timings: dict[str, float] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    mean: float | None
    std: float | None = None
    n: int  # questions that had a value


class LatencyStats(BaseModel):
    p50: float | None
    p95: float | None
    n: int


class RunReport(BaseModel):
    questions: int
    errors: int
    metrics: dict[str, MetricSummary]
    latency: dict[str, LatencyStats]
    not_computed: dict[str, str] = Field(default_factory=dict)  # metric -> reason, run-wide
    judge_calls: int = 0


class RunSummary(BaseModel):
    id: str
    dataset_id: str
    dataset_name: str
    state: RunState
    created_at: float
    finished_at: float | None = None
    progress: int = 0
    total: int = 0
    target: str = ""  # a short label for lists
    error: str = ""


class Run(RunSummary):
    started_at: float | None = None
    config: dict[str, Any] = Field(default_factory=dict)  # snapshot: target, ks, judge, versions
    report: RunReport | None = None
    results: list[QuestionResult] = Field(default_factory=list)


class MetricDelta(BaseModel):
    metric: str
    a: float | None
    b: float | None
    delta: float | None


class QuestionChange(BaseModel):
    item_id: str
    question: str
    metric: str
    a: float | None
    b: float | None
    delta: float


class Comparison(BaseModel):
    a: RunSummary
    b: RunSummary
    metrics: list[MetricDelta]
    changed_questions: list[QuestionChange]
    warnings: list[str] = Field(default_factory=list)


class JobStatus(BaseModel):
    job_id: str
    kind: Literal["run", "generate"]
    state: RunState
    created_at: float
    finished_at: float | None = None
    progress: int = 0
    total: int = 0
    result: dict[str, Any] = Field(default_factory=dict)  # e.g. {"dataset_id": "..."}
    error: str = ""


class JobAccepted(BaseModel):
    job_id: str
    state: RunState


class TargetsInfo(BaseModel):
    presets: list[str]
    allowed_target_hosts: list[str]
    allowed_header_refs: list[str]
    default_ks: list[int]
    judge_model: str
    judge_prompt_version: str


class EvalStatus(BaseModel):
    db_path: str
    judge_model: str
    judge_prompt_version: str
    presets: list[str]
