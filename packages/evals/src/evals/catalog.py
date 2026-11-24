"""The metrics an evaluation can compute, and what each one needs.

One list, used by the run options, the API (`GET /metrics`) and the UI picker, so the page can
say before a run which dataset fields and target outputs a metric needs.
"""

from typing import Literal

from pydantic import BaseModel, Field


class MetricInfo(BaseModel):
    id: str
    label: str
    group: Literal["retrieval", "answer", "citations"]
    description: str
    # Fields each dataset question must have for this metric to be computed.
    needs_dataset: list[Literal["relevant_chunk_ids", "reference_answer"]] = Field(
        default_factory=list
    )
    # What the evaluated system must return: "retrieved_chunk_ids", "answer", "contexts",
    # "cited_chunk_ids".
    needs_target: list[str] = Field(default_factory=list)
    judged: bool = False  # uses the LLM judge (costs model calls)


METRICS: list[MetricInfo] = [
    MetricInfo(
        id="recall", label="Recall@k", group="retrieval",
        description="Of the chunks that hold the answer, how many were retrieved in the top k.",
        needs_dataset=["relevant_chunk_ids"], needs_target=["retrieved_chunk_ids"],
    ),
    MetricInfo(
        id="precision", label="Precision@k", group="retrieval",
        description="Of the top k retrieved chunks, how many were relevant.",
        needs_dataset=["relevant_chunk_ids"], needs_target=["retrieved_chunk_ids"],
    ),
    MetricInfo(
        id="hit_rate", label="Hit rate@k", group="retrieval",
        description="Was at least one relevant chunk in the top k (1 or 0).",
        needs_dataset=["relevant_chunk_ids"], needs_target=["retrieved_chunk_ids"],
    ),
    MetricInfo(
        id="mrr", label="MRR", group="retrieval",
        description="1 divided by the rank of the first relevant chunk: rewards putting it first.",
        needs_dataset=["relevant_chunk_ids"], needs_target=["retrieved_chunk_ids"],
    ),
    MetricInfo(
        id="ndcg", label="nDCG@k", group="retrieval",
        description="Ranking quality: relevant chunks count more the higher they are ranked.",
        needs_dataset=["relevant_chunk_ids"], needs_target=["retrieved_chunk_ids"],
    ),
    MetricInfo(
        id="faithfulness", label="Faithfulness", group="answer", judged=True,
        description="Share of the answer's claims that the retrieved context supports.",
        needs_target=["answer", "contexts"],
    ),
    MetricInfo(
        id="completeness", label="Completeness", group="answer", judged=True,
        description="How many points of the reference answer the answer covers.",
        needs_dataset=["reference_answer"], needs_target=["answer"],
    ),
    MetricInfo(
        id="answer_accuracy", label="Answer accuracy", group="answer", judged=True,
        description="Does the answer agree with the reference answer (judged 0 to 1).",
        needs_dataset=["reference_answer"], needs_target=["answer"],
    ),
    MetricInfo(
        id="source_accuracy", label="Source accuracy", group="citations",
        description="Of the chunks the answer cites, how many are relevant.",
        needs_dataset=["relevant_chunk_ids"], needs_target=["cited_chunk_ids"],
    ),
]  # fmt: skip

METRIC_IDS = [m.id for m in METRICS]
RETRIEVAL_METRICS = tuple(m.id for m in METRICS if m.group == "retrieval")
JUDGED_METRICS = tuple(m.id for m in METRICS if m.judged)
ANSWER_SIDE_METRICS = tuple(m.id for m in METRICS if m.group != "retrieval")
