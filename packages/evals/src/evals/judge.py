"""The LLM judge: faithfulness, completeness and answer accuracy, as structured JSON.

An LLM judge has known biases (it prefers longer answers, confident tone, and its own style), so
the prompts say what to ignore, and the model and prompt version are recorded in every run.
Source accuracy is NOT here: it is deterministic and lives in `metrics.py`.
"""

from pydantic import BaseModel

from evals.models import (
    AccuracyResult,
    Claim,
    CompletenessResult,
    CoveragePoint,
    FaithfulnessResult,
)
from llm import LLM

PROMPT_VERSION = "1"
MAX_CONTEXT_CHARS = 24_000

SYSTEM = (
    "You are a strict, impartial evaluator of question answering. Judge only the content. Do not "
    "reward length, style, confidence or politeness. Write your notes in the language of the "
    "answer. Be brief."
)


class _Claims(BaseModel):
    claims: list[Claim]


class _Points(BaseModel):
    points: list[CoveragePoint]


class _Accuracy(BaseModel):
    score: float
    rationale: str


def _numbered(contexts: list[str]) -> str:
    out, used = [], 0
    for i, text in enumerate(contexts, start=1):
        room = MAX_CONTEXT_CHARS - used
        if room <= 0:
            out.append("[... further passages omitted: too long ...]")
            break
        piece = text[:room]
        out.append(f"[{i}] {piece}")
        used += len(piece)
    return "\n\n".join(out)


class LLMJudge:
    def __init__(self, llm: LLM, model: str, prompt_version: str = PROMPT_VERSION) -> None:
        self._llm = llm
        self.model = model
        self.prompt_version = prompt_version

    def score_faithfulness(self, answer: str, contexts: list[str]) -> FaithfulnessResult:
        """Split the answer into claims, check each against the retrieved passages.
        Score = supported claims / all claims. This is what catches hallucination."""
        if not answer.strip():
            return FaithfulnessResult(claims=[], score=None)
        prompt = (
            "Break the ANSWER into separate factual claims. For each claim decide whether the "
            "PASSAGES state it or clearly imply it. Outside knowledge does not count: if the "
            "passages do not support a claim, it is unsupported, even if it is true. For "
            "`evidence`, quote the supporting words, or write 'none'.\n\n"
            f"PASSAGES:\n{_numbered(contexts) or '(no passages were retrieved)'}\n\n"
            f"ANSWER:\n{answer}"
        )
        claims = self._llm.complete_json(prompt, _Claims, system=SYSTEM).claims
        if not claims:
            return FaithfulnessResult(claims=[], score=None)
        return FaithfulnessResult(
            claims=claims, score=sum(c.supported for c in claims) / len(claims)
        )

    def score_completeness(self, answer: str, reference: str) -> CompletenessResult:
        """List the points the REFERENCE makes, then check each is covered by the answer."""
        if not reference.strip():
            return CompletenessResult(points=[], score=None)
        prompt = (
            "List the separate points that the REFERENCE answer makes. For each point decide "
            "whether the ANSWER covers it (same meaning, wording may differ). Extra content in "
            "the answer is not a problem here.\n\n"
            f"REFERENCE:\n{reference}\n\nANSWER:\n{answer or '(empty)'}"
        )
        points = self._llm.complete_json(prompt, _Points, system=SYSTEM).points
        if not points:
            return CompletenessResult(points=[], score=None)
        return CompletenessResult(points=points, score=sum(p.covered for p in points) / len(points))

    def score_accuracy(self, question: str, answer: str, reference: str) -> AccuracyResult:
        """Is the answer correct, judged against the reference? 0 to 1, with a reason."""
        prompt = (
            "Score from 0 to 1 how correct the ANSWER is for the QUESTION, using the REFERENCE as "
            "the truth. 1 = same facts, 0.5 = partly right or partly wrong, 0 = wrong, "
            "contradicting, or empty. Ignore wording and length.\n\n"
            f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer or '(empty)'}"
        )
        out = self._llm.complete_json(prompt, _Accuracy, system=SYSTEM)
        return AccuracyResult(score=min(max(out.score, 0.0), 1.0), rationale=out.rationale)
