import json

import pytest

from evals.generator import QuestionGenerator, _Draft
from evals.judge import MAX_CONTEXT_CHARS, LLMJudge, _Accuracy, _Claims, _Points
from evals.protocols import ChunkRef
from llm import strict_schema
from llm.testing import FakeLLM


def judge(script) -> tuple[LLMJudge, FakeLLM]:
    llm = FakeLLM(script)
    return LLMJudge(llm, "judge-model", "v9"), llm


def test_every_model_the_judge_asks_for_is_valid_strict_structured_output():
    for model in (_Claims, _Points, _Accuracy, _Draft):
        schema = strict_schema(model)  # raises if OpenAI strict mode would refuse it
        assert schema["additionalProperties"] is False


def test_the_judge_records_its_model_and_prompt_version():
    j, _ = judge("{}")
    assert (j.model, j.prompt_version) == ("judge-model", "v9")


def test_faithfulness_is_supported_claims_over_all_claims():
    claims = [
        {"text": "a", "supported": True, "evidence": "quote"},
        {"text": "b", "supported": True, "evidence": "quote"},
        {"text": "c", "supported": True, "evidence": "quote"},
        {"text": "d", "supported": False, "evidence": "none"},
    ]
    j, llm = judge([json.dumps({"claims": claims})])
    out = j.score_faithfulness("The answer.", ["first passage", "second passage"])
    assert out.score == 0.75 and len(out.claims) == 4
    call = llm.calls[0]
    assert call.kind == "complete_json" and call.schema is _Claims
    assert "[1] first passage" in call.prompt and "[2] second passage" in call.prompt
    assert "ANSWER:\nThe answer." in call.prompt
    assert "Do not reward length" in call.system


def test_a_claim_the_passages_do_not_support_is_unsupported_even_if_true():
    j, llm = judge([json.dumps({"claims": []})])
    j.score_faithfulness("x", ["p"])
    assert "Outside knowledge does not count" in llm.calls[0].prompt


def test_faithfulness_without_an_answer_or_claims_is_not_scored():
    j, llm = judge([json.dumps({"claims": []})])
    assert j.score_faithfulness("   ", ["p"]).score is None and llm.calls == []  # no LLM call
    assert j.score_faithfulness("An answer with nothing checkable.", ["p"]).score is None


def test_faithfulness_with_no_passages_says_so_in_the_prompt():
    j, llm = judge([json.dumps({"claims": []})])
    j.score_faithfulness("x", [])
    assert "no passages were retrieved" in llm.calls[0].prompt


def test_very_long_contexts_are_truncated():
    j, llm = judge([json.dumps({"claims": []})])
    j.score_faithfulness("x", ["a" * (MAX_CONTEXT_CHARS - 10), "b" * 5000, "c" * 5000])
    prompt = llm.calls[0].prompt
    assert len(prompt) < MAX_CONTEXT_CHARS + 1000 and "omitted" in prompt


def test_completeness_is_covered_points_over_all_points():
    points = [
        {"point": "p1", "covered": True, "note": ""},
        {"point": "p2", "covered": False, "note": "missing"},
    ]
    j, llm = judge([json.dumps({"points": points})])
    out = j.score_completeness("answer", "the reference")
    assert out.score == 0.5 and out.points[1].covered is False
    assert "REFERENCE:\nthe reference" in llm.calls[0].prompt


def test_completeness_without_a_reference_is_not_scored_and_costs_nothing():
    j, llm = judge("{}")
    assert j.score_completeness("answer", "  ").score is None and llm.calls == []


@pytest.mark.parametrize(("raw", "expected"), [(0.9, 0.9), (1.7, 1.0), (-0.3, 0.0)])
def test_accuracy_is_clamped_to_zero_one(raw, expected):
    j, llm = judge([json.dumps({"score": raw, "rationale": "why"})])
    out = j.score_accuracy("the question", "the answer", "the reference")
    assert out.score == expected and out.rationale == "why"
    prompt = llm.calls[0].prompt
    assert all(s in prompt for s in ("the question", "the answer", "the reference"))
    assert "Ignore wording and length" in prompt


# ----- question generator ----------------------------------------------------------------------


def test_a_draft_question_links_back_to_its_chunk_and_awaits_review():
    llm = FakeLLM(['{"question": "  What is X?  ", "answer": " X is Y. "}'])
    chunk = ChunkRef(chunk_id="c9", text="X is Y, as everyone knows.", source_name="s.pdf", page=3)
    item = QuestionGenerator(llm).draft(chunk)
    assert item.status == "draft" and item.question == "What is X?"
    assert item.reference_answer == "X is Y." and item.relevant_chunk_ids == ["c9"]
    assert (item.evidence[0].chunk_id, item.evidence[0].source_name, item.evidence[0].page) == (
        "c9", "s.pdf", 3,
    )  # fmt: skip
    call = llm.calls[0]
    assert "X is Y, as everyone knows." in call.prompt
    assert "same language" in call.system and "Do not mention 'the passage'" in call.prompt
