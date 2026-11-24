"""Live tests for the judge: `make test-live`.

They need OPENAI_API_KEY in .env and cost a few cents. They run the judge prompts against a real
model. Read the printed results too: a judge that scores everything 1.0 is not working.
"""

import os

import pytest

pytestmark = pytest.mark.live


@pytest.fixture
def judge():
    from dotenv import dotenv_values

    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(".env").get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY is not set")
    from evals.judge import LLMJudge
    from llm import LLMSettings, OpenAILLM

    settings = LLMSettings(_env_file=None, openai_api_key=key)
    return LLMJudge(OpenAILLM(settings), settings.model)


PASSAGE = "The Eiffel Tower is 330 metres tall and was completed in 1889 for the World's Fair."


def test_faithfulness_separates_supported_from_invented_claims(judge):
    grounded = judge.score_faithfulness("The Eiffel Tower was completed in 1889.", [PASSAGE])
    invented = judge.score_faithfulness(
        "The Eiffel Tower was designed by Leonardo da Vinci.", [PASSAGE]
    )
    assert grounded.score == 1.0
    assert invented.score == 0.0


def test_completeness_notices_a_missing_point(judge):
    full = judge.score_completeness(
        "It is 330 metres tall and opened in 1889.", "330 metres tall; completed in 1889."
    )
    partial = judge.score_completeness(
        "It is 330 metres tall.", "330 metres tall; completed in 1889."
    )
    assert full.score > partial.score


def test_accuracy_prefers_the_correct_answer(judge):
    right = judge.score_accuracy("How tall is it?", "About 330 metres.", "330 metres")
    wrong = judge.score_accuracy("How tall is it?", "About 30 metres.", "330 metres")
    assert right.score > 0.7 and wrong.score < 0.4


def test_a_drafted_question_makes_sense_without_the_passage(judge):
    from evals.generator import QuestionGenerator
    from evals.protocols import ChunkRef

    item = QuestionGenerator(judge._llm).draft(ChunkRef(chunk_id="c1", text=PASSAGE))
    assert item.question.endswith("?") and "passage" not in item.question.lower()
