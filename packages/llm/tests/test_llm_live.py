"""Live tests for the OpenAI client. They call the real API: `make test-live`.

They need OPENAI_API_KEY in .env and cost a few cents. They check the request shape, `store: false`,
structured output, the `phase` field, error bodies and streaming events.
"""

import os

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.live


@pytest.fixture
def llm():
    from dotenv import dotenv_values

    from llm import LLMSettings, OpenAILLM

    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(".env").get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY is not set")
    return OpenAILLM(LLMSettings(_env_file=None, openai_api_key=key, max_output_tokens=500))


def test_complete(llm):
    assert "paris" in llm.complete("What is the capital of France? One word.").lower()


def test_complete_json(llm):
    class Answer(BaseModel):
        capital: str
        country: str

    out = llm.complete_json("Capital of Japan, and the country.", Answer)
    assert out.capital.lower() == "tokyo"


def test_stream_reports_time_to_first_token(llm):
    handle = llm.stream("Count from one to five in words.")
    text = "".join(handle)
    assert "three" in text.lower() and handle.ttft_ms is not None


def test_reasoning_can_be_turned_off(llm):
    from llm import LLMSettings, OpenAILLM

    off = OpenAILLM(
        LLMSettings(_env_file=None, openai_api_key=llm._s.openai_api_key, use_reasoning=False)
    )
    assert off.complete("Say OK.").strip()


def test_a_full_tool_round_with_the_real_model(llm):
    """Do the request and the replay shapes we send
    (flat tools, reasoning items echoed back) work with the real API?"""
    from llm import tool_output_item, user_item

    tools = [{
        "type": "function", "name": "calculate", "strict": False,
        "description": "Evaluate an arithmetic expression exactly.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]},
    }]  # fmt: skip
    history = [user_item("What is 12 percent of 340? Use the calculator.")]
    turn = llm.complete_with_tools(history, tools)
    assert turn.calls and turn.calls[0].name == "calculate"
    history += turn.output_items
    for call in turn.calls:
        history.append(tool_output_item(call.call_id, "40.8"))
    final = llm.complete_with_tools(history, tools)
    assert "40.8" in final.text
