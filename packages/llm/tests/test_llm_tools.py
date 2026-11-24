import json

import httpx
import pytest

from common import AppError
from llm import (
    LLMSettings,
    OpenAILLM,
    ToolTurn,
    build_tools_request,
    tool_output_item,
    user_item,
)
from llm.settings import FORBIDDEN_PARAMS
from llm.testing import FakeLLM
from observability import init_observability, shutdown_observability
from observability.exporters import InMemoryExporter
from observability.settings import ObsSettings

KEY = "sk-test-key-0123456789abcdef"

TOOLS = [
    {
        "type": "function",
        "name": "own__calculate",
        "description": "Evaluate arithmetic.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        "strict": False,
    }
]


def settings(**kw) -> LLMSettings:
    return LLMSettings(_env_file=None, openai_api_key=KEY, **kw)


def call_item(call_id, name="own__calculate", arguments='{"expression": "1+1"}', **extra):
    return {
        "type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": name,
        "arguments": arguments, **extra,
    }  # fmt: skip


def message(text, phase=None):
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    if phase:
        item["phase"] = phase
    return item


def reasoning(n=1):
    return {"type": "reasoning", "id": f"rs_{n}", "summary": [], "encrypted_content": "ENC" * 5}


def reply(*items, **top):
    return {
        "status": "completed", "output": list(items),
        "usage": {"input_tokens": 20, "output_tokens": 9, "output_tokens_details": {"reasoning_tokens": 4}},
        **top,
    }  # fmt: skip


class Server:
    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return httpx.Response(200, json=answer)


def llm_with(server, **kw) -> OpenAILLM:
    return OpenAILLM(
        settings(**kw),
        client=httpx.Client(transport=httpx.MockTransport(server)),
        sleep=lambda _: None,
    )


@pytest.fixture(autouse=True)
def _obs_reset():
    shutdown_observability()
    yield
    shutdown_observability()


# ----- the request ------------------------------------------------------------------------------


def test_the_request_offers_flat_tools_and_the_history_in_order():
    items = [user_item("what is 1+1?")]
    body = build_tools_request(settings(), items, TOOLS, system="be brief", max_output_tokens=50)
    assert body["input"] == [{"role": "system", "content": "be brief"}, user_item("what is 1+1?")]
    assert body["tools"] == TOOLS and body["tools"][0]["strict"] is False
    assert "function" not in body["tools"][0]  # flat, no nested wrapper
    assert body["store"] is False and body["reasoning"] == {"effort": "low"}
    assert body["max_output_tokens"] == 50
    assert "tool_choice" not in body and "parallel_tool_calls" not in body  # defaults, not guessed


@pytest.mark.parametrize("use_reasoning", [True, False])
def test_tool_requests_never_contain_sampling_parameters(use_reasoning):
    body = build_tools_request(settings(use_reasoning=use_reasoning), [user_item("x")], TOOLS)
    assert not set(body) & set(FORBIDDEN_PARAMS)
    assert body["reasoning"]["effort"] == ("low" if use_reasoning else "none")


def test_the_input_list_is_copied_not_modified():
    items = [user_item("x")]
    build_tools_request(settings(), items, TOOLS, system="s")
    assert items == [user_item("x")]


# ----- reading what the model said ------------------------------------------------------------------


def test_a_plain_answer_is_text_with_no_calls():
    turn = llm_with(Server(reply(message("It is 2.")))).complete_with_tools([user_item("x")], TOOLS)
    assert turn.text == "It is 2." and turn.calls == []
    assert turn.output_items == [message("It is 2.")]


def test_a_tool_call_is_parsed_with_its_arguments():
    server = Server(
        reply(reasoning(), call_item("call_1", arguments='{"expression": "340 * 0.12"}'))
    )
    turn = llm_with(server).complete_with_tools([user_item("12% of 340?")], TOOLS)
    assert turn.text == ""  # a turn that only calls tools has no text, which is not an error
    assert [(c.call_id, c.name, c.arguments, c.argument_error) for c in turn.calls] == [
        ("call_1", "own__calculate", {"expression": "340 * 0.12"}, "")
    ]


def test_several_calls_in_one_turn_are_all_returned_in_order():
    server = Server(
        reply(
            call_item("call_a", arguments='{"expression": "1+1"}'),
            call_item("call_b", name="own__search_corpus", arguments='{"query": "cats"}'),
            call_item("call_c", arguments="{}"),
        )
    )
    turn = llm_with(server).complete_with_tools([user_item("x")], TOOLS)
    assert [c.call_id for c in turn.calls] == ["call_a", "call_b", "call_c"]
    assert turn.calls[1].name == "own__search_corpus" and turn.calls[2].arguments == {}


def test_text_and_calls_can_arrive_together():
    server = Server(reply(message("Let me check."), call_item("call_1")))
    turn = llm_with(server).complete_with_tools([user_item("x")], TOOLS)
    assert turn.text == "Let me check." and len(turn.calls) == 1


@pytest.mark.parametrize(
    ("arguments", "why"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('"just a string"', "must be a JSON object"),
        ("42", "must be a JSON object"),
    ],
)
def test_malformed_arguments_are_flagged_so_the_caller_can_tell_the_model(arguments, why):
    server = Server(reply(call_item("call_1", arguments=arguments)))
    (call,) = llm_with(server).complete_with_tools([user_item("x")], TOOLS).calls
    assert call.arguments == {} and why in call.argument_error


def test_empty_arguments_mean_no_arguments_not_an_error():
    server = Server(reply(call_item("call_1", arguments="")))
    (call,) = llm_with(server).complete_with_tools([user_item("x")], TOOLS).calls
    assert call.arguments == {} and call.argument_error == ""


def test_the_working_notes_message_does_not_become_the_answer():
    server = Server(
        reply(message("notes...", phase="commentary"), message("Final.", phase="final_answer"))
    )
    assert llm_with(server).complete_with_tools([user_item("x")], TOOLS).text == "Final."


def test_a_call_without_an_id_or_name_is_a_protocol_error():
    for broken in ({"type": "function_call", "name": "x", "arguments": "{}"},
                   {"type": "function_call", "call_id": "c", "arguments": "{}"}):  # fmt: skip
        with pytest.raises(AppError) as err:
            llm_with(Server(reply(broken))).complete_with_tools([user_item("x")], TOOLS)
        assert err.value.code == "llm_bad_response"


def test_a_turn_with_neither_text_nor_calls_is_an_error():
    with pytest.raises(AppError) as err:
        llm_with(Server(reply())).complete_with_tools([user_item("x")], TOOLS)
    assert err.value.code == "llm_empty"


def test_refusals_and_cut_off_answers_are_still_errors():
    refusal = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": "no"}],
    }
    with pytest.raises(AppError) as err:
        llm_with(Server(reply(refusal))).complete_with_tools([user_item("x")], TOOLS)
    assert err.value.code == "llm_refused"
    cut = reply(
        call_item("c"), status="incomplete", incomplete_details={"reason": "max_output_tokens"}
    )
    with pytest.raises(AppError) as err:
        llm_with(Server(cut)).complete_with_tools([user_item("x")], TOOLS)
    assert err.value.code == "llm_incomplete"


# ----- continuing the conversation ------------------------------------------------------------------


def test_a_full_tool_round_replays_everything_the_model_said_then_the_results():
    first = reply(reasoning(1), call_item("call_1", arguments='{"expression": "340 * 0.12"}'))
    server = Server(first, reply(message("It is 40.8.")))
    llm = llm_with(server)

    history = [user_item("what is 12% of 340?")]
    turn = llm.complete_with_tools(history, TOOLS, system="sys")
    history += turn.output_items  # reasoning and the call, verbatim
    for call in turn.calls:
        history.append(tool_output_item(call.call_id, "40.8"))
    final = llm.complete_with_tools(history, TOOLS, system="sys")

    assert final.text == "It is 40.8." and final.calls == []
    sent = server.requests[1]["input"]
    assert [i.get("type") or i.get("role") for i in sent] == [
        "system", "user", "reasoning", "function_call", "function_call_output",
    ]  # fmt: skip
    assert sent[2] == reasoning(1)  # the encrypted reasoning item went back unchanged
    assert sent[4] == {"type": "function_call_output", "call_id": "call_1", "output": "40.8"}


def test_every_call_gets_its_own_output_item():
    assert tool_output_item("call_9", "result") == {
        "type": "function_call_output", "call_id": "call_9", "output": "result",
    }  # fmt: skip
    assert user_item("hi") == {"role": "user", "content": "hi"}


# ----- cross-cutting rules still hold ---------------------------------------------------------------------


def test_a_tool_turn_is_a_span_with_usage():
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    llm_with(Server(reply(call_item("c")))).complete_with_tools([user_item("x")], TOOLS)
    span = next(s for s in exporter.spans if s.name == "llm.complete_with_tools")
    assert span.attributes["gen_ai.usage.input_tokens"] == 20
    assert span.attributes["llm.reasoning_tokens"] == 4 and KEY not in str(span.model_dump())


def test_a_missing_key_fails_before_any_request():
    server = Server(reply(message("x")))
    llm = OpenAILLM(
        LLMSettings(_env_file=None, openai_api_key=""),
        client=httpx.Client(transport=httpx.MockTransport(server)),
    )
    with pytest.raises(AppError) as err:
        llm.complete_with_tools([user_item("x")], TOOLS)
    assert err.value.code == "llm_not_configured" and server.requests == []


def test_api_errors_use_the_same_clear_codes():
    llm = OpenAILLM(
        settings(),
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={}))),
    )
    with pytest.raises(AppError) as err:
        llm.complete_with_tools([user_item("x")], TOOLS)
    assert err.value.code == "llm_auth"


# ----- the fake ----------------------------------------------------------------------------------------------


def test_fake_llm_plays_a_scripted_tool_conversation_and_records_it():
    from llm import ToolCall

    fake = FakeLLM(
        tool_script=[
            ToolTurn(calls=[ToolCall(call_id="c1", name="t", arguments={"a": 1})]),
            "All done.",
        ]
    )
    first = fake.complete_with_tools([user_item("go")], TOOLS, system="s")
    assert first.calls[0].call_id == "c1"
    second = fake.complete_with_tools([user_item("go"), tool_output_item("c1", "ok")], TOOLS)
    assert second.text == "All done." and second.calls == []
    assert [c.kind for c in fake.calls] == ["complete_with_tools"] * 2
    assert fake.calls[0].system == "s" and fake.calls[0].tools == TOOLS
    assert fake.calls[1].items[-1]["type"] == "function_call_output"
    with pytest.raises(AssertionError, match="ran out"):
        fake.complete_with_tools([], TOOLS)


def test_fake_llm_without_a_tool_script_says_so():
    with pytest.raises(AssertionError, match="no tool_script"):
        FakeLLM().complete_with_tools([], TOOLS)


def test_the_reasoning_override_changes_effort_for_every_settings_object_until_cleared():
    from llm import LLMSettings, set_reasoning_override

    a, b = LLMSettings(_env_file=None, use_reasoning=True), LLMSettings(_env_file=None)
    try:
        assert a.effort == "low"
        set_reasoning_override(False)
        assert a.effort == "none" and b.effort == "none"
        set_reasoning_override(True)
        assert LLMSettings(_env_file=None, use_reasoning=False).effort == "low"
    finally:
        set_reasoning_override(None)
    assert a.effort == "low" and LLMSettings(_env_file=None, use_reasoning=False).effort == "none"
