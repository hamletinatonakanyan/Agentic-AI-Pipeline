import pytest
from agent_rig import Rig, run
from langchain_core.messages import AIMessage, HumanMessage

from agent_graph.models import RetrievalChoice, ToolChoice
from agent_graph.testing import Brain, FakeRetriever, FakeTools, calls_turn
from common import AppError

# ----- the three routes -----------------------------------------------------------------------


def test_with_nothing_enabled_the_agent_answers_directly_without_even_asking_the_planner():
    rig = Rig(Brain(answers=["Hello there!"]), retriever=False, tools=False)
    out = run(rig.say("hi"))
    assert out["route"] == "direct" and out["answer"] == "Hello there!"
    assert rig.brain.count["route"] == 0  # no planner call: there was nothing to choose between
    assert out["check"]["skipped"] and out["citations"] == []
    assert [type(m) for m in out["messages"]] == [HumanMessage, AIMessage]


def test_a_direct_route_chosen_by_the_planner_skips_search_and_check():
    rig = Rig(Brain(routes=["direct"], answers=["I am fine, thanks."]))
    out = run(rig.say("how are you?"))
    assert out["route"] == "direct" and rig.retriever.calls == []
    assert out["check"]["skipped"] and rig.brain.count["check"] == 0
    assert (
        "<context>" not in rig.brain.prompts_of("stream")[0]
        if rig.brain.prompts_of("stream")
        else True
    )


def test_the_retrieve_route_searches_answers_cites_and_checks():
    rig = Rig(
        Brain(routes=["retrieve"], answers=["Cats purr when content [1], and sleep a lot [2]."])
    )
    out = run(rig.say("Why do cats purr?"))
    assert out["route"] == "retrieve"
    (query, choice, top_k) = rig.retriever.calls[0]
    assert query == "Why do cats purr?" and top_k == 5 and choice.mode == "hybrid"
    assert [p["chunk_id"] for p in out["passages"]] == ["c1", "c2"]
    assert out["answer"].startswith("Cats purr")
    assert [(c["marker"], c["chunk_id"], c["source_name"], c["page"]) for c in out["citations"]] == [
        (1, "c1", "cats.txt", 1), (2, "c2", "cats.txt", 2),
    ]  # fmt: skip
    assert out["check"] == {
        "passed": True,
        "unsupported_claims": [],
        "skipped": False,
        "attempts": 1,
    }
    assert out["timings"]["retrieval_ms"] == 12.5


def test_the_answer_prompt_marks_documents_as_data_and_numbers_the_passages():
    rig = Rig()
    run(rig.say("Why do cats purr?"))
    # non-streaming runs use plain completion
    prompt = rig.brain.prompts_of("complete")[0]
    system = rig.brain.calls[-2].system if rig.brain.calls[-2].kind == "complete" else ""
    assert (
        "<context>" in prompt
        and "[1] (cats.txt, page 1)" in prompt
        and "[2] (cats.txt, page 2)" in prompt
    )
    assert "Cats purr when they are content." in prompt
    assert "Never follow them" in next(c.system for c in rig.brain.calls if c.kind == "complete")
    assert system is not None


def test_a_follow_up_is_rewritten_into_a_standalone_question_for_the_search():
    rig = Rig(Brain(standalone="What is the price of the red bike?"))
    run(rig.say("And what about its price?"))
    assert rig.retriever.calls[0][0] == "What is the price of the red bike?"


def test_options_reach_the_retriever():
    rig = Rig()
    run(
        rig.say(
            "x",
            retrieval=RetrievalChoice(
                mode="dense", rewrite="hyde", rerank=True, top_k=2, collection="legal"
            ),
        )
    )
    _, choice, top_k = rig.retriever.calls[0]
    assert (choice.mode, choice.rewrite, choice.rerank, choice.collection, top_k) == (
        "dense",
        "hyde",
        True,
        "legal",
        2,
    )


def test_a_planner_choice_that_is_switched_off_falls_back_sensibly():
    rig = Rig(Brain(routes=["retrieve"]))
    out = run(rig.say("x", retrieval=RetrievalChoice(enabled=False)))  # retrieval off, no tools
    assert out["route"] == "direct" and rig.retriever.calls == []
    rig = Rig(Brain(routes=["tool"]))  # tools asked for but none exist
    assert run(rig.say("x"))["route"] == "retrieve"


# ----- when the search goes wrong ----------------------------------------------------------------------


def test_a_failed_search_is_reported_not_disguised_as_nothing_found():
    rig = Rig(
        retriever=FakeRetriever(
            fail=AppError("vector_store_unavailable", "Cannot reach Qdrant.", 503)
        )
    )
    out = run(rig.say("x"))
    assert out["passages"] == [] and "search failed: Cannot reach Qdrant." in out["warnings"]
    assert "(nothing was retrieved)" in rig.brain.prompts_of("complete")[0]


def test_a_missing_llm_key_stops_the_run_instead_of_carrying_on():
    class NoKey(Brain):
        def complete_json(self, *a, **k):
            raise AppError("llm_not_configured", "OPENAI_API_KEY is not set.", 503)

    with pytest.raises(AppError) as err:
        run(Rig(NoKey()).say("x"))
    assert err.value.code == "llm_not_configured"


# ----- the self-check ------------------------------------------------------------------------------------


def test_a_failed_check_regenerates_with_the_feedback_and_then_passes():
    rig = Rig(Brain(
        answers=["Cats live 30 years [1].", "Cats purr when content [1]."],
        checks=[["Cats live 30 years"], []],
    ))  # fmt: skip
    out = run(rig.say("x"))
    assert out["answer"] == "Cats purr when content [1]." and out["retry_count"] == 1
    assert out["check"]["passed"] and out["check"]["attempts"] == 2
    second = rig.brain.prompts_of("complete")[1]
    assert "Cats live 30 years" in second and "previous answer had claims" in second


def test_when_every_attempt_fails_the_answer_is_still_delivered_with_the_problems_noted():
    rig = Rig(Brain(answers=["bad [1]"], checks=[["made up"]]), max_check_retries=2)
    out = run(rig.say("x"))
    assert rig.brain.count["answer"] == 3 and rig.brain.count["check"] == 3  # 1 + 2 retries
    assert out["check"] == {
        "passed": False,
        "unsupported_claims": ["made up"],
        "skipped": False,
        "attempts": 3,
    }
    assert out["answer"] == "bad [1]"


def test_a_request_can_change_or_switch_off_the_check():
    rig = Rig(Brain(checks=[["x"]]))
    assert run(rig.say("a", max_check_retries=0))["check"]["attempts"] == 1
    assert rig.brain.count["answer"] == 1  # no retry
    rig = Rig(Brain(checks=[["x"]]))
    out = run(rig.say("a", check=False))
    assert out["check"]["skipped"] and rig.brain.count["check"] == 0


def test_a_check_that_cannot_run_is_noted_and_does_not_block_the_answer():
    class FlakyCheck(Brain):
        def complete_json(self, prompt, schema, **k):
            if schema.__name__ == "CheckOutput":
                raise AppError("llm_invalid_output", "bad json", 502)
            return super().complete_json(prompt, schema, **k)

    out = run(Rig(FlakyCheck()).say("x"))
    assert (
        out["check"]["skipped"]
        and "the self-check could not run (llm_invalid_output)" in out["warnings"]
    )
    assert out["answer"]


def test_a_citation_that_does_not_exist_is_dropped_with_a_warning():
    out = run(Rig(Brain(answers=["Yes [1] and also [7]."])).say("x"))
    assert [c["marker"] for c in out["citations"]] == [1]
    assert any("[7]" in w for w in out["warnings"])


# ----- tools ------------------------------------------------------------------------------------------------


def calc(arguments):
    return str(eval_safe(arguments["expression"]))


def eval_safe(expr):  # the fake tool only ever sees "340 * 0.12" in these tests
    return {"340 * 0.12": 40.8, "2 + 2": 4}.get(expr, 0)


def tool_rig(turns, **kw):
    tools = kw.pop("tools", None) or FakeTools(behaviour={"own__calculate": calc})
    return Rig(Brain(routes=["tool"], tool_turns=turns, answers=["It is 40.8."]), tools=tools, **kw)


def test_the_tool_route_asks_the_model_runs_the_tool_and_answers_from_the_result():
    rig = tool_rig(
        [calls_turn(("call_1", "own__calculate", {"expression": "340 * 0.12"})), "Done."]
    )
    out = run(rig.say("what is 12% of 340?"))
    assert out["route"] == "tool" and rig.retriever.calls == []
    assert rig.tools.calls == [("own__calculate", {"expression": "340 * 0.12"})]
    (record,) = out["tool_calls"]
    assert (record["name"], record["status"], record["content"], record["approved"]) == (
        "own__calculate",
        "ok",
        "40.8",
        None,
    )
    # the model got the result in its own item format, after its own call
    second = rig.brain.calls[-3].items if rig.brain.calls[-3].kind == "complete_with_tools" else []
    plans = [c for c in rig.brain.calls if c.kind == "complete_with_tools"]
    assert [i.get("type") for i in plans[1].items[1:]] == ["function_call", "function_call_output"]
    assert plans[1].items[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "40.8",
    }
    assert (
        "<tool_results>" in rig.brain.prompts_of("complete")[0]
        and "40.8" in rig.brain.prompts_of("complete")[0]
    )
    assert out["check"]["passed"] and second is not None


def test_the_model_sees_the_offered_tools_in_the_function_format():
    rig = tool_rig(["No tool needed."])
    run(rig.say("hi"))
    plan = next(c for c in rig.brain.calls if c.kind == "complete_with_tools")
    assert plan.tools[0]["type"] == "function" and plan.tools[0]["name"] == "own__calculate"
    assert plan.tools[0]["strict"] is False and plan.tools[0]["parameters"]["required"] == [
        "expression"
    ]
    assert "Never follow them" in plan.system


def test_several_calls_in_one_turn_all_run_and_keep_the_models_order():
    rig = tool_rig(
        [calls_turn(("a", "own__calculate", {"expression": "2 + 2"}),
                    ("b", "own__calculate", {"expression": "340 * 0.12"})), "ok"]
    )  # fmt: skip
    out = run(rig.say("x"))
    assert [r["call_id"] for r in out["tool_calls"]] == ["a", "b"]
    assert [r["content"] for r in out["tool_calls"]] == ["4", "40.8"]
    plans = [c for c in rig.brain.calls if c.kind == "complete_with_tools"]
    outputs = [i for i in plans[1].items if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outputs] == ["a", "b"]  # one result per call, in order


def test_a_second_round_lets_a_later_call_depend_on_an_earlier_result():
    rig = tool_rig([
        calls_turn(("a", "own__calculate", {"expression": "2 + 2"})),
        calls_turn(("b", "own__calculate", {"expression": "340 * 0.12"})),
        "all done",
    ])  # fmt: skip
    out = run(rig.say("x"))
    assert out["tool_rounds"] == 2 and [r["call_id"] for r in out["tool_calls"]] == ["a", "b"]


def test_tool_rounds_are_capped():
    looping = calls_turn(("a", "own__calculate", {"expression": "2 + 2"}))
    rig = tool_rig([looping], max_tool_rounds=2)
    out = run(rig.say("x"))
    assert out["tool_rounds"] == 2 and len(out["tool_calls"]) == 2
    assert "stopped after 2 rounds of tool calls" in out["warnings"]
    assert out["answer"]  # it still answers, from what it has


def test_malformed_arguments_are_not_run_and_the_model_is_told():
    from llm import ToolCall, ToolTurn

    bad = ToolTurn(
        calls=[
            ToolCall(
                call_id="a",
                name="own__calculate",
                arguments={},
                argument_error="the arguments were not valid JSON",
            )
        ],
        output_items=[
            {
                "type": "function_call",
                "call_id": "a",
                "name": "own__calculate",
                "arguments": "{oops",
            }
        ],
    )
    rig = tool_rig([bad, "gave up"])
    out = run(rig.say("x"))
    assert rig.tools.calls == []  # never executed
    assert (
        out["tool_calls"][0]["status"] == "invalid"
        and out["tool_calls"][0]["error_kind"] == "invalid_arguments"
    )
    plans = [c for c in rig.brain.calls if c.kind == "complete_with_tools"]
    assert "not valid JSON" in plans[1].items[-1]["output"]


def test_a_tool_the_model_invented_is_refused_not_run():
    rig = tool_rig([calls_turn(("a", "own__delete_everything", {})), "ok"])
    out = run(rig.say("x"))
    assert rig.tools.calls == [] and out["tool_calls"][0]["error_kind"] == "unknown_tool"


def test_a_failing_tool_is_recorded_and_the_run_carries_on():
    from agent_graph.protocols import ToolOutcome

    tools = FakeTools(
        behaviour={
            "own__calculate": lambda a: ToolOutcome(
                ok=False, content="cannot divide by zero", error_kind="tool_error"
            )
        }
    )
    rig = tool_rig(
        [calls_turn(("a", "own__calculate", {"expression": "1/0"})), "sorry"], tools=tools
    )
    out = run(rig.say("x"))
    assert out["tool_calls"][0]["status"] == "error" and out["answer"]


def test_tools_can_be_switched_off_or_limited_per_request():
    rig = tool_rig(["No tool needed."])
    out = run(rig.say("x", tools=ToolChoice(enabled=False)))
    assert out["route"] == "retrieve" and out["tool_defs"] == []  # no tools to choose from
    rig = tool_rig(["No tool needed."])
    run(rig.say("x", tools=ToolChoice(allowlist=["something_else"])))
    assert out is not None and rig.brain.count["tool"] == 0  # nothing was offered, so no tool turn


def test_per_turn_fields_do_not_leak_into_the_next_turn():
    rig = tool_rig([calls_turn(("a", "own__calculate", {"expression": "2 + 2"})), "done"])
    first = run(rig.say("use the tool", thread="t"))
    assert len(first["tool_calls"]) == 1
    rig.brain._lists["route"] = ["direct"]
    rig.brain._i["route"] = 0
    second = run(rig.say("now just chat", thread="t"))
    assert second["tool_calls"] == [] and second["tool_rounds"] == 0 and second["warnings"] == []
    assert len(second["messages"]) == 4  # the conversation itself is kept


# ----- approval --------------------------------------------------------------------------------------------------


def risky_rig(approval="sensitive", read_only=None, **kw):
    from agent_graph.protocols import AgentTool

    tools = FakeTools(
        tools=[AgentTool(name="files__delete", description="Delete a file.", read_only=read_only,
                         parameters={"type": "object", "properties": {"path": {"type": "string"}}})],
        behaviour={"files__delete": lambda a: f"deleted {a['path']}"},
    )  # fmt: skip
    turns = [calls_turn(("c1", "files__delete", {"path": "/tmp/x"})), "done"]
    return Rig(Brain(routes=["tool"], tool_turns=turns), tools=tools, approval=approval, **kw)


def test_a_tool_not_marked_read_only_pauses_for_a_human():
    rig = risky_rig()
    out = run(rig.say("delete x"))
    assert "__interrupt__" in out
    asked = out["__interrupt__"][0].value
    assert asked["calls"][0]["tool"] == "files__delete" and asked["calls"][0]["arguments"] == {
        "path": "/tmp/x"
    }
    assert "did not say it only reads" in asked["calls"][0]["reason"]
    assert rig.tools.calls == []  # nothing ran yet
    assert run(rig.state()).next == ("approve",)


def test_approving_lets_the_tool_run_and_the_run_finish():
    rig = risky_rig()
    run(rig.say("delete x"))
    out = run(rig.resume({"decisions": {"c1": True}}))
    assert rig.tools.calls == [("files__delete", {"path": "/tmp/x"})]
    assert out["tool_calls"][0]["status"] == "ok" and out["tool_calls"][0]["approved"] is True
    assert out["answer"]


def test_declining_means_the_tool_never_runs_and_the_model_is_told():
    rig = risky_rig()
    run(rig.say("delete x"))
    out = run(rig.resume({"decisions": {"c1": False}}))
    assert rig.tools.calls == []
    assert (
        out["tool_calls"][0]["status"] == "declined" and out["tool_calls"][0]["approved"] is False
    )
    plans = [c for c in rig.brain.calls if c.kind == "complete_with_tools"]
    assert plans[1].items[-1]["output"] == "The user declined to run this tool."


def test_a_missing_decision_counts_as_no():
    rig = risky_rig()
    run(rig.say("delete x"))
    out = run(rig.resume({"decisions": {}}))
    assert rig.tools.calls == [] and out["tool_calls"][0]["status"] == "declined"


def test_read_only_tools_do_not_ask():
    rig = risky_rig(read_only=True)
    out = run(rig.say("delete x"))
    assert (
        "__interrupt__" not in out and rig.tools.calls and out["tool_calls"][0]["approved"] is None
    )


def test_the_approval_policy_can_be_everything_or_nothing():
    out = run(risky_rig(approval="all", read_only=True).say("x"))
    assert (
        "__interrupt__" in out
        and "every tool" in out["__interrupt__"][0].value["calls"][0]["reason"]
    )
    rig = risky_rig(approval="none")
    out = run(rig.say("x"))
    assert "__interrupt__" not in out and rig.tools.calls


def test_only_the_risky_calls_in_a_batch_ask():
    from agent_graph.protocols import AgentTool

    tools = FakeTools(
        tools=[
            AgentTool(name="own__calculate", read_only=True),
            AgentTool(name="files__delete", read_only=None),
        ],
        behaviour={"own__calculate": lambda a: "4", "files__delete": lambda a: "gone"},
    )
    turns = [
        calls_turn(
            ("a", "own__calculate", {"expression": "2 + 2"}), ("b", "files__delete", {"path": "p"})
        ),
        "ok",
    ]
    rig = Rig(Brain(routes=["tool"], tool_turns=turns), tools=tools)
    out = run(rig.say("x"))
    assert [c["call_id"] for c in out["__interrupt__"][0].value["calls"]] == ["b"]
    final = run(rig.resume({"decisions": {"b": False}}))
    assert [(r["call_id"], r["status"]) for r in final["tool_calls"]] == [
        ("a", "ok"),
        ("b", "declined"),
    ]
    assert rig.tools.calls == [("own__calculate", {"expression": "2 + 2"})]


# ----- memory --------------------------------------------------------------------------------------------------------


def test_recalled_facts_reach_the_prompts_only_for_their_own_user():
    rig = Rig(
        Brain(routes=["direct"], answers=["Hi Ana."]), memory=True, retriever=False, tools=False
    )
    rig.memory.add("ana", "Ana prefers short answers about tea")
    rig.memory.add("bob", "Bob lives in Berlin and likes tea")
    out = run(rig.say("tell me about tea", user_id="ana"))
    assert out["facts"] == ["Ana prefers short answers about tea"]
    assert (
        "<memory>" in rig.brain.prompts_of("complete")[0]
        and "Ana prefers" in rig.brain.prompts_of("complete")[0]
    )
    assert "Bob" not in str(rig.brain.calls)
    assert (
        run(rig.say("tell me about tea", user_id=None, thread="t2"))["facts"] == []
    )  # nobody to recall for


def test_a_broken_memory_is_a_warning_not_a_failure():
    rig = Rig(memory=True)

    def boom(*a, **k):
        raise RuntimeError("model download failed")

    rig.memory.search = boom
    out = run(rig.say("x", user_id="ana"))
    assert out["answer"] and "memory unavailable (RuntimeError)" in out["warnings"]


def test_facts_are_saved_only_when_asked():
    rig = Rig(Brain(facts=[["Ana works as a nurse"]]), memory=True)
    run(rig.say("I work as a nurse", user_id="ana"))
    assert rig.memory.all_facts("ana") == [] and rig.brain.count["facts"] == 0  # off by default

    run(rig.say("I work as a nurse", user_id="ana", thread="t2", remember=True))
    assert [f.text for f in rig.memory.all_facts("ana")] == ["Ana works as a nurse"]
    assert rig.memory.all_facts("bob") == []
    run(rig.say("I work as a nurse", user_id="ana", thread="t3", remember=True))
    assert len(rig.memory.all_facts("ana")) == 1  # the same fact twice is one fact


def test_the_setting_can_turn_remembering_on_for_everyone_and_a_request_can_still_refuse():
    rig = Rig(Brain(facts=[["a fact"]]), memory=True, auto_remember=True)
    run(rig.say("x", user_id="u"))
    assert len(rig.memory.all_facts("u")) == 1
    run(rig.say("y", user_id="u2", thread="t2", remember=False))
    assert rig.memory.all_facts("u2") == []


# ----- caching ------------------------------------------------------------------------------------------------------------


def test_identical_requests_reuse_cached_model_answers_and_the_cached_search():
    rig = Rig(cached=True)
    run(rig.say("Why do cats purr?", thread="a"))
    run(rig.say("Why do cats purr?", thread="b"))
    assert (rig.brain.count["route"], rig.brain.count["answer"], rig.brain.count["check"]) == (
        1,
        1,
        1,
    )
    assert len(rig.retriever.calls) == 1  # the retrieve node was cached too
    assert rig.cache.hits >= 3


def test_a_request_can_bypass_every_cache():
    rig = Rig(cached=True)
    run(rig.say("Why do cats purr?", thread="a", use_cache=False))
    run(rig.say("Why do cats purr?", thread="b", use_cache=False))
    assert (rig.brain.count["route"], rig.brain.count["answer"], rig.brain.count["check"]) == (
        2,
        2,
        2,
    )
    assert len(rig.retriever.calls) == 2


def test_the_setting_can_turn_caching_off_and_different_questions_do_not_share():
    rig = Rig(cached=True, use_cache=False)
    run(rig.say("Why do cats purr?", thread="a"))
    run(rig.say("Why do cats purr?", thread="b"))
    assert (
        rig.brain.count["route"] == 2 and len(rig.retriever.calls) == 2
    )  # search not cached either
    run(rig.say("Why do cats purr?", thread="c", use_cache=True))  # a request can opt back in
    run(rig.say("Why do cats purr?", thread="d", use_cache=True))
    assert len(rig.retriever.calls) == 3  # c searched, d reused c's search
    rig = Rig(cached=True)
    run(rig.say("Why do cats purr?", thread="a"))
    run(rig.say("How long do cats sleep?", thread="b"))
    assert rig.brain.count["route"] == 2 and len(rig.retriever.calls) == 2


# ----- streaming ----------------------------------------------------------------------------------------------------------------


def custom(events):
    return [chunk for _, mode, chunk in events if mode == "custom"]


def test_tokens_stream_in_order_and_rebuild_the_answer():
    rig = Rig(Brain(answers=["Cats purr when content [1]."]))
    events = run(rig.events("Why do cats purr?"))
    tokens = [c["text"] for c in custom(events) if c["type"] == "token"]
    assert "".join(tokens) == "Cats purr when content [1]." and len(tokens) > 3
    assert [c for c in custom(events) if c["type"] == "citation"][0]["chunk_id"] == "c1"
    assert rig.brain.count["answer"] == 1 and rig.brain.prompts_of(
        "stream"
    )  # streamed, not completed


def test_a_retry_tells_the_listener_to_discard_what_it_showed():
    rig = Rig(Brain(answers=["Wrong claim [1].", "Right claim [1]."], checks=[["Wrong claim"], []]))
    kinds = [c["type"] for c in custom(run(rig.events("x")))]
    first_token, replaced = kinds.index("token"), kinds.index("answer_replaced")
    assert first_token < replaced and kinds.count("answer_replaced") == 1
    assert kinds[replaced + 1] == "token"


def test_node_start_and_end_events_show_the_path_taken():
    events = run(Rig().events("x"))
    started = [
        chunk["name"] for _, mode, chunk in events if mode == "tasks" and "result" not in chunk
    ]
    assert started[:4] == ["begin", "recall", "load_tools", "route"]
    assert (
        "retrieve" in started
        and "respond" in started
        and "generate" in started
        and "check" in started
    )
    namespaces = {ns for ns, mode, _ in events if mode == "tasks"}
    assert () in namespaces and any(
        ns and ns[0].startswith("answer") for ns in namespaces
    )  # the subgraph


def test_ttft_is_recorded_for_streamed_answers_only():
    streamed = run(Rig().say("x", stream=True))
    assert streamed["timings"]["ttft_ms"] == 1.0
    assert "ttft_ms" not in run(Rig().say("x"))["timings"]


# ----- the conversation ---------------------------------------------------------------------------------------------------------


def test_the_second_turn_sees_the_first():
    rig = Rig(
        Brain(routes=["retrieve"], answers=["First answer [1].", "Second answer [1]."]),
        retriever=True,
    )
    run(rig.say("Tell me about cats", thread="t"))
    out = run(rig.say("and dogs?", thread="t"))
    assert [type(m).__name__ for m in out["messages"]] == ["HumanMessage", "AIMessage"] * 2
    route_prompt = rig.brain.prompts_of("complete_json")[2]  # the second route call
    assert (
        "User: Tell me about cats" in route_prompt
        and "Assistant: First answer [1]." in route_prompt
    )


def test_only_the_most_recent_turns_are_shown_to_the_model():
    rig = Rig(
        Brain(routes=["direct"], answers=[f"reply {i}" for i in range(12)]),
        retriever=False,
        tools=False,
        history_turns=2,
    )
    for i in range(6):
        run(rig.say(f"message {i}", thread="t"))
    last = rig.brain.prompts_of("complete")[-1]
    assert "message 4" in last and "reply 4" in last
    assert "message 0" not in last and "reply 0" not in last


def test_threads_are_separate_conversations():
    rig = Rig(Brain(routes=["direct"]), retriever=False, tools=False)
    run(rig.say("I am thread one", thread="one"))
    out = run(rig.say("hello", thread="two"))
    assert len(out["messages"]) == 2
    assert "thread one" not in rig.brain.prompts_of("complete")[-1]


def test_cached_search_results_are_not_reused_after_the_documents_change():
    rig = Rig(cached=True)
    run(rig.say("Why do cats purr?", thread="a"))
    run(rig.say("Why do cats purr?", thread="b"))
    assert len(rig.retriever.calls) == 1  # the second question reused the first search
    rig.retriever.data_version = "2"  # someone ingested into, or deleted, a collection
    run(rig.say("Why do cats purr?", thread="c"))
    assert len(rig.retriever.calls) == 2  # searched again, on the new data


def test_when_the_data_version_cannot_be_read_the_search_cache_is_not_trusted():
    rig = Rig(cached=True)
    rig.retriever.version_error = RuntimeError("qdrant is down")
    run(rig.say("Why do cats purr?", thread="a"))
    run(rig.say("Why do cats purr?", thread="b"))
    assert len(rig.retriever.calls) == 2
