"""The agent as a LangGraph graph.

    begin -> recall -> load_tools -> route --+--> retrieve ------------+
                                             |                         v
                                             +--> plan_tools <-+     answer -> respond -> remember
                                             |       |  ^      |   (generate <-> check: a subgraph)
                                             |       v  |      |         ^
                                             |    approve  collect_tools |
                                             |       |        ^          |
                                             |       +--run_tool (parallel, Send)
                                             +--> (direct) --------------+

Each node is a small function. Nodes that wait on the network (`load_tools`, `run_tool`) are
async. The others are plain functions, which LangGraph runs in worker threads.
"""

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, Overwrite, Send, interrupt

from agent_graph import prompts
from agent_graph.memory import FactMemory
from agent_graph.models import AgentOptions, ApprovalItem, PendingApproval
from agent_graph.protocols import Retriever, ToolProvider
from agent_graph.settings import AgentSettings
from agent_graph.state import AgentState, AnswerInput, AnswerOutput, AnswerState
from agent_graph.tracing import StepDetail
from common import AppError
from llm import LLM, tool_output_item, user_item
from observability import span

# Errors that mean every further model call would fail too: stop the run.
_FATAL = {"llm_not_configured", "llm_auth", "llm_model_not_found"}
_MARKER = re.compile(r"\[(\d+)\]")


@dataclass
class Deps:
    """Everything the graph needs from outside, injected so tests can use fakes."""

    llm: LLM  # always a real call (used for streaming)
    settings: AgentSettings
    cached_llm: LLM | None = None  # the same, with the response cache in front
    retriever: Retriever | None = None
    tools: ToolProvider | None = None
    memory: FactMemory | None = None

    def llm_for(self, options: AgentOptions) -> LLM:
        use = self.settings.use_cache if options.use_cache is None else options.use_cache
        return self.cached_llm if (use and self.cached_llm is not None) else self.llm


# ----- small helpers -------------------------------------------------------------------


def _options(state: AgentState) -> AgentOptions:
    return AgentOptions.model_validate(state.get("options") or {})


def _history(messages: list, turns: int, limit: int) -> list[tuple[str, str]]:
    """The earlier messages (not the latest user message), newest `turns` of them."""
    earlier = messages[:-1] if messages and isinstance(messages[-1], HumanMessage) else messages
    out = []
    for m in earlier[-turns:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        out.append((role, prompts.clip(str(m.content), limit)))
    return out


def _retrieval_key(state: AgentState, default_on: bool, version: str = "") -> str:
    """The node-cache key for `retrieve`. A run that has the cache off (by request, or by the
    setting when the request does not say) gets a key nobody else has, so it can never hit."""
    options = _options(state)
    use = default_on if options.use_cache is None else options.use_cache
    if not use:
        return uuid.uuid4().hex
    return json.dumps(
        {
            "q": state.get("standalone_question") or state.get("question"),
            "retrieval": options.retrieval.model_dump(),
            "data": version,  # changes when the documents change, so old results are not reused
        },
        sort_keys=True,
    )


def _merge_timings(state: AgentState, **values: float | None) -> dict[str, float]:
    merged = dict(state.get("timings") or {})
    merged.update({k: v for k, v in values.items() if v is not None})
    return merged


def _offered(state: AgentState) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            "strict": False,
        }
        for t in state.get("tool_defs", [])
    ]


def build_graph(deps: Deps) -> StateGraph:
    """The uncompiled graph. Compile it with a checkpointer, a store and a cache."""
    cfg = deps.settings
    detail = StepDetail(cfg)

    # ===== start of a turn =============================================================

    def retrieval_key(state: AgentState) -> str:
        """The retrieve node's cache key. It includes the retriever's data version (if it has
        one), so ingesting or deleting documents never lets stale passages come back."""
        version = ""
        fn = getattr(deps.retriever, "version", None)
        if fn is not None:
            try:
                version = str(fn(_options(state).retrieval))
            except Exception:  # cannot tell: do not trust the cache for this question
                version = uuid.uuid4().hex
        return _retrieval_key(state, cfg.use_cache, version)

    def begin(state: AgentState) -> dict[str, Any]:
        """Reset everything that belongs to one turn. The messages stay (they are the memory of
        the conversation). Lists with an `append` reducer are reset with `Overwrite`."""
        last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        question = str(last.content) if last else ""
        return {
            "question": question,
            "standalone_question": question,
            "route": None,
            "route_reason": "",
            "facts": [],
            "passages": [],
            "rewritten_queries": [],
            "tool_defs": [],
            "tool_items": [],
            "tool_rounds": 0,
            "pending_calls": [],
            "tool_results": Overwrite([]),
            "tool_calls": Overwrite([]),
            "tool_notes": "",
            "answer": None,
            "citations": [],
            "check": None,
            "retry_count": 0,
            "feedback": "",
            "warnings": Overwrite([]),
            "timings": {},
        }

    def recall(state: AgentState) -> dict[str, Any]:
        """Long-term memory: the facts about this user closest in meaning to the question."""
        user_id = state.get("user_id")
        if deps.memory is None or not user_id:
            return {}
        with span("agent.recall") as sp:
            try:
                found = deps.memory.search(
                    user_id, state["question"], cfg.memory_k, cfg.memory_min_score
                )
            except Exception as exc:  # memory is a help, never a reason to fail the answer
                return {"warnings": [f"memory unavailable ({type(exc).__name__})"]}
            sp.set_attribute("agent.facts", len(found))
            detail.value(sp, "agent.fact_texts", [f.text for f in found])
        return {"facts": [f.text for f in found]}

    async def load_tools(state: AgentState) -> dict[str, Any]:
        """Which tools may be offered to the model this turn."""
        options = _options(state)
        if deps.tools is None or not options.tools.enabled:
            return {}
        with span("agent.load_tools") as sp:
            try:
                available = await deps.tools.tools()
            except Exception as exc:
                return {"warnings": [f"tools unavailable ({type(exc).__name__})"]}
            allow = options.tools.allowlist
            chosen = [t for t in available if allow is None or t.name in allow]
            sp.set_attribute("agent.tools", len(chosen))
            sp.set_attribute("agent.tool_names", [t.name for t in chosen])
        return {"tool_defs": [t.model_dump() for t in chosen]}

    # ===== deciding what to do =========================================================

    def route(state: AgentState) -> dict[str, Any]:
        options = _options(state)
        can_retrieve = options.retrieval.enabled and deps.retriever is not None
        tool_lines = [f"{t['name']}: {t.get('description', '')[:160]}" for t in state["tool_defs"]]
        question = state["question"]

        with span("agent.route") as sp:
            if not can_retrieve and not tool_lines:
                decision = ("direct", question, "no document search or tools are enabled")
            else:
                history = _history(state["messages"], cfg.history_turns, 600)
                out = deps.llm_for(options).complete_json(
                    prompts.route_prompt(
                        question, history, state.get("facts", []), can_retrieve, tool_lines
                    ),
                    prompts.RouteDecision,
                    system=prompts.ROUTE_SYSTEM,
                )
                chosen = out.route
                # The model may pick something that is switched off: fall back sensibly.
                if chosen == "retrieve" and not can_retrieve:
                    chosen = "tool" if tool_lines else "direct"
                if chosen == "tool" and not tool_lines:
                    chosen = "retrieve" if can_retrieve else "direct"
                decision = (chosen, out.standalone_question.strip() or question, out.reason)
            sp.set_attributes(
                {
                    "agent.route": decision[0],
                    "agent.can_retrieve": can_retrieve,
                    "agent.tools_offered": len(tool_lines),
                }
            )
            detail.text(sp, "agent.reason", decision[2])
            detail.text(sp, "agent.standalone_question", decision[1])
        return {
            "route": decision[0],
            "standalone_question": decision[1],
            "route_reason": decision[2],
        }

    def after_route(state: AgentState) -> str:
        return {"retrieve": "retrieve", "tool": "plan_tools"}.get(state["route"] or "", "answer")

    # ===== searching the documents =====================================================

    def retrieve(state: AgentState) -> dict[str, Any]:
        options = _options(state)
        top_k = options.retrieval.top_k or cfg.top_k
        with span("agent.retrieve") as sp:
            try:
                got = deps.retriever.retrieve(
                    state["standalone_question"], options.retrieval, top_k
                )
            except AppError as exc:
                # A search that fails must not look like "nothing found". Say what happened.
                if exc.code in _FATAL:
                    raise
                sp.set_attribute("agent.error", exc.message)
                return {"warnings": [f"search failed: {exc.message}"]}
            sp.set_attributes(
                {
                    "agent.passage_count": len(got.passages),
                    "agent.mode": options.retrieval.mode,
                    "agent.rerank": options.retrieval.rerank,
                    "agent.top_k": top_k,
                    "agent.collection": options.retrieval.collection or "",
                }
            )
            detail.text(sp, "agent.query", state["standalone_question"])
            detail.value(sp, "agent.rewritten_queries", got.rewritten_queries)
            detail.passages(sp, [p.model_dump() for p in got.passages])
        return {
            "passages": [p.model_dump() for p in got.passages],
            "rewritten_queries": got.rewritten_queries,
            "timings": _merge_timings(state, retrieval_ms=got.retrieval_ms),
        }

    # ===== using tools =================================================================

    def plan_tools(state: AgentState) -> dict[str, Any]:
        """Ask the model which tools to call (or whether it is done)."""
        options = _options(state)
        items = list(state.get("tool_items") or [])
        if not items:
            history = _history(state["messages"], cfg.history_turns, 600)
            items = [
                user_item(
                    prompts.plan_prompt(
                        state["standalone_question"],
                        history,
                        state.get("facts", []),
                        options.retrieval.collection,
                    )
                )
            ]
        rounds = state.get("tool_rounds", 0)
        with span("agent.plan_tools", {"agent.round": rounds + 1}) as sp:
            turn = deps.llm_for(options).complete_with_tools(
                items, _offered(state), system=prompts.PLAN_SYSTEM
            )
            sp.set_attributes(
                {
                    "agent.calls": len(turn.calls),
                    "agent.call_names": [c.name for c in turn.calls],
                }
            )
            detail.value(sp, "agent.call_arguments", [{c.name: c.arguments} for c in turn.calls])
            detail.text(sp, "agent.model_note", turn.text)
        out: dict[str, Any] = {"tool_items": items + turn.output_items, "tool_notes": turn.text}
        calls = [c.model_dump() for c in turn.calls]
        pinned = options.retrieval.collection
        if pinned:
            # Domains stay apart: a tool that takes a `collection` may only use the active one,
            # whatever the model asked for. (The model is told this too, but it is enforced here.)
            takes = {
                t["name"]
                for t in state.get("tool_defs", [])
                if "collection" in (t.get("parameters") or {}).get("properties", {})
            }
            for c in calls:
                if c["name"] in takes and not c.get("argument_error"):
                    c["arguments"] = {**c["arguments"], "collection": pinned}
        if calls and rounds >= cfg.max_tool_rounds:
            out["warnings"] = [f"stopped after {cfg.max_tool_rounds} rounds of tool calls"]
            calls = []
        out["pending_calls"] = calls
        return out

    def after_plan(state: AgentState) -> str:
        return "approve" if state.get("pending_calls") else "answer"

    def needs_approval(tool: dict[str, Any] | None) -> str:
        """Why a human must approve this call, or '' if no one has to."""
        if tool is None or cfg.approval == "none":
            return ""
        if cfg.approval == "all":
            return "approval is required for every tool"
        if tool.get("read_only") is not True:
            return "the tool's server did not say it only reads data"
        return ""

    def approve(state: AgentState) -> dict[str, Any]:
        """Pause for a human on any call to a tool that could change something."""
        defs = {t["name"]: t for t in state.get("tool_defs", [])}
        calls = [dict(c) for c in state["pending_calls"]]
        asks = [
            ApprovalItem(
                call_id=c["call_id"],
                tool=c["name"],
                arguments=c["arguments"],
                reason=reason,
            )
            for c in calls
            if not c.get("argument_error") and (reason := needs_approval(defs.get(c["name"])))
        ]
        decisions: dict[str, bool] = {}
        if asks:
            # Pauses the run and saves it. It continues, from this node's start, when the caller
            # resumes the thread with a decision per call.
            reply = interrupt(PendingApproval(calls=asks).model_dump())
            # Always sent as {"decisions": {call_id: bool}}: LangGraph treats an empty resume
            # value as "nothing was sent", so the payload must never be a bare empty dict.
            decisions = (reply or {}).get("decisions", {})
        asked = {a.call_id for a in asks}
        for c in calls:
            c["approved"] = (
                bool(decisions.get(c["call_id"], False)) if c["call_id"] in asked else None
            )
        return {"pending_calls": calls}

    def fan_out(state: AgentState) -> list[Send]:
        """One `run_tool` per call, all at once."""
        offered = {t["name"] for t in state.get("tool_defs", [])}
        return [
            Send("run_tool", {"call": c, "offered": sorted(offered)})
            for c in state["pending_calls"]
        ]

    async def run_tool(payload: dict[str, Any]) -> dict[str, Any]:
        call, offered = payload["call"], set(payload["offered"])
        name, arguments = call["name"], call.get("arguments", {})
        record: dict[str, Any] = {
            "call_id": call["call_id"], "name": name, "arguments": arguments,
            "approved": call.get("approved"), "status": "ok", "content": "",
            "error_kind": None, "duration_ms": 0.0,
        }  # fmt: skip
        with span("agent.tool", {"agent.tool": name}) as sp:
            if call.get("argument_error"):
                record.update(status="invalid", error_kind="invalid_arguments")
                output = f"Error: {call['argument_error']}. Send the arguments as a JSON object."
            elif name not in offered:
                record.update(status="invalid", error_kind="unknown_tool")
                output = f"Error: there is no tool named '{name}'."
            elif call.get("approved") is False:
                record.update(status="declined", error_kind="declined")
                output = "The user declined to run this tool."
            else:
                outcome = await deps.tools.call(name, arguments)
                record.update(
                    status="ok" if outcome.ok else "error",
                    error_kind=outcome.error_kind,
                    duration_ms=outcome.duration_ms,
                )
                output = prompts.clip(outcome.content, cfg.tool_output_chars)
            record["content"] = output
            sp.set_attributes(
                {
                    "agent.tool_status": record["status"],
                    "agent.tool_error_kind": record["error_kind"] or "",
                    "agent.tool_approved": record["approved"],
                }
            )
            detail.value(sp, "agent.tool_arguments", arguments)
            detail.text(sp, "agent.tool_result", output)
        return {"tool_results": [{"call_id": call["call_id"], "output": output, "record": record}]}

    def collect_tools(state: AgentState) -> dict[str, Any]:
        """Put the results back in the model's own item format, in the order it asked."""
        by_id = {r["call_id"]: r for r in state.get("tool_results", [])}
        items = list(state["tool_items"])
        records = []
        for call in state["pending_calls"]:
            result = by_id.get(call["call_id"])
            if result is None:  # cannot happen, but the model must get an answer for every call
                continue
            items.append(tool_output_item(call["call_id"], result["output"]))
            records.append(result["record"])
        return {
            "tool_items": items,
            "tool_calls": records,
            "tool_results": Overwrite([]),
            "pending_calls": [],
            "tool_rounds": state.get("tool_rounds", 0) + 1,
        }

    # ===== writing and checking the answer (a subgraph) ================================

    def sources(state: AgentState) -> tuple[str, str]:
        context = prompts.context_block(state.get("passages", []), cfg.context_chars)
        tools = prompts.tool_results_block(state.get("tool_calls", []), cfg.tool_output_chars)
        return context, tools

    def generate(state: AgentState) -> dict[str, Any]:
        options = _options(state)
        history = _history(state["messages"], cfg.history_turns, 600)
        question = state["standalone_question"] or state["question"]
        retry = state.get("retry_count", 0)
        if state.get("route") == "direct":
            system = prompts.DIRECT_SYSTEM
            prompt = prompts.direct_prompt(state["question"], history, state.get("facts", []))
        else:
            context, tools = sources(state)
            system = prompts.ANSWER_SYSTEM
            prompt = prompts.answer_prompt(
                question, history, state.get("facts", []), context, tools, state.get("feedback", "")
            )

        with span("agent.generate", {"agent.attempt": retry + 1}) as sp:
            if state.get("stream"):
                writer = get_stream_writer()
                if retry:
                    writer({"type": "answer_replaced"})  # the text shown so far is being redone
                handle = deps.llm.stream(prompt, system=system)
                pieces = []
                for piece in handle:
                    pieces.append(piece)
                    writer({"type": "token", "text": piece})
                text = "".join(pieces)
                ttft = handle.ttft_ms if "ttft_ms" not in (state.get("timings") or {}) else None
            else:
                text = deps.llm_for(options).complete(prompt, system=system)
                ttft = None
            sp.set_attributes(
                {
                    "agent.answer_chars": len(text),
                    "agent.prompt_chars": len(prompt),
                    "agent.streamed": bool(state.get("stream")),
                }
            )
            detail.text(sp, "agent.feedback", state.get("feedback", "") if retry else "")
            detail.text(sp, "agent.answer", text.strip())
        return {"answer": text.strip(), "timings": _merge_timings(state, ttft_ms=ttft)}

    def check(state: AgentState) -> dict[str, Any]:
        options = _options(state)
        attempts = state.get("retry_count", 0) + 1
        context, tools = sources(state)
        if state.get("route") == "direct" or not options.check or not (context or tools):
            return {"check": {"passed": True, "unsupported_claims": [], "skipped": True,
                              "attempts": attempts}, "feedback": ""}  # fmt: skip
        with span("agent.check") as sp:
            try:
                out = deps.llm_for(options).complete_json(
                    prompts.check_prompt(state["question"], state["answer"] or "", context, tools),
                    prompts.CheckOutput,
                    system=prompts.CHECK_SYSTEM,
                )
            except AppError as exc:
                if exc.code in _FATAL:
                    raise
                return {
                    "check": {"passed": True, "unsupported_claims": [], "skipped": True,
                              "attempts": attempts},
                    "warnings": [f"the self-check could not run ({exc.code})"],
                    "feedback": "",
                }  # fmt: skip
            claims = [c.strip() for c in out.unsupported_claims if c.strip()]
            sp.set_attributes({"agent.unsupported": len(claims), "agent.passed": not claims})
            detail.value(sp, "agent.claims", claims)
        return {
            "check": {"passed": not claims, "unsupported_claims": claims, "skipped": False,
                      "attempts": attempts},
            "feedback": "\n".join(f"- {c}" for c in claims),
        }  # fmt: skip

    def after_check(state: AgentState) -> str:
        limit = _options(state).max_check_retries
        limit = cfg.max_check_retries if limit is None else limit
        failed = not state["check"]["passed"]
        return "retry" if failed and state.get("retry_count", 0) < limit else "done"

    def bump_retry(state: AgentState) -> dict[str, Any]:
        return {"retry_count": state.get("retry_count", 0) + 1}

    answer = StateGraph(AnswerState, input_schema=AnswerInput, output_schema=AnswerOutput)
    answer.add_node("generate", generate)
    answer.add_node("check", check)
    answer.add_node("bump_retry", bump_retry)
    answer.add_edge(START, "generate")
    answer.add_edge("generate", "check")
    answer.add_conditional_edges("check", after_check, {"retry": "bump_retry", "done": END})
    answer.add_edge("bump_retry", "generate")

    # ===== finishing ===================================================================

    def respond(state: AgentState) -> dict[str, Any]:
        text = state.get("answer") or ""
        passages = state.get("passages", [])
        citations, warnings, seen = [], [], set()
        with span("agent.respond") as sp:
            if state.get("route") != "direct":
                for number in (int(n) for n in _MARKER.findall(text)):
                    if number in seen:
                        continue
                    seen.add(number)
                    if 1 <= number <= len(passages):
                        p = passages[number - 1]
                        citations.append(
                            {
                                "marker": number,
                                "chunk_id": p["chunk_id"],
                                "source_name": p.get("source_name", ""),
                                "page": p.get("page"),
                            }
                        )
                    else:
                        warnings.append(
                            f"the answer cites [{number}], which is not a retrieved passage"
                        )
            sp.set_attributes({"agent.citations": len(citations), "agent.warnings": len(warnings)})
            detail.value(sp, "agent.citation_list", citations)
        if state.get("stream") and citations:
            writer = get_stream_writer()
            for c in citations:
                writer({"type": "citation", **c})
        # An explicit id ties this message to its run, so the saved details can be found again.
        run_id = state.get("run_id")
        message = AIMessage(content=text, id=f"ai-{run_id}" if run_id else None)
        return {"citations": citations, "warnings": warnings, "messages": [message]}

    def remember(state: AgentState) -> dict[str, Any]:
        """Optionally save durable facts the user stated. Off unless asked for."""
        options = _options(state)
        wanted = cfg.auto_remember if options.remember is None else options.remember
        user_id = state.get("user_id")
        if not wanted or deps.memory is None or not user_id:
            return {}
        with span("agent.remember") as sp:
            try:
                out = deps.llm_for(options).complete_json(
                    f"User message: {state['question']}",
                    prompts.FactsOutput,
                    system=prompts.REMEMBER_SYSTEM,
                )
                saved = [deps.memory.add(user_id, f) for f in out.facts[:5] if f.strip()]
            except AppError as exc:
                if exc.code in _FATAL:
                    raise
                return {"warnings": [f"could not save facts ({exc.code})"]}
            sp.set_attribute("agent.facts_saved", len(saved))
            detail.value(sp, "agent.facts", [f.text for f in saved])
        return {}

    # ===== the graph ===================================================================

    g = StateGraph(AgentState)
    g.add_node("begin", begin)
    g.add_node("recall", recall)
    g.add_node("load_tools", load_tools)
    g.add_node("route", route)
    g.add_node(
        "retrieve",
        retrieve,
        cache_policy=CachePolicy(key_func=retrieval_key, ttl=cfg.cache_ttl_s),
    )
    g.add_node("plan_tools", plan_tools)
    g.add_node("approve", approve)
    g.add_node("run_tool", run_tool)
    g.add_node("collect_tools", collect_tools)
    g.add_node("answer", answer.compile())
    g.add_node("respond", respond)
    g.add_node("remember", remember)

    g.add_edge(START, "begin")
    g.add_edge("begin", "recall")
    g.add_edge("recall", "load_tools")
    g.add_edge("load_tools", "route")
    g.add_conditional_edges("route", after_route, ["retrieve", "plan_tools", "answer"])
    g.add_edge("retrieve", "answer")
    g.add_conditional_edges("plan_tools", after_plan, ["approve", "answer"])
    g.add_conditional_edges("approve", fan_out, ["run_tool"])
    g.add_edge("run_tool", "collect_tools")
    g.add_edge("collect_tools", "plan_tools")
    g.add_edge("answer", "respond")
    g.add_edge("respond", "remember")
    g.add_edge("remember", END)
    return g
