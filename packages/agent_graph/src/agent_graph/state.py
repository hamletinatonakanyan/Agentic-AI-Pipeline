"""The agent's state: one shared notebook that every node reads and writes.

A field with a reducer is combined with each write instead of replaced. `messages` appends (and
understands message ids), and `tool_results` appends within a round. Fields that must start fresh
on every user message are reset by the `begin` node, using `Overwrite` for the list that has a
reducer.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # ----- the conversation (kept for the whole thread) --------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]

    # ----- this turn --------------------------------------------------------------------------
    run_id: str
    user_id: str | None
    options: dict[str, Any]  # AgentOptions, as a dict
    question: str  # the user's latest message
    standalone_question: str  # the same question, rewritten so it makes sense on its own
    route: str | None  # "retrieve" | "tool" | "direct"
    route_reason: str
    facts: list[str]  # long-term memory recalled for this question
    passages: list[dict[str, Any]]  # replaced on each retrieval
    rewritten_queries: list[str]

    stream: bool  # stream the answer's tokens (and so do not use the response cache for it)
    tool_defs: list[dict[str, Any]]  # the tools offered this turn: name, description, parameters

    # ----- the tool conversation -------------------------------------------------------
    tool_items: list[dict[str, Any]]  # the model's tool conversation, in the API's item format
    tool_rounds: int
    pending_calls: list[dict[str, Any]]  # calls the model asked for in this round
    tool_results: Annotated[list[dict[str, Any]], operator.add]  # filled in parallel, per round
    tool_calls: Annotated[list[dict[str, Any]], operator.add]  # a record of every call this turn
    tool_notes: str  # what the model said while using tools

    # ----- the answer ------------------------------------------------------------------
    answer: str | None
    citations: list[dict[str, Any]]
    check: dict[str, Any] | None
    retry_count: int
    feedback: str  # what the self-check wants fixed, for the next attempt
    warnings: Annotated[list[str], operator.add]
    timings: dict[str, float]


# ----- the answer subgraph: narrow schemas on purpose ----------------------------------
#
# A subgraph hands back every key of its output schema, and the parent applies each key's reducer.
# If the subgraph shared the parent's full schema, `tool_calls` (an append list) would be handed
# back unchanged and appended AGAIN, duplicating every record. So the subgraph reads only what it
# needs and returns only what it produces.


class AnswerInput(TypedDict, total=False):
    # A key shared with the parent must have the SAME type and reducer as the parent's, or the
    # graph refuses to build. These two are read here and, being absent from AnswerOutput, never
    # handed back.
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    standalone_question: str
    route: str | None
    facts: list[str]
    passages: list[dict[str, Any]]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    options: dict[str, Any]
    stream: bool
    retry_count: int
    feedback: str
    timings: dict[str, float]


class AnswerOutput(TypedDict, total=False):
    answer: str | None
    check: dict[str, Any] | None
    retry_count: int
    feedback: str
    timings: dict[str, float]
    warnings: Annotated[list[str], operator.add]  # starts empty here, so only NEW ones go back


class AnswerState(AnswerInput, AnswerOutput):
    """Everything the subgraph itself uses, including what stays private to it."""

    attempts: list[dict[str, Any]]  # private: the draft and verdict of each attempt
