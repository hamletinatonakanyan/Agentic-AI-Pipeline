"""The contract: options, responses, events, and the pieces the UI shows."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

RouteName = Literal["retrieve", "tool", "direct"]
SearchMode = Literal["dense", "sparse", "hybrid"]
RewriteStrategy = Literal["none", "multi_query", "hyde", "step_back"]


# ----- what a caller can choose --------------------------------------------------------


class RetrievalChoice(BaseModel):
    enabled: bool = True
    collection: str | None = None
    mode: SearchMode = "hybrid"
    rewrite: RewriteStrategy = "none"
    rerank: bool = False
    top_k: int | None = Field(default=None, ge=1, le=50)


class ToolChoice(BaseModel):
    enabled: bool = True
    # Names as the model sees them (for example `own__calculate`). None = every allowed tool.
    allowlist: list[str] | None = None


class AgentOptions(BaseModel):
    retrieval: RetrievalChoice = Field(default_factory=RetrievalChoice)
    tools: ToolChoice = Field(default_factory=ToolChoice)
    use_cache: bool | None = None  # None = the setting. Evals should pass False.
    check: bool = True  # verify the answer against its sources
    remember: bool | None = None  # None = the setting (off by default)
    max_check_retries: int | None = Field(default=None, ge=0, le=5)


class MessageRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str | None = Field(default=None, max_length=100)
    options: AgentOptions = Field(default_factory=AgentOptions)


class ResumeRequest(BaseModel):
    # Either one answer for every pending call, or a decision per call id.
    approve: bool | None = None
    decisions: dict[str, bool] | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "ResumeRequest":
        if (self.approve is None) == (self.decisions is None):
            raise ValueError("send either `approve` or `decisions`, not both and not neither")
        return self


class ForkRequest(BaseModel):
    checkpoint_id: str
    message: str | None = Field(default=None, min_length=1)  # replace the user's message
    options: AgentOptions = Field(default_factory=AgentOptions)


class ThreadCreate(BaseModel):
    user_id: str | None = Field(default=None, max_length=100)
    title: str | None = Field(default=None, max_length=200)


class FactCreate(BaseModel):
    text: str = Field(min_length=1, max_length=500)


# ----- what comes back -----------------------------------------------------------------


class Passage(BaseModel):
    chunk_id: str
    text: str
    source_name: str = ""
    page: int | None = None
    score: float = 0.0


class Citation(BaseModel):
    marker: int  # the [n] in the answer
    chunk_id: str
    source_name: str = ""
    page: int | None = None


class ToolCallRecord(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "declined", "invalid"]
    content: str = ""  # what the tool returned (as given to the model, possibly trimmed)
    error_kind: str | None = None
    duration_ms: float = 0.0
    approved: bool | None = None  # None = no approval was needed


class CheckResult(BaseModel):
    passed: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    skipped: bool = False  # nothing to check against (a direct answer), or checking was off
    attempts: int = 1


class ApprovalItem(BaseModel):
    call_id: str
    tool: str
    arguments: dict[str, Any]
    reason: str  # why a human is asked, for example "the server did not mark it read-only"


class PendingApproval(BaseModel):
    interrupt_id: str | None = None
    calls: list[ApprovalItem]


class Timings(BaseModel):
    ttft_ms: float | None = None  # time to the first token of the answer
    retrieval_ms: float | None = None
    total_ms: float = 0.0


class AgentResponse(BaseModel):
    thread_id: str
    run_id: str
    status: Literal["completed", "interrupted"] = "completed"
    answer: str | None = None
    route: RouteName | None = None
    citations: list[Citation] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    check: CheckResult | None = None
    interrupt: PendingApproval | None = None
    timings: Timings = Field(default_factory=Timings)
    checkpoint_id: str | None = None
    trace_id: str | None = None
    warnings: list[str] = Field(default_factory=list)


# ----- events, threads, state, memory --------------------------------------------------

EventType = Literal[
    "run_started",
    "node_start",
    "node_end",
    "token",
    "answer_replaced",  # a failed self-check is being retried: discard the text shown so far
    "tool_call",
    "citation",
    "interrupt",
    "done",
    "error",
]


class Event(BaseModel):
    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


class ThreadInfo(BaseModel):
    thread_id: str
    user_id: str | None = None
    title: str = ""
    created_at: float
    updated_at: float
    messages: int = 0
    collection: str | None = None  # the domain this conversation last searched


class RunRecord(BaseModel):
    """Everything about one answer that the chat shows again when a conversation is reopened."""

    question: str = ""
    options: dict[str, Any] = Field(default_factory=dict)
    response: "AgentResponse"
    created_at: float = 0.0


class MessageView(BaseModel):
    id: str | None = None
    role: Literal["user", "assistant"]
    content: str
    run: RunRecord | None = None  # for an answer: its sources, tool calls, timings, trace id


class ThreadDetail(ThreadInfo):
    history: list[MessageView] = Field(default_factory=list)


class StateView(BaseModel):
    thread_id: str
    checkpoint_id: str | None
    next: list[str]
    values: dict[str, Any]
    created_at: str | None = None
    interrupts: list[PendingApproval] = Field(default_factory=list)


class HistoryItem(BaseModel):
    checkpoint_id: str
    parent_checkpoint_id: str | None = None
    step: int | None = None
    source: str | None = None  # "input", "loop", "update", "fork"
    next: list[str]
    created_at: str | None = None
    summary: str = ""  # a short description for the UI


class Fact(BaseModel):
    id: str
    text: str
    created_at: float | None = None
    score: float | None = None


class GraphNode(BaseModel):
    id: str
    label: str
    kind: Literal["start", "end", "node", "subgraph"] = "node"


class GraphEdge(BaseModel):
    source: str
    target: str
    conditional: bool = False


class GraphInfo(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    mermaid: str


class AgentStatus(BaseModel):
    checkpoint_backend: str
    data_dir: str
    approval: str
    max_check_retries: int
    max_tool_rounds: int
    use_cache: bool
    auto_remember: bool
    memory_embed_model: str
    tools_available: int


class ReasoningSwitch(BaseModel):
    use_reasoning: bool


class ReasoningState(BaseModel):
    use_reasoning: bool  # what is in effect now
    effort: str  # "low" or "none"
    overridden: bool  # True when set from the UI (lasts until restart)
