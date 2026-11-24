"""AgentService: run, stream, resume and fork conversations, and read their state and history.

One conversation is one `thread_id`. Every step of every run is saved by the checkpointer, so a
conversation survives a restart, a paused run can be resumed days later, and any past step can be
used as the start of a new branch.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agent_graph.memory import FactMemory
from agent_graph.models import (
    AgentOptions,
    AgentResponse,
    AgentStatus,
    ApprovalItem,
    CheckResult,
    Citation,
    Event,
    Fact,
    ForkRequest,
    GraphEdge,
    GraphInfo,
    GraphNode,
    HistoryItem,
    MessageRequest,
    MessageView,
    Passage,
    PendingApproval,
    ResumeRequest,
    RunRecord,
    StateView,
    ThreadCreate,
    ThreadDetail,
    ThreadInfo,
    Timings,
    ToolCallRecord,
)
from agent_graph.protocols import ToolProvider
from agent_graph.settings import AgentSettings
from agent_graph.threads import USER_ID, ThreadStore
from agent_graph.tracing import INPUT, OUTPUT, SESSION, StepDetail
from common import AppError
from observability import current_trace_id, span

_CUSTOM = {"token", "answer_replaced", "citation"}


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


class AgentService:
    def __init__(
        self,
        settings: AgentSettings,
        graph: Any,  # the compiled graph
        saver: Any,
        threads: ThreadStore,
        memory: FactMemory | None,
        tools: ToolProvider | None = None,
    ) -> None:
        self._s = settings
        self._graph = graph
        self._saver = saver
        self._threads = threads
        self._memory = memory
        self._tools = tools
        self._busy: set[str] = set()
        self._detail = StepDetail(settings)

    # ===== threads ==============================================================================

    def create_thread(self, req: ThreadCreate) -> ThreadInfo:
        self._check_user(req.user_id)
        return self._threads.create(req.user_id, req.title)

    def list_threads(self, user_id: str | None = None) -> list[ThreadInfo]:
        return self._threads.list(user_id)

    def _thread(self, thread_id: str) -> ThreadInfo:
        info = self._threads.get(thread_id)
        if info is None:
            raise AppError("thread_not_found", f"No thread {thread_id}.", 404)
        return info

    @staticmethod
    def _config(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        configurable = {"thread_id": thread_id, "checkpoint_ns": ""}
        if checkpoint_id:
            configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable, "recursion_limit": 100}

    @staticmethod
    def _check_user(user_id: str | None) -> None:
        if user_id is not None and not USER_ID.match(user_id):
            raise AppError(
                "invalid_user", "user_id may use letters, digits, '_', '.', '@' and '-'.", 422
            )

    async def get_thread(self, thread_id: str) -> ThreadDetail:
        info = self._thread(thread_id)
        snap = await self._graph.aget_state(self._config(thread_id))
        messages = (snap.values or {}).get("messages", [])
        runs = self._threads.runs(thread_id)

        def view(m: Any) -> MessageView:
            mid = getattr(m, "id", None)
            run = None
            if isinstance(m, AIMessage) and mid and mid.startswith("ai-") and mid[3:] in runs:
                run = RunRecord.model_validate_json(runs[mid[3:]][1])
            return MessageView(
                id=mid,
                role="user" if isinstance(m, HumanMessage) else "assistant",
                content=_text(m),
                run=run,
            )

        return ThreadDetail(**info.model_dump(), history=[view(m) for m in messages])

    async def delete_thread(self, thread_id: str) -> None:
        self._thread(thread_id)
        await self._saver.adelete_thread(thread_id)
        self._threads.delete(thread_id)

    # ===== one run at a time per thread =========================================================

    @asynccontextmanager
    async def _exclusive(self, thread_id: str) -> AsyncIterator[None]:
        """Two runs on one thread at once would interleave their saves and corrupt it."""
        if thread_id in self._busy:
            raise AppError(
                "thread_busy", "This conversation is still working on a previous message.", 409
            )
        self._busy.add(thread_id)
        try:
            yield
        finally:
            self._busy.discard(thread_id)

    # ===== input ================================================================================

    def _validate(self, req: MessageRequest, thread: ThreadInfo) -> str | None:
        if len(req.message) > self._s.max_message_chars:
            raise AppError(
                "invalid_message",
                f"The message is longer than {self._s.max_message_chars} characters.",
                422,
            )
        self._check_user(req.user_id)
        return thread.user_id or req.user_id  # a thread keeps the owner it was created with

    def _input(
        self, message: str, user_id: str | None, options: AgentOptions, stream: bool
    ) -> dict:
        return {
            "messages": [HumanMessage(message)],
            "run_id": uuid.uuid4().hex[:10],
            "user_id": user_id,
            "options": options.model_dump(),
            "stream": stream,
        }

    # ===== running ==============================================================================

    async def run(self, thread_id: str, req: MessageRequest) -> AgentResponse:
        thread = self._thread(thread_id)
        user_id = self._validate(req, thread)
        payload = self._input(req.message, user_id, req.options, stream=False)
        return await self._execute(thread_id, payload, title_hint=req.message)

    async def resume(self, thread_id: str, req: ResumeRequest) -> AgentResponse:
        self._thread(thread_id)
        command = await self._resume_command(thread_id, req)
        return await self._execute(thread_id, command)

    async def fork(self, thread_id: str, req: ForkRequest) -> AgentResponse:
        self._thread(thread_id)
        payload, config = await self._fork_input(thread_id, req, stream=False)
        return await self._execute(thread_id, payload, config=config)

    async def _execute(
        self,
        thread_id: str,
        payload: Any,
        config: dict | None = None,
        title_hint: str | None = None,
    ) -> AgentResponse:
        config = config or self._config(thread_id)
        started = time.perf_counter()
        async with self._exclusive(thread_id):
            with span("agent.run", {"agent.thread": thread_id, SESSION: thread_id}) as root:
                trace_id = current_trace_id()
                try:
                    async with asyncio.timeout(self._s.run_timeout_s):
                        await self._graph.ainvoke(payload, config)
                except TimeoutError:
                    raise AppError(
                        "run_timeout",
                        f"The run took longer than {self._s.run_timeout_s:.0f}s.",
                        504,
                    ) from None
                response = await self._response(thread_id, trace_id, started)
                await self._finish(root, thread_id, response, title_hint)
        return response

    # ===== streaming ============================================================================

    async def stream(self, thread_id: str, req: MessageRequest) -> AsyncIterator[Event]:
        thread = self._thread(thread_id)
        user_id = self._validate(req, thread)
        payload = self._input(req.message, user_id, req.options, stream=True)
        async for event in self._stream(thread_id, payload, title_hint=req.message):
            yield event

    async def stream_resume(self, thread_id: str, req: ResumeRequest) -> AsyncIterator[Event]:
        self._thread(thread_id)
        command = await self._resume_command(thread_id, req)
        async for event in self._stream(thread_id, command):
            yield event

    async def stream_fork(self, thread_id: str, req: ForkRequest) -> AsyncIterator[Event]:
        self._thread(thread_id)
        payload, config = await self._fork_input(thread_id, req, stream=True)
        async for event in self._stream(thread_id, payload, config=config):
            yield event

    async def _stream(
        self,
        thread_id: str,
        payload: Any,
        config: dict | None = None,
        title_hint: str | None = None,
    ) -> AsyncIterator[Event]:
        """The run as it happens: which node starts and ends, each token of the answer, each
        tool call, and finally the whole response."""
        config = config or self._config(thread_id)
        started = time.perf_counter()
        async with self._exclusive(thread_id):
            with span(
                "agent.run", {"agent.thread": thread_id, "agent.stream": True, SESSION: thread_id}
            ) as root:
                trace_id = current_trace_id()
                yield Event(type="run_started", data={"thread_id": thread_id, "trace_id": trace_id})
                try:
                    async with asyncio.timeout(self._s.run_timeout_s):
                        async for event in self._events(payload, config):
                            yield event
                except TimeoutError:
                    yield Event(
                        type="error", data={"code": "run_timeout", "message": "The run timed out."}
                    )
                    return
                except AppError as exc:
                    yield Event(type="error", data={"code": exc.code, "message": exc.message})
                    return
                except Exception as exc:
                    yield Event(
                        type="error",
                        data={
                            "code": "agent_failed",
                            "message": f"{type(exc).__name__}: {exc}"[:300],
                        },
                    )
                    return
                response = await self._response(thread_id, trace_id, started)
                await self._finish(root, thread_id, response, title_hint)
                if response.status == "interrupted":
                    yield Event(type="interrupt", data=response.interrupt.model_dump())
                yield Event(type="done", data=response.model_dump())

    async def _events(self, payload: Any, config: dict) -> AsyncIterator[Event]:
        async for namespace, mode, chunk in self._graph.astream(
            payload, config, stream_mode=["updates", "custom", "tasks"], subgraphs=True
        ):
            inner = bool(namespace)
            if mode == "tasks":
                if "result" not in chunk:
                    yield Event(type="node_start", data={"node": chunk["name"], "subgraph": inner})
                else:
                    yield Event(
                        type="node_end",
                        data={
                            "node": chunk["name"],
                            "subgraph": inner,
                            "error": bool(chunk.get("error")),
                        },
                    )
            elif mode == "custom" and chunk.get("type") in _CUSTOM:
                yield Event(
                    type=chunk["type"], data={k: v for k, v in chunk.items() if k != "type"}
                )
            elif mode == "updates":
                for node, update in chunk.items():
                    if node == "collect_tools":
                        for record in (update or {}).get("tool_calls", []):
                            yield Event(
                                type="tool_call",
                                data=ToolCallRecord.model_validate(record).model_dump(),
                            )

    # ===== resume and fork ======================================================================

    async def _pending(self, thread_id: str) -> tuple[list[ApprovalItem], str | None]:
        snap = await self._graph.aget_state(self._config(thread_id))
        for task in snap.tasks:
            for itr in task.interrupts:
                value = itr.value or {}
                return [ApprovalItem.model_validate(c) for c in value.get("calls", [])], getattr(
                    itr, "id", None
                )
        return [], None

    async def _resume_command(self, thread_id: str, req: ResumeRequest) -> Command:
        pending, _ = await self._pending(thread_id)
        if not pending:
            raise AppError(
                "nothing_to_resume", "This conversation is not waiting for an approval.", 409
            )
        ids = [c.call_id for c in pending]
        if req.approve is not None:
            decisions = dict.fromkeys(ids, req.approve)
        else:
            unknown = set(req.decisions or {}) - set(ids)
            if unknown:
                raise AppError(
                    "invalid_decision", f"Unknown call ids: {', '.join(sorted(unknown))}.", 422
                )
            decisions = {i: bool((req.decisions or {}).get(i, False)) for i in ids}  # silence = no
        return Command(resume={"decisions": decisions})

    async def _fork_input(self, thread_id: str, req: ForkRequest, stream: bool) -> tuple[Any, dict]:
        """Branch from a past step. The old path stays in the history, and the thread continues
        down the new one (this is "edit my message and regenerate")."""
        config = self._config(thread_id, req.checkpoint_id)
        snap = await self._graph.aget_state(config)
        if not snap.created_at and not snap.values:
            raise AppError("checkpoint_not_found", f"No checkpoint {req.checkpoint_id}.", 404)
        if req.message and len(req.message) > self._s.max_message_chars:
            raise AppError("invalid_message", "The message is too long.", 422)
        info = self._thread(thread_id)

        if not snap.next:  # a finished turn: the new message starts the next turn from here
            if not req.message:
                raise AppError(
                    "message_required", "Give the message to continue from this point.", 422
                )
            return self._input(req.message, info.user_id, req.options, stream), config

        updates: dict[str, Any] = {
            "run_id": uuid.uuid4().hex[:10],
            "options": req.options.model_dump(),
            "stream": stream,
        }
        if req.message:
            last = next(
                (m for m in reversed(snap.values["messages"]) if isinstance(m, HumanMessage)), None
            )
            updates["messages"] = [HumanMessage(req.message, id=last.id if last else None)]
        new_config = await self._graph.aupdate_state(config, updates)
        return None, new_config  # continue from the edited checkpoint

    # ===== reading results ======================================================================

    async def _response(
        self, thread_id: str, trace_id: str | None, started: float
    ) -> AgentResponse:
        snap = await self._graph.aget_state(self._config(thread_id))
        v = snap.values or {}
        timings = v.get("timings") or {}
        base = {
            "thread_id": thread_id,
            "run_id": v.get("run_id", ""),
            "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
            "trace_id": trace_id,
            "route": v.get("route"),
            "passages": [Passage.model_validate(p) for p in v.get("passages", [])],
            "warnings": list(v.get("warnings", [])),
            "timings": Timings(
                ttft_ms=timings.get("ttft_ms"),
                retrieval_ms=timings.get("retrieval_ms"),
                total_ms=(time.perf_counter() - started) * 1000,
            ),
        }
        pending, interrupt_id = await self._pending(thread_id)
        if pending:
            return AgentResponse(
                **base, status="interrupted",
                interrupt=PendingApproval(interrupt_id=interrupt_id, calls=pending),
                tool_calls=[ToolCallRecord.model_validate(r) for r in v.get("tool_calls", [])],
            )  # fmt: skip
        return AgentResponse(
            **base,
            status="completed",
            answer=v.get("answer"),
            citations=[Citation.model_validate(c) for c in v.get("citations", [])],
            tool_calls=[ToolCallRecord.model_validate(r) for r in v.get("tool_calls", [])],
            check=CheckResult.model_validate(v["check"]) if v.get("check") else None,
        )

    async def _finish(
        self, root: Any, thread_id: str, response: AgentResponse, title_hint: str | None
    ) -> None:
        """After a run: describe it on its span, and keep what the chat will need to show again."""
        snap = await self._graph.aget_state(self._config(thread_id))
        v = snap.values or {}
        options = v.get("options") or {}
        collection = (options.get("retrieval") or {}).get("collection")
        root.set_attributes(
            {
                "agent.route": response.route or "",
                "agent.status": response.status,
                "agent.passages": len(response.passages),
                "agent.tool_calls": len(response.tool_calls),
                "agent.collection": collection or "",
                "agent.retries": (response.check.attempts - 1) if response.check else 0,
            }
        )
        self._detail.text(root, INPUT, v.get("question"))
        if response.status == "completed":
            self._detail.text(root, OUTPUT, response.answer)
        self._threads.touch(thread_id, len(v.get("messages", [])), title_hint, collection)
        if response.status == "completed" and response.run_id:
            record = RunRecord(
                question=v.get("question", ""),
                options=options,
                response=response,
                created_at=time.time(),
            )
            self._threads.save_run(thread_id, response.run_id, record.model_dump_json())

    @staticmethod
    def _jsonable(values: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key, value in values.items():
            if key == "messages":
                out[key] = [
                    {
                        "role": "user" if isinstance(m, HumanMessage) else "assistant",
                        "content": _text(m),
                    }
                    for m in value
                ]
            else:
                out[key] = value
        return out

    async def get_state(self, thread_id: str, checkpoint_id: str | None = None) -> StateView:
        self._thread(thread_id)
        snap = await self._graph.aget_state(self._config(thread_id, checkpoint_id))
        pending, interrupt_id = await self._pending(thread_id) if not checkpoint_id else ([], None)
        return StateView(
            thread_id=thread_id,
            checkpoint_id=snap.config["configurable"].get("checkpoint_id"),
            next=list(snap.next),
            values=self._jsonable(snap.values or {}),
            created_at=snap.created_at,
            interrupts=[PendingApproval(interrupt_id=interrupt_id, calls=pending)]
            if pending
            else [],
        )

    async def get_history(self, thread_id: str) -> list[HistoryItem]:
        """Every saved step, newest first. A fork shows up as a second child of one step."""
        self._thread(thread_id)
        items = []
        async for snap in self._graph.aget_state_history(self._config(thread_id)):
            meta = snap.metadata or {}
            parent = (snap.parent_config or {}).get("configurable", {}).get("checkpoint_id")
            values = snap.values or {}
            last_user = next(
                (
                    _text(m)
                    for m in reversed(values.get("messages", []))
                    if isinstance(m, HumanMessage)
                ),
                "",
            )
            summary = (f"next: {', '.join(snap.next)}" if snap.next else "finished") + (
                f" | {last_user[:50]}" if last_user else ""
            )
            items.append(
                HistoryItem(
                    checkpoint_id=snap.config["configurable"]["checkpoint_id"],
                    parent_checkpoint_id=parent,
                    step=meta.get("step"),
                    source=meta.get("source"),
                    next=list(snap.next),
                    created_at=snap.created_at,
                    summary=summary,
                )
            )
        return items

    # ===== memory ===============================================================================

    def _need_memory(self) -> FactMemory:
        if self._memory is None:
            raise AppError("memory_unavailable", "Long-term memory is not configured.", 503)
        return self._memory

    async def list_facts(self, user_id: str) -> list[Fact]:
        self._check_user(user_id)
        return await asyncio.to_thread(self._need_memory().all_facts, user_id)

    async def add_fact(self, user_id: str, text: str) -> Fact:
        self._check_user(user_id)
        return await asyncio.to_thread(self._need_memory().add, user_id, text)

    async def delete_fact(self, user_id: str, fact_id: str) -> None:
        self._check_user(user_id)
        if not await asyncio.to_thread(self._need_memory().delete, user_id, fact_id):
            raise AppError("fact_not_found", f"No fact {fact_id}.", 404)

    async def search_facts(self, user_id: str, query: str) -> list[Fact]:
        self._check_user(user_id)
        return await asyncio.to_thread(
            self._need_memory().search, user_id, query, self._s.memory_k, self._s.memory_min_score
        )

    # ===== information ==========================================================================

    def graph_info(self) -> GraphInfo:
        drawn = self._graph.get_graph()
        nodes = [
            GraphNode(
                id=n.id,
                label=n.id.strip("_"),
                kind="start"
                if n.id == "__start__"
                else "end"
                if n.id == "__end__"
                else "subgraph"
                if n.id == "answer"
                else "node",
            )
            for n in drawn.nodes.values()
        ]
        edges = [
            GraphEdge(source=e.source, target=e.target, conditional=bool(e.conditional))
            for e in drawn.edges
        ]
        return GraphInfo(nodes=nodes, edges=edges, mermaid=drawn.draw_mermaid())

    async def status(self) -> AgentStatus:
        s = self._s
        try:
            n_tools = len(await self._tools.tools()) if self._tools else 0
        except Exception:
            n_tools = 0
        return AgentStatus(
            checkpoint_backend=s.checkpoint_backend, data_dir=s.data_dir, approval=s.approval,
            max_check_retries=s.max_check_retries, max_tool_rounds=s.max_tool_rounds,
            use_cache=s.use_cache, auto_remember=s.auto_remember,
            memory_embed_model=s.memory_embed_model, tools_available=n_tools,
        )  # fmt: skip
