"""Routes. Three routers so each app decides where each is mounted:

- `health_router`: public liveness.
- `status_router`: protected `/status`.
- `router`:        protected, mounted at `/api/agent`.

The streaming routes answer with Server-Sent Events: one `data: {json}` line per event.
"""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from agent_graph.models import (
    AgentResponse,
    AgentStatus,
    Event,
    Fact,
    FactCreate,
    ForkRequest,
    GraphInfo,
    HistoryItem,
    MessageRequest,
    ReasoningState,
    ReasoningSwitch,
    ResumeRequest,
    StateView,
    ThreadCreate,
    ThreadDetail,
    ThreadInfo,
)
from agent_graph.service import AgentService
from common import AppError, require_api_key
from llm import LLMSettings, reasoning_override, set_reasoning_override

health_router = APIRouter()
status_router = APIRouter(dependencies=[Depends(require_api_key)])
router = APIRouter(dependencies=[Depends(require_api_key)])


def get_service(request: Request) -> AgentService:
    service = request.app.state.agent.service
    if service is None:  # only before startup finished or after shutdown
        raise AppError("agent_not_ready", "The agent is not running.", 503)
    return service


@health_router.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@status_router.get("/status", response_model=AgentStatus)
async def status(svc: AgentService = Depends(get_service)) -> AgentStatus:
    return await svc.status()


# ----- threads ------------------------------------------------------------------------------


@router.post("/threads", response_model=ThreadInfo, status_code=201)
def create_thread(body: ThreadCreate, svc: AgentService = Depends(get_service)) -> ThreadInfo:
    return svc.create_thread(body)


@router.get("/threads", response_model=list[ThreadInfo])
def list_threads(user_id: str | None = None, svc: AgentService = Depends(get_service)):
    return svc.list_threads(user_id)


@router.get("/threads/{thread_id}", response_model=ThreadDetail)
async def get_thread(thread_id: str, svc: AgentService = Depends(get_service)) -> ThreadDetail:
    return await svc.get_thread(thread_id)


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(thread_id: str, svc: AgentService = Depends(get_service)) -> None:
    await svc.delete_thread(thread_id)


# ----- running ------------------------------------------------------------------------------


@router.post("/threads/{thread_id}/messages", response_model=AgentResponse)
async def send(
    thread_id: str, body: MessageRequest, svc: AgentService = Depends(get_service)
) -> AgentResponse:
    return await svc.run(thread_id, body)


@router.post("/threads/{thread_id}/resume", response_model=AgentResponse)
async def resume(
    thread_id: str, body: ResumeRequest, svc: AgentService = Depends(get_service)
) -> AgentResponse:
    return await svc.resume(thread_id, body)


@router.post("/threads/{thread_id}/fork", response_model=AgentResponse)
async def fork(
    thread_id: str, body: ForkRequest, svc: AgentService = Depends(get_service)
) -> AgentResponse:
    return await svc.fork(thread_id, body)


def _sse(events: AsyncIterator[Event]) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        async for event in events:
            yield f"data: {json.dumps(event.model_dump(), default=str)}\n\n"

    return StreamingResponse(
        body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


async def _checked(first: Event, rest: AsyncIterator[Event]) -> AsyncIterator[Event]:
    yield first
    async for event in rest:
        yield event


async def _prime(events: AsyncIterator[Event]) -> AsyncIterator[Event]:
    """Take the first event before answering, so a refusal that happens up front (unknown
    thread, a busy thread, a bad request) is a normal HTTP error and not a stream that
    opens with an error event."""
    first = await anext(events)
    return _checked(first, events)


@router.post("/threads/{thread_id}/messages/stream")
async def send_stream(
    thread_id: str, body: MessageRequest, svc: AgentService = Depends(get_service)
) -> StreamingResponse:
    return _sse(await _prime(svc.stream(thread_id, body)))


@router.post("/threads/{thread_id}/resume/stream")
async def resume_stream(
    thread_id: str, body: ResumeRequest, svc: AgentService = Depends(get_service)
) -> StreamingResponse:
    return _sse(await _prime(svc.stream_resume(thread_id, body)))


@router.post("/threads/{thread_id}/fork/stream")
async def fork_stream(
    thread_id: str, body: ForkRequest, svc: AgentService = Depends(get_service)
) -> StreamingResponse:
    return _sse(await _prime(svc.stream_fork(thread_id, body)))


# ----- looking inside -----------------------------------------------------------------------


@router.get("/threads/{thread_id}/state", response_model=StateView)
async def state(
    thread_id: str, checkpoint_id: str | None = None, svc: AgentService = Depends(get_service)
) -> StateView:
    return await svc.get_state(thread_id, checkpoint_id)


@router.get("/threads/{thread_id}/history", response_model=list[HistoryItem])
async def history(thread_id: str, svc: AgentService = Depends(get_service)):
    return await svc.get_history(thread_id)


@router.get("/graph", response_model=GraphInfo)
def graph(svc: AgentService = Depends(get_service)) -> GraphInfo:
    return svc.graph_info()


# ----- long-term memory ---------------------------------------------------------------------


@router.get("/memory/{user_id}", response_model=list[Fact])
async def list_facts(
    user_id: str, query: str | None = None, svc: AgentService = Depends(get_service)
):
    return await (svc.search_facts(user_id, query) if query else svc.list_facts(user_id))


@router.post("/memory/{user_id}", response_model=Fact, status_code=201)
async def add_fact(
    user_id: str, body: FactCreate, svc: AgentService = Depends(get_service)
) -> Fact:
    return await svc.add_fact(user_id, body.text)


@router.delete("/memory/{user_id}/{fact_id}", status_code=204)
async def delete_fact(user_id: str, fact_id: str, svc: AgentService = Depends(get_service)) -> None:
    await svc.delete_fact(user_id, fact_id)


# ----- the global reasoning switch -----------------------------------------------------------


def _reasoning() -> ReasoningState:
    configured = LLMSettings().use_reasoning
    override = reasoning_override()
    on = configured if override is None else override
    return ReasoningState(
        use_reasoning=on, effort="low" if on else "none", overridden=override is not None
    )


@router.get("/reasoning", response_model=ReasoningState)
def get_reasoning() -> ReasoningState:
    return _reasoning()


@router.put("/reasoning", response_model=ReasoningState)
def put_reasoning(body: ReasoningSwitch) -> ReasoningState:
    """Flips the one global setting for every component in this process, until it restarts."""
    set_reasoning_override(body.use_reasoning)
    return _reasoning()
