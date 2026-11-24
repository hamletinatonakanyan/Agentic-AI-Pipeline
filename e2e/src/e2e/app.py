"""The e2e FastAPI app.

    uv run uvicorn e2e.app:create_app --factory --port 8000      (make run-e2e)

A factory, so importing this module has no side effects (no SQLite file is created).
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import agent_graph.app as agent_app
import evals.app as evals_app
import ingest.app as ingest_app
import mcp_tools.app as mcp_app
import observability.app as obs_app
import retrieval.app as retrieval_app
from agent_graph import api as agent_api
from agent_graph.runtime import Agent, build_agent
from agent_graph.settings import AgentSettings
from common import install_error_handlers, mount_ui_kit, setup_logging
from e2e.api import health_router, router
from e2e.protocols import DependencyProbe
from e2e.service import BuiltComponent, QdrantProbe
from e2e.settings import E2ESettings
from e2e.wiring import (
    InProcessCorpus,
    InProcessRetriever,
    McpToolProvider,
    agent_preset,
    retrieval_preset,
)
from evals import api as evals_api
from evals.factory import Evals, build_evals
from evals.settings import EvalSettings
from ingest import api as ingest_api
from ingest.factory import Ingest, build_ingest
from ingest.settings import IngestSettings
from llm import LLMSettings, OpenAILLM, llm_status
from mcp_tools import api as mcp_api
from mcp_tools.factory import Mcp, build_mcp
from mcp_tools.settings import McpSettings
from observability import api as obs_api
from observability import init_observability, shutdown_observability
from observability.service import ObservabilityService
from observability.settings import ObsSettings
from retrieval import api as retrieval_api
from retrieval.factory import Retrieval, build_retrieval
from retrieval.settings import RetrievalSettings

UI_DIR = Path(__file__).parent / "ui"


def _mount_observability(
    app: FastAPI, obs_settings: ObsSettings, built: dict[str, BuiltComponent]
) -> None:
    handle = init_observability(obs_settings, force=True)
    service = ObservabilityService(handle)
    app.state.obs_service = service
    # /api/obs/health is public, /api/obs/status and /api/obs/traces need the key.
    app.include_router(obs_api.health_router, prefix="/api/obs")
    app.include_router(obs_api.status_router, prefix="/api/obs")
    app.include_router(obs_api.router, prefix="/api/obs")
    app.mount("/ui/observability", StaticFiles(directory=obs_app.UI_DIR, html=True))

    st = service.status()
    detail = f"store: {st.store}, exporters: {', '.join(st.exporters) or 'none'}"
    if not st.enabled:
        detail = "tracing is off (OBS_ENABLED=false)"
    built["observability"] = BuiltComponent(detail=detail, ui_path="/ui/observability/")


def _mount_ingest(app: FastAPI, ingest: Ingest, built: dict[str, BuiltComponent]) -> None:
    app.state.ingest = ingest
    # /api/ingest/health is public. Everything else needs the key.
    app.include_router(ingest_api.health_router, prefix="/api/ingest")
    app.include_router(ingest_api.status_router, prefix="/api/ingest")
    app.include_router(ingest_api.jobs_router, prefix="/api/ingest")
    app.include_router(ingest_api.router, prefix="/api/ingest")
    app.mount("/ui/ingest", StaticFiles(directory=ingest_app.UI_DIR, html=True))

    st = ingest.service.status()
    detail = f"embedding: {st.embed_model}, PDF parser: {st.pdf_parser}"
    if st.pdf_parser == "llamaparse" and not st.llamaparse_configured:
        detail += " (LLAMA_CLOUD_API_KEY not set: scanned PDF pages are skipped)"
    built["ingest"] = BuiltComponent(detail=detail, ui_path="/ui/ingest/")


def _mount_retrieval(app: FastAPI, retrieval: Retrieval, built: dict[str, BuiltComponent]) -> None:
    app.state.retrieval = retrieval
    app.include_router(retrieval_api.health_router, prefix="/api/retrieval")
    app.include_router(retrieval_api.status_router, prefix="/api/retrieval")
    app.include_router(retrieval_api.router, prefix="/api/retrieval")
    app.mount("/ui/retrieval", StaticFiles(directory=retrieval_app.UI_DIR, html=True))
    st = retrieval.service.status()
    built["retrieval"] = BuiltComponent(
        detail=f"default collection: {st.default_collection}, rerank model: {st.rerank_model}",
        ui_path="/ui/retrieval/",
    )


def _mount_evals(app: FastAPI, evals: Evals, built: dict[str, BuiltComponent]) -> None:
    app.state.evals = evals
    app.include_router(evals_api.health_router, prefix="/api/evals")
    app.include_router(evals_api.status_router, prefix="/api/evals")
    app.include_router(evals_api.jobs_router, prefix="/api/evals")
    app.include_router(evals_api.router, prefix="/api/evals")
    app.mount("/ui/evals", StaticFiles(directory=evals_app.UI_DIR, html=True))
    st = evals.service.status()
    detail = f"judge: {st.judge_model}, targets: {', '.join(st.presets)}"
    built["evals"] = BuiltComponent(detail=detail, ui_path="/ui/evals/")


def _mount_mcp(app: FastAPI, mcp: Mcp, built: dict[str, BuiltComponent]) -> None:
    app.state.mcp = mcp
    app.include_router(mcp_api.health_router, prefix="/api/mcp")
    app.include_router(mcp_api.status_router, prefix="/api/mcp")
    app.include_router(mcp_api.router, prefix="/api/mcp")
    # The MCP protocol itself: /mcp, behind the same key. The SDK's session manager must run in
    # this app's lifespan, or every request to /mcp fails.
    mcp.endpoint.install(app)
    app.state.lifespan_hooks.append(mcp.endpoint.lifespan)
    app.mount("/ui/mcp", StaticFiles(directory=mcp_app.UI_DIR, html=True))
    n = len(mcp.registry.list())
    built["mcp"] = BuiltComponent(
        detail=f"our server at /mcp (needs the key), {n} registered server(s)", ui_path="/ui/mcp/"
    )


def _mount_agent(app: FastAPI, agent: Agent, built: dict[str, BuiltComponent]) -> None:
    app.state.agent = agent
    app.state.loop_holder = holder = {}

    @asynccontextmanager
    async def remember_loop() -> AsyncIterator[None]:
        # Eval jobs run in worker threads and must hand each question to this loop.
        holder["loop"] = asyncio.get_running_loop()
        yield
        holder.pop("loop", None)

    app.state.lifespan_hooks.append(remember_loop)
    app.include_router(agent_api.health_router, prefix="/api/agent")
    app.include_router(agent_api.status_router, prefix="/api/agent")
    app.include_router(agent_api.router, prefix="/api/agent")
    app.state.lifespan_hooks.append(agent.lifespan)  # opens the checkpointer at startup
    app.mount("/ui/agent", StaticFiles(directory=agent_app.UI_DIR, html=True))
    s = agent.settings
    missing = built["llm"].missing if "llm" in built else ()  # the agent needs the model
    built["agent"] = BuiltComponent(
        detail=f"checkpoints: {s.checkpoint_backend}, approval: {s.approval}",
        ui_path="/ui/agent/",
        missing=missing,
    )


def _register_llm(built: dict[str, BuiltComponent], settings: LLMSettings) -> None:
    st = llm_status(settings)
    built["llm"] = BuiltComponent(
        detail=f"{st.model}, reasoning effort: {st.reasoning_effort}",
        missing=() if st.configured else ("OPENAI_API_KEY",),
    )


def create_app(
    settings: E2ESettings | None = None,
    qdrant: DependencyProbe | None = None,
    obs_settings: ObsSettings | None = None,
    ingest: Ingest | None = None,
    llm_settings: LLMSettings | None = None,
    retrieval: Retrieval | None = None,
    evals: Evals | None = None,
    mcp: Mcp | None = None,
    agent: Agent | None = None,
) -> FastAPI:
    """Build the app. Tests pass fakes for `qdrant` and `ingest`, and in-memory `obs_settings`."""
    setup_logging()
    settings = settings or E2ESettings()
    qdrant = qdrant or QdrantProbe(settings.qdrant_url, settings.probe_timeout_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for hook in app.state.lifespan_hooks:  # things a mounted component needs running
                await stack.enter_async_context(hook())
            yield
        for name in ("ingest", "evals", "mcp"):
            if getattr(app.state, name, None) is not None:
                getattr(app.state, name).close()
        shutdown_observability()

    app = FastAPI(title="Agentic Blocks", lifespan=lifespan)
    install_error_handlers(app)
    built: dict[str, BuiltComponent] = {}
    app.state.settings = settings
    app.state.qdrant = qdrant
    app.state.built = built
    app.state.lifespan_hooks = []

    app.include_router(health_router)
    app.include_router(router)

    # Components, in mount order. Observability goes first so that everything after it is traced.
    if settings.enable_obs:
        _mount_observability(app, obs_settings or ObsSettings(), built)
    _register_llm(built, llm_settings or LLMSettings())
    if settings.enable_ingest:
        _mount_ingest(app, ingest or build_ingest(IngestSettings()), built)
    if settings.enable_retrieval:
        retrieval = retrieval or build_retrieval(RetrievalSettings())
        _mount_retrieval(app, retrieval, built)
    if settings.enable_mcp:
        # Our mounted server reads the corpus through the services directly, not over HTTP.
        ingest_service = getattr(getattr(app.state, "ingest", None), "service", None)
        mcp_settings = McpSettings()
        corpus = InProcessCorpus(
            retrieval.service if retrieval else None,
            ingest_service,
            default_collection=mcp_settings.default_collection,
            max_chunks=mcp_settings.table_max_chunks,
        )
        _mount_mcp(app, mcp or build_mcp(mcp_settings, corpus=corpus), built)

    if settings.enable_agent:
        # The agent reads documents and calls tools through the services directly (no HTTP).
        mcp_built = getattr(app.state, "mcp", None)
        if agent is None:
            agent_llm_settings = llm_settings or LLMSettings()
            agent = build_agent(
                AgentSettings(),
                llm=OpenAILLM(agent_llm_settings),
                retriever=InProcessRetriever(retrieval.service) if retrieval else None,
                tools=McpToolProvider(mcp_built.manager) if mcp_built else None,
            )
        _mount_agent(app, agent, built)

    # Evals come last, so they can offer both in-process targets: retrieval, and the whole agent.
    if settings.enable_evals:
        presets = {"retrieval": retrieval_preset(retrieval.service)} if retrieval else {}
        evals = evals or build_evals(EvalSettings(), presets=presets)
        if agent is not None and settings.enable_agent:
            evals.service.register_preset("agent", agent_preset(agent, app.state.loop_holder))
        _mount_evals(app, evals, built)

    # Static UI, all public: the pages load, then ask for the key. Order matters: the more
    # specific mounts (kit, each package) must come before the catch-all "/" mount.
    mount_ui_kit(app)
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
