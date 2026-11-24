"""Builds the standalone agent from settings. Nothing connects at build time."""

from agent_graph.http_adapters import HttpRetriever, HttpToolProvider
from agent_graph.protocols import FactEmbedder
from agent_graph.runtime import Agent, build_agent
from agent_graph.settings import AgentSettings
from common import get_app_settings
from llm import LLMSettings, OpenAILLM


def build_standalone(
    settings: AgentSettings,
    llm_settings: LLMSettings | None = None,
    embedder: FactEmbedder | None = None,
) -> Agent:
    key = lambda: get_app_settings().app_api_key  # noqa: E731  (read at call time, not stored)
    return build_agent(
        settings,
        llm=OpenAILLM(llm_settings or LLMSettings()),
        retriever=HttpRetriever(settings.retrieval_url, key, settings.http_timeout_s),
        tools=HttpToolProvider(settings.mcp_url, key, settings.http_timeout_s),
        embedder=embedder,
    )
