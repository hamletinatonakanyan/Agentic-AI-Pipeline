from pydantic import BaseModel

from llm.settings import LLMSettings


class LLMStatus(BaseModel):
    model: str
    reasoning_effort: str
    configured: bool  # is OPENAI_API_KEY set


def llm_status(settings: LLMSettings) -> LLMStatus:
    """For a status card. Never contains the key, or any part of it."""
    return LLMStatus(
        model=settings.model,
        reasoning_effort=settings.effort,
        configured=bool(settings.openai_api_key),
    )
