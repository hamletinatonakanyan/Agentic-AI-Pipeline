from pydantic import AliasChoices, Field
from pydantic_settings import SettingsConfigDict

from common import BaseAppSettings

# Parameters this model rejects unless effort is "none". We never send them at all, and a test
# proves no request ever contains them.
FORBIDDEN_PARAMS = ("temperature", "top_p", "logprobs", "top_logprobs")


# A runtime override of `use_reasoning`, set by the UI's "Use reasoning" checkbox. It lasts until
# the process restarts, then the .env value applies again. None = no override.
_reasoning_override: bool | None = None


def set_reasoning_override(value: bool | None) -> None:
    global _reasoning_override
    _reasoning_override = value


def reasoning_override() -> bool | None:
    return _reasoning_override


class LLMSettings(BaseAppSettings):
    """Reads LLM_* variables, plus the shared OPENAI_API_KEY."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", extra="ignore", populate_by_name=True
    )

    model: str = "gpt-6-luna"
    # one global switch. On sends reasoning.effort = "low", off sends "none".
    use_reasoning: bool = True
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = Field(default=120.0, gt=0)
    # Retries for rate limits (429), server errors (5xx) and network failures.
    max_retries: int = Field(default=2, ge=0, le=6)
    max_output_tokens: int | None = Field(default=None, ge=1)

    # Prices in USD per million tokens, only used to ESTIMATE the cost shown in traces. Empty
    # means no estimate. Never guessed here: take them from your provider's price list.
    price_input_per_m: float | None = Field(default=None, ge=0)
    price_output_per_m: float | None = Field(default=None, ge=0)

    openai_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_OPENAI_API_KEY"),
    )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        """USD, or None when no prices are configured. Reasoning tokens are part of the output
        tokens. This ignores discounts (for example cached input), so it is an estimate."""
        if self.price_input_per_m is None and self.price_output_per_m is None:
            return None
        return (
            input_tokens * (self.price_input_per_m or 0.0)
            + output_tokens * (self.price_output_per_m or 0.0)
        ) / 1_000_000

    @property
    def effort(self) -> str:
        on = self.use_reasoning if _reasoning_override is None else _reasoning_override
        return "low" if on else "none"
