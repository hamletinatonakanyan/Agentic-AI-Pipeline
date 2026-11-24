# llm

The one LLM adapter: OpenAI `gpt-6-luna` through the Responses API. A shared library with no UI
and no routes. Retrieval (query rewriting), evals (judge, question drafting) and the agent use it,
so every model call follows the same rules.

## Usage

```python
from llm import OpenAILLM, LLMSettings

llm = OpenAILLM(LLMSettings())
llm.complete("Summarise this.", system="Be brief.")  # -> str
llm.complete_json("Judge this.", Verdict)  # -> a validated pydantic Verdict
for piece in (h := llm.stream("Tell me a story")):  # text pieces as they arrive
    ...  # h.ttft_ms and h.total_ms afterwards
```

Native tool calling, for agents:

```python
from llm import user_item, tool_output_item

turn = llm.complete_with_tools([user_item("What is 12% of 340?")], tools=[...])
turn.calls  # the tool calls the model asked for (call_id, name, arguments)
turn.output_items  # replay these, then add tool_output_item(call_id, result), and call again
```

## The rules it enforces

- `reasoning.effort` comes only from `LLM_USE_REASONING`: **low** when on, **none** when off.
  There is no per-call override. A process-wide switch (`set_reasoning_override`) exists for a UI
  toggle and lasts until the process restarts.
- `temperature`, `top_p` and `logprobs` are **never sent**. A test checks every request body.
- Every call is a span (`llm.complete`, `llm.complete_json`, `llm.complete_with_tools`,
  `llm.stream`) with model, token counts, reasoning tokens, an estimated cost and timings.
  Streaming also records time to first token.
- `store: false` is sent, so OpenAI does not keep your documents.
- Errors become `AppError`s with clear codes (`llm_not_configured`, `llm_auth`,
  `llm_rate_limited`, `llm_timeout`, `llm_refused`, `llm_incomplete`, `llm_invalid_output`, ...).
  Key-like text in an error is redacted.
- 429, 5xx and network errors are retried with backoff and `Retry-After`. Streaming is not
  retried.

## Response handling

The model sometimes returns an extra `commentary` message beside the real `final_answer`. The
parser prefers `final_answer`, falls back to the last message, and `complete_json` retries once
if nothing parses.

## Structured output

`strict_schema(Model)` rewrites a pydantic schema into what OpenAI's strict mode accepts
(`additionalProperties: false`, everything required, no defaults). Free-form `dict` fields are
refused: use a list of `{key, value}` objects.

## Tool calling

`complete_with_tools` uses the Responses API function format: flat tool definitions, the model's
own output items replayed verbatim (including reasoning items) together with
`function_call_output` items, and several calls per turn. Malformed JSON arguments are reported
on the call (`argument_error`) instead of raising, so a caller can tell the model.

## Cost estimate

Set `LLM_PRICE_INPUT_PER_M` and `LLM_PRICE_OUTPUT_PER_M` (USD per million tokens, from your
provider's price list) and each call records `llm.cost_usd` on its span. Empty means no
estimate. It is an estimate: reasoning tokens count as output, and discounts such as cached
input are ignored.

## Configuration

`LLM_MODEL`, `LLM_USE_REASONING`, `LLM_BASE_URL`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`,
`LLM_MAX_OUTPUT_TOKENS`, `LLM_PRICE_INPUT_PER_M`, `LLM_PRICE_OUTPUT_PER_M`, and the shared
`OPENAI_API_KEY`. See `.env.example`.

## Tests

`FakeLLM` (in `llm.testing`) is a scripted model for tests in any package.

```bash
uv run pytest packages/llm
make test-live     # checks the request and response shapes against the real API
```
