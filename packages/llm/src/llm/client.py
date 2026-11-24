"""OpenAI Responses API client (httpx, no SDK) for gpt-6-luna.

The client follows OpenAI's published API. Details the documentation leaves open (the error body
shape, rate-limit headers, whether `response.completed` carries usage while streaming, and the
exact `store` / `phase` behaviour) are handled defensively below and exercised by the live tests
in `tests/test_llm_live.py`.

The rules this file exists to enforce:
- `reasoning.effort` comes only from `LLM_USE_REASONING` ("low" when on, "none" when off).
- `temperature`, `top_p` and `logprobs` are never sent, whatever the effort.
"""

import json
import re
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from common import AppError
from llm.schema import strict_schema
from llm.settings import LLMSettings
from llm.tools import ToolCall, ToolTurn, parse_arguments
from observability import record_llm_call, span

T = TypeVar("T", bound=BaseModel)

_RETRY_STATUS = {429, 500, 502, 503, 504}
_KEY_LIKE = re.compile(r"sk-[A-Za-z0-9_\-*]{6,}")


def _clean(text: str, limit: int = 300) -> str:
    """Error text from the API can echo part of a key. Never pass that on."""
    return _KEY_LIKE.sub("[REDACTED]", text)[:limit]


def build_request(
    settings: LLMSettings,
    prompt: str,
    system: str | None = None,
    schema: type[BaseModel] | None = None,
    stream: bool = False,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """The request body. Pure, so tests can check every field."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body: dict[str, Any] = {
        "model": settings.model,
        "input": messages,
        "reasoning": {"effort": settings.effort},
        "store": False,  # do not keep our documents on OpenAI's side
    }
    limit = max_output_tokens or settings.max_output_tokens
    if limit:
        body["max_output_tokens"] = limit
    if schema is not None:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": re.sub(r"[^A-Za-z0-9_-]", "_", schema.__name__)[:64],
                "schema": strict_schema(schema),
                "strict": True,
            }
        }
    if stream:
        body["stream"] = True
    return body


def build_tools_request(
    settings: LLMSettings,
    items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system: str | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """A request that offers tools. `items` is the conversation in the API's own item format."""
    history = ([{"role": "system", "content": system}] if system else []) + list(items)
    body: dict[str, Any] = {
        "model": settings.model,
        "input": history,
        "reasoning": {"effort": settings.effort},
        "store": False,
        "tools": tools,
    }
    limit = max_output_tokens or settings.max_output_tokens
    if limit:
        body["max_output_tokens"] = limit
    return body


class TextStream:
    """Returned by `OpenAILLM.stream`. Iterate it for text pieces, then read the timings."""

    def __init__(self, run: Callable[["TextStream"], Iterator[str]]) -> None:
        self._run = run
        self.ttft_ms: float | None = None
        self.total_ms: float | None = None

    def __iter__(self) -> Iterator[str]:
        return self._run(self)


class OpenAILLM:
    def __init__(
        self,
        settings: LLMSettings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._s = settings
        self._http = client or httpx.Client(timeout=settings.timeout_s)
        self._sleep = sleep

    # ----- public API ---------------------------------------------------------------------------

    def complete(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> str:
        body = build_request(self._s, prompt, system, max_output_tokens=max_output_tokens)
        data = self._call("llm.complete", body)
        return self._candidates(data)[0]

    def complete_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        max_output_tokens: int | None = None,
    ) -> T:
        body = build_request(
            self._s, prompt, system, schema=schema, max_output_tokens=max_output_tokens
        )
        problems: list[str] = []
        for _attempt in range(2):  # gpt-6-luna occasionally returns unusable JSON: try once more
            data = self._call("llm.complete_json", body)
            for text in self._candidates(data):
                try:
                    return schema.model_validate_json(text)
                except ValidationError as exc:
                    problems.append(_clean(str(exc.errors(include_input=False)[:2]), 200))
        raise AppError(
            "llm_invalid_output",
            "The model did not return JSON matching the expected shape, twice in a row.",
            502,
            {"problems": problems[:4]},
        )

    def complete_with_tools(
        self,
        items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        system: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ToolTurn:
        """One turn of a tool conversation: the model answers, or asks for tools."""
        body = build_tools_request(self._s, items, tools, system, max_output_tokens)
        data = self._call("llm.complete_with_tools", body)
        return self._tool_turn(data)

    def stream(
        self, prompt: str, *, system: str | None = None, max_output_tokens: int | None = None
    ) -> TextStream:
        body = build_request(
            self._s, prompt, system, stream=True, max_output_tokens=max_output_tokens
        )
        return TextStream(lambda handle: self._stream(body, handle))

    # ----- one HTTP call, with retries ---------------------------------------------------------

    def _require_key(self) -> dict[str, str]:
        if not self._s.openai_api_key:
            raise AppError(
                "llm_not_configured",
                "OPENAI_API_KEY is not set. Add it to .env and restart.",
                503,
            )
        return {"Authorization": f"Bearer {self._s.openai_api_key}"}

    def _call(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = self._require_key()
        started = time.perf_counter()
        with span(name, {"gen_ai.request.model": self._s.model}) as sp:
            data = self._post_with_retries(body, headers)
            usage = data.get("usage") or {}
            tokens_in = int(usage.get("input_tokens") or 0)
            tokens_out = int(usage.get("output_tokens") or 0)
            record_llm_call(
                self._s.model,
                tokens_in,
                tokens_out,
                total_ms=(time.perf_counter() - started) * 1000,
                cost_usd=self._s.estimate_cost(tokens_in, tokens_out),
            )
            reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
            if reasoning is not None:
                sp.set_attribute("llm.reasoning_tokens", reasoning)
            sp.set_attribute("llm.reasoning_effort", self._s.effort)
            return data

    def _post_with_retries(self, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        url = f"{self._s.base_url.rstrip('/')}/responses"
        for attempt in range(self._s.max_retries + 1):
            last = attempt == self._s.max_retries
            try:
                response = self._http.post(url, json=body, headers=headers)
            except httpx.TimeoutException as exc:
                if last:
                    raise AppError(
                        "llm_timeout",
                        f"OpenAI did not answer within {self._s.timeout_s:.0f}s.",
                        504,
                    ) from exc
                self._sleep(2.0**attempt)
                continue
            except httpx.HTTPError as exc:
                if last:
                    raise AppError(
                        "llm_unavailable", f"Cannot reach OpenAI ({type(exc).__name__}).", 502
                    ) from exc
                self._sleep(2.0**attempt)
                continue

            if response.status_code in _RETRY_STATUS and not last:
                self._sleep(self._delay(response, attempt))
                continue
            if response.status_code >= 400:
                self._raise_for(response)
            try:
                return response.json()
            except ValueError as exc:
                raise AppError(
                    "llm_bad_response", "OpenAI returned a non-JSON response.", 502
                ) from exc
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _delay(response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after", "")
        try:
            return min(float(retry_after), 60.0)
        except ValueError:
            return 2.0**attempt

    @staticmethod
    def _raise_for(response: httpx.Response) -> None:
        status = response.status_code
        try:
            detail = _clean(str((response.json().get("error") or {}).get("message") or ""))
        except (ValueError, AttributeError):
            detail = _clean(response.text)
        if status in (401, 403):
            raise AppError("llm_auth", f"OpenAI rejected the API key (HTTP {status}).", 502)
        if status == 404:
            raise AppError(
                "llm_model_not_found",
                f"OpenAI says the model or endpoint is unknown: {detail}",
                502,
            )
        if status == 429:
            raise AppError("llm_rate_limited", f"OpenAI rate limit reached. {detail}", 503)
        if status >= 500:
            raise AppError("llm_unavailable", f"OpenAI had a server error (HTTP {status}).", 502)
        raise AppError("llm_bad_request", f"OpenAI rejected the request: {detail}", 502)

    # ----- reading a response ---------------------------------------------------------------------

    @staticmethod
    def _candidates(data: dict[str, Any]) -> list[str]:
        """Texts the model may have answered with, best first.

        gpt-6-luna can return more than one message item (a "commentary" item with working
        notes beside the real "final_answer"). So prefer `final_answer`, then the last message.
        """
        if data.get("status") == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason", "unknown")
            raise AppError(
                "llm_incomplete",
                f"The answer was cut off ({reason}). Raise LLM_MAX_OUTPUT_TOKENS or shorten the "
                "input.",
                502,
            )
        if data.get("status") == "failed" or data.get("error"):
            message = _clean(str((data.get("error") or {}).get("message", "")))
            raise AppError("llm_failed", f"The model call failed. {message}", 502)

        messages = [item for item in data.get("output", []) if item.get("type") == "message"]
        for item in messages:
            for part in item.get("content", []):
                if part.get("type") == "refusal":
                    raise AppError(
                        "llm_refused", f"The model refused: {_clean(part.get('refusal', ''))}", 502
                    )

        def text_of(item: dict[str, Any]) -> str:
            return "".join(
                p.get("text", "") for p in item.get("content", []) if p.get("type") == "output_text"
            )

        final = [m for m in messages if m.get("phase") == "final_answer"]
        rest = [m for m in reversed(messages) if m.get("phase") != "final_answer"]
        texts = [t for m in [*final, *rest] if (t := text_of(m).strip())]
        if not texts:
            raise AppError("llm_empty", "The model returned no text.", 502)
        return texts

    # ----- reading a tool turn --------------------------------------------------------------

    def _tool_turn(self, data: dict[str, Any]) -> ToolTurn:
        output = [item for item in data.get("output", []) if isinstance(item, dict)]
        calls: list[ToolCall] = []
        for item in output:
            if item.get("type") != "function_call":
                continue
            call_id, name = item.get("call_id"), item.get("name")
            if not call_id or not name:
                raise AppError(
                    "llm_bad_response", "A tool call from the model was incomplete.", 502
                )
            arguments, error = parse_arguments(item.get("arguments"))
            calls.append(
                ToolCall(call_id=call_id, name=name, arguments=arguments, argument_error=error)
            )
        try:
            text = self._candidates(data)[0]
        except AppError as exc:
            # A turn that only calls tools has no text, which is not an error. Refusals and cut-off
            # answers still are.
            if exc.code != "llm_empty" or not calls:
                raise
            text = ""
        return ToolTurn(text=text, calls=calls, output_items=output)

    # ----- streaming ------------------------------------------------------------------------------

    def _stream(self, body: dict[str, Any], handle: TextStream) -> Iterator[str]:
        headers = self._require_key()
        url = f"{self._s.base_url.rstrip('/')}/responses"
        started = time.perf_counter()
        usage: dict[str, Any] = {}
        commentary: set[str] = set()  # items to skip: the model's working notes, not the answer

        with span("llm.stream", {"gen_ai.request.model": self._s.model}) as sp:
            try:
                with self._http.stream("POST", url, json=body, headers=headers) as response:
                    if response.status_code >= 400:
                        response.read()
                        self._raise_for(response)
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            event = json.loads(line[5:].strip())
                        except ValueError:
                            continue  # e.g. a "[DONE]" marker
                        kind = event.get("type", "")
                        if kind == "response.output_item.added":
                            item = event.get("item") or {}
                            if item.get("type") == "message" and item.get("phase") == "commentary":
                                commentary.add(item.get("id", ""))
                        elif kind == "response.output_text.delta":
                            if event.get("item_id") in commentary:
                                continue
                            piece = event.get("delta", "")
                            if piece and handle.ttft_ms is None:
                                handle.ttft_ms = (time.perf_counter() - started) * 1000
                            if piece:
                                yield piece
                        elif kind == "response.completed":
                            usage = (event.get("response") or {}).get("usage") or {}
                        elif kind in ("response.failed", "response.incomplete", "error"):
                            detail = event.get("message") or str(
                                ((event.get("response") or {}).get("error") or {}).get(
                                    "message", ""
                                )
                            )
                            raise AppError(
                                "llm_failed", f"The stream ended early: {_clean(str(detail))}", 502
                            )
            except httpx.TimeoutException as exc:
                raise AppError("llm_timeout", "OpenAI stopped answering mid-stream.", 504) from exc
            except httpx.HTTPError as exc:
                raise AppError(
                    "llm_unavailable", f"Cannot reach OpenAI ({type(exc).__name__}).", 502
                ) from exc

            handle.total_ms = (time.perf_counter() - started) * 1000
            tokens_in = int(usage.get("input_tokens") or 0)
            tokens_out = int(usage.get("output_tokens") or 0)
            record_llm_call(
                self._s.model,
                tokens_in,
                tokens_out,
                ttft_ms=handle.ttft_ms,
                total_ms=handle.total_ms,
                cost_usd=self._s.estimate_cost(tokens_in, tokens_out),
            )
            sp.set_attribute("llm.reasoning_effort", self._s.effort)
