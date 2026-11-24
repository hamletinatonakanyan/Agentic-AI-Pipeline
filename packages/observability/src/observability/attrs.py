"""Attribute names in one place.

They follow the OpenTelemetry GenAI semantic conventions where those exist. The conventions
are still evolving, so if a name changes upstream, change it here and nowhere else.
"""

MODEL = "gen_ai.request.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Not part of the standard: our own names.
TTFT_MS = "llm.ttft_ms"
TOTAL_MS = "llm.total_ms"
COST_USD = "llm.cost_usd"

# Conventional names (OpenInference / OpenTelemetry). Whoever starts a trace may set them on the
# root span: the conversation it belongs to, and what went in and came out.
SESSION_ID = "session.id"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
