.PHONY: help sync up down test test-integration test-live lint format run-e2e run-observability run-ingest run-retrieval run-evals run-mcp run-agent inspect-mcp

help:
	@echo "sync              create/refresh .venv with uv"
	@echo "up / down         start / stop Qdrant (Docker)"
	@echo "test              unit tests (no Docker, no network)"
	@echo "test-integration  tests that need Qdrant running"
	@echo "test-live         tests that call real providers (need keys, may spend credits)"
	@echo "lint / format     ruff"
	@echo "run-ingest        serve the ingest UI on http://localhost:8001"
	@echo "run-retrieval     serve the retrieval UI on http://localhost:8002"
	@echo "run-evals         serve the evals UI on http://localhost:8003"
	@echo "run-agent         serve the agent UI on http://localhost:8006 (start retrieval and MCP too)"
	@echo "run-mcp           serve the MCP UI and /mcp on http://localhost:8004"
	@echo "inspect-mcp       open the MCP Inspector on our stdio server (needs Node)"
	@echo "run-observability serve the trace UI on http://localhost:8005"
	@echo "run-e2e           serve the app on http://localhost:8000"

sync:
	uv sync

up:
	docker compose -f infra/docker-compose.yml up -d

down:
	docker compose -f infra/docker-compose.yml down

test:
	uv run pytest

test-integration:
	uv run pytest -m integration -o addopts=""

# Never part of `make test`. Needs keys in .env and may spend credits.
test-live:
	uv run pytest -m live -o addopts=""

lint:
	uv run ruff check .

format:
	uv run ruff format .

run-mcp:
	uv run uvicorn mcp_tools.app:create_app --factory --port 8004

# Opens the MCP Inspector (a web UI from the MCP project) on our stdio server. Needs Node, and
# downloads the Inspector the first time.
inspect-mcp:
	npx @modelcontextprotocol/inspector uv run python -m mcp_tools.server

run-evals:
	uv run uvicorn evals.app:create_app --factory --port 8003

run-retrieval:
	uv run uvicorn retrieval.app:create_app --factory --port 8002

run-ingest:
	uv run uvicorn ingest.app:create_app --factory --port 8001

run-observability:
	uv run uvicorn observability.app:create_app --factory --port 8005

run-agent:
	uv run uvicorn agent_graph.app:create_app --factory --port 8006

run-e2e:
	uv run uvicorn e2e.app:create_app --factory --port 8000
