import threading
import time

import pytest
import uvicorn

from observability import shutdown_observability


@pytest.fixture(autouse=True)
def clean_observability():
    """create_app turns tracing on globally. Reset it so tests cannot leak into each other."""
    shutdown_observability()
    yield
    shutdown_observability()


@pytest.fixture(autouse=True)
def never_load_the_real_embedding_model(request, monkeypatch):
    """No test may download the real embedding model (~0.3 GB). Live tests are exempt."""
    if request.node.get_closest_marker("live"):
        return
    from agent_graph.memory import FastEmbedFactEmbedder

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to load the real embedding model: pass FakeEmbedder")

    monkeypatch.setattr(FastEmbedFactEmbedder, "embed", refuse)
    monkeypatch.setattr(FastEmbedFactEmbedder, "dim", property(refuse))


class LiveServer:
    """The app over real HTTP on a free port, for tests that need a real socket (MCP over HTTP)."""

    def __init__(self, app) -> None:
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="on")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "LiveServer":
        self._thread.start()
        for _ in range(100):
            if self._server.started:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("the test server did not start")
        self.url = f"http://127.0.0.1:{self._server.servers[0].sockets[0].getsockname()[1]}"
        return self

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


@pytest.fixture
def live():
    return LiveServer
