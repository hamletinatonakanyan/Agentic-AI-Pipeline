import pytest

from observability import shutdown_observability


@pytest.fixture(autouse=True)
def clean_observability():
    """Every test starts and ends with observability off, so tests cannot leak state."""
    shutdown_observability()
    yield
    shutdown_observability()
