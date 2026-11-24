from typing import Protocol


class DependencyProbe(Protocol):
    """Checks whether an outside system answers. Tests pass a fake, so no Docker is needed."""

    def is_reachable(self) -> bool: ...
