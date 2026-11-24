from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    """State of one component, as shown on a status card."""

    name: str
    enabled: bool
    configured: bool
    # None means "not checked" (for example, the component is not mounted).
    reachable: bool | None = None
    missing: list[str] = Field(default_factory=list)
    detail: str = ""
    # Where this component's UI is served, when it has one that is mounted.
    ui_path: str | None = None


class StatusReport(BaseModel):
    components: list[ComponentStatus]
