"""Serving the shared UI kit (plain CSS and JS, no build step)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def ui_kit_dir() -> Path:
    return Path(__file__).parent / "ui_kit"


def mount_ui_kit(app: FastAPI, path: str = "/ui/kit") -> None:
    """Serve kit.css and kit.js publicly. Static files hold no secrets.

    Mount this BEFORE any broader `/ui` mount, because the first matching mount wins.
    """
    app.mount(path, StaticFiles(directory=ui_kit_dir()), name="ui-kit")
