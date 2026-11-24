"""Looking up a secret by NAME.

Registrations and targets store only the name of a variable (for example `APP_API_KEY`), never
its value. The value is read at the moment it is needed. A name is looked up in the process
environment first, then in the `.env` file, because pydantic-settings reads `.env` for settings
but does not put its contents into the environment.
"""

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values


class Environment:
    """A read-only view: `environment.get("APP_API_KEY")`. Empty values count as unset."""

    def __init__(self, env_file: str | Path = ".env", environ: Mapping[str, str] | None = None):
        self._file = Path(env_file)
        self._environ = os.environ if environ is None else environ

    def get(self, name: str, default: str | None = None) -> str | None:
        value = self._environ.get(name)
        if value:
            return value
        if self._file.is_file():
            value = dotenv_values(self._file).get(name)
            if value:
                return value
        return default
