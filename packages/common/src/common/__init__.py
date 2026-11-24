"""Shared infrastructure for every package. No domain logic lives here."""

from common.auth import require_api_key
from common.env import Environment
from common.errors import AppError, ErrorBody, ErrorEnvelope, install_error_handlers
from common.logging import setup_logging
from common.settings import AppSettings, BaseAppSettings, get_app_settings, mask_secret
from common.ui import mount_ui_kit, ui_kit_dir

__all__ = [
    "AppError",
    "AppSettings",
    "Environment",
    "ErrorBody",
    "BaseAppSettings",
    "ErrorEnvelope",
    "get_app_settings",
    "install_error_handlers",
    "mask_secret",
    "mount_ui_kit",
    "require_api_key",
    "setup_logging",
    "ui_kit_dir",
]
