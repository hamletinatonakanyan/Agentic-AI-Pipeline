"""The access-key check.

Attach `require_api_key` to a router (or an app) to protect every route on it:

    router = APIRouter(dependencies=[Depends(require_api_key)])

Exempt by design: all `health` routes and the static UI files. They are
simply not placed behind this dependency.
"""

import hmac

from fastapi import Depends, Header

from common.errors import AppError
from common.settings import AppSettings, get_app_settings


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: AppSettings = Depends(get_app_settings),
) -> None:
    configured = settings.app_api_key
    if not configured:
        # No key configured: refuse everything rather than run open.
        raise AppError(
            "not_configured",
            "APP_API_KEY is not set. Add it to .env and restart.",
            status_code=503,
        )
    # compare_digest takes the same time however many characters match.
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), configured.encode()):
        raise AppError("unauthorized", "Missing or invalid X-API-Key.", status_code=401)
