# common

Shared infrastructure for every package. No domain logic.

| Module | What it gives you |
|---|---|
| `settings` | `BaseAppSettings` (subclass it and set an `env_prefix`), `AppSettings` (`APP_API_KEY`), `mask_secret` |
| `env` | `Environment`: environment variables with a fallback to `.env`, so secrets are found either way |
| `errors` | the one error envelope, `AppError`, and `install_error_handlers(app)` |
| `logging` | `setup_logging()`: JSON lines, idempotent |
| `auth` | `require_api_key`: the `X-API-Key` dependency (503 if the key is unset, 401 if wrong) |
| `ui` and `ui_kit/` | `mount_ui_kit(app)` serves the shared `kit.css` and `kit.js` at `/ui/kit` |

## Usage

```python
from fastapi import APIRouter, Depends, FastAPI
from pydantic_settings import SettingsConfigDict
from common import AppError, BaseAppSettings, install_error_handlers, require_api_key


class MySettings(BaseAppSettings):
    model_config = SettingsConfigDict(env_prefix="MY_", env_file=".env", extra="ignore")
    limit: int = 10  # read from MY_LIMIT


app = FastAPI()
install_error_handlers(app)  # every error becomes the same JSON envelope
router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/thing")
def thing():
    raise AppError("not_found", "No such thing.", 404)
```

Errors always look like `{"error": {"code": "...", "message": "...", "details": {...}}}`.

## Access rules

Put `Depends(require_api_key)` on a router to protect it. `health` routes and static UI files are
never placed behind it, so a page can load and then ask for the key.

## UI kit

`kit.css` is the shared palette and component styles (light and dark, no build step, nothing
loaded from the web). `kit.js` provides `kit.apiFetch` (adds the key and unwraps the error
envelope), `kit.streamSSE`, tabs, the access-key bar, and the active-domain helpers. Every
package UI uses them.

## Tests

```bash
uv run pytest packages/common
```
