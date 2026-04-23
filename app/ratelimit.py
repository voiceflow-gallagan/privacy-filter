"""
Per-IP rate limiting via slowapi.

Design notes:
- A single module-level Limiter singleton is shared across all route modules,
  so per-IP counters are unified. Two limiter instances would mean each route
  has its own counter, defeating the per-IP cap.
- The limit string is provided as a CALLABLE so it is re-evaluated per request.
  This lets test fixtures override env vars after the module is imported.
- `RATE_LIMIT_ENABLED=false` is honored by returning a sky-high limit
  ("1000000/second") rather than disabling the limiter — keeps slowapi's
  decorator wiring uniform.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings


def _key_func(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return get_remote_address(request)


limiter: Limiter = Limiter(key_func=_key_func, enabled=True)


def current_limit() -> str:
    """Used as a callable in @limiter.limit so env overrides are honored per request."""
    s = get_settings()
    if not s.rate_limit_enabled:
        return "1000000/second"
    return s.rate_limit_per_ip


def install(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def reset_for_tests() -> None:
    """Clear in-memory counters between tests."""
    limiter.reset()
