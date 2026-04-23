from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app import model as model_module
from app.mcp_app import build_mcp
from app.ratelimit import install, reset_for_tests
from app.routes import batch as batch_routes
from app.routes import detect as detect_routes
from app.routes import health as health_routes


def create_app(load_at_startup: bool = True) -> FastAPI:
    mcp = build_mcp()
    mcp_asgi_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_at_startup:
            model_module.load_model()
        # The streamable-HTTP session manager has its own lifespan that must
        # run for the /mcp mount to accept requests. Nest it under ours.
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="pii-filter", version="1.0.0", lifespan=lifespan)

    install(app)
    reset_for_tests()

    app.include_router(health_routes.router)
    app.include_router(detect_routes.router)
    app.include_router(batch_routes.router)
    app.mount("/mcp", mcp_asgi_app)
    return app


app = create_app(load_at_startup=True)
