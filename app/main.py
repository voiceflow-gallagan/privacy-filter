from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app import model as model_module
from app.routes import health as health_routes


def create_app(load_at_startup: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_at_startup:
            model_module.load_model()
        yield

    app = FastAPI(
        title="pii-filter",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(health_routes.router)
    return app


app = create_app(load_at_startup=True)
