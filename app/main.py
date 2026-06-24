from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import init_db
from .routers import ask, datasets, llm, search, ui


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title="RAG Service", version="0.2.0", lifespan=lifespan)


@app.get("/api/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


app.mount(
    "/static",
    StaticFiles(directory=str(settings.static_dir)),
    name="static",
)

app.include_router(datasets.router)
app.include_router(search.router)
app.include_router(ask.router)
app.include_router(llm.router)
app.include_router(ui.router)
