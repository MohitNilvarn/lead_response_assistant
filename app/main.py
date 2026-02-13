"""
FastAPI application factory and entry point.

Run with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.core.knowledge_base import get_knowledge_base

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _configure_logging() -> None:
    """Set up structured logging to stdout."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    - On startup: configure logging, warm up the knowledge base & embedding model.
    - On shutdown: clean-up (currently no-op).
    """
    _configure_logging()
    logger = logging.getLogger(__name__)

    settings = get_settings()
    logger.info("Starting %s v%s …", settings.app_name, settings.app_version)

    # Warm-up: load knowledge base + build FAISS index on startup
    logger.info("Warming up knowledge base and embedding model…")
    kb = get_knowledge_base()
    logger.info(
        "Knowledge base ready — %d documents indexed.", kb.document_count
    )

    yield  # ← application is running

    logger.info("Shutting down %s.", settings.app_name)


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-powered Lead Response Assistant for UrbanRoof. "
            "Reads customer enquiries and drafts helpful, guardrail-checked replies."
        ),
        lifespan=lifespan,
    )

    # ── CORS (permissive for dev; lock down in production) ───────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register routes ──────────────────────────────────────────────
    app.include_router(router)

    # ── Serve static frontend ────────────────────────────────────────
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
