"""
Application configuration module.

Centralizes all environment variables, secrets, and app-level settings.
Uses pydantic-settings for validation, type coercion, and .env file support.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable, validated application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ─────────────────────────────────────────────────────────
    groq_api_key: str
    huggingface_api_token: Optional[str] = None  # only needed for gated models

    # ── Groq LLM Settings ────────────────────────────────────────────────
    groq_model_name: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.4
    groq_max_tokens: int = 1024
    groq_top_p: float = 0.9

    # ── HuggingFace Embedding Settings ───────────────────────────────────
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # must match the model above

    # ── RAG / Vector Store Settings ──────────────────────────────────────
    knowledge_top_k: int = 5
    similarity_threshold: float = 0.35

    # ── Application Settings ─────────────────────────────────────────────
    app_name: str = "UrbanRoof Lead Response Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── Server Settings ──────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the validated settings."""
    return Settings()
