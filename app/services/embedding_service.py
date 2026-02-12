"""
HuggingFace Embedding Service.

Provides a singleton wrapper around sentence-transformers to generate
dense vector embeddings for queries and knowledge-base documents.
Thread-safe and lazy-loaded so the model is only downloaded once.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

_model_instance: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model_instance
    if _model_instance is None:
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model_name)
        _model_instance = SentenceTransformer(
            settings.embedding_model_name,
            token=settings.huggingface_api_token,
        )
        logger.info("Embedding model loaded successfully.")
    return _model_instance


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of texts into dense vectors.

    Parameters
    ----------
    texts : List[str]
        Raw text strings to embed.
    batch_size : int
        Encoding batch size (tune for GPU memory if applicable).

    Returns
    -------
    np.ndarray
        Matrix of shape ``(len(texts), embedding_dim)``.
    """
    model = _get_model()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # unit‑norm → cosine sim == dot product
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string.

    Returns
    -------
    np.ndarray
        1-D vector of shape ``(embedding_dim,)``.
    """
    return embed_texts([query])[0]


def get_embedding_dimension() -> int:
    """Return the dimensionality of the loaded model."""
    return _get_model().get_sentence_embedding_dimension()
