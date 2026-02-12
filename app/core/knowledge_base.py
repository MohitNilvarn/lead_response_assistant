"""
Knowledge Base – FAISS-backed Retrieval-Augmented Generation (RAG) store.

Loads UrbanRoof domain documents at startup, embeds them with the
HuggingFace embedding service, and exposes a cosine-similarity retrieval
method to supply the response-generation pipeline with grounded context.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from app.config import get_settings
from app.services.embedding_service import embed_query, embed_texts

logger = logging.getLogger(__name__)

# ── Data class for a single knowledge document ──────────────────────────

@dataclass
class KnowledgeDocument:
    """A single piece of contextual knowledge."""

    topic: str
    content: str
    category: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def text_for_embedding(self) -> str:
        """Concatenate topic + content for richer semantic encoding."""
        return f"{self.topic}: {self.content}"


# ── Retrieval result ─────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """A document retrieved from the knowledge base with its score."""

    document: KnowledgeDocument
    score: float  # cosine similarity (higher = more relevant)


# ── Knowledge Base Manager ───────────────────────────────────────────────

class KnowledgeBase:
    """In-memory FAISS vector store built from the UrbanRoof knowledge JSON."""

    def __init__(self) -> None:
        self._documents: List[KnowledgeDocument] = []
        self._index: Optional[faiss.IndexFlatIP] = None  # inner-product ≈ cosine
        self._is_initialised = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def initialise(self) -> None:
        """Load knowledge JSON, compute embeddings, and build the FAISS index."""
        if self._is_initialised:
            logger.debug("Knowledge base already initialised – skipping.")
            return

        data_path = Path(__file__).resolve().parent.parent / "data" / "urbanroof_knowledge.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {data_path}")

        logger.info("Loading knowledge base from %s", data_path)
        with open(data_path, "r", encoding="utf-8") as fh:
            raw: dict = json.load(fh)

        # Flatten all categories into a single document list
        for category, items in raw.items():
            for item in items:
                self._documents.append(
                    KnowledgeDocument(
                        topic=item["topic"],
                        content=item["content"],
                        category=category,
                    )
                )

        logger.info("Loaded %d knowledge documents across %d categories.",
                     len(self._documents), len(raw))

        # Embed all documents in one batched call
        texts = [doc.text_for_embedding for doc in self._documents]
        embeddings = embed_texts(texts)

        for doc, emb in zip(self._documents, embeddings):
            doc.embedding = emb

        # Build FAISS index (inner-product on L2-normalised vectors ≡ cosine similarity)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))

        self._is_initialised = True
        logger.info("FAISS index built — %d vectors, dim=%d.", self._index.ntotal, dim)

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant knowledge documents for the given query.

        Parameters
        ----------
        query : str
            The customer enquiry (or a derived search query).
        top_k : int, optional
            Number of nearest neighbours to return (default from settings).
        threshold : float, optional
            Minimum cosine similarity to include (default from settings).

        Returns
        -------
        List[RetrievalResult]
            Ranked list of relevant documents, highest similarity first.
        """
        if not self._is_initialised or self._index is None:
            raise RuntimeError("Knowledge base not initialised. Call .initialise() first.")

        settings = get_settings()
        top_k = top_k or settings.knowledge_top_k
        threshold = threshold if threshold is not None else settings.similarity_threshold

        query_vec = embed_query(query).astype(np.float32).reshape(1, -1)
        scores, indices = self._index.search(query_vec, top_k)

        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < threshold:
                continue
            results.append(
                RetrievalResult(document=self._documents[idx], score=float(score))
            )

        logger.debug(
            "Retrieved %d documents for query (top score=%.3f).",
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    # ── Utilities ────────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def is_ready(self) -> bool:
        return self._is_initialised


# ── Module-level singleton ───────────────────────────────────────────────

_instance: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """Return a lazily-initialised singleton of the knowledge base."""
    global _instance
    if _instance is None:
        _instance = KnowledgeBase()
        _instance.initialise()
    return _instance
