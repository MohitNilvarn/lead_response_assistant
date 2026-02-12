"""
FastAPI route definitions for the Lead Response Assistant.

Exposes:
  POST /api/v1/enquiry       — process a customer enquiry
  GET  /api/v1/health        — health / readiness check
  POST /api/v1/intent        — classify intent only (debug / test)
  POST /api/v1/knowledge     — retrieve relevant knowledge (debug / test)
"""

from __future__ import annotations

import logging
import time
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.config import get_settings
from app.core.intent_classifier import classify_intent
from app.core.knowledge_base import get_knowledge_base
from app.core.response_generator import generate_lead_response
from app.models.schemas import (
    ErrorResponse,
    HealthResponse,
    IntentAnalysis,
    LeadEnquiryRequest,
    LeadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Lead Response Assistant"])


# ── Health Check ─────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & readiness check",
)
async def health_check() -> HealthResponse:
    """Return the application health status."""
    settings = get_settings()
    return HealthResponse(version=settings.app_version)


# ── Main Enquiry Endpoint ────────────────────────────────────────────────

@router.post(
    "/enquiry",
    response_model=LeadResponse,
    summary="Process a customer enquiry and draft a response",
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal processing error"},
    },
)
async def process_enquiry(request: LeadEnquiryRequest) -> LeadResponse:
    """
    Accept a customer enquiry and return a structured, guardrail-checked reply.

    The pipeline:
    1. Classify the customer's intent.
    2. Retrieve relevant UrbanRoof knowledge (RAG).
    3. Generate a draft reply via Groq LLM.
    4. Apply guardrails to remove hallucinations / unsafe claims.
    5. Return the full response with intent analysis, follow-ups, and sources.
    """
    start = time.perf_counter()
    try:
        response = generate_lead_response(
            message=request.message,
            customer_name=request.customer_name,
        )
        elapsed = time.perf_counter() - start
        logger.info(
            "Enquiry processed in %.2fs (id=%s, intent=%s).",
            elapsed,
            response.request_id,
            response.intent_analysis.primary_intent.value,
        )
        return response
    except Exception as exc:
        logger.exception("Failed to process enquiry.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# ── Intent-only Endpoint (utility / debugging) ──────────────────────────

@router.post(
    "/intent",
    response_model=IntentAnalysis,
    summary="Classify intent only (utility endpoint)",
)
async def classify_intent_endpoint(request: LeadEnquiryRequest) -> IntentAnalysis:
    """Classify the customer's intent without generating a full response."""
    try:
        return classify_intent(request.message)
    except Exception as exc:
        logger.exception("Intent classification failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# ── Knowledge Retrieval Endpoint (utility / debugging) ───────────────────

@router.post(
    "/knowledge",
    summary="Retrieve relevant knowledge base entries (utility endpoint)",
)
async def retrieve_knowledge(request: LeadEnquiryRequest) -> List[dict]:
    """Return the top-matching knowledge base documents for the query."""
    try:
        kb = get_knowledge_base()
        results = kb.retrieve(request.message)
        return [
            {
                "topic": r.document.topic,
                "category": r.document.category,
                "content": r.document.content,
                "similarity_score": round(r.score, 4),
            }
            for r in results
        ]
    except Exception as exc:
        logger.exception("Knowledge retrieval failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
