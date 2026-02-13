"""
Pydantic request / response schemas for the Lead Response Assistant API.

Every public contract lives here so that API, service, and test layers
share a single source of truth.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class CustomerIntent(str, Enum):
    """Supported customer intent categories."""

    WATER_DAMAGE = "water_damage"
    WATERPROOFING_INQUIRY = "waterproofing_inquiry"
    LEAKAGE_SEEPAGE = "leakage_seepage"
    ROOF_REPAIR = "roof_repair"
    WALL_REPAIR = "wall_repair"
    THERMAL_SCANNING = "thermal_scanning"
    PRICING_INQUIRY = "pricing_inquiry"
    SERVICE_BOOKING = "service_booking"
    COMPLAINT = "complaint"
    GENERAL_INQUIRY = "general_inquiry"


class Urgency(str, Enum):
    """Estimated urgency of the customer issue."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Request Models ───────────────────────────────────────────────────────

class LeadEnquiryRequest(BaseModel):
    """Incoming customer enquiry payload."""

    message: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        description="The raw customer enquiry text.",
        examples=[
            "Hi, I am getting damp patches on my bedroom wall after rains. What should I do?"
        ],
    )
    customer_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional customer name for personalisation.",
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional metadata such as source channel, location, etc.",
    )


# ── Response Models ──────────────────────────────────────────────────────

class IntentAnalysis(BaseModel):
    """Result of intent classification on the enquiry."""

    primary_intent: CustomerIntent
    confidence: float = Field(ge=0.0, le=1.0)
    secondary_intents: List[CustomerIntent] = Field(default_factory=list)
    urgency: Urgency = Urgency.MEDIUM
    key_topics: List[str] = Field(default_factory=list)


class FollowUpQuestion(BaseModel):
    """A contextual follow-up question to ask the customer."""

    question: str
    reason: str = Field(description="Why this question helps provide a better answer.")


class GuardrailCheck(BaseModel):
    """Outcome of the hallucination / safety guardrail scan."""

    passed: bool = True
    flags: List[str] = Field(default_factory=list)
    modifications_applied: List[str] = Field(default_factory=list)


class LeadResponse(BaseModel):
    """Full structured response returned to the caller."""

    request_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    original_message: str
    intent_analysis: IntentAnalysis
    drafted_reply: str
    follow_up_questions: List[FollowUpQuestion] = Field(default_factory=list)
    guardrail_check: GuardrailCheck = Field(default_factory=GuardrailCheck)
    sources_used: List[str] = Field(
        default_factory=list,
        description="Knowledge-base sources that informed the reply.",
    )


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standardised error envelope."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
