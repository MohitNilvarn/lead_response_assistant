"""
Intent Classifier.

Uses the Groq LLM with a structured JSON prompt to determine the
customer's primary intent, urgency, and key topics from their enquiry.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from app.models.schemas import CustomerIntent, IntentAnalysis, Urgency
from app.services.groq_service import get_groq_service

logger = logging.getLogger(__name__)

# ── System prompt for intent classification ──────────────────────────────

_SYSTEM_PROMPT = """You are an intent classification engine for UrbanRoof, a home repair and waterproofing company.

Analyse the customer message and return a JSON object with EXACTLY these fields:
{
  "primary_intent": "<one of: water_damage, waterproofing_inquiry, leakage_seepage, roof_repair, wall_repair, thermal_scanning, pricing_inquiry, service_booking, complaint, general_inquiry>",
  "confidence": <float 0.0-1.0>,
  "secondary_intents": [<list of other applicable intents from the same set, can be empty>],
  "urgency": "<one of: low, medium, high, critical>",
  "key_topics": [<short phrases describing the main topics mentioned, max 5>]
}

Classification guidelines:
- water_damage: Active water damage, flooding, water stains
- waterproofing_inquiry: Questions about waterproofing services/solutions
- leakage_seepage: Leaks, seepage, dampness, moisture issues
- roof_repair: Roof-related issues (leaks, cracks, waterproofing)
- wall_repair: Wall-related issues (cracks, dampness, paint peeling)
- thermal_scanning: Questions about thermal/infrared inspection
- pricing_inquiry: Cost, rates, pricing questions
- service_booking: Want to schedule an inspection or service
- complaint: Dissatisfaction with existing service
- general_inquiry: General questions about the company/services

Urgency guidelines:
- critical: Active flooding, electrical safety risk, structural danger
- high: Active leakage during rains, spreading damage
- medium: Recurring dampness, seepage, intermittent issues
- low: General enquiry, preventive check, information gathering

Return ONLY the JSON object, no other text."""


def classify_intent(message: str) -> IntentAnalysis:
    """
    Classify the intent of a customer enquiry.

    Parameters
    ----------
    message : str
        The raw customer message.

    Returns
    -------
    IntentAnalysis
        Structured intent classification result.
    """
    groq = get_groq_service()

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    result: Dict = groq.chat_completion_json(messages, temperature=0.1, max_tokens=300)

    # ── Parse and validate ──────────────────────────────────────────
    try:
        primary = CustomerIntent(result.get("primary_intent", "general_inquiry"))
    except ValueError:
        logger.warning("Unknown primary intent '%s' — falling back to general_inquiry.",
                       result.get("primary_intent"))
        primary = CustomerIntent.GENERAL_INQUIRY

    secondary: List[CustomerIntent] = []
    for raw_intent in result.get("secondary_intents", []):
        try:
            secondary.append(CustomerIntent(raw_intent))
        except ValueError:
            pass  # silently skip unknown secondary intents

    try:
        urgency = Urgency(result.get("urgency", "medium"))
    except ValueError:
        urgency = Urgency.MEDIUM

    analysis = IntentAnalysis(
        primary_intent=primary,
        confidence=min(max(float(result.get("confidence", 0.7)), 0.0), 1.0),
        secondary_intents=secondary,
        urgency=urgency,
        key_topics=result.get("key_topics", [])[:5],
    )

    logger.info(
        "Intent classified → %s (conf=%.2f, urgency=%s)",
        analysis.primary_intent.value,
        analysis.confidence,
        analysis.urgency.value,
    )
    return analysis
