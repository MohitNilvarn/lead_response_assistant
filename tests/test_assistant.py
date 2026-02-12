"""
Unit and integration tests for the Lead Response Assistant.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import re

import pytest

from app.core.guardrails import apply_guardrails
from app.models.schemas import (
    CustomerIntent,
    LeadEnquiryRequest,
    Urgency,
)


# ═══════════════════════════════════════════════════════════════════════════
# Guardrails Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGuardrails:
    """Verify guardrail rules catch hallucinated content."""

    def test_exact_price_is_replaced(self):
        draft = "The waterproofing will cost Rs. 25,000 for your terrace."
        cleaned, check = apply_guardrails(draft)
        assert "Rs. 25,000" not in cleaned
        assert "on-site assessment" in cleaned.lower() or "assessment" in cleaned.lower()
        assert not check.passed

    def test_inr_price_is_replaced(self):
        draft = "Total cost would be INR 15000 approximately."
        cleaned, check = apply_guardrails(draft)
        assert "INR 15000" not in cleaned
        assert not check.passed

    def test_rupee_symbol_price_is_replaced(self):
        draft = "The inspection fee is ₹2,500."
        cleaned, check = apply_guardrails(draft)
        assert "₹2,500" not in cleaned
        assert not check.passed

    def test_fabricated_timeline_is_replaced(self):
        draft = "We will complete the work within 3 days."
        cleaned, check = apply_guardrails(draft)
        assert "within 3 days" not in cleaned
        assert not check.passed

    def test_absolute_guarantee_is_replaced(self):
        draft = "We provide a 100% guarantee on all waterproofing work."
        cleaned, check = apply_guardrails(draft)
        assert "100% guarantee" not in cleaned
        assert not check.passed

    def test_lifetime_warranty_is_replaced(self):
        draft = "This comes with a lifetime warranty."
        cleaned, check = apply_guardrails(draft)
        assert "lifetime warranty" not in cleaned
        assert not check.passed

    def test_ai_self_reference_is_removed(self):
        draft = "As an AI language model, I can help you with your query."
        cleaned, check = apply_guardrails(draft)
        assert "ai language model" not in cleaned.lower()
        assert not check.passed

    def test_clean_text_passes(self):
        draft = (
            "Thank you for reaching out! We understand your concern about "
            "damp patches on your wall. Our team of civil engineers can help "
            "diagnose the issue using thermal scanning technology."
        )
        cleaned, check = apply_guardrails(draft)
        assert check.passed
        assert cleaned == draft

    def test_multiple_violations_all_caught(self):
        draft = (
            "The repair will cost Rs. 10,000 and will be completed within 2 days. "
            "We offer a 100% guarantee."
        )
        cleaned, check = apply_guardrails(draft)
        assert "Rs. 10,000" not in cleaned
        assert "within 2 days" not in cleaned
        assert "100% guarantee" not in cleaned
        assert len(check.flags) >= 3


# ═══════════════════════════════════════════════════════════════════════════
# Schema Validation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemas:
    """Verify Pydantic schema validation."""

    def test_valid_enquiry(self):
        req = LeadEnquiryRequest(
            message="I have damp patches on my wall after rains."
        )
        assert req.message == "I have damp patches on my wall after rains."
        assert req.customer_name is None

    def test_enquiry_with_name(self):
        req = LeadEnquiryRequest(
            message="My roof is leaking during monsoon.",
            customer_name="Rahul",
        )
        assert req.customer_name == "Rahul"

    def test_short_message_rejected(self):
        with pytest.raises(Exception):
            LeadEnquiryRequest(message="Hi")

    def test_customer_intent_enum(self):
        assert CustomerIntent.WATER_DAMAGE.value == "water_damage"
        assert CustomerIntent.LEAKAGE_SEEPAGE.value == "leakage_seepage"

    def test_urgency_enum(self):
        assert Urgency.CRITICAL.value == "critical"
        assert Urgency.LOW.value == "low"


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests (require GROQ_API_KEY in environment)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegration:
    """
    Integration tests that call external APIs.

    Run with: pytest tests/ -v -m integration
    Skip in CI without keys: pytest tests/ -v -m "not integration"
    """

    def test_full_pipeline(self):
        """Smoke-test the full enquiry → response pipeline."""
        import os
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set — skipping integration test.")

        from app.core.response_generator import generate_lead_response

        response = generate_lead_response(
            message="Hi, I am getting damp patches on my bedroom wall after rains. What should I do?",
            customer_name="Mohit",
        )

        # Basic structural checks
        assert response.request_id
        assert response.drafted_reply
        assert len(response.drafted_reply) > 100
        assert response.intent_analysis.primary_intent in (
            CustomerIntent.WATER_DAMAGE,
            CustomerIntent.LEAKAGE_SEEPAGE,
            CustomerIntent.WALL_REPAIR,
        )
        assert len(response.sources_used) > 0

        # Guardrail: no fabricated prices should remain
        assert not re.search(r"Rs\.?\s*[\d,]+", response.drafted_reply)
