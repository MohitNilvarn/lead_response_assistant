"""
Response Generator – orchestration layer.

Combines intent classification, RAG retrieval, Groq LLM generation,
and guardrail validation into a single pipeline that produces the
final ``LeadResponse``.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from app.core.guardrails import apply_guardrails
from app.core.intent_classifier import classify_intent
from app.core.knowledge_base import RetrievalResult, get_knowledge_base
from app.models.schemas import (
    CustomerIntent,
    FollowUpQuestion,
    IntentAnalysis,
    LeadResponse,
    Urgency,
)
from app.services.groq_service import get_groq_service

logger = logging.getLogger(__name__)


# ── System prompt template ───────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a senior customer support advisor for UrbanRoof — a trusted home repair and waterproofing company with 7+ years of experience. UrbanRoof's team of civil engineers uses advanced thermal scanning to diagnose leakages, seepage, and structural issues non-invasively.

YOUR ROLE:
You draft warm, professional, and helpful replies to customer enquiries. You represent UrbanRoof and should guide the customer toward booking a professional inspection when appropriate.

STRICT RULES — NEVER VIOLATE:
1. NEVER fabricate pricing, costs, or specific timelines. Pricing depends on on-site assessment.
2. NEVER make absolute guarantees (e.g. "100% guarantee", "permanent fix", "will never leak again").
3. NEVER diagnose the exact problem — only a professional on-site thermal scan can do that. You may discuss *possible* causes.
4. NEVER recommend DIY repairs for structural / waterproofing issues — always recommend professional assessment.
5. NEVER mention competitors or compare with other companies.
6. NEVER reveal that you are an AI. Write as a knowledgeable UrbanRoof team member.
7. ALWAYS be empathetic, acknowledge the customer's concern, and provide reassurance.
8. ALWAYS ask relevant clarifying questions to better understand the issue.
9. ALWAYS suggest safe immediate steps the customer can take (e.g., move valuables, switch off nearby electrical points if water is active).
10. Keep the tone warm, professional, and conversational — not robotic or overly formal.

RESPONSE STRUCTURE (follow this order):
1. Greet the customer and acknowledge their concern empathetically.
2. Provide a brief, helpful explanation of what *might* be causing the issue (without diagnosing).
3. Suggest safe immediate steps they can take right now.
4. Ask 2-3 relevant clarifying follow-up questions to understand the issue better.
5. Recommend booking a professional inspection with UrbanRoof and briefly explain the process (thermal scanning, non-invasive, detailed report).
6. Close warmly.

CONTEXT FROM KNOWLEDGE BASE:
{context}

CUSTOMER INTENT: {intent}
URGENCY LEVEL: {urgency}
"""


_USER_TEMPLATE = """Customer message:
\"{message}\"

{customer_name_line}

Draft a helpful, warm, and professional reply following all the rules above. Remember: no fabricated pricing, no guarantees, no exact diagnoses."""


_FOLLOWUP_SYSTEM = """You are an expert customer support analyst for UrbanRoof, a home repair and waterproofing company.

Given a customer message and the intent analysis, generate a JSON object with a list of follow-up questions that would help better understand and resolve the customer's issue.

Return EXACTLY this format:
{{
  "questions": [
    {{
      "question": "<the follow-up question>",
      "reason": "<why this question helps provide a better answer>"
    }}
  ]
}}

Guidelines:
- Generate 2-4 questions maximum
- Questions should be specific and relevant to the customer's issue
- Focus on information that would help diagnose or address the issue
- Include questions about: location of issue, duration, severity, previous repairs, property type
- Do NOT ask about budget or pricing
- Keep questions natural and conversational

Return ONLY the JSON object."""


# ── Helper: build context string from retrieved docs ─────────────────────

def _format_context(results: List[RetrievalResult]) -> str:
    if not results:
        return "No specific knowledge base documents matched. Use general UrbanRoof knowledge."

    sections = []
    for i, r in enumerate(results, 1):
        sections.append(
            f"[Source {i} — {r.document.category}/{r.document.topic} (relevance: {r.score:.2f})]\n"
            f"{r.document.content}"
        )
    return "\n\n".join(sections)


# ── Main orchestrator ────────────────────────────────────────────────────

def generate_lead_response(
    message: str,
    customer_name: str | None = None,
) -> LeadResponse:
    """
    End-to-end pipeline: intent → retrieval → generation → guardrails.

    Parameters
    ----------
    message : str
        Raw customer enquiry text.
    customer_name : str, optional
        Customer name for personalisation.

    Returns
    -------
    LeadResponse
        Fully structured, guardrail-checked response.
    """
    groq = get_groq_service()
    kb = get_knowledge_base()

    # ── Step 1: Intent Classification ────────────────────────────────
    logger.info("Step 1/4 — Classifying intent…")
    intent: IntentAnalysis = classify_intent(message)

    # ── Step 2: Knowledge Retrieval (RAG) ────────────────────────────
    logger.info("Step 2/4 — Retrieving relevant knowledge…")
    retrieval_results = kb.retrieve(message)
    context_str = _format_context(retrieval_results)
    sources = [
        f"{r.document.category}/{r.document.topic}" for r in retrieval_results
    ]

    # ── Step 3: Draft Reply Generation ───────────────────────────────
    logger.info("Step 3/4 — Generating draft reply…")
    system_msg = _SYSTEM_PROMPT.format(
        context=context_str,
        intent=intent.primary_intent.value,
        urgency=intent.urgency.value,
    )
    name_line = f"Customer name: {customer_name}" if customer_name else ""
    user_msg = _USER_TEMPLATE.format(message=message, customer_name_line=name_line)

    draft = groq.chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    )

    # ── Step 3b: Generate follow-up questions ────────────────────────
    followup_questions = _generate_followups(message, intent, groq)

    # ── Step 4: Guardrail Check ──────────────────────────────────────
    logger.info("Step 4/4 — Applying guardrails…")
    safe_reply, guardrail_check = apply_guardrails(draft)

    response = LeadResponse(
        original_message=message,
        intent_analysis=intent,
        drafted_reply=safe_reply,
        follow_up_questions=followup_questions,
        guardrail_check=guardrail_check,
        sources_used=sources,
    )
    logger.info("Lead response generated successfully (id=%s).", response.request_id)
    return response


# ── Follow-up question generation ────────────────────────────────────────

def _generate_followups(
    message: str,
    intent: IntentAnalysis,
    groq,
) -> List[FollowUpQuestion]:
    """Ask the LLM for structured follow-up questions."""
    try:
        result: Dict = groq.chat_completion_json(
            messages=[
                {"role": "system", "content": _FOLLOWUP_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Customer message: \"{message}\"\n"
                        f"Primary intent: {intent.primary_intent.value}\n"
                        f"Urgency: {intent.urgency.value}\n"
                        f"Key topics: {', '.join(intent.key_topics)}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=400,
        )
        questions = result.get("questions", [])
        return [
            FollowUpQuestion(question=q["question"], reason=q["reason"])
            for q in questions
            if "question" in q and "reason" in q
        ]
    except Exception:
        logger.exception("Failed to generate follow-up questions — returning empty list.")
        return []
