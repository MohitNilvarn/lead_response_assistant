"""
Guardrails Module.

Post-processes the LLM-generated draft reply to detect and neutralise:
  - Hallucinated pricing / timelines / guarantees
  - Unsupported factual claims
  - Safety-critical misinformation
  - Overly aggressive sales language

Returns a sanitised reply together with an audit trail of modifications.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from app.models.schemas import GuardrailCheck

logger = logging.getLogger(__name__)


# ── Pattern definitions ──────────────────────────────────────────────────

@dataclass
class _Rule:
    """A single guardrail rule: a regex pattern, its replacement, and a label."""

    name: str
    pattern: re.Pattern
    replacement: str


_RULES: List[_Rule] = [
    # ── Fabricated pricing ──────────────────────────────────────────
    _Rule(
        name="hallucinated_exact_price",
        pattern=re.compile(
            r"(?:(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d{1,2})?|[\d,]+(?:\.\d{1,2})?\s*(?:Rs\.?|INR|₹|rupees))",
            re.IGNORECASE,
        ),
        replacement="(pricing will be determined after a professional on-site assessment)",
    ),
    # ── Fabricated timelines ────────────────────────────────────────
    _Rule(
        name="hallucinated_timeline",
        pattern=re.compile(
            r"\b(?:within|in)\s+\d+\s*(?:hours?|days?|weeks?|months?)\b",
            re.IGNORECASE,
        ),
        replacement="(timeline will be confirmed after assessment)",
    ),
    # ── Absolute guarantees ─────────────────────────────────────────
    _Rule(
        name="absolute_guarantee",
        pattern=re.compile(
            r"\b(?:100\s*%\s*guarant(?:ee|y)|lifetime\s+(?:guarant(?:ee|y)|warranty)|"
            r"permanent(?:ly)?\s+(?:fix|solv|cur)|never\s+(?:leak|seep|damage)\s+again)\b",
            re.IGNORECASE,
        ),
        replacement="(long-lasting solutions backed by professional assessment)",
    ),
    # ── False urgency / scare tactics ───────────────────────────────
    _Rule(
        name="scare_tactics",
        pattern=re.compile(
            r"\b(?:your\s+(?:house|home|building)\s+(?:will|could)\s+collapse|"
            r"extremely\s+dangerous|life[- ]threatening\s+(?:situation|condition))\b",
            re.IGNORECASE,
        ),
        replacement="(we recommend a professional assessment to evaluate the situation)",
    ),
    # ── Competitor bashing ──────────────────────────────────────────
    _Rule(
        name="competitor_bashing",
        pattern=re.compile(
            r"\b(?:other\s+companies\s+(?:will|would)\s+(?:cheat|scam|overcharge|rip\s*off))\b",
            re.IGNORECASE,
        ),
        replacement="(we focus on transparency and professional service)",
    ),
]


# ── Phrase-level blocklist (exact substring match) ───────────────────────

_BLOCKED_PHRASES: List[Tuple[str, str]] = [
    ("i am an ai", ""),
    ("as an ai language model", ""),
    ("as a large language model", ""),
    ("i don't have access to real-time", ""),
]


# ── Public API ───────────────────────────────────────────────────────────

def apply_guardrails(draft_reply: str) -> Tuple[str, GuardrailCheck]:
    """
    Run all guardrail rules against the draft reply.

    Parameters
    ----------
    draft_reply : str
        The raw LLM-generated reply text.

    Returns
    -------
    Tuple[str, GuardrailCheck]
        The (possibly modified) reply and an audit object.
    """
    flags: List[str] = []
    modifications: List[str] = []
    cleaned = draft_reply

    # 1. Regex-based rules
    for rule in _RULES:
        matches = rule.pattern.findall(cleaned)
        if matches:
            flags.append(f"{rule.name}: matched {len(matches)} occurrence(s)")
            cleaned = rule.pattern.sub(rule.replacement, cleaned)
            modifications.append(f"Replaced {rule.name} matches with safe alternative.")
            logger.info("Guardrail [%s] triggered — %d match(es).", rule.name, len(matches))

    # 2. Blocked phrases
    lower_cleaned = cleaned.lower()
    for phrase, repl in _BLOCKED_PHRASES:
        if phrase in lower_cleaned:
            flags.append(f"blocked_phrase: '{phrase}'")
            cleaned = re.sub(re.escape(phrase), repl, cleaned, flags=re.IGNORECASE)
            modifications.append(f"Removed blocked phrase: '{phrase}'")
            logger.info("Guardrail blocked phrase removed: '%s'", phrase)

    # 3. Remove any double-spacing artefacts from replacements
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    passed = len(flags) == 0

    check = GuardrailCheck(
        passed=passed,
        flags=flags,
        modifications_applied=modifications,
    )

    if not passed:
        logger.warning("Guardrail check FAILED — %d flag(s) raised.", len(flags))
    else:
        logger.debug("Guardrail check PASSED — no issues found.")

    return cleaned, check
