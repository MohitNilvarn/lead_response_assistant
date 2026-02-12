"""
Groq LLM Service.

Thin, production-hardened wrapper around the Groq chat-completion API.
Handles retries, timeouts, structured prompt assembly, and token tracking.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from groq import Groq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


class GroqService:
    """Manages interactions with the Groq inference API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model_name
        self._temperature = settings.groq_temperature
        self._max_tokens = settings.groq_max_tokens
        self._top_p = settings.groq_top_p

    # ── Public API ───────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Send a chat-completion request to Groq.

        Parameters
        ----------
        messages : list of dicts
            OpenAI-compatible message list (role + content).
        temperature : float, optional
            Override the default temperature.
        max_tokens : int, optional
            Override the default max_tokens.
        response_format : dict, optional
            Pass ``{"type": "json_object"}`` to enforce JSON output.

        Returns
        -------
        str
            The assistant's reply text.
        """
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
            "top_p": self._top_p,
        }
        if response_format:
            kwargs["response_format"] = response_format

        logger.debug("Groq request → model=%s, msgs=%d", self._model, len(messages))

        response = self._client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        usage = response.usage
        logger.info(
            "Groq response → tokens(prompt=%s, completion=%s, total=%s)",
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
            usage.total_tokens if usage else "?",
        )
        return content

    def chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Chat completion that returns a parsed JSON dict.

        Automatically sets ``response_format`` to ``json_object`` and
        parses the reply. Raises ``ValueError`` on malformed output.
        """
        raw = self.chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Groq returned non-JSON: %s", raw[:300])
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc


# ── Module-level singleton ───────────────────────────────────────────────

_instance: GroqService | None = None


def get_groq_service() -> GroqService:
    """Return a cached singleton of the Groq service."""
    global _instance
    if _instance is None:
        _instance = GroqService()
    return _instance
