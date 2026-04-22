"""OpenRouter LLM client for Astrion DQ.

OpenRouter exposes a unified API (OpenAI wire format) that routes to Claude,
GPT-4o, Gemini, Llama, and others through a single key and endpoint.

Usage:
    from astrion_dq.llm.client import chat, LLMUnavailable

    try:
        summary = chat("Summarise these issues: ...")
    except LLMUnavailable:
        summary = ""   # fall back to template
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        summary = ""

The client is a module-level singleton initialised lazily on the first call.
Set OPENROUTER_API_KEY in config/.env or as an environment variable.
Without the key, every call raises LLMUnavailable immediately.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client: Optional[object] = None


class LLMUnavailable(RuntimeError):
    """Raised when OPENROUTER_API_KEY is not configured."""


def _get_client():
    """Return the singleton OpenAI client pointed at OpenRouter."""
    global _client
    if _client is not None:
        return _client

    from astrion_dq.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

    if not OPENROUTER_API_KEY:
        raise LLMUnavailable(
            "OPENROUTER_API_KEY is not set. "
            "Add it to config/.env or export it as an environment variable. "
            "The system continues in template-only mode without it."
        )

    from openai import OpenAI
    _client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/astrion-dq",
            "X-Title": "Astrion DQ",
        },
    )
    logger.info("OpenRouter client initialised (model: %s)", _get_model())
    return _client


def _get_model() -> str:
    from astrion_dq.config import OPENROUTER_MODEL
    return OPENROUTER_MODEL


def chat(prompt: str, max_tokens: int = 500) -> str:
    """Send a single-turn prompt and return the response text.

    Raises:
        LLMUnavailable: OPENROUTER_API_KEY is not configured.
        openai.APIError: Network or upstream API failure.
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model=_get_model(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def chat_with_history(
    messages: list,
    system: str = "",
    max_tokens: int = 1000,
) -> str:
    """Send a multi-turn conversation and return the assistant reply text.

    Args:
        messages: List of dicts with ``role`` ("user" | "assistant") and
                  ``content`` keys representing the conversation so far.
                  The latest message should be the user's current turn.
        system:   Optional system prompt prepended to every conversation.
        max_tokens: Maximum tokens in the response.

    Raises:
        LLMUnavailable: OPENROUTER_API_KEY is not configured.
        openai.APIError: Network or upstream API failure.
    """
    client = _get_client()

    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages.extend(messages)

    resp = client.chat.completions.create(
        model=_get_model(),
        messages=api_messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def reset_client() -> None:
    """Reset the singleton. Used in tests that mock the client."""
    global _client
    _client = None
