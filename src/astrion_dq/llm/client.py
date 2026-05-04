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
import re
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


def _extract_affordable_max_tokens(error_text: str) -> Optional[int]:
    """Parse OpenRouter's budget error and return the affordable max_tokens."""
    match = re.search(r"can only afford\s+(\d+)", error_text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _create_completion_with_budget_retry(client, **kwargs):
    """Retry once with a lower max_tokens when OpenRouter returns a 402 budget error."""
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        affordable = _extract_affordable_max_tokens(str(exc))
        requested = int(kwargs.get("max_tokens") or 0)
        if affordable is None or affordable >= requested:
            raise

        # Leave a little headroom below the provider-reported allowance.
        retry_max_tokens = max(64, affordable - 32)
        if retry_max_tokens >= requested:
            raise

        logger.warning(
            "OpenRouter budget retry: requested max_tokens=%s, retrying with %s",
            requested,
            retry_max_tokens,
        )
        retry_kwargs = {**kwargs, "max_tokens": retry_max_tokens}
        return client.chat.completions.create(**retry_kwargs)


def chat(prompt: str, max_tokens: int = 500) -> str:
    """Send a single-turn prompt and return the response text.

    Raises:
        LLMUnavailable: OPENROUTER_API_KEY is not configured.
        openai.APIError: Network or upstream API failure.
    """
    client = _get_client()
    resp = _create_completion_with_budget_retry(
        client,
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

    resp = _create_completion_with_budget_retry(
        client,
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
