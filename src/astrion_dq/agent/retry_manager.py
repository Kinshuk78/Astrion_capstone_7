"""Exponential-backoff retry wrapper for OpenAI API calls.

Wraps any callable that accepts api_key as a keyword argument. On retriable
errors (rate limit, timeout, connection failure) it reports the failure to the
key manager, rotates to the next available key, waits, and retries.

Non-retriable errors (bad request, auth failures from bad keys, value errors
in the caller) propagate immediately without consuming retry budget.

Environment variables:
    AGENT_MAX_RETRIES   Maximum attempts per request (default 3)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, TypeVar

from .key_manager import APIKeyManager, AllKeysExhausted

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "3"))
_BASE_DELAY_SECS = 1.0


def call_with_retry(
    fn: Callable[..., T],
    key_manager: APIKeyManager,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Call fn(*args, api_key=<current_key>, **kwargs) with retry and key rotation.

    Retry behaviour:
        - Up to _MAX_RETRIES attempts total.
        - Exponential backoff: 1s, 2s, 4s between attempts.
        - Each failure reports to key_manager and rotates to the next key.
        - Circuit breaker: stops immediately when all keys are dead.

    Args:
        fn:          Callable that accepts api_key as a keyword argument.
        key_manager: Pool of API keys.
        *args:       Positional arguments forwarded to fn.
        **kwargs:    Keyword arguments forwarded to fn (must NOT include api_key).

    Returns:
        Whatever fn returns on success.

    Raises:
        AllKeysExhausted: When all keys are dead or max retries are exceeded.
        Any non-retriable exception raised by fn.
    """
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    _RETRIABLE = (RateLimitError, APITimeoutError, APIConnectionError)

    last_exc: Exception = AllKeysExhausted("No attempts made.")

    for attempt in range(_MAX_RETRIES):
        try:
            key = key_manager.get_key()
        except AllKeysExhausted:
            raise

        try:
            result = fn(*args, api_key=key, **kwargs)
            key_manager.report_success(key)
            return result

        except AllKeysExhausted:
            raise

        except _RETRIABLE as exc:
            last_exc = exc
            key_manager.report_failure(key)

            if key_manager.all_dead():
                raise AllKeysExhausted(
                    "All API keys are exhausted (rate-limited or timed out)."
                ) from exc

            delay = _BASE_DELAY_SECS * (2 ** attempt)
            logger.warning(
                "API call attempt %d/%d failed (%s). Rotating key and retrying in %.1fs.",
                attempt + 1,
                _MAX_RETRIES,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)

        except Exception:
            raise

    raise AllKeysExhausted(
        f"Request failed after {_MAX_RETRIES} attempt(s). Last error: {last_exc}"
    ) from last_exc
