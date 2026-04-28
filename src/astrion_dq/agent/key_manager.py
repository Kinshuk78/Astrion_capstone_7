"""Multi-key API key manager with per-key circuit breaker.

Accepts multiple OpenAI API keys from the environment and rotates through them
on failure. A key is marked dead after AGENT_CIRCUIT_BREAKER_THRESHOLD
consecutive failures. Raises AllKeysExhausted when every key is dead or when
no keys are configured.

Environment variables:
    OPENAI_API_KEYS       Comma-separated list of keys (e.g. sk-a,sk-b,sk-c)
    OPENAI_API_KEY        Single key fallback when OPENAI_API_KEYS is unset
    AGENT_CIRCUIT_BREAKER_THRESHOLD   Consecutive failures before a key is dead (default 3)
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("AGENT_CIRCUIT_BREAKER_THRESHOLD", "3"))


class AllKeysExhausted(Exception):
    """Raised when every configured API key has exceeded the failure threshold
    or when no keys are configured at all."""


@dataclass
class _KeyState:
    key: str
    consecutive_failures: int = 0
    dead: bool = False


class APIKeyManager:
    """Thread-safe pool of OpenAI API keys with rotation and circuit breaking."""

    def __init__(self, keys: list[str]) -> None:
        self._states: list[_KeyState] = [
            _KeyState(key=k) for k in keys if k.strip()
        ]
        self._current: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        primary_var: str = "OPENAI_API_KEYS",
        fallback_var: str = "OPENAI_API_KEY",
    ) -> "APIKeyManager":
        """Build a manager from environment variables.

        Reads OPENAI_API_KEYS (comma-separated) first; falls back to
        OPENAI_API_KEY (single key). Returns a manager with whatever keys
        are available — callers will receive AllKeysExhausted from get_key()
        if the list is empty.
        """
        raw = os.getenv(primary_var, "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            single = os.getenv(fallback_var, "").strip()
            if single:
                keys = [single]
        return cls(keys)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_key(self) -> str:
        """Return the current available key.

        Rotates through the pool looking for the first non-dead key.
        Raises AllKeysExhausted if none are available.
        """
        with self._lock:
            if not self._states:
                raise AllKeysExhausted(
                    "No OpenAI API keys configured. "
                    "Set OPENAI_API_KEYS or OPENAI_API_KEY in your environment."
                )
            for _ in range(len(self._states)):
                state = self._states[self._current]
                if not state.dead:
                    return state.key
                self._current = (self._current + 1) % len(self._states)
            raise AllKeysExhausted(
                f"All {len(self._states)} configured API key(s) have exceeded "
                f"the failure threshold ({_CIRCUIT_BREAKER_THRESHOLD} consecutive failures). "
                "Rotate or replace keys and call reset() to recover."
            )

    def report_success(self, key: str) -> None:
        """Reset the failure counter for a key after a successful call."""
        with self._lock:
            for state in self._states:
                if state.key == key:
                    state.consecutive_failures = 0
                    state.dead = False
                    return

    def report_failure(self, key: str) -> None:
        """Increment the failure counter for a key.

        When the counter reaches _CIRCUIT_BREAKER_THRESHOLD the key is marked
        dead and the manager rotates to the next slot.
        """
        with self._lock:
            for idx, state in enumerate(self._states):
                if state.key == key:
                    state.consecutive_failures += 1
                    if state.consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                        state.dead = True
                        logger.warning(
                            "API key ...%s marked dead after %d consecutive failures.",
                            key[-4:],
                            state.consecutive_failures,
                        )
                    self._current = (idx + 1) % len(self._states)
                    return

    def all_dead(self) -> bool:
        """Return True if every key is currently dead."""
        with self._lock:
            return bool(self._states) and all(s.dead for s in self._states)

    def reset(self) -> None:
        """Revive all keys (used in tests or after manual key rotation)."""
        with self._lock:
            for state in self._states:
                state.consecutive_failures = 0
                state.dead = False
            self._current = 0
        logger.info("APIKeyManager reset — all %d key(s) marked alive.", len(self._states))

    def key_count(self) -> int:
        return len(self._states)

    def live_key_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._states if not s.dead)
