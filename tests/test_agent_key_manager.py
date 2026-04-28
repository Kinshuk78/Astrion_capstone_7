"""Tests for the multi-key API key manager and circuit breaker.

Covers:
    - Single and multi-key construction
    - from_env() reading OPENAI_API_KEYS and OPENAI_API_KEY
    - Key rotation after failure
    - Circuit breaker: key marked dead after threshold failures
    - AllKeysExhausted when all keys are dead
    - reset() revives all keys
    - Success resets the failure counter
    - all_dead() and live_key_count()
"""
from __future__ import annotations

import pytest

from astrion_dq.agent.key_manager import (
    APIKeyManager,
    AllKeysExhausted,
    _CIRCUIT_BREAKER_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_single_key_construction():
    mgr = APIKeyManager(["sk-test-1"])
    assert mgr.key_count() == 1
    assert mgr.live_key_count() == 1
    assert mgr.get_key() == "sk-test-1"


def test_multi_key_construction():
    mgr = APIKeyManager(["sk-a", "sk-b", "sk-c"])
    assert mgr.key_count() == 3
    assert mgr.live_key_count() == 3


def test_empty_keys_raises_on_get_key():
    mgr = APIKeyManager([])
    with pytest.raises(AllKeysExhausted):
        mgr.get_key()


def test_whitespace_only_keys_ignored():
    mgr = APIKeyManager(["  ", "", "sk-valid"])
    assert mgr.key_count() == 1
    assert mgr.get_key() == "sk-valid"


# ---------------------------------------------------------------------------
# from_env()
# ---------------------------------------------------------------------------

def test_from_env_reads_multiple_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEYS", "sk-1,sk-2,sk-3")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mgr = APIKeyManager.from_env()
    assert mgr.key_count() == 3


def test_from_env_falls_back_to_single_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-solo")
    mgr = APIKeyManager.from_env()
    assert mgr.key_count() == 1
    assert mgr.get_key() == "sk-solo"


def test_from_env_no_keys_raises_on_get(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mgr = APIKeyManager.from_env()
    with pytest.raises(AllKeysExhausted):
        mgr.get_key()


def test_from_env_strips_whitespace(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEYS", " sk-x , sk-y ")
    mgr = APIKeyManager.from_env()
    assert mgr.key_count() == 2


# ---------------------------------------------------------------------------
# Rotation and failure
# ---------------------------------------------------------------------------

def test_rotation_on_failure():
    mgr = APIKeyManager(["sk-a", "sk-b"])
    first = mgr.get_key()
    assert first == "sk-a"
    mgr.report_failure("sk-a")
    # After one failure sk-a is not dead yet (threshold is 3), but manager
    # still rotates the current index so next get_key() returns sk-b
    second = mgr.get_key()
    assert second == "sk-b"


def test_success_resets_failure_counter():
    mgr = APIKeyManager(["sk-only"])
    mgr.report_failure("sk-only")
    mgr.report_failure("sk-only")
    mgr.report_success("sk-only")
    # Counter reset — key should still be alive
    assert mgr.live_key_count() == 1
    assert mgr.get_key() == "sk-only"


def test_success_revives_dead_key():
    mgr = APIKeyManager(["sk-only"])
    for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
        mgr.report_failure("sk-only")
    assert mgr.all_dead()
    mgr.report_success("sk-only")
    assert mgr.live_key_count() == 1


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

def test_circuit_breaker_marks_key_dead():
    mgr = APIKeyManager(["sk-dead"])
    for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
        mgr.report_failure("sk-dead")
    assert mgr.all_dead()
    with pytest.raises(AllKeysExhausted):
        mgr.get_key()


def test_circuit_breaker_threshold_not_yet_dead():
    mgr = APIKeyManager(["sk-alive"])
    for _ in range(_CIRCUIT_BREAKER_THRESHOLD - 1):
        mgr.report_failure("sk-alive")
    assert not mgr.all_dead()
    assert mgr.get_key() == "sk-alive"


def test_multi_key_one_dead_falls_through():
    mgr = APIKeyManager(["sk-a", "sk-b"])
    for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
        mgr.report_failure("sk-a")
    # sk-a is dead, sk-b should be returned
    assert mgr.get_key() == "sk-b"
    assert mgr.live_key_count() == 1


def test_all_dead_raises():
    mgr = APIKeyManager(["sk-1", "sk-2"])
    for key in ["sk-1", "sk-2"]:
        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            mgr.report_failure(key)
    assert mgr.all_dead()
    with pytest.raises(AllKeysExhausted):
        mgr.get_key()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_revives_all_dead_keys():
    mgr = APIKeyManager(["sk-1", "sk-2"])
    for key in ["sk-1", "sk-2"]:
        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            mgr.report_failure(key)
    assert mgr.all_dead()
    mgr.reset()
    assert not mgr.all_dead()
    assert mgr.live_key_count() == 2
    assert mgr.get_key() in ("sk-1", "sk-2")


def test_reset_on_fresh_manager_is_safe():
    mgr = APIKeyManager(["sk-x"])
    mgr.reset()
    assert mgr.get_key() == "sk-x"


# ---------------------------------------------------------------------------
# all_dead() and live_key_count()
# ---------------------------------------------------------------------------

def test_all_dead_false_when_keys_alive():
    mgr = APIKeyManager(["sk-a"])
    assert not mgr.all_dead()


def test_live_key_count_decreases_on_circuit_break():
    mgr = APIKeyManager(["sk-1", "sk-2", "sk-3"])
    for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
        mgr.report_failure("sk-1")
    assert mgr.live_key_count() == 2
