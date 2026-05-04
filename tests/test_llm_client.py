"""LLM client unit tests.

Tests the graceful-degradation path -- LLMUnavailable is raised when
OPENROUTER_API_KEY is absent, and callers (summariser_node) handle it
by falling back to the template summary.

These tests never make real network calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from astrion_dq.llm.client import LLMUnavailable, chat, chat_with_history, reset_client


@pytest.fixture(autouse=True)
def _reset():
    """Reset the singleton before and after each test."""
    reset_client()
    yield
    reset_client()


def test_llm_unavailable_when_no_api_key(monkeypatch):
    """chat() raises LLMUnavailable when OPENROUTER_API_KEY is empty."""
    monkeypatch.setattr("astrion_dq.config.OPENROUTER_API_KEY", "")
    with pytest.raises(LLMUnavailable, match="OPENROUTER_API_KEY"):
        chat("test prompt")


def test_llm_unavailable_is_runtime_error():
    """LLMUnavailable is a RuntimeError subclass so callers can catch broadly."""
    assert issubclass(LLMUnavailable, RuntimeError)


def test_chat_returns_response_when_key_set(monkeypatch):
    """chat() returns model response text when API key is configured."""
    monkeypatch.setattr("astrion_dq.config.OPENROUTER_API_KEY", "sk-or-test-key")

    fake_message = MagicMock()
    fake_message.content = "  Executive summary text.  "
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("astrion_dq.llm.client._client", fake_client):
        result = chat("Summarise these issues.")

    assert result == "Executive summary text."
    fake_client.chat.completions.create.assert_called_once()


def test_summariser_node_graceful_without_api_key(monkeypatch):
    """summariser_node produces a report even when OPENROUTER_API_KEY is absent."""
    monkeypatch.setattr("astrion_dq.config.OPENROUTER_API_KEY", "")

    from astrion_dq.graph.nodes import summariser_node

    state = {
        "source": "injected",
        "sensitivity": "normal",
        "ranked_issues": [
            {
                "issue_id": "X001",
                "issue_type": "missing_values",
                "table": "fact_sales",
                "columns": ["sales_sk"],
                "severity": "high",
                "evidence_rows": 20,
                "impact_score": 1.5,
                "confidence": 1.0,
                "affected_reports": ["daily_sales_summary"],
                "description": "Column 'sales_sk' has 2.00% missing values.",
                "agent_trace": [],
                "metric": 0.02,
            }
        ],
        "agent_trace": ["data_loader", "ranker"],
        "timing": {"data_loader": 0.1},
    }

    result = summariser_node(state)

    assert "report_md" in result
    assert "Astrion Data Quality Triage Report" in result["report_md"]
    assert "missing_values" in result["report_md"]
    # No executive summary section without API key
    assert "Executive Summary" not in result["report_md"]


def test_chat_with_history_retries_on_openrouter_budget_error(monkeypatch):
    """A 402 budget error should trigger one retry with a lower max_tokens."""
    monkeypatch.setattr("astrion_dq.config.OPENROUTER_API_KEY", "sk-or-test-key")

    fake_message = MagicMock()
    fake_message.content = "retry success"
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    calls = []

    def _create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "Error code: 402 - {'error': {'message': 'This request requires more credits, "
                "or fewer max_tokens. You requested up to 1200 tokens, but can only afford 1109.'}}"
            )
        return fake_response

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _create

    with patch("astrion_dq.llm.client._client", fake_client):
        result = chat_with_history(
            [{"role": "user", "content": "help me"}],
            system="You are helpful",
            max_tokens=1200,
        )

    assert result == "retry success"
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 1200
    assert calls[1]["max_tokens"] < 1200
