"""Tests for the new agent-layer REST endpoints.

Endpoints:
    POST /analyze      -- triage + AI analysis (async, 202)
    POST /explain      -- business explanations for issues
    POST /prioritise   -- AI-ranked priority list
    POST /generate-fix -- SQL + Python fix code for one issue
    POST /report       -- executive business report

All endpoints use the fallback path (no LLM calls) because no OPENAI_API_KEYS
are configured in the test environment. Tests verify:
    - Correct HTTP status codes
    - Response structure matches contract
    - Auth is enforced (401 on missing/wrong token)
    - 422 validation on bad input
    - Fallback produces structurally correct output
"""
from __future__ import annotations

import pytest
from unittest.mock import patch


_AUTH = {"Authorization": "Bearer test-agent-token"}


def _sample_issue(
    issue_type: str = "missing_values",
    issue_id: str = "X001",
) -> dict:
    return {
        "issue_id": issue_id,
        "issue_type": issue_type,
        "table": "fact_sales",
        "columns": ["amount"],
        "severity": "high",
        "metric": 0.15,
        "evidence_rows": 1500,
        "impact_score": 7.2,
        "confidence": 0.95,
        "description": "15% of rows have null amounts",
        "dim_table": "",
        "dim_pk": "",
    }


def _sample_issues() -> list[dict]:
    return [
        _sample_issue("missing_values", "X001"),
        _sample_issue("duplicate_rows", "X002"),
    ]


# ---------------------------------------------------------------------------
# Fixture: TestClient with token configured and no OpenAI keys
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent_client(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRION_API_TOKEN", "test-agent-token")
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)

    from fastapi.testclient import TestClient
    from astrion_dq.api.app import app
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# POST /explain
# ---------------------------------------------------------------------------

def test_explain_returns_200(agent_client):
    resp = agent_client.post(
        "/explain",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    assert resp.status_code == 200


def test_explain_response_has_explanations_key(agent_client):
    resp = agent_client.post(
        "/explain",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    body = resp.json()
    assert "explanations" in body
    assert isinstance(body["explanations"], list)


def test_explain_used_fallback_flag_present(agent_client):
    resp = agent_client.post(
        "/explain",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    assert "used_fallback" in resp.json()


def test_explain_fallback_source_when_no_keys(agent_client):
    resp = agent_client.post(
        "/explain",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    body = resp.json()
    assert body["used_fallback"] is True
    for exp in body["explanations"]:
        assert exp["source"] == "fallback"


def test_explain_each_explanation_has_required_fields(agent_client):
    resp = agent_client.post(
        "/explain",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    for exp in resp.json()["explanations"]:
        assert "issue_id" in exp
        assert "business_explanation" in exp


def test_explain_empty_issues_returns_422(agent_client):
    resp = agent_client.post(
        "/explain",
        json={"issues": []},
        headers=_AUTH,
    )
    assert resp.status_code == 422


def test_explain_requires_auth(agent_client):
    resp = agent_client.post("/explain", json={"issues": _sample_issues()})
    assert resp.status_code == 401


def test_explain_wrong_token_returns_401(agent_client):
    resp = agent_client.post(
        "/explain",
        json={"issues": _sample_issues()},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /prioritise
# ---------------------------------------------------------------------------

def test_prioritise_returns_200(agent_client):
    resp = agent_client.post(
        "/prioritise",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    assert resp.status_code == 200


def test_prioritise_response_has_prioritised_issues(agent_client):
    resp = agent_client.post(
        "/prioritise",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    body = resp.json()
    assert "prioritised_issues" in body
    assert isinstance(body["prioritised_issues"], list)


def test_prioritise_result_count_matches_input(agent_client):
    issues = _sample_issues()
    resp = agent_client.post(
        "/prioritise",
        json={"issues": issues},
        headers=_AUTH,
    )
    assert len(resp.json()["prioritised_issues"]) == len(issues)


def test_prioritise_each_item_has_priority_rank(agent_client):
    resp = agent_client.post(
        "/prioritise",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    for item in resp.json()["prioritised_issues"]:
        assert "priority_rank" in item


def test_prioritise_empty_issues_returns_422(agent_client):
    resp = agent_client.post(
        "/prioritise",
        json={"issues": []},
        headers=_AUTH,
    )
    assert resp.status_code == 422


def test_prioritise_requires_auth(agent_client):
    resp = agent_client.post("/prioritise", json={"issues": _sample_issues()})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /generate-fix
# ---------------------------------------------------------------------------

def test_generate_fix_returns_200(agent_client):
    resp = agent_client.post(
        "/generate-fix",
        json={"issue": _sample_issue()},
        headers=_AUTH,
    )
    assert resp.status_code == 200


def test_generate_fix_has_sql_and_python(agent_client):
    resp = agent_client.post(
        "/generate-fix",
        json={"issue": _sample_issue()},
        headers=_AUTH,
    )
    body = resp.json()
    assert "sql_fix" in body
    assert "python_fix" in body
    assert body["sql_fix"]
    assert body["python_fix"]


def test_generate_fix_fallback_source_when_no_keys(agent_client):
    resp = agent_client.post(
        "/generate-fix",
        json={"issue": _sample_issue()},
        headers=_AUTH,
    )
    assert resp.json().get("source") == "fallback"


def test_generate_fix_empty_issue_returns_422(agent_client):
    resp = agent_client.post(
        "/generate-fix",
        json={"issue": {}},
        headers=_AUTH,
    )
    assert resp.status_code == 422


def test_generate_fix_requires_auth(agent_client):
    resp = agent_client.post("/generate-fix", json={"issue": _sample_issue()})
    assert resp.status_code == 401


@pytest.mark.parametrize("issue_type", [
    "missing_values", "duplicate_rows", "numeric_outliers",
    "invalid_future_dates", "referential_integrity_break",
    "empty_table", "statistical_drift",
])
def test_generate_fix_all_issue_types_return_200(agent_client, issue_type):
    resp = agent_client.post(
        "/generate-fix",
        json={"issue": _sample_issue(issue_type)},
        headers=_AUTH,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sql_fix"]
    assert body["python_fix"]


# ---------------------------------------------------------------------------
# POST /report
# ---------------------------------------------------------------------------

def test_report_returns_200(agent_client):
    resp = agent_client.post(
        "/report",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    assert resp.status_code == 200


def test_report_has_executive_summary(agent_client):
    resp = agent_client.post(
        "/report",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    body = resp.json()
    assert "executive_summary" in body
    assert body["executive_summary"]


def test_report_has_required_fields(agent_client):
    resp = agent_client.post(
        "/report",
        json={"issues": _sample_issues()},
        headers=_AUTH,
    )
    body = resp.json()
    assert "top_risks" in body
    assert "recommended_actions" in body
    assert "overall_data_health" in body
    assert "total_issues" in body


def test_report_empty_issues_returns_safe_response(agent_client):
    resp = agent_client.post(
        "/report",
        json={"issues": []},
        headers=_AUTH,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_data_health"] == "excellent"


def test_report_requires_auth(agent_client):
    resp = agent_client.post("/report", json={"issues": _sample_issues()})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /assistant/chat and /assistant/summary
# ---------------------------------------------------------------------------

def test_assistant_chat_returns_200(agent_client):
    with patch("astrion_dq.llm.client.chat_with_history", return_value="```sql\nSELECT 1;\n```"):
        resp = agent_client.post(
            "/assistant/chat",
            json={
                "message": "Show me a query",
                "history": [{"role": "user", "content": "Previous turn"}],
                "schema_desc": "dq_retail.fact_sales(amount DOUBLE)",
                "issues": _sample_issues(),
            },
            headers=_AUTH,
        )

    assert resp.status_code == 200
    assert resp.json()["response"]
    assert resp.json()["used_fallback"] is False


def test_assistant_chat_uses_fallback_when_llm_unavailable(agent_client):
    from astrion_dq.llm.client import LLMUnavailable

    with patch("astrion_dq.llm.client.chat_with_history", side_effect=LLMUnavailable("missing key")):
        resp = agent_client.post(
            "/assistant/chat",
            json={"message": "Explain the top issue", "issues": _sample_issues()},
            headers=_AUTH,
        )

    assert resp.status_code == 200
    assert resp.json()["used_fallback"] is True
    assert "template-only" in resp.json()["response"]


def test_assistant_chat_requires_auth(agent_client):
    resp = agent_client.post("/assistant/chat", json={"message": "hello"})
    assert resp.status_code == 401


def test_assistant_summary_returns_200(agent_client):
    with patch("astrion_dq.graph.nodes._llm_executive_summary", return_value="Executive summary text"):
        resp = agent_client.post(
            "/assistant/summary",
            json={"issues": _sample_issues(), "source": "injected"},
            headers=_AUTH,
        )

    assert resp.status_code == 200
    assert resp.json()["summary"] == "Executive summary text"
    assert resp.json()["used_fallback"] is False


def test_assistant_summary_uses_fallback_when_llm_returns_empty(agent_client):
    with patch("astrion_dq.graph.nodes._llm_executive_summary", return_value=""):
        resp = agent_client.post(
            "/assistant/summary",
            json={"issues": _sample_issues(), "source": "injected"},
            headers=_AUTH,
        )

    assert resp.status_code == 200
    assert resp.json()["used_fallback"] is True
    assert "Recommended Actions" in resp.json()["summary"]


# ---------------------------------------------------------------------------
# POST /analyze (async job)
# ---------------------------------------------------------------------------

def test_analyze_returns_202(agent_client):
    from astrion_dq.api.app import TriageResult

    def _fake_analyze(source: str) -> TriageResult:
        return TriageResult(
            run_id="analyzerun123",
            source=source,
            issue_count=1,
            ranked_issues=[_sample_issue()],
            agent_trace=["data_loader", "ranker"],
        )

    with patch("astrion_dq.api.app._execute_triage", side_effect=_fake_analyze):
        resp = agent_client.post(
            "/analyze",
            json={"source": "injected"},
            headers=_AUTH,
        )

    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "running"
    assert body["poll_url"].startswith("/jobs/")


def test_analyze_invalid_source_returns_422(agent_client):
    resp = agent_client.post(
        "/analyze",
        json={"source": "invalid"},
        headers=_AUTH,
    )
    assert resp.status_code == 422


def test_analyze_requires_auth(agent_client):
    resp = agent_client.post("/analyze", json={"source": "injected"})
    assert resp.status_code == 401
