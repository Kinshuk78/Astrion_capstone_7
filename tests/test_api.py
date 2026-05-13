"""FastAPI REST endpoint tests.

Endpoints:
  POST /triage               -- submit async triage job, returns 202 + job_id
  GET  /jobs/{job_id}        -- poll job status and result
  GET  /runs/{run_id}        -- look up a past run from run_log.jsonl
  GET  /health               -- unauthenticated liveness probe

Auth: Bearer token from ASTRION_API_TOKEN env var.
  - Missing / wrong token -> 401 on protected endpoints.
  - /health is always open.

Async pattern:
  POST /triage  -> 202 {"job_id": ..., "status": "running", "poll_url": ...}
  GET  /jobs/{job_id}  ->  {"status": "done", "result": {...}}

Tests patch _execute_triage so the job completes synchronously in the thread pool.
"""
from __future__ import annotations

import json
import time
from unittest.mock import patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ranked_issues():
    return [
        {"issue_id": "X001", "issue_type": "missing_values", "severity": "high",
         "table": "fact_sales", "columns": ["amount"], "metric": 0.1,
         "evidence_rows": 10, "description": "10% nulls", "impact_score": 5.0,
         "affected_reports": [], "agent_trace": [], "confidence": 0.95,
         "dim_table": "", "dim_pk": ""},
    ]


def _fake_triage_result(source: str = "injected"):
    """Return a TriageResult-like object matching the async worker's return type."""
    from astrion_dq.api.app import TriageResult
    return TriageResult(
        run_id="testrun123456",
        source=source,
        issue_count=len(_ranked_issues()),
        ranked_issues=_ranked_issues(),
        agent_trace=["data_loader", "profiler", "detector", "ranker", "summariser"],
    )


def _poll_job(client, job_id: str, headers: dict, timeout_secs: float = 5.0) -> object:
    """Poll GET /jobs/{job_id} until status != 'running' or timeout."""
    deadline = time.monotonic() + timeout_secs
    resp = None
    while time.monotonic() < deadline:
        resp = client.get(f"/jobs/{job_id}", headers=headers)
        if resp.status_code == 200 and resp.json().get("status") != "running":
            return resp
        time.sleep(0.05)
    return resp  # return last response even on timeout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """TestClient with a patched OUTPUTS_DIR and a required API token."""
    monkeypatch.setenv("ASTRION_API_TOKEN", "test-token-abc")

    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SNAPSHOTS_DIR", tmp_path / "drift_snapshots")

    from fastapi.testclient import TestClient
    from astrion_dq.api.app import app

    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def api_client_no_token(tmp_path, monkeypatch):
    """TestClient without a configured API token (token guard disabled)."""
    monkeypatch.delenv("ASTRION_API_TOKEN", raising=False)

    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SNAPSHOTS_DIR", tmp_path / "drift_snapshots")

    from fastapi.testclient import TestClient
    from astrion_dq.api.app import app

    return TestClient(app, raise_server_exceptions=True)


_AUTH = {"Authorization": "Bearer test-token-abc"}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_no_auth(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_triage_requires_auth(api_client):
    resp = api_client.post("/triage", json={"source": "injected"})
    assert resp.status_code == 401


def test_triage_wrong_token_returns_401(api_client):
    resp = api_client.post(
        "/triage",
        json={"source": "injected"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_no_token_configured_allows_all(api_client_no_token):
    """When ASTRION_API_TOKEN is unset, POST /triage accepts any request."""
    with patch("astrion_dq.api.app._execute_triage", return_value=_fake_triage_result()):
        resp = api_client_no_token.post("/triage", json={"source": "injected"})

    assert resp.status_code == 202
    assert resp.json()["status"] == "running"
    assert "job_id" in resp.json()


# ---------------------------------------------------------------------------
# POST /triage → 202 async
# ---------------------------------------------------------------------------

def test_post_triage_returns_202_and_job_id(api_client):
    """POST /triage must return 202 with job_id immediately."""
    with patch("astrion_dq.api.app._execute_triage", return_value=_fake_triage_result()):
        resp = api_client.post("/triage", json={"source": "injected"}, headers=_AUTH)

    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "running"
    assert body["poll_url"].startswith("/jobs/")


def test_post_triage_job_completes(api_client):
    """Polling GET /jobs/{job_id} after submission must return status='done'."""
    with patch("astrion_dq.api.app._execute_triage", return_value=_fake_triage_result()):
        submit = api_client.post("/triage", json={"source": "injected"}, headers=_AUTH)

    job_id = submit.json()["job_id"]
    result_resp = _poll_job(api_client, job_id, _AUTH)

    assert result_resp.status_code == 200
    body = result_resp.json()
    assert body["status"] == "done"
    assert "result" in body
    assert body["result"]["issue_count"] == 1
    assert body["result"]["ranked_issues"][0]["issue_id"] == "X001"
    assert "agent_trace" in body["result"]


def test_post_triage_returns_ranked_issues(api_client):
    """The job result must contain ranked_issues and run_id."""
    with patch("astrion_dq.api.app._execute_triage", return_value=_fake_triage_result()):
        submit = api_client.post("/triage", json={"source": "injected"}, headers=_AUTH)

    job_id = submit.json()["job_id"]
    result_resp = _poll_job(api_client, job_id, _AUTH)

    result = result_resp.json()["result"]
    assert "run_id" in result
    assert len(result["ranked_issues"]) == 1
    assert result["issue_count"] == 1


def test_post_triage_writes_run_log(api_client, tmp_path, monkeypatch):
    """Completed triage job must append a record to run_log.jsonl.

    _execute_triage calls _write_run_log internally. Use a side_effect that
    replicates that call so the test can verify the file is written.
    """
    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)

    def _fake_with_log(source: str):
        from astrion_dq.api.app import _write_run_log
        r = _fake_triage_result(source)
        _write_run_log({
            "run_id": r.run_id,
            "source": source,
            "sensitivity": "high",
            "timestamp": "2026-04-22T00:00:00+00:00",
            "issue_count": r.issue_count,
            "agent_trace": r.agent_trace,
        })
        return r

    with patch("astrion_dq.api.app._execute_triage", side_effect=_fake_with_log):
        submit = api_client.post("/triage", json={"source": "injected"}, headers=_AUTH)

    job_id = submit.json()["job_id"]
    _poll_job(api_client, job_id, _AUTH)

    log_path = tmp_path / "run_log.jsonl"
    assert log_path.exists(), "Completed job must write run_log.jsonl"
    entry = json.loads(log_path.read_text().strip())
    assert entry["source"] == "injected"
    assert entry["issue_count"] == 1


def test_post_triage_invalid_source(api_client):
    resp = api_client.post("/triage", json={"source": "xyz"}, headers=_AUTH)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------

def test_get_job_not_found(api_client):
    resp = api_client.get("/jobs/nonexistent123", headers=_AUTH)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs/{run_id}
# ---------------------------------------------------------------------------

def test_get_run_returns_entry(api_client, tmp_path, monkeypatch):
    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)

    entry = {
        "run_id": "abc123def456",
        "source": "injected",
        "sensitivity": "high",
        "timestamp": "2026-04-21T10:00:00+00:00",
        "issue_count": 3,
        "agent_trace": ["data_loader", "ranker"],
    }
    (tmp_path / "run_log.jsonl").write_text(json.dumps(entry) + "\n")

    resp = api_client.get("/runs/abc123def456", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "abc123def456"
    assert resp.json()["issue_count"] == 3


def test_get_run_not_found(api_client, tmp_path, monkeypatch):
    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)

    (tmp_path / "run_log.jsonl").write_text("")

    resp = api_client.get("/runs/doesnotexist", headers=_AUTH)
    assert resp.status_code == 404


def test_persist_triage_artifacts_writes_ranked_and_report(tmp_path, monkeypatch):
    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)

    from astrion_dq.api.app import _persist_triage_artifacts

    _persist_triage_artifacts(
        "injected",
        _ranked_issues(),
        "# Demo report\n\nThis is a report.",
    )

    ranked_path = tmp_path / "ranked_issues_injected.json"
    report_path = tmp_path / "triage_report_injected.md"
    assert ranked_path.exists()
    assert report_path.exists()
    assert json.loads(ranked_path.read_text())[0]["issue_id"] == "X001"
    assert "Demo report" in report_path.read_text()


def test_outputs_ranked_issues_returns_artifact(api_client, tmp_path, monkeypatch):
    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)

    (tmp_path / "ranked_issues_injected.json").write_text(json.dumps(_ranked_issues()))
    resp = api_client.get("/outputs/ranked-issues?source=injected", headers=_AUTH)

    assert resp.status_code == 200
    assert resp.json()["issues"][0]["issue_id"] == "X001"


def test_outputs_report_returns_content(api_client, tmp_path, monkeypatch):
    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)

    (tmp_path / "triage_report_injected.md").write_text("# Report\n\nHello")
    resp = api_client.get("/outputs/report?source=injected", headers=_AUTH)

    assert resp.status_code == 200
    assert resp.json()["content"].startswith("# Report")


def test_outputs_evaluation_returns_results(api_client, tmp_path, monkeypatch):
    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)

    payload = [{"strategy": "A_baseline", "precision": 1.0}]
    (tmp_path / "evaluation_comparison.json").write_text(json.dumps(payload))
    resp = api_client.get("/outputs/evaluation", headers=_AUTH)

    assert resp.status_code == 200
    assert resp.json()["results"][0]["strategy"] == "A_baseline"


def test_outputs_run_log_returns_newest_first(api_client, tmp_path, monkeypatch):
    import astrion_dq.config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)

    entries = [
        {"run_id": "old", "timestamp": "2026-01-01T00:00:00+00:00"},
        {"run_id": "new", "timestamp": "2026-01-02T00:00:00+00:00"},
    ]
    (tmp_path / "run_log.jsonl").write_text("\n".join(json.dumps(item) for item in entries))
    resp = api_client.get("/outputs/run-log?limit=10", headers=_AUTH)

    assert resp.status_code == 200
    assert resp.json()["entries"][0]["run_id"] == "new"


def test_outputs_status_reports_readiness(api_client, tmp_path, monkeypatch):
    import astrion_dq.config as cfg
    snapshots_dir = tmp_path / "drift_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SNAPSHOTS_DIR", snapshots_dir)

    (tmp_path / "ranked_issues_injected.json").write_text(json.dumps(_ranked_issues()))
    (tmp_path / "triage_report_injected.md").write_text("# Report")
    (tmp_path / "evaluation_comparison.json").write_text("[]")
    (tmp_path / "run_log.jsonl").write_text("{}\n")
    (tmp_path / "retail_injected_issues.json").write_text("[]")
    (snapshots_dir / "snapshot_baseline.json").write_text("{}")

    resp = api_client.get("/outputs/status?source=injected", headers=_AUTH)
    body = resp.json()
    assert resp.status_code == 200
    assert body["baseline_snapshot"] is True
    assert body["ranked_issues"] is True
    assert body["report_md"] is True
    assert body["evaluation"] is True
    assert body["run_log"] is True
    assert body["ground_truth"] is True


def test_assistant_context_returns_schema(api_client):
    tables = {
        "fact_sales": pd.DataFrame(
            [{"transaction_id": 1, "amount": 42.0}]
        )
    }
    with patch("astrion_dq.warehouse.loader.load_retail_tables", return_value=tables):
        resp = api_client.get("/assistant/context?source=injected", headers=_AUTH)

    assert resp.status_code == 200
    body = resp.json()
    assert body["sql_ready"] is True
    assert "dq_retail.fact_sales" in body["schema_desc"]


def test_assistant_sql_executes_against_requested_source(api_client):
    tables = {
        "fact_sales": pd.DataFrame(
            [{"transaction_id": 1, "amount": 42.0}]
        )
    }
    with patch("astrion_dq.warehouse.loader.load_retail_tables", return_value=tables):
        resp = api_client.post(
            "/assistant/sql",
            json={
                "source": "injected",
                "sql_blocks": [
                    "SELECT transaction_id, amount FROM dq_retail.fact_sales ORDER BY transaction_id"
                ],
                "max_rows": 10,
            },
            headers=_AUTH,
        )

    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["columns"] == ["transaction_id", "amount"]
    assert result["rows"][0]["transaction_id"] == 1
    assert result["rows"][0]["amount"] == 42.0


def test_assistant_chat_populates_schema_when_missing(api_client):
    tables = {
        "fact_sales": pd.DataFrame(
            [{"transaction_id": 1, "amount": 42.0}]
        )
    }

    def _fake_chat(messages, system: str, max_tokens: int):
        assert "dq_retail.fact_sales" in system
        return "ok"

    with patch("astrion_dq.warehouse.loader.load_retail_tables", return_value=tables), \
         patch("astrion_dq.llm.client.chat_with_history", side_effect=_fake_chat):
        resp = api_client.post(
            "/assistant/chat",
            json={
                "message": "Show schema",
                "source": "injected",
                "history": [],
                "issues": [],
                "schema_desc": "",
            },
            headers=_AUTH,
        )

    assert resp.status_code == 200
    assert resp.json()["response"] == "ok"


def test_snapshot_endpoint_returns_path(api_client):
    with patch("astrion_dq.warehouse.loader.load_retail_tables", return_value={"fact_sales": []}), \
         patch("astrion_dq.checks.drift.save_snapshot", return_value="outputs/drift_snapshots/snapshot_baseline.json"):
        resp = api_client.post("/snapshot", json={"tag": "baseline"}, headers=_AUTH)

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "snapshot_baseline.json" in resp.json()["path"]


def test_inject_endpoint_returns_issue_count(api_client):
    with patch("astrion_dq.warehouse.loader.load_retail_tables", return_value={"fact_sales": []}), \
         patch("astrion_dq.injectors.retail_issues.inject_retail_issues", return_value=({}, [{"issue_id": "X001"}])):
        resp = api_client.post("/inject", json={"seed": 42}, headers=_AUTH)

    assert resp.status_code == 200
    assert resp.json()["issue_count"] == 1


def test_evaluate_endpoint_returns_results(api_client):
    payload = [{"strategy": "A_baseline", "precision": 1.0}]
    with patch("astrion_dq.evaluation.metrics.evaluate_all", return_value=payload):
        resp = api_client.post("/evaluate", json={"source": "injected"}, headers=_AUTH)

    assert resp.status_code == 200
    assert resp.json()["results"][0]["strategy"] == "A_baseline"
