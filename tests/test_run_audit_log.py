"""P6a: run audit log written to outputs/run_log.jsonl.

Pre-fix expected failures:
  test_triage_appends_run_log_entry -- run_log.jsonl not written yet
  test_run_log_entry_schema         -- run_id field not present yet
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _fake_state(source: str = "injected") -> dict:
    return {
        "source": source,
        "sensitivity": "normal",
        "tables": {},
        "table_sizes": {},
        "db_path": "",
        "metadata": {},
        "raw_issues": [],
        "drift_issues": [],
        "all_issues": [],
        "verified_issues": [],
        "ranked_issues": [
            {"issue_id": "X001", "issue_type": "missing_values"},
            {"issue_id": "X002", "issue_type": "duplicate_rows"},
        ],
        "data_loaded": False,
        "metadata_ready": False,
        "detection_done": False,
        "drift_done": False,
        "debug_done": False,
        "review_done": False,
        "ranking_done": False,
        "needs_human_review": False,
        "human_decision": None,
        "agent_trace": ["data_loader", "profiler", "detector", "ranker", "summariser"],
        "timing": {},
        "report_md": "# test",
        "error": None,
    }


def test_triage_appends_run_log_entry(tmp_path):
    """triage must append one JSONL record to outputs/run_log.jsonl."""
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_state("injected")

    import astrion_dq.config as cfg
    from typer.testing import CliRunner
    from astrion_dq.cli import app

    runner = CliRunner()
    with (
        patch("astrion_dq.graph.workflow.build_graph", return_value=fake_graph),
        patch("astrion_dq.graph.state.initial_state", return_value=_fake_state("injected")),
        patch("astrion_dq.warehouse.loader.close_connection"),
        patch.object(cfg, "OUTPUTS_DIR", tmp_path),
    ):
        result = runner.invoke(app, ["triage", "--source", "injected", "--auto-approve"])

    assert result.exit_code == 0, f"triage failed:\n{result.output}"

    log_path = tmp_path / "run_log.jsonl"
    assert log_path.exists(), "triage must create outputs/run_log.jsonl"

    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected 1 log entry, got {len(lines)}"

    entry = json.loads(lines[0])
    assert "run_id" in entry, "log entry must have run_id"
    assert "source" in entry
    assert "sensitivity" in entry
    assert "timestamp" in entry
    assert "issue_count" in entry
    assert "agent_trace" in entry


def test_run_log_entry_schema(tmp_path):
    """Log entry values must match the triage run parameters."""
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_state("clean")

    import astrion_dq.config as cfg
    from typer.testing import CliRunner
    from astrion_dq.cli import app

    runner = CliRunner()
    with (
        patch("astrion_dq.graph.workflow.build_graph", return_value=fake_graph),
        patch("astrion_dq.graph.state.initial_state", return_value=_fake_state("clean")),
        patch("astrion_dq.warehouse.loader.close_connection"),
        patch.object(cfg, "OUTPUTS_DIR", tmp_path),
    ):
        result = runner.invoke(app, ["triage", "--source", "clean", "--auto-approve"])

    assert result.exit_code == 0, f"triage failed:\n{result.output}"

    log_path = tmp_path / "run_log.jsonl"
    entry = json.loads(log_path.read_text().strip())

    assert entry["source"] == "clean"
    assert entry["sensitivity"] == "high"   # always "high" — sensitivity removed as user choice
    assert entry["issue_count"] == 2  # _fake_state has 2 ranked_issues
    assert len(entry["run_id"]) == 12, "run_id must be a 12-char hex string"
    assert isinstance(entry["agent_trace"], list)


def test_run_log_appends_not_overwrites(tmp_path):
    """Two successive triage calls must append, not overwrite, run_log.jsonl."""
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_state("injected")

    import astrion_dq.config as cfg
    from typer.testing import CliRunner
    from astrion_dq.cli import app

    runner = CliRunner()
    ctx = (
        patch("astrion_dq.graph.workflow.build_graph", return_value=fake_graph),
        patch("astrion_dq.graph.state.initial_state", return_value=_fake_state("injected")),
        patch("astrion_dq.warehouse.loader.close_connection"),
        patch.object(cfg, "OUTPUTS_DIR", tmp_path),
    )

    with ctx[0], ctx[1], ctx[2], ctx[3]:
        runner.invoke(app, ["triage", "--source", "injected", "--auto-approve"])
        runner.invoke(app, ["triage", "--source", "injected", "--auto-approve"])

    log_path = tmp_path / "run_log.jsonl"
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2, "two triage runs must produce two log entries"
