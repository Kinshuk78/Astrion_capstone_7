"""P3-A: source-scoped output filenames (test-first for F-03).

Pre-fix expected failures:
  test_triage_writes_source_scoped_ranked_issues  -- triage writes ranked_issues.json (no suffix)
  test_report_command_reads_source_scoped_files   -- report command has no --source option
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _fake_state(source: str = "clean") -> dict:
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
        "ranked_issues": [{"issue_id": "X001", "issue_type": "missing_values"}],
        "data_loaded": False,
        "metadata_ready": False,
        "detection_done": False,
        "drift_done": False,
        "debug_done": False,
        "review_done": False,
        "ranking_done": False,
        "needs_human_review": False,
        "human_decision": None,
        "agent_trace": ["data_loader", "ranker"],
        "timing": {},
        "report_md": "# test report",
        "error": None,
    }


def test_triage_writes_source_scoped_ranked_issues(tmp_path):
    """triage must write ranked_issues_{source}.json, not ranked_issues.json."""
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
    assert (tmp_path / "ranked_issues_clean.json").exists(), (
        "triage must write ranked_issues_clean.json (source-scoped)"
    )
    assert not (tmp_path / "ranked_issues.json").exists(), (
        "triage must NOT write legacy ranked_issues.json"
    )
    assert (tmp_path / "triage_report_clean.md").exists(), (
        "triage must write triage_report_clean.md (source-scoped)"
    )
    assert not (tmp_path / "triage_report.md").exists(), (
        "triage must NOT write legacy triage_report.md"
    )


def test_report_command_reads_source_scoped_files(tmp_path):
    """report --source must read from ranked_issues_{source}.json."""
    source = "clean"
    ranked_data = [{"issue_id": "X001", "issue_type": "missing_values", "table": "fact_sales"}]
    (tmp_path / f"ranked_issues_{source}.json").write_text(
        json.dumps(ranked_data), encoding="utf-8"
    )
    (tmp_path / f"triage_report_{source}.md").write_text("# clean report", encoding="utf-8")

    import astrion_dq.config as cfg
    from typer.testing import CliRunner
    from astrion_dq.cli import app

    mock_generate = MagicMock(return_value=tmp_path / "triage_report.pdf")
    runner = CliRunner()
    with (
        patch("astrion_dq.report.pdf.generate_triage_report", mock_generate),
        patch.object(cfg, "OUTPUTS_DIR", tmp_path),
    ):
        result = runner.invoke(app, ["report", "--source", source])

    assert result.exit_code == 0, f"report failed:\n{result.output}"
    call_args = mock_generate.call_args
    assert call_args is not None, "generate_triage_report was not called"
    passed_ranked = call_args[0][0]
    assert passed_ranked == ranked_data, (
        f"report must read ranked_issues_{source}.json; got: {passed_ranked}"
    )
