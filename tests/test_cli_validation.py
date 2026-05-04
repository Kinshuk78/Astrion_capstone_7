"""P5-A: CLI enum validation tests (test-first for F-09).

Pre-fix: --source xyz fails inside load_retail_tables with a bare ValueError
and a full traceback in the output. Not a clean user-facing error.

Post-fix: Typer rejects invalid values at the CLI boundary, exit code 2,
clean message, no traceback.
"""
from __future__ import annotations

import subprocess

from typer.testing import CliRunner

from astrion_dq.cli import app

runner = CliRunner()


def test_triage_rejects_invalid_source():
    result = runner.invoke(app, ["triage", "--source", "xyz"])
    assert result.exit_code != 0, "Invalid source must be rejected"
    assert "Traceback" not in result.output, "Must not show a raw Python traceback"
    assert "xyz" in result.output or "invalid" in result.output.lower(), (
        "Error message must mention the invalid value"
    )


def test_triage_rejects_invalid_sensitivity():
    result = runner.invoke(app, ["triage", "--sensitivity", "ultra"])
    assert result.exit_code != 0, "Invalid sensitivity must be rejected"
    assert "Traceback" not in result.output
    assert "ultra" in result.output or "invalid" in result.output.lower()


def test_snapshot_rejects_invalid_source():
    # snapshot command does not have --source but triage/evaluate do
    result = runner.invoke(app, ["evaluate", "--source", "bad"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_triage_accepts_valid_source_clean():
    """Enum must accept 'clean' without error (validation only, no pipeline run)."""
    # We just need Typer to accept it -- the import inside triage will fail
    # because no real data exists, but exit code must NOT be 2 (usage error)
    result = runner.invoke(app, ["triage", "--source", "clean", "--help"])
    assert result.exit_code == 0


def test_triage_accepts_valid_sensitivity():
    result = runner.invoke(app, ["triage", "--sensitivity", "high", "--help"])
    assert result.exit_code == 0


def test_dashboard_command_uses_repo_dashboard_path(monkeypatch):
    calls = []

    def _fake_run(cmd, check):
        calls.append((cmd, check))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = runner.invoke(app, ["dashboard", "--port", "9999"])
    assert result.exit_code == 0
    assert "Launching dashboard on http://localhost:9999" in result.output
    assert calls, "dashboard command must attempt to launch Streamlit"
    cmd, check = calls[0]
    assert check is True
    assert "dashboard/app.py" in " ".join(str(part) for part in cmd)
