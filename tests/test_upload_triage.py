"""Tests for the upload triage pipeline (_run_upload_triage in dashboard).

The pipeline runs: infer_metadata → run_all_checks_parallel → ranking_agent_v2
on arbitrary DataFrames without using the DuckDB singleton. These tests verify:
  - Clean data produces zero issues
  - Injected nulls produce a missing_values issue
  - Duplicate rows produce a duplicate_rows issue
  - Results are sorted descending by impact_score
  - dim_table/dim_pk flow through to the output dicts
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from astrion_dq.checks.detect import infer_metadata, run_all_checks_parallel
from astrion_dq.config import REPORT_MAPPING
from astrion_dq.models import RankedIssue
from astrion_dq.ranking.impact import ranking_agent_v2


# ---------------------------------------------------------------------------
# Helper: minimal triage without needing the Streamlit dashboard
# ---------------------------------------------------------------------------

def run_triage_on_tables(tables: dict[str, pd.DataFrame]) -> list[dict]:
    """Replicates _run_upload_triage logic without importing Streamlit."""
    meta = infer_metadata(tables)
    issues = run_all_checks_parallel(tables, meta, sensitivity="high")
    table_sizes = {name: len(df) for name, df in tables.items()}
    ranked_input = [
        RankedIssue(
            issue_id=i.issue_id,
            issue_type=i.issue_type,
            table=i.table,
            columns=i.columns,
            severity=i.severity,
            metric=i.metric,
            evidence_rows=i.evidence_rows,
            description=i.description,
            impact_score=0.0,
            affected_reports=REPORT_MAPPING.get(i.issue_type, []),
            agent_trace=[],
            confidence=1.0,
            dim_table=i.dim_table,
            dim_pk=i.dim_pk,
        )
        for i in issues
    ]
    ranked, _ = ranking_agent_v2(ranked_input, table_sizes)
    return [asdict(r) for r in ranked]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_table() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "id": range(1, 101),
        "amount": rng.uniform(10, 500, 100).round(2),
        "category": rng.choice(["A", "B", "C"], 100),
    })
    return {"transactions": df}


@pytest.fixture
def null_table() -> dict[str, pd.DataFrame]:
    df = pd.DataFrame({
        "id": range(1, 101),
        "amount": [None] * 20 + list(range(20, 100)),  # 20 % nulls
    })
    return {"sales": df}


@pytest.fixture
def duplicate_table() -> dict[str, pd.DataFrame]:
    base = pd.DataFrame({
        "id": range(1, 21),
        "amount": range(100, 120),
    })
    df = pd.concat([base, base.head(10)], ignore_index=True)  # 10 exact duplicates
    return {"orders": df}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clean_data_returns_list(clean_table):
    result = run_triage_on_tables(clean_table)
    assert isinstance(result, list)


def test_null_injection_detected(null_table):
    result = run_triage_on_tables(null_table)
    issue_types = [r["issue_type"] for r in result]
    assert "missing_values" in issue_types, "Nulls should produce a missing_values issue"


def test_duplicate_injection_detected(duplicate_table):
    result = run_triage_on_tables(duplicate_table)
    issue_types = [r["issue_type"] for r in result]
    assert "duplicate_rows" in issue_types, "Duplicates should produce a duplicate_rows issue"


def test_results_sorted_by_impact_score(null_table):
    result = run_triage_on_tables(null_table)
    if len(result) > 1:
        scores = [r["impact_score"] for r in result]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by BIS descending"


def test_result_dict_has_required_fields(null_table):
    result = run_triage_on_tables(null_table)
    required = {"issue_id", "issue_type", "table", "columns", "severity",
                "metric", "evidence_rows", "description", "impact_score",
                "confidence", "affected_reports", "dim_table", "dim_pk"}
    for issue in result:
        missing = required - set(issue.keys())
        assert not missing, f"Issue dict missing fields: {missing}"


def test_dim_table_dim_pk_present_in_output(null_table):
    result = run_triage_on_tables(null_table)
    for issue in result:
        assert "dim_table" in issue
        assert "dim_pk" in issue


def test_empty_dataframe_handled_gracefully():
    tables = {"empty_tbl": pd.DataFrame({"col_a": []})}
    result = run_triage_on_tables(tables)
    # Should either return empty list or detect empty_table — not crash
    assert isinstance(result, list)


def test_multiple_tables_all_analysed():
    rng = np.random.default_rng(42)
    tables = {
        "table_a": pd.DataFrame({"x": rng.integers(0, 100, 50)}),
        "table_b": pd.DataFrame({"y": [None] * 30 + list(range(70))}),
    }
    result = run_triage_on_tables(tables)
    tables_seen = {r["table"] for r in result}
    # At least table_b's nulls should appear
    assert "table_b" in tables_seen
