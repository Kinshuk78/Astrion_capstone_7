"""Tests for the deterministic fallback functions.

All fallback functions must:
    - never raise exceptions
    - return structurally valid output matching the LLM output format
    - include "source": "fallback" on every result
    - handle all 7 known issue types
    - handle unknown issue types gracefully
"""
from __future__ import annotations

import pytest

from astrion_dq.agent.fallback import (
    fallback_explain,
    fallback_generate_fix,
    fallback_prioritise,
    fallback_report,
)

_ISSUE_TYPES = [
    "missing_values",
    "duplicate_rows",
    "numeric_outliers",
    "invalid_future_dates",
    "referential_integrity_break",
    "empty_table",
    "statistical_drift",
]


def _make_issue(issue_type: str, idx: int = 0) -> dict:
    return {
        "issue_id": f"X{idx:03d}",
        "issue_type": issue_type,
        "table": "fact_sales",
        "columns": ["amount"],
        "severity": "high",
        "metric": 0.10,
        "evidence_rows": 1000,
        "impact_score": float(7 - idx),
        "confidence": 0.90,
        "dim_table": "dim_store",
        "dim_pk": "store_id",
    }


def _sample_issues() -> list[dict]:
    return [_make_issue(t, i) for i, t in enumerate(_ISSUE_TYPES)]


# ---------------------------------------------------------------------------
# fallback_explain
# ---------------------------------------------------------------------------

def test_explain_returns_list_for_all_issue_types():
    result = fallback_explain(_sample_issues())
    assert isinstance(result, list)
    assert len(result) == len(_ISSUE_TYPES)


def test_explain_each_item_has_required_fields():
    required = {"issue_id", "issue_type", "table", "columns", "severity",
                "business_explanation", "risk_summary", "recommended_action", "source"}
    for item in fallback_explain(_sample_issues()):
        missing = required - set(item.keys())
        assert not missing, f"Missing fields: {missing}"


def test_explain_source_is_fallback():
    for item in fallback_explain(_sample_issues()):
        assert item["source"] == "fallback"


def test_explain_business_explanation_is_non_empty():
    for item in fallback_explain(_sample_issues()):
        assert len(item["business_explanation"]) > 20


def test_explain_handles_unknown_issue_type():
    issue = _make_issue("completely_unknown_type")
    result = fallback_explain([issue])
    assert len(result) == 1
    assert result[0]["business_explanation"]  # must not be empty


def test_explain_empty_list_returns_empty_list():
    assert fallback_explain([]) == []


def test_explain_referential_integrity_mentions_records():
    issue = _make_issue("referential_integrity_break")
    result = fallback_explain([issue])
    assert "record" in result[0]["business_explanation"].lower()


# ---------------------------------------------------------------------------
# fallback_prioritise
# ---------------------------------------------------------------------------

def test_prioritise_returns_list():
    result = fallback_prioritise(_sample_issues())
    assert isinstance(result, list)


def test_prioritise_sorted_by_impact_score_descending():
    issues = _sample_issues()
    result = fallback_prioritise(issues)
    scores = [r["impact_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_prioritise_priority_ranks_start_at_one():
    result = fallback_prioritise(_sample_issues())
    ranks = [r["priority_rank"] for r in result]
    assert min(ranks) == 1


def test_prioritise_priority_ranks_are_sequential():
    result = fallback_prioritise(_sample_issues())
    ranks = sorted(r["priority_rank"] for r in result)
    assert ranks == list(range(1, len(result) + 1))


def test_prioritise_source_is_fallback():
    for item in fallback_prioritise(_sample_issues()):
        assert item["source"] == "fallback"


def test_prioritise_each_item_has_justification():
    for item in fallback_prioritise(_sample_issues()):
        assert item.get("priority_justification")
        assert len(item["priority_justification"]) > 10


def test_prioritise_urgency_field_present():
    for item in fallback_prioritise(_sample_issues()):
        assert item["urgency"] in ("critical", "high", "medium", "low")


def test_prioritise_empty_list_returns_empty_list():
    assert fallback_prioritise([]) == []


# ---------------------------------------------------------------------------
# fallback_generate_fix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("issue_type", _ISSUE_TYPES)
def test_fix_code_generated_for_all_issue_types(issue_type):
    issue = _make_issue(issue_type)
    result = fallback_generate_fix(issue)
    assert result["sql_fix"]
    assert result["python_fix"]
    assert result["source"] == "fallback"


def test_fix_required_fields_present():
    issue = _make_issue("missing_values")
    result = fallback_generate_fix(issue)
    assert "issue_id" in result
    assert "issue_type" in result
    assert "sql_fix" in result
    assert "python_fix" in result


def test_fix_sql_references_correct_table():
    issue = _make_issue("missing_values")
    result = fallback_generate_fix(issue)
    assert "fact_sales" in result["sql_fix"]


def test_fix_sql_references_correct_column():
    issue = _make_issue("missing_values")
    result = fallback_generate_fix(issue)
    assert "amount" in result["sql_fix"]


def test_fix_python_references_correct_column():
    issue = _make_issue("missing_values")
    result = fallback_generate_fix(issue)
    assert "amount" in result["python_fix"]


def test_fix_referential_integrity_uses_dim_table():
    issue = _make_issue("referential_integrity_break")
    result = fallback_generate_fix(issue)
    assert "dim_store" in result["sql_fix"]


def test_fix_unknown_issue_type_returns_comment():
    issue = _make_issue("unknown_type_xyz")
    result = fallback_generate_fix(issue)
    assert "--" in result["sql_fix"] or "#" in result["python_fix"]


def test_fix_empty_columns_does_not_crash():
    issue = {
        "issue_id": "X000",
        "issue_type": "empty_table",
        "table": "some_table",
        "columns": [],
        "severity": "high",
        "metric": 0.0,
        "evidence_rows": 0,
        "impact_score": 5.0,
    }
    result = fallback_generate_fix(issue)
    assert result["sql_fix"]
    assert result["python_fix"]


# ---------------------------------------------------------------------------
# fallback_report
# ---------------------------------------------------------------------------

def test_report_no_issues_returns_excellent_health():
    result = fallback_report([])
    assert result["overall_data_health"] == "excellent"
    assert result["total_issues"] == 0


def test_report_many_critical_returns_poor_health():
    issues = [_make_issue("missing_values", i) for i in range(5)]
    result = fallback_report(issues)
    assert result["overall_data_health"] == "poor"


def test_report_has_required_fields():
    result = fallback_report(_sample_issues())
    required = {"run_id", "executive_summary", "top_risks",
                "recommended_actions", "overall_data_health",
                "total_issues", "critical_issues", "source"}
    assert required.issubset(result.keys())


def test_report_source_is_fallback():
    result = fallback_report(_sample_issues())
    assert result["source"] == "fallback"


def test_report_run_id_passed_through():
    result = fallback_report([], run_id="abc123")
    assert result["run_id"] == "abc123"


def test_report_top_risks_is_list():
    result = fallback_report(_sample_issues())
    assert isinstance(result["top_risks"], list)


def test_report_recommended_actions_is_list():
    result = fallback_report(_sample_issues())
    assert isinstance(result["recommended_actions"], list)
    assert len(result["recommended_actions"]) >= 1
