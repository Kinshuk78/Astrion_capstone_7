"""Tests for input validation and LLM response scoring.

Covers:
    validate_input()    — pre-call checks on issues and schema
    build_allowed_columns() — set of valid identifiers from schema
    validate_response() — four-dimensional scoring + pass/fail
"""
from __future__ import annotations

import pytest

from astrion_dq.agent.validation import (
    ValidationResult,
    build_allowed_columns,
    validate_input,
    validate_response,
)

_SAMPLE_SCHEMA = [
    {"table": "fact_sales", "columns": ["sale_id", "amount", "sale_date", "store_id"]},
    {"table": "dim_store", "columns": ["store_id", "store_name", "region"]},
]

_SAMPLE_ISSUES = [
    {
        "issue_id": "X001",
        "issue_type": "missing_values",
        "table": "fact_sales",
        "columns": ["amount"],
        "severity": "high",
        "metric": 0.15,
        "evidence_rows": 1500,
        "impact_score": 7.2,
        "confidence": 0.95,
    },
    {
        "issue_id": "X002",
        "issue_type": "duplicate_rows",
        "table": "fact_sales",
        "columns": ["sale_id"],
        "severity": "medium",
        "metric": 0.03,
        "evidence_rows": 300,
        "impact_score": 4.1,
        "confidence": 0.90,
    },
]


# ---------------------------------------------------------------------------
# validate_input()
# ---------------------------------------------------------------------------

def test_valid_input_does_not_raise():
    validate_input(_SAMPLE_ISSUES, _SAMPLE_SCHEMA)


def test_empty_issues_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_input([], _SAMPLE_SCHEMA)


def test_empty_schema_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_input(_SAMPLE_ISSUES, [])


def test_missing_issue_type_raises():
    bad = [{"issue_id": "X000", "table": "fact_sales", "columns": []}]
    with pytest.raises(ValueError, match="issue_type"):
        validate_input(bad, _SAMPLE_SCHEMA)


def test_unknown_issue_type_raises():
    bad = [{"issue_id": "X000", "issue_type": "invented_type", "table": "fact_sales"}]
    with pytest.raises(ValueError, match="Unknown issue_type"):
        validate_input(bad, _SAMPLE_SCHEMA)


def test_all_known_issue_types_accepted():
    known = [
        "missing_values", "duplicate_rows", "numeric_outliers",
        "invalid_future_dates", "referential_integrity_break",
        "empty_table", "statistical_drift",
    ]
    for t in known:
        validate_input(
            [{"issue_id": "X", "issue_type": t, "table": "t", "columns": []}],
            [{"table": "t", "columns": []}],
        )


# ---------------------------------------------------------------------------
# build_allowed_columns()
# ---------------------------------------------------------------------------

def test_build_allowed_columns_includes_tables_and_cols():
    allowed = build_allowed_columns(_SAMPLE_SCHEMA)
    assert "fact_sales" in allowed
    assert "amount" in allowed
    assert "store_id" in allowed
    assert "dim_store" in allowed
    assert "region" in allowed


def test_build_allowed_columns_all_lowercase():
    schema = [{"table": "FactSales", "columns": ["SaleID", "Amount"]}]
    allowed = build_allowed_columns(schema)
    assert "factsales" in allowed
    assert "saleid" in allowed
    assert "amount" in allowed


def test_build_allowed_columns_empty_schema_returns_empty_set():
    assert build_allowed_columns([]) == set()


# ---------------------------------------------------------------------------
# validate_response() — basic
# ---------------------------------------------------------------------------

def test_empty_response_fails():
    result = validate_response("", _SAMPLE_ISSUES, set())
    assert not result.passed
    assert result.overall_score == 0.0


def test_whitespace_response_fails():
    result = validate_response("   \n  ", _SAMPLE_ISSUES, set())
    assert not result.passed


def test_valid_response_passes():
    # Prose response — mentions correct tables, columns, and issue types
    response = (
        "The fact_sales table has a missing_values problem affecting the amount column. "
        "This will cause revenue reports to undercount totals. "
        "There are also duplicate_rows in the sale_id column that will inflate daily summaries."
    )
    allowed = build_allowed_columns(_SAMPLE_SCHEMA)
    result = validate_response(response, _SAMPLE_ISSUES, allowed)
    assert isinstance(result, ValidationResult)
    assert 0.0 <= result.overall_score <= 1.0
    assert result.passed


def test_hallucinated_column_lowers_schema_score():
    # Response mentions a made-up column name
    response = (
        '{"issue_id": "X001", "business_explanation": '
        '"The `fake_column_xyz` has issues in `invented_table`."}'
    )
    allowed = build_allowed_columns(_SAMPLE_SCHEMA)
    result = validate_response(response, _SAMPLE_ISSUES, allowed)
    assert result.schema_compliance_score < 1.0


def test_no_column_mentions_gives_perfect_schema_score():
    # Response with no quoted identifiers at all
    response = "The data has some missing values. Please investigate."
    allowed = build_allowed_columns(_SAMPLE_SCHEMA)
    result = validate_response(response, _SAMPLE_ISSUES, allowed)
    assert result.schema_compliance_score == 1.0


# ---------------------------------------------------------------------------
# validate_response() — scoring dimensions
# ---------------------------------------------------------------------------

def test_factual_consistency_drops_when_known_type_not_in_input():
    # Response mentions numeric_outliers — a real issue type, but NOT in _SAMPLE_ISSUES.
    # _SAMPLE_ISSUES only has missing_values and duplicate_rows.
    # factual = valid_types / mentioned_types = 2/3 < 1.0
    response = (
        "There are missing_values and duplicate_rows issues in fact_sales. "
        "There is also a numeric_outliers issue in the amount column."
    )
    allowed = build_allowed_columns(_SAMPLE_SCHEMA)
    result = validate_response(response, _SAMPLE_ISSUES, allowed)
    assert result.factual_consistency_score < 1.0


def test_completeness_high_when_top_issues_addressed():
    # Mentions the table for both top issues
    response = (
        "The fact_sales table has two main issues: missing values in the amount field "
        "and duplicate rows in sale_id. Both need immediate attention."
    )
    allowed = build_allowed_columns(_SAMPLE_SCHEMA)
    result = validate_response(response, _SAMPLE_ISSUES, allowed, top_k=2)
    assert result.completeness_score == 1.0


def test_confidence_score_is_average_of_input_confidences():
    response = "The data has issues."
    allowed = build_allowed_columns(_SAMPLE_SCHEMA)
    result = validate_response(response, _SAMPLE_ISSUES, allowed)
    expected_conf = (0.95 + 0.90) / 2
    assert abs(result.confidence_score - expected_conf) < 0.01


def test_overall_score_is_weighted_combination():
    response = "fact_sales has missing_values and duplicate_rows problems."
    allowed = build_allowed_columns(_SAMPLE_SCHEMA)
    result = validate_response(response, _SAMPLE_ISSUES, allowed)
    expected = (
        result.factual_consistency_score * 0.30
        + result.schema_compliance_score * 0.40
        + result.completeness_score * 0.20
        + result.confidence_score * 0.10
    )
    assert abs(result.overall_score - round(min(expected, 1.0), 4)) < 0.001


def test_failure_reason_populated_on_low_score():
    result = validate_response("", _SAMPLE_ISSUES, set())
    assert result.failure_reason != ""


def test_pass_reason_empty_on_success():
    response = (
        "The fact_sales table has missing_values in the amount column "
        "and duplicate_rows in sale_id. Both require remediation."
    )
    allowed = build_allowed_columns(_SAMPLE_SCHEMA)
    result = validate_response(response, _SAMPLE_ISSUES, allowed)
    if result.passed:
        assert result.failure_reason == ""
