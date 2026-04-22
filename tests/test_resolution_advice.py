"""Tests for _resolution_advice — deterministic SQL template generation.

Verifies that every known issue_type produces:
  - A non-empty list of lines
  - At least one ```sql ... ``` block
  - The correct table name in the SQL
  - dim_table / dim_pk used directly when available (no regex fallback needed)
"""
from __future__ import annotations

import pytest

from astrion_dq.graph.nodes import _resolution_advice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_sql_block(lines: list[str]) -> bool:
    text = "\n".join(lines)
    return "```sql" in text


def _sql_text(lines: list[str]) -> str:
    text = "\n".join(lines)
    import re
    blocks = re.findall(r"```sql\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Parametrised: every issue type must produce SQL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("issue_type", [
    "referential_integrity_break",
    "duplicate_rows",
    "numeric_outliers",
    "missing_values",
    "invalid_future_dates",
    "statistical_drift",
    "empty_table",
])
def test_all_types_produce_sql_blocks(issue_type):
    issue = {
        "issue_type": issue_type,
        "table": "fact_sales",
        "columns": ["customer_sk"],
        "severity": "high",
        "metric": 0.05,
        "evidence_rows": 150,
        "description": f"Test issue of type {issue_type}",
        "impact_score": 1.234,
        "confidence": 0.95,
        "affected_reports": ["daily_sales_report"],
        "dim_table": "dim_customer",
        "dim_pk": "customer_key",
    }
    lines = _resolution_advice(issue, rank=1)
    assert lines, "Expected non-empty lines"
    assert _has_sql_block(lines), f"{issue_type}: no ```sql block found"


def test_unknown_type_still_returns_lines():
    issue = {
        "issue_type": "custom_unknown_type",
        "table": "my_table",
        "columns": ["col_a"],
        "severity": "low",
        "metric": 0.01,
        "evidence_rows": 5,
        "description": "Some unknown issue",
        "impact_score": 0.5,
        "confidence": 1.0,
        "affected_reports": [],
        "dim_table": "",
        "dim_pk": "",
    }
    lines = _resolution_advice(issue, rank=3)
    assert lines
    assert _has_sql_block(lines), "Unknown type must still emit an investigative SQL block"


# ---------------------------------------------------------------------------
# RI break: direct dim_table/dim_pk (no regex)
# ---------------------------------------------------------------------------

def test_ri_uses_dim_table_from_field():
    issue = {
        "issue_type": "referential_integrity_break",
        "table": "fact_orders",
        "columns": ["product_fk"],
        "severity": "high",
        "metric": 0.1,
        "evidence_rows": 300,
        "description": "FK violations exist",  # no 'not found in X.Y' pattern
        "impact_score": 2.5,
        "confidence": 0.98,
        "affected_reports": [],
        "dim_table": "dim_product",
        "dim_pk": "product_key",
    }
    lines = _resolution_advice(issue, rank=1)
    sql = _sql_text(lines)
    assert "dim_product" in sql, "dim_table should appear in SQL"
    assert "product_key" in sql, "dim_pk should appear in SQL"


def test_ri_regex_fallback_when_fields_empty():
    issue = {
        "issue_type": "referential_integrity_break",
        "table": "fact_orders",
        "columns": ["product_fk"],
        "severity": "high",
        "metric": 0.1,
        "evidence_rows": 300,
        "description": "Column 'product_fk' has 300 values not found in dim_product.product_key.",
        "impact_score": 2.5,
        "confidence": 0.98,
        "affected_reports": [],
        "dim_table": "",  # empty — must fall back to description parsing
        "dim_pk": "",
    }
    lines = _resolution_advice(issue, rank=1)
    sql = _sql_text(lines)
    assert "dim_product" in sql, "Regex fallback should extract dim_product from description"
    assert "product_key" in sql, "Regex fallback should extract product_key from description"


def test_ri_defaults_when_no_info():
    issue = {
        "issue_type": "referential_integrity_break",
        "table": "fact_orders",
        "columns": ["product_fk"],
        "severity": "medium",
        "metric": 0.05,
        "evidence_rows": 50,
        "description": "FK violations with no table info",  # no pattern
        "impact_score": 1.0,
        "confidence": 0.8,
        "affected_reports": [],
        "dim_table": "",
        "dim_pk": "",
    }
    lines = _resolution_advice(issue, rank=2)
    sql = _sql_text(lines)
    # Should fall back to placeholder names
    assert "dimension_table" in sql
    assert "pk_column" in sql


# ---------------------------------------------------------------------------
# Table name always appears in output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("issue_type", [
    "duplicate_rows",
    "numeric_outliers",
    "missing_values",
    "invalid_future_dates",
    "empty_table",
])
def test_table_name_in_sql(issue_type):
    table = "my_special_table"
    issue = {
        "issue_type": issue_type,
        "table": table,
        "columns": ["amount"],
        "severity": "medium",
        "metric": 0.03,
        "evidence_rows": 100,
        "description": "Test",
        "impact_score": 0.8,
        "confidence": 1.0,
        "affected_reports": [],
        "dim_table": "",
        "dim_pk": "",
    }
    lines = _resolution_advice(issue, rank=1)
    sql = _sql_text(lines)
    assert table in sql, f"Table name '{table}' should appear in SQL for {issue_type}"


# ---------------------------------------------------------------------------
# Priority / rank header
# ---------------------------------------------------------------------------

def test_rank_appears_in_header():
    issue = {
        "issue_type": "missing_values",
        "table": "dim_store",
        "columns": ["store_name"],
        "severity": "low",
        "metric": 0.02,
        "evidence_rows": 10,
        "description": "Nulls in store_name",
        "impact_score": 0.3,
        "confidence": 1.0,
        "affected_reports": [],
        "dim_table": "",
        "dim_pk": "",
    }
    for rank in (1, 5, 12):
        lines = _resolution_advice(issue, rank=rank)
        text = "\n".join(lines)
        assert f"#{rank}" in text, f"Rank #{rank} should appear in header"
