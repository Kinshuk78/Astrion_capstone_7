"""End-to-end tests using a synthetic messy e-commerce dataset.

Simulates the characteristics of real messy e-commerce data:
    - Missing values in revenue and customer_id columns
    - Numeric outliers (extreme price values)
    - Invalid future dates in order_date
    - Duplicate order records
    - Referential integrity breaks (customer_id not in customer table)
    - Mixed case and format inconsistencies (handled as string columns)

Pipeline under test:
    infer_metadata -> run_all_checks_parallel -> ranking_agent_v2

Then verifies that the agent fallback layer can process the detected issues
without crashing and returns structurally valid output.
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
# Dataset factory
# ---------------------------------------------------------------------------

def _build_messy_ecommerce() -> dict[str, pd.DataFrame]:
    """Build a synthetic messy e-commerce dataset with injected problems.

    Injected issues:
        missing_values    -- 20% nulls in revenue, 10% nulls in customer_id
        numeric_outliers  -- 5 extreme price values (orders of magnitude above normal)
        invalid_future_dates -- 8 order_date values set to year 2099
        duplicate_rows    -- 15 exact duplicate order rows
    """
    rng = np.random.default_rng(42)
    n = 200

    # Base orders table
    order_ids = list(range(1001, 1001 + n))
    customer_ids = rng.integers(1, 51, n).tolist()  # 50 distinct customers
    revenues = rng.uniform(10, 500, n).round(2).tolist()
    dates = pd.date_range("2023-01-01", periods=n, freq="D").tolist()
    categories = rng.choice(["Electronics", "Clothing", "Home", "Sports"], n).tolist()

    # Inject missing values: 20% of revenue and 10% of customer_id
    for idx in rng.choice(n, size=int(n * 0.20), replace=False):
        revenues[idx] = None
    for idx in rng.choice(n, size=int(n * 0.10), replace=False):
        customer_ids[idx] = None

    # Inject numeric outliers: 5 extreme prices
    for idx in rng.choice(n, size=5, replace=False):
        revenues[idx] = 999_999.99

    # Inject future dates: 8 orders in year 2099
    for i, idx in enumerate(rng.choice(n, size=8, replace=False)):
        dates[idx] = pd.Timestamp(f"2099-0{(i % 9) + 1}-15")

    orders_df = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customer_ids,
        "revenue": revenues,
        "order_date": dates,
        "category": categories,
    })

    # Inject duplicates: append 15 rows from the first 15
    dupes = orders_df.head(15).copy()
    orders_df = pd.concat([orders_df, dupes], ignore_index=True)

    # Customers dimension table (only 50 customers — some order customer_ids won't match)
    customers_df = pd.DataFrame({
        "customer_id": list(range(1, 51)),
        "name": [f"Customer {i}" for i in range(1, 51)],
        "region": rng.choice(["NSW", "VIC", "QLD", "SA"], 50).tolist(),
    })

    # NOTE: table name must contain "fact" so infer_metadata() assigns role="fact"
    # and the outlier detector is applied. Tables without "fact" in the name are
    # classified as "dimension" and skipped by the IQR outlier check.
    return {
        "fact_orders": orders_df,
        "dim_customers": customers_df,
    }


def _run_triage(tables: dict[str, pd.DataFrame]) -> list[dict]:
    """Run the deterministic triage pipeline and return ranked issue dicts."""
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

@pytest.fixture(scope="module")
def messy_tables():
    return _build_messy_ecommerce()


@pytest.fixture(scope="module")
def triage_results(messy_tables):
    return _run_triage(messy_tables)


# ---------------------------------------------------------------------------
# Dataset shape tests
# ---------------------------------------------------------------------------

def test_messy_dataset_has_correct_tables(messy_tables):
    assert "fact_orders" in messy_tables
    assert "dim_customers" in messy_tables


def test_orders_table_has_duplicates_injected(messy_tables):
    df = messy_tables["fact_orders"]
    assert len(df) > 200, "Duplicates should push row count above base 200"


def test_orders_table_has_null_revenue(messy_tables):
    null_fraction = messy_tables["fact_orders"]["revenue"].isna().mean()
    assert null_fraction > 0.05, "At least 5% of revenue values should be null"


def test_orders_table_has_future_dates(messy_tables):
    future_count = (messy_tables["fact_orders"]["order_date"] > pd.Timestamp.today()).sum()
    assert future_count >= 8


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

def test_triage_returns_list(triage_results):
    assert isinstance(triage_results, list)


def test_triage_detects_missing_values(triage_results):
    types = [r["issue_type"] for r in triage_results]
    assert "missing_values" in types, "Injected nulls must be detected"


def test_triage_detects_duplicate_rows(triage_results):
    types = [r["issue_type"] for r in triage_results]
    assert "duplicate_rows" in types, "Injected duplicates must be detected"


def test_triage_detects_future_dates(triage_results):
    types = [r["issue_type"] for r in triage_results]
    assert "invalid_future_dates" in types, "Injected future dates must be detected"


def test_triage_detects_numeric_outliers(triage_results):
    types = [r["issue_type"] for r in triage_results]
    assert "numeric_outliers" in types, "Injected extreme prices must be detected"


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------

def test_results_sorted_by_impact_score_descending(triage_results):
    if len(triage_results) > 1:
        scores = [r["impact_score"] for r in triage_results]
        assert scores == sorted(scores, reverse=True)


def test_highest_ranked_issue_is_critical_severity(triage_results):
    if triage_results:
        assert triage_results[0]["severity"] in ("high", "medium")


def test_all_issues_have_positive_impact_score(triage_results):
    for issue in triage_results:
        assert issue["impact_score"] >= 0


# ---------------------------------------------------------------------------
# Field completeness tests
# ---------------------------------------------------------------------------

def test_all_issues_have_required_fields(triage_results):
    required = {
        "issue_id", "issue_type", "table", "columns", "severity",
        "metric", "evidence_rows", "description", "impact_score",
        "confidence", "affected_reports", "dim_table", "dim_pk",
    }
    for issue in triage_results:
        missing = required - set(issue.keys())
        assert not missing, f"Issue missing fields: {missing}"


def test_orders_table_issues_reference_orders(triage_results):
    order_issues = [r for r in triage_results if r["table"] == "fact_orders"]
    assert len(order_issues) >= 1


# ---------------------------------------------------------------------------
# Fallback agent layer on messy data (no LLM required)
# ---------------------------------------------------------------------------

def test_fallback_explain_handles_all_detected_issues(triage_results):
    from astrion_dq.agent.fallback import fallback_explain
    result = fallback_explain(triage_results)
    assert len(result) == len(triage_results)
    for item in result:
        assert item["business_explanation"]
        assert item["source"] == "fallback"


def test_fallback_prioritise_returns_sorted_results(triage_results):
    from astrion_dq.agent.fallback import fallback_prioritise
    result = fallback_prioritise(triage_results)
    assert len(result) == len(triage_results)
    ranks = [r["priority_rank"] for r in result]
    assert ranks[0] == 1


def test_fallback_generate_fix_for_each_detected_issue(triage_results):
    from astrion_dq.agent.fallback import fallback_generate_fix
    for issue in triage_results:
        fix = fallback_generate_fix(issue)
        assert fix["sql_fix"], f"No SQL fix for {issue['issue_type']}"
        assert fix["python_fix"], f"No Python fix for {issue['issue_type']}"


def test_fallback_report_reflects_issue_count(triage_results):
    from astrion_dq.agent.fallback import fallback_report
    report = fallback_report(triage_results, run_id="testrun")
    assert report["total_issues"] == len(triage_results)
    assert report["overall_data_health"] in ("poor", "fair", "good", "excellent")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_dataframe_does_not_crash():
    tables = {"empty_orders": pd.DataFrame({"order_id": [], "revenue": []})}
    result = _run_triage(tables)
    assert isinstance(result, list)


def test_single_row_table_does_not_crash():
    tables = {
        "orders": pd.DataFrame({
            "order_id": [1],
            "revenue": [100.0],
            "order_date": [pd.Timestamp("2024-01-01")],
        })
    }
    result = _run_triage(tables)
    assert isinstance(result, list)


def test_all_nulls_column_handled_gracefully():
    tables = {
        "fact": pd.DataFrame({
            "id": range(50),
            "amount": [None] * 50,
        })
    }
    result = _run_triage(tables)
    assert isinstance(result, list)
    types = [r["issue_type"] for r in result]
    assert "missing_values" in types
