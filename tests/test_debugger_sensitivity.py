"""Regression test: IssueVerifier must use the same IQR multiplier as the detector.

Dataset design
--------------
65 rows:
  25 values = 0   (bulk lower cluster)
  25 values = 10  (bulk upper cluster)
  10 values = 30  (moderate outlier: > Q3 + 1.5*IQR = 25, NOT > Q3 + 3*IQR = 40)
   5 values = 100 (extreme outlier: caught by both multipliers)

With this data:
  Q1 = 0,  Q3 = 10,  IQR = 10

  HIGH sensitivity (IQR_MULT_HIGH = 1.5):
    lower = -15,  upper = 25
    outliers: 30 (10 rows) + 100 (5 rows) = 15 rows

  NORMAL sensitivity (IQR_MULT_NORMAL = 3.0):
    lower = -30,  upper = 40
    outliers: 100 (5 rows) = 5 rows

Without the fix, IssueVerifier always used IQR_MULT_NORMAL, so the SQL query
for a HIGH-sensitivity issue would count 5 rows while pandas found 15 rows:
confidence = 5/15 = 0.33.  With the fix: sql_count == pd_count, confidence = 1.0.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from astrion_dq.checks.detect import detect_outliers
from astrion_dq.config import DUCKDB_SCHEMA
from astrion_dq.graph.debugger import IssueVerifier
from astrion_dq.metadata import infer_metadata
from astrion_dq.warehouse import loader as warehouse_loader


@pytest.fixture()
def outlier_tables_and_conn(tmp_path):
    """Build the test DataFrame and an in-memory DuckDB with the table registered."""
    values = [0] * 25 + [10] * 25 + [30] * 10 + [100] * 5
    df = pd.DataFrame({"fact_sales": ["s"] * 65, "amount": values})

    tables = {"fact_sales": df}

    db_file = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{DUCKDB_SCHEMA}"')
    conn.register("_tmp", df)
    conn.execute(f'CREATE TABLE "{DUCKDB_SCHEMA}"."fact_sales" AS SELECT * FROM _tmp')
    conn.unregister("_tmp")

    # Inject the connection into the warehouse loader singleton so IssueVerifier
    # can call get_connection() without a full data_loader_node run.
    warehouse_loader._CONN = conn

    yield tables, conn

    warehouse_loader._CONN = None
    conn.close()


def test_high_sensitivity_confidence_is_1(outlier_tables_and_conn):
    """IssueVerifier(sensitivity='high') must agree with detect_outliers(sensitivity='high')."""
    tables, _ = outlier_tables_and_conn
    meta = infer_metadata(tables)

    issues_high = detect_outliers(tables, meta, sensitivity="high")
    assert issues_high, "detect_outliers should find at least one issue at high sensitivity"

    amount_issues = [i for i in issues_high if i.columns == ["amount"]]
    assert amount_issues, "Expected an outlier issue for column 'amount'"
    issue = amount_issues[0]
    pd_count = issue.evidence_rows
    assert pd_count == 15, f"Expected 15 outliers at high sensitivity, got {pd_count}"

    verifier = IssueVerifier(sensitivity="high")
    verified = verifier._verify_one(issue)

    assert verified.sql_count == pd_count, (
        f"SQL count ({verified.sql_count}) should equal pandas count ({pd_count}) "
        f"when sensitivity matches. confidence={verified.confidence}"
    )
    assert verified.confidence == 1.0, f"Expected confidence=1.0, got {verified.confidence}"


def test_normal_sensitivity_confidence_is_1(outlier_tables_and_conn):
    """IssueVerifier(sensitivity='normal') must agree with detect_outliers(sensitivity='normal')."""
    tables, _ = outlier_tables_and_conn
    meta = infer_metadata(tables)

    issues_normal = detect_outliers(tables, meta, sensitivity="normal")
    assert issues_normal, "detect_outliers should find at least one issue at normal sensitivity"

    amount_issues = [i for i in issues_normal if i.columns == ["amount"]]
    assert amount_issues, "Expected an outlier issue for column 'amount'"
    issue = amount_issues[0]
    pd_count = issue.evidence_rows
    assert pd_count == 5, f"Expected 5 outliers at normal sensitivity, got {pd_count}"

    verifier = IssueVerifier(sensitivity="normal")
    verified = verifier._verify_one(issue)

    assert verified.sql_count == pd_count, (
        f"SQL count ({verified.sql_count}) should equal pandas count ({pd_count}) "
        f"when sensitivity matches. confidence={verified.confidence}"
    )
    assert verified.confidence == 1.0, f"Expected confidence=1.0, got {verified.confidence}"


def test_ri_break_with_empty_dim_table_falls_back(outlier_tables_and_conn):
    """P5-B: _verify_ri_break must return SQL_FALLBACK_CONFIDENCE when dim_table is empty.

    Pre-fix: the regex path returns SQL_FALLBACK_CONFIDENCE only when the
    description string does not match the pattern. Post-fix: dim_table field
    is checked directly -- no regex needed.
    """
    from astrion_dq.config import SQL_FALLBACK_CONFIDENCE
    from astrion_dq.models import QualityIssue

    _, _ = outlier_tables_and_conn

    issue = QualityIssue(
        issue_id="DFK_0001",
        issue_type="referential_integrity_break",
        table="fact_sales",
        columns=["campaign_sk"],
        severity="high",
        metric=0.05,
        evidence_rows=10,
        description="Column 'campaign_sk' has 10 values not found in dim_campaigns.campaign_sk.",
        dim_table="",   # empty -- should trigger fallback
        dim_pk="",
    )
    verifier = IssueVerifier()
    result = verifier._verify_one(issue)
    assert result.confidence == SQL_FALLBACK_CONFIDENCE, (
        f"Expected SQL_FALLBACK_CONFIDENCE ({SQL_FALLBACK_CONFIDENCE}) "
        f"when dim_table is empty, got {result.confidence}"
    )


def test_mismatched_sensitivity_lowers_confidence(outlier_tables_and_conn):
    """Using normal verifier on a high-sensitivity issue should produce confidence < 1.0.

    This test documents the original bug: without the fix, ALL runs would exhibit
    this behaviour when sensitivity='high' was used.
    """
    tables, _ = outlier_tables_and_conn
    meta = infer_metadata(tables)

    # Detect with high sensitivity -> 15 outliers
    issues_high = detect_outliers(tables, meta, sensitivity="high")
    amount_issues = [i for i in issues_high if i.columns == ["amount"]]
    issue = amount_issues[0]

    # Verify with normal multiplier -> SQL will find only 5 rows
    verifier_wrong = IssueVerifier(sensitivity="normal")
    verified = verifier_wrong._verify_one(issue)

    assert verified.confidence < 1.0, (
        "Using the wrong IQR multiplier should produce confidence < 1.0"
    )
