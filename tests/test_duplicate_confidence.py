"""Regression test for F-05: detect_duplicates must count excess copies only.

Before the fix, detect_duplicates used keep=False, which counts ALL rows in
duplicate groups. For 5 inserted duplicate pairs (5 originals + 5 copies = 10
rows), evidence_rows was 10. The SQL verifier counts excess copies via
  COUNT(*) - COUNT(DISTINCT pk) = 10 - 5 = 5.
Confidence = min(10, 5) / max(10, 5) = 0.50, below CONFIDENCE_THRESHOLD (0.70).
Every correctly detected duplicate triggered mandatory human review.

After the fix, keep='first' counts excess copies only: evidence_rows = 5.
SQL returns 5. Confidence = 1.0 >= 0.95.

BREAKING: evidence_rows for duplicate_rows issues is approximately halved
relative to v0.2.0. See CHANGELOG.md.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

import astrion_dq.warehouse.loader as warehouse_loader
from astrion_dq.checks.detect import detect_duplicates
from astrion_dq.config import DUCKDB_SCHEMA
from astrion_dq.graph.debugger import IssueVerifier
from astrion_dq.metadata import infer_metadata
from astrion_dq.models import QualityIssue


N_PAIRS = 5


@pytest.fixture()
def dup_setup(tmp_path):
    """5 original rows + 5 exact duplicate rows = 10 rows total.

    All columns are duplicated (no PK can be inferred because every value
    appears twice). This is the worst-case scenario the fix must handle.
    """
    sales_sk = list(range(1, N_PAIRS + 1)) * 2    # [1,2,3,4,5, 1,2,3,4,5]
    amounts = [float(i * 10) for i in range(1, N_PAIRS + 1)] * 2
    tables = {
        "fact_sales": pd.DataFrame({
            "sales_sk": sales_sk,
            "amount": amounts,
        })
    }

    conn = duckdb.connect(str(tmp_path / "dup_test.duckdb"))
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{DUCKDB_SCHEMA}"')
    for name, df in tables.items():
        conn.register("_tmp", df)
        conn.execute(
            f'CREATE OR REPLACE TABLE "{DUCKDB_SCHEMA}"."{name}" AS SELECT * FROM _tmp'
        )
        conn.unregister("_tmp")
    warehouse_loader._CONN = conn

    yield tables

    warehouse_loader._CONN = None
    conn.close()


def test_evidence_rows_counts_excess_copies(dup_setup):
    """detect_duplicates must report N_PAIRS excess copies, not 2 * N_PAIRS.

    With keep=False (old behaviour): dup_count = 10 (all rows in groups).
    With keep='first' (fixed behaviour): dup_count = 5 (excess only).
    """
    tables = dup_setup
    meta = infer_metadata(tables)

    issues = detect_duplicates(tables, meta)
    dup_issues = [i for i in issues if i.table == "fact_sales"]

    assert dup_issues, "Expected at least one duplicate_rows issue"
    assert dup_issues[0].evidence_rows == N_PAIRS, (
        f"Expected evidence_rows={N_PAIRS} (excess copies only), "
        f"got {dup_issues[0].evidence_rows}. "
        f"With keep=False this would be {N_PAIRS * 2}."
    )


def test_verifier_confidence_is_high_when_counts_agree(dup_setup):
    """IssueVerifier confidence must be >= 0.95 when pandas and SQL counts agree.

    This test constructs a QualityIssue with columns=["sales_sk"] to exercise
    the SQL verification path (not the fallback). SQL gives:
      COUNT(*) - COUNT(DISTINCT "sales_sk") = 10 - 5 = 5 (excess copies).
    With evidence_rows=5 (post-fix): confidence = min(5,5)/max(5,5,1) = 1.0.
    With evidence_rows=10 (pre-fix): confidence = min(5,10)/max(5,10,1) = 0.50.
    """
    issue_post_fix = QualityIssue(
        issue_id="DDUP_TEST",
        issue_type="duplicate_rows",
        table="fact_sales",
        columns=["sales_sk"],
        severity="high",
        metric=0.5,
        evidence_rows=N_PAIRS,          # 5: post-fix count
        description=f"Detected {N_PAIRS} duplicate excess copies in 'fact_sales'.",
    )

    verifier = IssueVerifier(sensitivity="normal")
    verified = verifier.verify_all([issue_post_fix])

    assert verified[0].confidence >= 0.95, (
        f"Expected confidence >= 0.95, got {verified[0].confidence}. "
        f"sql_count={verified[0].sql_count}, pd_count={verified[0].pd_count}."
    )
