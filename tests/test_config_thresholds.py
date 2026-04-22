"""Regression test: detect_nulls must use NULL_THRESHOLD_IMPORTANT (0.01) for key
columns in normal-sensitivity mode, not the old hardcoded 0.02.

A column with exactly 1.5% nulls:
  - Old threshold 0.02: 1.5% < 2%  -> no issue raised (BUG)
  - New threshold 0.01: 1.5% > 1%  -> issue raised   (CORRECT)
"""
from __future__ import annotations

import pandas as pd

from astrion_dq.checks.detect import detect_nulls
from astrion_dq.config import NULL_THRESHOLD_IMPORTANT
from astrion_dq.metadata import infer_metadata


def _make_tables_with_fk_nulls(frac: float, n_rows: int = 1000):
    """Return tables where the fact FK column has exactly frac * n_rows nulls."""
    dim_customers = pd.DataFrame({"customer_sk": list(range(1, n_rows + 1))})

    n_null = int(frac * n_rows)
    customer_sk_vals = list(range(1, n_rows - n_null + 1)) + [None] * n_null

    fact_sales = pd.DataFrame({
        "sales_sk": list(range(1, n_rows + 1)),
        "customer_sk": customer_sk_vals,
        "amount": [1.0] * n_rows,
    })

    return {"dim_customers": dim_customers, "fact_sales": fact_sales}


def test_null_threshold_important_fires_at_1_5_percent():
    """detect_nulls in normal mode must fire on a 1.5% null FK column.

    NULL_THRESHOLD_IMPORTANT = 0.01.  1.5% > 1%  -> issue expected.
    The old _NULL_KEY_NORMAL = 0.02 would NOT have fired (1.5% < 2%).
    """
    # Verify the constant itself to catch config regressions.
    assert NULL_THRESHOLD_IMPORTANT == 0.01, (
        f"NULL_THRESHOLD_IMPORTANT changed from 0.01 to {NULL_THRESHOLD_IMPORTANT}; "
        "update this test accordingly."
    )

    tables = _make_tables_with_fk_nulls(frac=0.015, n_rows=1000)
    meta = infer_metadata(tables)

    # customer_sk is an FK column -> goes into the 'important' set
    fk_cols = meta["fact_sales"].foreign_keys
    assert "customer_sk" in fk_cols, (
        "customer_sk should be detected as a FK column by infer_metadata"
    )

    issues = detect_nulls(tables, meta, sensitivity="normal")
    fact_null_issues = [
        i for i in issues
        if i.table == "fact_sales" and "customer_sk" in i.columns
    ]

    assert fact_null_issues, (
        "detect_nulls should raise an issue for customer_sk at 1.5% nulls "
        "(NULL_THRESHOLD_IMPORTANT = 0.01). The old threshold of 0.02 would "
        "have missed this."
    )


def test_null_threshold_does_not_fire_below_1_percent():
    """detect_nulls in normal mode must NOT fire when null fraction is below 1%."""
    tables = _make_tables_with_fk_nulls(frac=0.009, n_rows=1000)
    meta = infer_metadata(tables)

    issues = detect_nulls(tables, meta, sensitivity="normal")
    fact_null_issues = [
        i for i in issues
        if i.table == "fact_sales" and "customer_sk" in i.columns
    ]

    assert not fact_null_issues, (
        "detect_nulls should not fire when null fraction (0.9%) is below "
        "NULL_THRESHOLD_IMPORTANT (1.0%)."
    )


def test_high_sensitivity_fires_at_0_5_percent():
    """In high-sensitivity mode the uniform threshold is NULL_THRESHOLD_HIGH_SENS (0.01).

    A 0.5% null fraction on a non-key column should not fire (< 1%) but any
    column at 1.1% should fire.
    """
    from astrion_dq.config import NULL_THRESHOLD_HIGH_SENS

    tables = _make_tables_with_fk_nulls(frac=0.011, n_rows=1000)
    meta = infer_metadata(tables)

    issues = detect_nulls(tables, meta, sensitivity="high")
    customer_sk_issues = [
        i for i in issues
        if i.table == "fact_sales" and "customer_sk" in i.columns
    ]

    assert customer_sk_issues, (
        f"High-sensitivity mode (threshold={NULL_THRESHOLD_HIGH_SENS}) should "
        "fire on 1.1% nulls."
    )
