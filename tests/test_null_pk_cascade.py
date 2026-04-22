"""Regression test for F-16: detect_nulls must apply the strict threshold
to key-like columns even when PK inference fails on corrupted data.

Root cause:
  metadata.py:163 detects a PK by checking:
    df[c].nunique(dropna=True) == len(df)
  When the injector inserts 2% nulls into sales_sk AFTER inserting duplicate
  rows, len(df) > nunique(dropna=True) and PK inference fails (primary_keys=[]).

  detect.py:84-88 builds the `important` set from:
    primary_keys + foreign_keys.keys() + date_cols
  With primary_keys=[], sales_sk is NOT in important. It is tested against
  NULL_THRESHOLD_OTHER (0.05). At 2% nulls: 0.02 < 0.05 -> issue silenced.

Fix:
  Augment important with all is_key_col(c) columns regardless of PK inference.
  sales_sk ends with _sk, so is_key_col("sales_sk") == True. It then uses
  NULL_THRESHOLD_IMPORTANT (0.01). 0.02 > 0.01 -> issue raised.
"""
from __future__ import annotations

import pandas as pd
import pytest

from astrion_dq.checks.detect import detect_nulls
from astrion_dq.config import NULL_THRESHOLD_IMPORTANT, NULL_THRESHOLD_OTHER
from astrion_dq.metadata import infer_metadata


NULL_FRAC = 0.02    # 2% nulls: above NULL_THRESHOLD_IMPORTANT (0.01), below NULL_THRESHOLD_OTHER (0.05)
N_ROWS = 1000


@pytest.fixture()
def corrupted_fact():
    """Fact table where PK inference fails due to injected duplicates + nulls.

    sales_sk values: 950 unique + 30 duplicates of values 1-30 + 20 nulls.
    nunique(dropna=True) = 950, len(df) = 1000. PK check: 950 != 1000 -> fails.
    Null fraction for sales_sk = 20/1000 = 2%.
    """
    n_unique = 950
    n_dup = 30
    n_null = N_ROWS - n_unique - n_dup    # 20

    sales_sk = (
        list(range(1, n_unique + 1))      # 950 unique values
        + list(range(1, n_dup + 1))       # 30 duplicates of 1-30
        + [None] * n_null                 # 20 nulls
    )
    assert len(sales_sk) == N_ROWS

    tables = {
        "fact_sales": pd.DataFrame({
            "sales_sk": sales_sk,
            "amount": [1.0] * N_ROWS,
        })
    }
    return tables


def test_pk_inference_fails_on_corrupted_data(corrupted_fact):
    """Confirm the precondition: PK inference must fail on the corrupted table."""
    meta = infer_metadata(corrupted_fact)
    assert meta["fact_sales"].primary_keys == [], (
        "Test precondition failed: expected primary_keys=[] on corrupted data. "
        "If PK inference was fixed elsewhere, this test needs updating."
    )


def test_null_threshold_important_fires_on_key_col_despite_pk_inference_failure(corrupted_fact):
    """detect_nulls must raise an issue for sales_sk at 2% nulls.

    Thresholds:
      NULL_THRESHOLD_IMPORTANT = 0.01 (fires at 2%)
      NULL_THRESHOLD_OTHER     = 0.05 (does NOT fire at 2%)

    Before fix: sales_sk not in important (primary_keys=[]) -> threshold=0.05
                0.02 < 0.05 -> no issue raised. Assertion FAILS.
    After fix:  sales_sk in important via is_key_col -> threshold=0.01
                0.02 > 0.01 -> issue raised. Assertion PASSES.
    """
    # Verify constants have not drifted from expected values.
    assert NULL_THRESHOLD_IMPORTANT == 0.01
    assert NULL_THRESHOLD_OTHER == 0.05

    meta = infer_metadata(corrupted_fact)
    assert meta["fact_sales"].primary_keys == [], "Precondition: PK inference must have failed"

    issues = detect_nulls(corrupted_fact, meta, sensitivity="normal")
    sales_sk_issues = [
        i for i in issues
        if i.table == "fact_sales" and "sales_sk" in i.columns
    ]

    assert sales_sk_issues, (
        f"detect_nulls must raise an issue for sales_sk at {NULL_FRAC:.0%} nulls "
        f"(NULL_THRESHOLD_IMPORTANT={NULL_THRESHOLD_IMPORTANT}). "
        f"Without the fix, primary_keys=[] causes threshold={NULL_THRESHOLD_OTHER} "
        f"and the issue is silenced."
    )


def test_null_threshold_does_not_fire_below_important_threshold(corrupted_fact):
    """Below NULL_THRESHOLD_IMPORTANT the fix must NOT raise a spurious issue.

    Build a table with 0.5% nulls on sales_sk (below 0.01 threshold).
    Neither threshold would fire.
    """
    n_null_low = 5    # 0.5% of 1000
    n_clean = N_ROWS - n_null_low

    tables_low = {
        "fact_sales": pd.DataFrame({
            "sales_sk": list(range(1, n_clean + 1)) + [None] * n_null_low,
            "amount": [1.0] * N_ROWS,
        })
    }
    meta = infer_metadata(tables_low)
    issues = detect_nulls(tables_low, meta, sensitivity="normal")
    sales_sk_issues = [
        i for i in issues
        if i.table == "fact_sales" and "sales_sk" in i.columns
    ]

    assert not sales_sk_issues, (
        f"detect_nulls must NOT fire on sales_sk at 0.5% nulls "
        f"(below NULL_THRESHOLD_IMPORTANT={NULL_THRESHOLD_IMPORTANT})."
    )
