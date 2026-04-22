"""Regression test for F-01: drift detector must gate columns via meta.numeric_cols.

Before the fix, detect_drift iterated cur_df.columns and filtered only by
is_numeric_dtype. Surrogate key columns (sales_sk, customer_sk) are numeric
but are NOT meaningful distribution targets. On injected data their value
ranges shift, producing 16 false-positive drift issues (confirmed at F1=0.308
for C_full in the pre-fix evaluation baseline).

After the fix, detect_drift requires a `meta` argument and gates each column:
    m = meta.get(table)
    if m is not None and col not in m.numeric_cols:
        continue

metadata.numeric_cols already excludes key columns via is_key_col() at
metadata.py:138. The fix makes detect_drift consistent with detect_outliers,
which already uses meta.numeric_cols (detect.py:159).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astrion_dq.checks.drift import detect_drift
from astrion_dq.metadata import infer_metadata


def _make_tables(sales_sk_start: int, amount_values: list) -> dict:
    """Return a minimal star schema fact + dim pair.

    sales_sk and customer_sk are surrogate key columns (both excluded from
    numeric_cols by is_key_col). amount is the only genuine numeric column.
    """
    n = len(amount_values)
    return {
        "fact_sales": pd.DataFrame({
            "sales_sk": list(range(sales_sk_start, sales_sk_start + n)),
            "customer_sk": [i % 50 + 1 for i in range(n)],
            "amount": amount_values,
        }),
        "dim_customers": pd.DataFrame({
            "customer_sk": list(range(1, 51)),
        }),
    }


@pytest.fixture()
def schema_pair():
    """Reference and current tables.

    sales_sk range shifts by 500 (PSI would fire if key columns are scanned).
    amount values are identical across both (no genuine drift).
    """
    rng = np.random.default_rng(42)
    n = 500
    amount = rng.normal(loc=100.0, scale=10.0, size=n).tolist()
    ref_tables = _make_tables(1, amount)
    cur_tables = _make_tables(501, amount)   # sales_sk shifted -- no amount change
    return ref_tables, cur_tables


def test_no_false_positives_on_key_columns(schema_pair):
    """Shifted key columns must not trigger drift issues after the fix.

    Before fix: detect_drift(cur_tables, meta, ...) raises TypeError because
    the old signature is detect_drift(current_tables, reference_tables=None, ...)
    and passing `meta` as the second positional argument conflicts with
    the keyword `reference_tables=ref_tables`.
    """
    ref_tables, cur_tables = schema_pair
    meta = infer_metadata(cur_tables)

    # Verify setup: key columns must NOT appear in numeric_cols.
    fact_meta = meta["fact_sales"]
    assert "sales_sk" not in fact_meta.numeric_cols
    assert "customer_sk" not in fact_meta.numeric_cols
    assert "amount" in fact_meta.numeric_cols

    issues = detect_drift(cur_tables, meta, reference_tables=ref_tables)

    key_issues = [
        i for i in issues
        if any(c in ("sales_sk", "customer_sk") for c in i.columns)
    ]
    assert not key_issues, (
        f"detect_drift must not fire on surrogate key columns; found: {key_issues}"
    )


def test_genuine_signal_still_detected(schema_pair):
    """A large shift in a real numeric column must still produce a drift issue.

    Multiplying amount by 5 (mean shifts from ~100 to ~500) must trigger PSI
    or KS. This confirms the meta gate does not suppress valid detections.
    """
    ref_tables, cur_tables = schema_pair
    # Apply a 5x multiplier to amount in current tables.
    shifted_tables = {
        name: df.copy() for name, df in cur_tables.items()
    }
    shifted_tables["fact_sales"]["amount"] = (
        shifted_tables["fact_sales"]["amount"] * 5.0
    )
    meta = infer_metadata(shifted_tables)

    issues = detect_drift(shifted_tables, meta, reference_tables=ref_tables)

    amount_issues = [i for i in issues if "amount" in i.columns]
    assert amount_issues, (
        "detect_drift must report drift when a genuine numeric column shifts by 5x"
    )
