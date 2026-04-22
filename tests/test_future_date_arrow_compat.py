"""Regression test for F-17: detect_future_dates must handle ArrowStringArray
columns with mixed ISO 8601 formats (datetime-with-time and date-only strings).

Root cause (pandas 2.x + pyarrow):
  pd.to_datetime(s, errors='coerce') on an ArrowStringArray with mixed formats
  infers the format from the first value. If the first value is a datetime-with-
  time string ("2024-02-19T15:28:02"), pandas infers that format and then fails
  to parse date-only strings ("2099-01-01"), silently coercing them to NaT.
  NaT > Timestamp("2050-01-01") returns False. Future dates are not detected.

Fix:
  Add format='ISO8601' to the pd.to_datetime call in detect.py:205.
  ISO8601 mode handles both datetime-with-time and date-only strings correctly.

Confirmed by live test on fact_sales_normalized.sales_date:
  Without fix: future_count=0,  NaT=20400
  With fix:    future_count=20400, NaT=0
"""
from __future__ import annotations

import pandas as pd
import pytest

from astrion_dq.checks.detect import detect_future_dates
from astrion_dq.metadata import infer_metadata


N_ROWS = 200
FUTURE_FRAC = 0.05    # 5% future dates -> 10 future rows


@pytest.fixture()
def mixed_format_tables():
    """Fact table with an ArrowStringArray date column in mixed ISO 8601 format.

    The column contains:
      - 190 past datetime-with-time strings: "2024-02-19T15:28:02"
      - 10 future date-only strings:         "2099-01-01"

    Using dtype='string[pyarrow]' (ArrowStringArray) replicates the real-world
    behaviour: pandas does NOT use ArrowStringArray by default, but pyarrow-
    backed string columns appear when loading CSV files with certain pandas +
    pyarrow version combinations (including pandas >= 2.0 + pyarrow >= 12).
    """
    n_future = int(N_ROWS * FUTURE_FRAC)
    n_past = N_ROWS - n_future

    sales_date = (
        ["2024-02-19T15:28:02"] * n_past
        + ["2099-01-01"] * n_future
    )

    tables = {
        "fact_sales": pd.DataFrame({
            "sales_sk": list(range(1, N_ROWS + 1)),
            "sales_date": pd.array(sales_date, dtype="string[pyarrow]"),
            "amount": [1.0] * N_ROWS,
        })
    }
    return tables


def test_future_dates_detected_in_arrow_string_column(mixed_format_tables):
    """detect_future_dates must find future dates in a mixed-format ArrowStringArray.

    Before fix: pd.to_datetime without format coerces "2099-01-01" to NaT.
                bad_count = 0. No issue raised. Assertion FAILS.
    After fix:  format='ISO8601' parses both formats. bad_count = 10.
                Issue raised with evidence_rows=10. Assertion PASSES.
    """
    tables = mixed_format_tables
    meta = infer_metadata(tables)

    # Confirm sales_date is in date_cols.
    assert "sales_date" in meta["fact_sales"].date_cols, (
        "sales_date must be in date_cols (column name contains 'date')"
    )

    issues = detect_future_dates(tables, meta)

    date_issues = [
        i for i in issues
        if i.table == "fact_sales" and "sales_date" in i.columns
    ]

    assert date_issues, (
        "detect_future_dates must detect future dates in an ArrowStringArray column "
        "with mixed ISO 8601 formats. "
        "Without format='ISO8601', pandas 2.x coerces date-only strings to NaT."
    )
    assert date_issues[0].evidence_rows > 0, (
        f"evidence_rows must be > 0, got {date_issues[0].evidence_rows}"
    )


def test_future_dates_evidence_rows_count(mixed_format_tables):
    """evidence_rows must equal the number of future-date rows injected."""
    n_expected = int(N_ROWS * FUTURE_FRAC)   # 10

    tables = mixed_format_tables
    meta = infer_metadata(tables)
    issues = detect_future_dates(tables, meta)

    date_issues = [
        i for i in issues
        if i.table == "fact_sales" and "sales_date" in i.columns
    ]
    assert date_issues, "Expected at least one future-date issue"
    assert date_issues[0].evidence_rows == n_expected, (
        f"Expected evidence_rows={n_expected}, got {date_issues[0].evidence_rows}"
    )
