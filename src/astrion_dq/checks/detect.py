"""Rule-based data quality detectors.

All five detector functions accept the same signature:
  (tables, meta, sensitivity) -> List[QualityIssue]

``infer_metadata`` is re-exported here for backwards compatibility with
existing import sites (``from astrion_dq.checks.detect import infer_metadata``).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_integer_dtype

from astrion_dq.config import (
    DUP_MIN_FRACTION_HIGH,
    DUP_MIN_FRACTION_NORMAL,
    FUTURE_DATE_SENTINEL_INT,
    FUTURE_DATE_SENTINEL_TS,
    IQR_MULT_HIGH,
    IQR_MULT_NORMAL,
    MIN_ROWS_FOR_STATS,
    NULL_THRESHOLD_HIGH_SENS,
    NULL_THRESHOLD_IMPORTANT,
    NULL_THRESHOLD_OTHER,
    OUTLIER_MIN_FRAC,
    OUTLIER_MIN_FRAC_HIGH,
    SEVERITY_HIGH_THRESHOLD,
    SEVERITY_MEDIUM_THRESHOLD,
)
from astrion_dq.metadata import infer_metadata, is_key_col  # noqa: F401 — re-exported for callers
from astrion_dq.models import QualityIssue, TableMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _null_thresholds(sensitivity: str) -> Tuple[float, float]:
    """Return ``(key_threshold, other_threshold)`` for the given sensitivity level.

    In normal mode key columns use ``NULL_THRESHOLD_IMPORTANT`` (0.01), not the
    old hardcoded 0.02. In high-sensitivity mode both thresholds collapse to the
    same strict value (``NULL_THRESHOLD_HIGH_SENS``).
    """
    if sensitivity == "high":
        return NULL_THRESHOLD_HIGH_SENS, NULL_THRESHOLD_HIGH_SENS
    return NULL_THRESHOLD_IMPORTANT, NULL_THRESHOLD_OTHER


def _severity(metric: float) -> str:
    if metric >= SEVERITY_HIGH_THRESHOLD:
        return "high"
    if metric >= SEVERITY_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def detect_nulls(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    """Detect columns with null fractions above threshold.

    Key columns (PKs, FKs, date cols) use ``NULL_THRESHOLD_IMPORTANT`` (0.01)
    in normal mode. Other columns use ``NULL_THRESHOLD_OTHER`` (0.05).
    High-sensitivity mode applies ``NULL_THRESHOLD_HIGH_SENS`` uniformly.
    """
    issues: List[QualityIssue] = []
    ctr = 1
    key_threshold, other_threshold = _null_thresholds(sensitivity)

    for table, df in tables.items():
        important = set(
            meta[table].primary_keys
            + list(meta[table].foreign_keys.keys())
            + meta[table].date_cols
            + [c for c in df.columns if is_key_col(c)]   # key-like cols always strict
        )
        for col in df.columns:
            frac = float(df[col].isna().mean())
            threshold = key_threshold if col in important else other_threshold
            if frac >= threshold:
                issues.append(QualityIssue(
                    issue_id=f"DNULL_{ctr:04d}",
                    issue_type="missing_values",
                    table=table,
                    columns=[col],
                    severity=_severity(frac),
                    metric=round(frac, 6),
                    evidence_rows=int(df[col].isna().sum()),
                    description=f"Column '{col}' has {frac:.2%} missing values.",
                ))
                ctr += 1
    return issues


def detect_duplicates(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    """Detect duplicate rows, scoping deduplication to primary key columns when available."""
    issues: List[QualityIssue] = []
    ctr = 1
    min_frac = DUP_MIN_FRACTION_HIGH if sensitivity == "high" else DUP_MIN_FRACTION_NORMAL

    for table, df in tables.items():
        subset = meta[table].primary_keys or None
        dup_count = int(df.duplicated(subset=subset, keep="first").sum())
        if dup_count == 0:
            continue
        frac = dup_count / max(1, len(df))
        if frac >= min_frac or dup_count >= 2:
            if subset:
                desc = (
                    f"Detected {dup_count} duplicate excess copies in '{table}' "
                    f"on primary key columns."
                )
            else:
                desc = (
                    f"Detected {dup_count} duplicate excess copies in '{table}' "
                    f"on all columns (no primary key found)."
                )
            issues.append(QualityIssue(
                issue_id=f"DDUP_{ctr:04d}",
                issue_type="duplicate_rows",
                table=table,
                columns=subset or [],
                severity=_severity(frac),
                metric=round(frac, 6),
                evidence_rows=dup_count,
                description=desc,
            ))
            ctr += 1
    return issues


def detect_outliers(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    """Detect IQR-based numeric outliers in fact table numeric columns."""
    issues: List[QualityIssue] = []
    ctr = 1
    iqr_mult = IQR_MULT_HIGH if sensitivity == "high" else IQR_MULT_NORMAL
    min_frac = OUTLIER_MIN_FRAC_HIGH if sensitivity == "high" else OUTLIER_MIN_FRAC

    for table, df in tables.items():
        if meta[table].role != "fact":
            continue
        for col in meta[table].numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < MIN_ROWS_FOR_STATS:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
            out_count = int(((series < lower) | (series > upper)).sum())
            frac = out_count / max(1, len(series))
            if frac >= min_frac:
                issues.append(QualityIssue(
                    issue_id=f"DOUT_{ctr:04d}",
                    issue_type="numeric_outliers",
                    table=table,
                    columns=[col],
                    severity=_severity(frac),
                    metric=round(frac, 6),
                    evidence_rows=out_count,
                    description=f"Column '{col}' has {out_count} IQR outliers (mult={iqr_mult}).",
                ))
                ctr += 1
    return issues


def detect_future_dates(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    """Detect date column values beyond the configured future-date sentinel (year 2050).

    ``sensitivity`` is accepted for API uniformity but does not alter the sentinel
    value — the threshold (year 2050) is an absolute business rule, not a dial.
    """
    del sensitivity  # uniform interface; threshold is not sensitivity-dependent
    issues: List[QualityIssue] = []
    ctr = 1

    for table, df in tables.items():
        for col in meta[table].date_cols:
            s = df[col]
            if is_integer_dtype(s):
                bad_mask = pd.to_numeric(s, errors="coerce") > FUTURE_DATE_SENTINEL_INT
            else:
                bad_mask = pd.to_datetime(s, errors="coerce", format="ISO8601") > pd.Timestamp(FUTURE_DATE_SENTINEL_TS)

            bad_count = int(bad_mask.fillna(False).sum())
            if bad_count > 0:
                frac = bad_count / max(1, len(df))
                issues.append(QualityIssue(
                    issue_id=f"DDATE_{ctr:04d}",
                    issue_type="invalid_future_dates",
                    table=table,
                    columns=[col],
                    severity=_severity(frac),
                    metric=round(frac, 6),
                    evidence_rows=bad_count,
                    description=(
                        f"Column '{col}' contains {bad_count} values "
                        f"beyond {FUTURE_DATE_SENTINEL_TS}."
                    ),
                ))
                ctr += 1
    return issues


def detect_referential_breaks(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    """Detect fact rows whose FK values do not appear in the referenced dimension PK.

    ``sensitivity`` is accepted for API uniformity; referential integrity is a
    binary check (value present or absent) and has no sensitivity dial.
    """
    del sensitivity  # uniform interface; RI check is binary, not sensitivity-dependent
    issues: List[QualityIssue] = []
    ctr = 1

    for table, df in tables.items():
        if meta[table].role != "fact":
            continue
        for fk_col, (dim_table, dim_pk) in meta[table].foreign_keys.items():
            if (
                dim_table not in tables
                or dim_pk not in tables[dim_table].columns
                or fk_col not in df.columns
            ):
                continue
            valid = set(tables[dim_table][dim_pk].dropna().unique().tolist())
            mask = ~df[fk_col].isna() & ~df[fk_col].isin(valid)
            bad_count = int(mask.sum())
            if bad_count > 0:
                frac = bad_count / max(1, len(df))
                issues.append(QualityIssue(
                    issue_id=f"DFK_{ctr:04d}",
                    issue_type="referential_integrity_break",
                    table=table,
                    columns=[fk_col],
                    severity=_severity(frac),
                    metric=round(frac, 6),
                    evidence_rows=bad_count,
                    description=(
                        f"Column '{fk_col}' has {bad_count} values "
                        f"not found in {dim_table}.{dim_pk}."
                    ),
                    dim_table=dim_table,
                    dim_pk=dim_pk,
                ))
                ctr += 1
    return issues


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def run_all_checks_parallel(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    """Run all five quality checks in parallel using a thread pool.

    Empty tables are flagged immediately and excluded from per-check functions.
    Each check runs in its own thread; exceptions are logged and skipped.
    """
    all_issues: List[QualityIssue] = []
    non_empty: Dict[str, pd.DataFrame] = {}

    for name, df in tables.items():
        if len(df) == 0:
            all_issues.append(QualityIssue(
                issue_id=f"EMPTY_{name.upper()[:20]}",
                issue_type="empty_table",
                table=name,
                columns=[],
                severity="high",
                metric=1.0,
                evidence_rows=0,
                description=f"Table '{name}' contains zero rows.",
            ))
        else:
            non_empty[name] = df

    if not non_empty:
        return all_issues

    checks = [
        detect_nulls,
        detect_duplicates,
        detect_outliers,
        detect_future_dates,
        detect_referential_breaks,
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fn, non_empty, meta, sensitivity): fn.__name__
            for fn in checks
        }
        for future in as_completed(futures):
            check_name = futures[future]
            try:
                all_issues.extend(future.result())
            except Exception as exc:
                logger.warning("Check '%s' raised: %s", check_name, exc)

    return all_issues
