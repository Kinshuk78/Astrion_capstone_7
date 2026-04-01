from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    table: str
    column: str
    dtype: str
    null_frac: float
    distinct_count: int
    is_unique: bool


def profile_tables(tables: Dict[str, pd.DataFrame]) -> List[ColumnProfile]:
    profiles: List[ColumnProfile] = []
    for tname, df in tables.items():
        for col in df.columns:
            series = df[col]
            null_frac = float(series.isna().mean())
            distinct_count = int(series.nunique(dropna=True))
            is_unique = bool(series.is_unique)
            profiles.append(
                ColumnProfile(
                    table=tname,
                    column=col,
                    dtype=str(series.dtype),
                    null_frac=null_frac,
                    distinct_count=distinct_count,
                    is_unique=is_unique,
                )
            )
    return profiles


@dataclass
class QualityIssue:
    issue_id: str
    issue_type: str
    table: str
    columns: List[str]
    severity: str
    metric_value: float
    evidence_rows: int
    description: str


def detect_null_issues(
    tables: Dict[str, pd.DataFrame],
    profiles: List[ColumnProfile],
    threshold: float = 0.1,
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    counter = 1
    for p in profiles:
        if p.null_frac >= threshold:
            df = tables[p.table]
            evidence_rows = int((df[p.column].isna()).sum())
            issues.append(
                QualityIssue(
                    issue_id=f"QNULL{counter:04d}",
                    issue_type="high_null_fraction",
                    table=p.table,
                    columns=[p.column],
                    severity="medium" if p.null_frac < 0.5 else "high",
                    metric_value=p.null_frac,
                    evidence_rows=evidence_rows,
                    description=f"{p.null_frac:.2%} nulls in {p.table}.{p.column}",
                )
            )
            counter += 1
    return issues


def detect_duplicate_rows(
    tables: Dict[str, pd.DataFrame],
    key_columns_by_table: Dict[str, List[str]],
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    counter = 1
    for tname, df in tables.items():
        keys = key_columns_by_table.get(tname)
        if not keys:
            continue
        dup_mask = df.duplicated(subset=keys, keep=False)
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            frac = dup_count / len(df)
            severity = "low"
            if frac > 0.01:
                severity = "medium"
            if frac > 0.05:
                severity = "high"
            issues.append(
                QualityIssue(
                    issue_id=f"QDUP{counter:04d}",
                    issue_type="duplicate_rows",
                    table=tname,
                    columns=keys,
                    severity=severity,
                    metric_value=frac,
                    evidence_rows=dup_count,
                    description=f"{dup_count} duplicate rows on keys {keys} in {tname}",
                )
            )
            counter += 1
    return issues


def detect_outliers(
    tables: Dict[str, pd.DataFrame],
    numeric_cols_by_table: Dict[str, List[str]],
    z_threshold: float = 4.0,
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    counter = 1
    for tname, df in tables.items():
        cols = numeric_cols_by_table.get(tname, [])
        for col in cols:
            series = df[col].dropna()
            if len(series) < 20:
                continue
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            if std == 0:
                continue
            z = np.abs((series - mean) / std)
            outliers = int((z > z_threshold).sum())
            if outliers > 0:
                frac = outliers / len(df)
                severity = "low"
                if frac > 0.01:
                    severity = "medium"
                if frac > 0.05:
                    severity = "high"
                issues.append(
                    QualityIssue(
                        issue_id=f"QOUT{counter:04d}",
                        issue_type="numeric_outliers",
                        table=tname,
                        columns=[col],
                        severity=severity,
                        metric_value=frac,
                        evidence_rows=outliers,
                        description=f"{outliers} outliers in {tname}.{col}",
                    )
                )
                counter += 1
    return issues

