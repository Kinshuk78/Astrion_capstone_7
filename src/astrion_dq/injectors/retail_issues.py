"""Synthetic issue injector for the retail star schema.

Injects seven controlled issue types into a copy of the retail tables and writes:
  - Modified CSVs to ``data/injected/retail/``
  - Ground-truth JSON to ``outputs/retail_injected_issues.json``

Issue types (exact strings used as ground-truth labels by the evaluator):
  missing_key_values        -- nulls in the fact PK or first FK column
  duplicate_transactions    -- duplicated fact rows
  invalid_future_dates      -- date values beyond year 2099
  referential_integrity_break -- fact FK values absent from the dim PK
  numeric_outliers          -- numeric values multiplied by 10
  promotion_drift           -- promo/campaign column values corrupted
  dimension_missing_values  -- nulls in a dimension text column
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

from astrion_dq.config import INJECTED_DIR, OUTPUTS_DIR
from astrion_dq.metadata import infer_metadata, is_key_col


@dataclass
class InjectedIssue:
    issue_id: str
    issue_type: str
    table: str
    columns: List[str]
    row_count: int
    severity: str
    business_area: str
    intended_report_impact: str
    parameters: Dict[str, str | int | float]


def _choose_indices(n_rows: int, frac: float, rng: random.Random) -> List[int]:
    if n_rows <= 0:
        return []
    k = max(1, int(n_rows * frac))
    k = min(k, n_rows)
    return rng.sample(list(range(n_rows)), k)


def _future_date_value(series: pd.Series):
    if is_datetime64_any_dtype(series):
        return pd.Timestamp("2099-01-01")
    if is_integer_dtype(series):
        return 20990101
    return "2099-01-01"


def _invalid_fk_value(series: pd.Series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return 999999
    if is_integer_dtype(series):
        return int(non_null.max()) + 999
    return "__INVALID_FK__"


def _write_tables(output_dir, tables: Dict[str, pd.DataFrame]) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def inject_retail_issues(
    tables: Dict[str, pd.DataFrame],
    seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], List[InjectedIssue]]:
    """Inject synthetic quality issues and return (modified_tables, ground_truth).

    Uses ``infer_metadata`` to select columns by role rather than by suffix
    pattern. This handles datasets that use ``_sk`` surrogate keys for FKs
    and ``_id`` for natural/business keys.

    Args:
        tables: Clean tables loaded from the retail directory.
        seed:   Random seed for reproducibility.

    Returns:
        Tuple of (injected table dict, list of InjectedIssue ground-truth records).
    """
    rng = random.Random(seed)
    injected = {name: df.copy() for name, df in tables.items()}
    issues: List[InjectedIssue] = []
    issue_counter = 1

    # Infer metadata so column selection is schema-aware.
    meta = infer_metadata(injected)

    # Identify the fact table (first table with role=="fact", else first table).
    fact_name: Optional[str] = None
    for name in injected:
        if meta[name].role == "fact":
            fact_name = name
            break
    if fact_name is None:
        fact_name = next(iter(injected.keys()))

    fact_df = injected[fact_name]
    n_rows = len(fact_df)
    fact_meta = meta[fact_name]

    date_cols = fact_meta.date_cols
    numeric_cols = fact_meta.numeric_cols
    promo_cols = fact_meta.promo_cols

    # Fallback key-like column list for the missing-key injection when no PK detected.
    key_like_cols = [c for c in fact_df.columns if is_key_col(c)]

    # 1. Missing key values
    # Priority: fact PK -> first FK -> first key-like column.
    pk_col: Optional[str] = None
    if fact_meta.primary_keys:
        pk_col = fact_meta.primary_keys[0]
    elif fact_meta.foreign_keys:
        pk_col = next(iter(fact_meta.foreign_keys))
    elif key_like_cols:
        pk_col = key_like_cols[0]

    if pk_col is not None:
        idx = _choose_indices(n_rows, 0.02, rng)
        fact_df.loc[idx, pk_col] = pd.NA
        issues.append(InjectedIssue(
            issue_id=f"I{issue_counter:04d}",
            issue_type="missing_key_values",
            table=fact_name,
            columns=[pk_col],
            row_count=len(idx),
            severity="high",
            business_area="sales",
            intended_report_impact="sales by store / product joins may break",
            parameters={"fraction": 0.02, "column": pk_col},
        ))
        issue_counter += 1

    # 2. Duplicate transactions
    dup_idx = _choose_indices(n_rows, 0.02, rng)
    if dup_idx:
        dup_rows = fact_df.iloc[dup_idx].copy()
        fact_df = pd.concat([fact_df, dup_rows], ignore_index=True)
        injected[fact_name] = fact_df
        issues.append(InjectedIssue(
            issue_id=f"I{issue_counter:04d}",
            issue_type="duplicate_transactions",
            table=fact_name,
            columns=[],
            row_count=len(dup_idx),
            severity="high",
            business_area="sales",
            intended_report_impact="daily sales and top products may be overstated",
            parameters={"fraction": 0.02},
        ))
        issue_counter += 1

    # Recompute after duplication.
    fact_df = injected[fact_name]
    n_rows = len(fact_df)

    # 3. Invalid / future dates
    if date_cols:
        col = date_cols[0]
        idx = _choose_indices(n_rows, 0.02, rng)
        future_value = _future_date_value(fact_df[col])
        fact_df.loc[idx, col] = future_value
        issues.append(InjectedIssue(
            issue_id=f"I{issue_counter:04d}",
            issue_type="invalid_future_dates",
            table=fact_name,
            columns=[col],
            row_count=len(idx),
            severity="medium",
            business_area="time",
            intended_report_impact="daily sales trend and date aggregations become unreliable",
            parameters={"fraction": 0.02, "column": col, "value": str(future_value)},
        ))
        issue_counter += 1

    # 4. Referential integrity break
    # Use the first FK column identified by metadata (a _sk column, not the natural key).
    fk_col: Optional[str] = (
        next(iter(fact_meta.foreign_keys)) if fact_meta.foreign_keys else None
    )
    if fk_col is not None:
        idx = _choose_indices(n_rows, 0.02, rng)
        bad_value = _invalid_fk_value(fact_df[fk_col])
        fact_df.loc[idx, fk_col] = bad_value
        issues.append(InjectedIssue(
            issue_id=f"I{issue_counter:04d}",
            issue_type="referential_integrity_break",
            table=fact_name,
            columns=[fk_col],
            row_count=len(idx),
            severity="high",
            business_area="warehouse joins",
            intended_report_impact="dimension joins fail and grouped reporting becomes incomplete",
            parameters={"fraction": 0.02, "column": fk_col, "bad_value": str(bad_value)},
        ))
        issue_counter += 1

    # 5. Numeric outliers
    if numeric_cols:
        col = numeric_cols[0]
        idx = _choose_indices(n_rows, 0.02, rng)
        fact_df.loc[idx, col] = fact_df.loc[idx, col] * 10
        issues.append(InjectedIssue(
            issue_id=f"I{issue_counter:04d}",
            issue_type="numeric_outliers",
            table=fact_name,
            columns=[col],
            row_count=len(idx),
            severity="high",
            business_area="sales metrics",
            intended_report_impact="sales totals and product rankings become distorted",
            parameters={"fraction": 0.02, "column": col, "multiplier": 10},
        ))
        issue_counter += 1

    # 6. Promotion drift
    if promo_cols:
        col = promo_cols[0]
        idx = _choose_indices(n_rows, 0.02, rng)
        if is_numeric_dtype(fact_df[col]):
            fact_df.loc[idx, col] = fact_df.loc[idx, col] * 3
        else:
            fact_df.loc[idx, col] = "__PROMO_DRIFT__"
        issues.append(InjectedIssue(
            issue_id=f"I{issue_counter:04d}",
            issue_type="promotion_drift",
            table=fact_name,
            columns=[col],
            row_count=len(idx),
            severity="medium",
            business_area="campaign performance",
            intended_report_impact="promotion performance reporting becomes misleading",
            parameters={"fraction": 0.02, "column": col},
        ))
        issue_counter += 1

    # 7. Dimension missing values (optional — only when a customer dimension exists)
    dim_customer: Optional[str] = None
    for name in injected:
        if "customer" in name.lower():
            dim_customer = name
            break

    if dim_customer is not None:
        df = injected[dim_customer]
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if obj_cols:
            col = obj_cols[0]
            idx = _choose_indices(len(df), 0.03, rng)
            df.loc[idx, col] = None
            issues.append(InjectedIssue(
                issue_id=f"I{issue_counter:04d}",
                issue_type="dimension_missing_values",
                table=dim_customer,
                columns=[col],
                row_count=len(idx),
                severity="medium",
                business_area="customer dimension",
                intended_report_impact="customer-level slicing may be incomplete",
                parameters={"fraction": 0.03, "column": col},
            ))
            issue_counter += 1

    _write_tables(INJECTED_DIR, injected)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "retail_injected_issues.json", "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in issues], f, indent=2)

    return injected, issues
