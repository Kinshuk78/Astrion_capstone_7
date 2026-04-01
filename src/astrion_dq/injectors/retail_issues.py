from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INJECTED_DIR = PROJECT_ROOT / "data" / "injected" / "retail"


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


def _write_tables(output_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def inject_retail_issues(
    tables: Dict[str, pd.DataFrame],
    seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], List[InjectedIssue]]:
    rng = random.Random(seed)
    injected = {name: df.copy() for name, df in tables.items()}
    issues: List[InjectedIssue] = []
    issue_counter = 1

    fact_name = None
    for name in injected:
        lower = name.lower()
        if "fact" in lower and "denormalized" not in lower:
            fact_name = name
            break

    if fact_name is None:
        fact_name = next(iter(injected.keys()))

    fact_df = injected[fact_name]
    n_rows = len(fact_df)

    cols = list(fact_df.columns)
    id_cols = [c for c in cols if c.lower().endswith("_id")]
    date_cols = [c for c in cols if "date" in c.lower() or "day" in c.lower()]
    numeric_cols = [c for c in cols if is_numeric_dtype(fact_df[c]) and not c.lower().endswith("_id")]

    campaign_cols = [c for c in cols if any(tok in c.lower() for tok in ["campaign", "promo", "discount"])]

    # 1. Missing key values
    if id_cols:
        col = id_cols[0]
        idx = _choose_indices(n_rows, 0.02, rng)
        fact_df.loc[idx, col] = pd.NA
        issues.append(
            InjectedIssue(
                issue_id=f"I{issue_counter:04d}",
                issue_type="missing_key_values",
                table=fact_name,
                columns=[col],
                row_count=len(idx),
                severity="high",
                business_area="sales",
                intended_report_impact="sales by store / product joins may break",
                parameters={"fraction": 0.02, "column": col},
            )
        )
        issue_counter += 1

    # 2. Duplicate transactions
    dup_idx = _choose_indices(n_rows, 0.02, rng)
    if dup_idx:
        dup_rows = fact_df.iloc[dup_idx].copy()
        fact_df = pd.concat([fact_df, dup_rows], ignore_index=True)
        injected[fact_name] = fact_df
        issues.append(
            InjectedIssue(
                issue_id=f"I{issue_counter:04d}",
                issue_type="duplicate_transactions",
                table=fact_name,
                columns=[],
                row_count=len(dup_idx),
                severity="high",
                business_area="sales",
                intended_report_impact="daily sales and top products may be overstated",
                parameters={"fraction": 0.02},
            )
        )
        issue_counter += 1

    # recompute after duplication
    fact_df = injected[fact_name]
    n_rows = len(fact_df)

    # 3. Invalid / future dates
    if date_cols:
        col = date_cols[0]
        idx = _choose_indices(n_rows, 0.02, rng)
        future_value = _future_date_value(fact_df[col])
        fact_df.loc[idx, col] = future_value
        issues.append(
            InjectedIssue(
                issue_id=f"I{issue_counter:04d}",
                issue_type="invalid_future_dates",
                table=fact_name,
                columns=[col],
                row_count=len(idx),
                severity="medium",
                business_area="time",
                intended_report_impact="daily sales trend and date aggregations become unreliable",
                parameters={"fraction": 0.02, "column": col, "value": str(future_value)},
            )
        )
        issue_counter += 1

    # 4. Referential integrity break
    if len(id_cols) >= 2:
        col = id_cols[1]
        idx = _choose_indices(n_rows, 0.02, rng)
        bad_value = _invalid_fk_value(fact_df[col])
        fact_df.loc[idx, col] = bad_value
        issues.append(
            InjectedIssue(
                issue_id=f"I{issue_counter:04d}",
                issue_type="referential_integrity_break",
                table=fact_name,
                columns=[col],
                row_count=len(idx),
                severity="high",
                business_area="warehouse joins",
                intended_report_impact="dimension joins fail and grouped reporting becomes incomplete",
                parameters={"fraction": 0.02, "column": col, "bad_value": str(bad_value)},
            )
        )
        issue_counter += 1

    # 5. Numeric outliers
    if numeric_cols:
        col = numeric_cols[0]
        idx = _choose_indices(n_rows, 0.02, rng)
        fact_df.loc[idx, col] = fact_df.loc[idx, col] * 10
        issues.append(
            InjectedIssue(
                issue_id=f"I{issue_counter:04d}",
                issue_type="numeric_outliers",
                table=fact_name,
                columns=[col],
                row_count=len(idx),
                severity="high",
                business_area="sales metrics",
                intended_report_impact="sales totals and product rankings become distorted",
                parameters={"fraction": 0.02, "column": col, "multiplier": 10},
            )
        )
        issue_counter += 1

    # 6. Promotion drift
    if campaign_cols:
        col = campaign_cols[0]
        idx = _choose_indices(n_rows, 0.02, rng)
        if is_numeric_dtype(fact_df[col]):
            fact_df.loc[idx, col] = fact_df.loc[idx, col] * 3
        else:
            fact_df.loc[idx, col] = "__PROMO_DRIFT__"
        issues.append(
            InjectedIssue(
                issue_id=f"I{issue_counter:04d}",
                issue_type="promotion_drift",
                table=fact_name,
                columns=[col],
                row_count=len(idx),
                severity="medium",
                business_area="campaign performance",
                intended_report_impact="promotion performance reporting becomes misleading",
                parameters={"fraction": 0.02, "column": col},
            )
        )
        issue_counter += 1

    # Optional dimension corruption
    dim_customer = None
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
            issues.append(
                InjectedIssue(
                    issue_id=f"I{issue_counter:04d}",
                    issue_type="dimension_missing_values",
                    table=dim_customer,
                    columns=[col],
                    row_count=len(idx),
                    severity="medium",
                    business_area="customer dimension",
                    intended_report_impact="customer-level slicing may be incomplete",
                    parameters={"fraction": 0.03, "column": col},
                )
            )
            issue_counter += 1

    _write_tables(INJECTED_DIR, injected)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "retail_injected_issues.json", "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in issues], f, indent=2)

    return injected, issues
