"""Deterministic fallback functions used when the LLM is unavailable.

Every agent function has a corresponding fallback that:
    - never crashes
    - returns a structurally identical response to the LLM version
    - is marked with "source": "fallback" so callers know which path was taken

These are also used as the safety net when the LLM response fails validation.
"""
from __future__ import annotations

_SEVERITY_LABELS = {"high": "Critical", "medium": "Moderate", "low": "Minor"}

_ISSUE_EXPLANATIONS: dict[str, str] = {
    "missing_values": (
        "Some records in this table are missing required values. Any report that "
        "calculates totals, averages, or counts on the affected fields will produce "
        "incomplete or incorrect results."
    ),
    "duplicate_rows": (
        "Duplicate records exist in this table. Revenue figures, customer counts, and "
        "transaction totals will be inflated in every report that reads from this data."
    ),
    "numeric_outliers": (
        "Some numeric values are far outside the expected range for this column. These "
        "extreme values distort averages and totals in any aggregated report."
    ),
    "invalid_future_dates": (
        "Date values beyond today have been found. These records will appear in "
        "future-period reports and skew trend analysis and forecasting."
    ),
    "referential_integrity_break": (
        "Some records reference values in a lookup table that do not exist. When "
        "reports join these tables, the affected rows are silently dropped, which "
        "causes undercounting in sales and performance summaries."
    ),
    "empty_table": (
        "A table that is expected to contain data is completely empty. Any report "
        "that depends on this table will return no results or break entirely."
    ),
    "statistical_drift": (
        "The distribution of values in this column has shifted significantly compared "
        "to the historical baseline. This may indicate a pipeline failure, a source "
        "system change, or unusual business activity."
    ),
}

_RECOMMENDED_ACTIONS: dict[str, str] = {
    "missing_values": "Remove or impute the missing rows before loading into reporting tables.",
    "duplicate_rows": "Deduplicate the table by keeping the earliest or most complete record.",
    "numeric_outliers": "Investigate the outlier rows and either correct or exclude them from aggregations.",
    "invalid_future_dates": "Set future dates to NULL or today's date and investigate the source system.",
    "referential_integrity_break": "Remove orphan rows or fix the foreign key references in the fact table.",
    "empty_table": "Investigate the upstream pipeline step that should be populating this table.",
    "statistical_drift": "Compare the current load to the baseline snapshot and contact the data source team.",
}


# ---------------------------------------------------------------------------
# Fallback: explain
# ---------------------------------------------------------------------------

def fallback_explain(issues: list[dict]) -> list[dict]:
    """Return deterministic business explanations for all issues."""
    result = []
    for issue in issues:
        issue_type = issue.get("issue_type", "unknown")
        explanation = _ISSUE_EXPLANATIONS.get(
            issue_type,
            "An unrecognised data quality issue was found. Manual investigation is required.",
        )
        action = _RECOMMENDED_ACTIONS.get(issue_type, "Investigate and remediate manually.")
        severity = issue.get("severity", "")
        result.append({
            "issue_id": issue.get("issue_id", ""),
            "issue_type": issue_type,
            "table": issue.get("table", ""),
            "columns": issue.get("columns", []),
            "severity": severity,
            "business_explanation": explanation,
            "risk_summary": (
                f"{_SEVERITY_LABELS.get(severity, 'Unknown')} risk — "
                f"{issue.get('evidence_rows', 0):,} affected records "
                f"({issue.get('metric', 0) * 100:.1f}% of table)."
            ),
            "recommended_action": action,
            "source": "fallback",
        })
    return result


# ---------------------------------------------------------------------------
# Fallback: prioritise
# ---------------------------------------------------------------------------

def fallback_prioritise(issues: list[dict]) -> list[dict]:
    """Return issues sorted by impact_score with deterministic justifications."""
    sorted_issues = sorted(
        issues, key=lambda x: float(x.get("impact_score", 0)), reverse=True
    )
    result = []
    for rank, issue in enumerate(sorted_issues, 1):
        severity = issue.get("severity", "")
        result.append({
            **issue,
            "priority_rank": rank,
            "urgency": (
                "critical" if severity == "high" else
                "high" if severity == "medium" else "medium"
            ),
            "priority_justification": (
                f"Ranked {rank} by Business Impact Score "
                f"({float(issue.get('impact_score', 0)):.2f}). "
                f"Severity: {_SEVERITY_LABELS.get(severity, 'Unknown')}. "
                f"Affects {issue.get('evidence_rows', 0):,} records."
            ),
            "source": "fallback",
        })
    return result


# ---------------------------------------------------------------------------
# Fallback: generate fix code
# ---------------------------------------------------------------------------

def fallback_generate_fix(issue: dict) -> dict:
    """Return deterministic SQL and Python fix code for a single issue."""
    table = issue.get("table", "your_table")
    columns = issue.get("columns", [])
    issue_type = issue.get("issue_type", "")
    col = columns[0] if columns else "your_column"
    dim_table = issue.get("dim_table", "") or "dimension_table"
    dim_pk = issue.get("dim_pk", "") or "id"

    return {
        "issue_id": issue.get("issue_id", ""),
        "issue_type": issue_type,
        "sql_fix": _sql_fix(issue_type, table, col, columns, dim_table, dim_pk),
        "python_fix": _python_fix(issue_type, table, col, columns),
        "source": "fallback",
    }


def _sql_fix(
    issue_type: str,
    table: str,
    col: str,
    columns: list[str],
    dim_table: str,
    dim_pk: str,
) -> str:
    if issue_type == "missing_values":
        return f"DELETE FROM {table}\nWHERE {col} IS NULL;"

    if issue_type == "duplicate_rows":
        cols_str = ", ".join(columns) if columns else col
        return (
            f"DELETE FROM {table}\n"
            f"WHERE rowid NOT IN (\n"
            f"    SELECT MIN(rowid)\n"
            f"    FROM {table}\n"
            f"    GROUP BY {cols_str}\n"
            f");"
        )

    if issue_type == "numeric_outliers":
        return (
            f"SELECT *\nFROM {table}\n"
            f"WHERE {col} < (SELECT AVG({col}) - 3 * STDDEV_POP({col}) FROM {table})\n"
            f"   OR {col} > (SELECT AVG({col}) + 3 * STDDEV_POP({col}) FROM {table});"
        )

    if issue_type == "invalid_future_dates":
        return (
            f"UPDATE {table}\n"
            f"SET {col} = NULL\n"
            f"WHERE {col} > CURRENT_DATE;"
        )

    if issue_type == "referential_integrity_break":
        return (
            f"SELECT f.*\n"
            f"FROM {table} f\n"
            f"LEFT JOIN {dim_table} d ON f.{col} = d.{dim_pk}\n"
            f"WHERE d.{dim_pk} IS NULL;"
        )

    if issue_type == "empty_table":
        return (
            f"SELECT COUNT(*) AS row_count FROM {table};\n"
            f"-- Table is empty. Investigate the upstream pipeline."
        )

    if issue_type == "statistical_drift":
        return (
            f"SELECT\n"
            f"    AVG({col}) AS current_avg,\n"
            f"    STDDEV_POP({col}) AS current_std,\n"
            f"    MIN({col}) AS current_min,\n"
            f"    MAX({col}) AS current_max\n"
            f"FROM {table};\n"
            f"-- Compare these values against your baseline snapshot."
        )

    return f"-- No automated fix available for issue type: {issue_type}"


def _python_fix(
    issue_type: str,
    table: str,
    col: str,
    columns: list[str],
) -> str:
    if issue_type == "missing_values":
        return (
            f"# Remove rows with missing values in '{col}'\n"
            f"df = df.dropna(subset=['{col}'])\n"
            f"print(f'Removed {{orig_len - len(df)}} rows with null {col}')"
        )

    if issue_type == "duplicate_rows":
        cols_repr = repr(columns) if columns else f"['{col}']"
        return (
            f"orig_len = len(df)\n"
            f"df = df.drop_duplicates(subset={cols_repr}, keep='first')\n"
            f"print(f'Removed {{orig_len - len(df)}} duplicate rows')"
        )

    if issue_type == "numeric_outliers":
        return (
            f"Q1 = df['{col}'].quantile(0.25)\n"
            f"Q3 = df['{col}'].quantile(0.75)\n"
            f"IQR = Q3 - Q1\n"
            f"lower = Q1 - 1.5 * IQR\n"
            f"upper = Q3 + 1.5 * IQR\n"
            f"df = df[df['{col}'].between(lower, upper)]\n"
            f"print(f'Retained {{len(df)}} rows within IQR bounds')"
        )

    if issue_type == "invalid_future_dates":
        return (
            f"import pandas as pd\n"
            f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')\n"
            f"future_mask = df['{col}'] > pd.Timestamp.today()\n"
            f"df.loc[future_mask, '{col}'] = pd.NaT\n"
            f"print(f'Nulled {{future_mask.sum()}} future dates in {col}')"
        )

    if issue_type == "referential_integrity_break":
        return (
            f"# Replace 'dim_df' with your actual dimension DataFrame\n"
            f"valid_keys = set(dim_df['{col}'])\n"
            f"orig_len = len(df)\n"
            f"df = df[df['{col}'].isin(valid_keys)]\n"
            f"print(f'Removed {{orig_len - len(df)}} orphan rows')"
        )

    if issue_type == "empty_table":
        return (
            f"if df.empty:\n"
            f"    raise ValueError(\n"
            f"        'Table {table} is empty. Check the upstream pipeline.'\n"
            f"    )"
        )

    if issue_type == "statistical_drift":
        return (
            f"from scipy import stats\n"
            f"# Replace 'baseline_df' with your snapshot DataFrame\n"
            f"ks_stat, p_value = stats.ks_2samp(\n"
            f"    baseline_df['{col}'].dropna(),\n"
            f"    df['{col}'].dropna(),\n"
            f")\n"
            f"print(f'KS stat: {{ks_stat:.4f}}, p-value: {{p_value:.4f}}')"
        )

    return f"# No automated fix available for issue type: {issue_type}"


# ---------------------------------------------------------------------------
# Fallback: report
# ---------------------------------------------------------------------------

def fallback_report(issues: list[dict], run_id: str = "") -> dict:
    """Return a deterministic executive summary when the LLM is unavailable."""
    high = [i for i in issues if i.get("severity") == "high"]
    total = len(issues)

    if total == 0:
        health = "excellent"
        summary = (
            "No data quality issues were detected in the current data load. "
            "The data appears to be clean and ready for reporting."
        )
        risks = []
        actions = ["Continue monitoring on the next scheduled data load."]
    elif len(high) >= 3:
        health = "poor"
        summary = (
            f"{total} data quality issue(s) were found, including {len(high)} critical ones. "
            "Immediate action is recommended before this data is used in reports."
        )
        risks = [
            f"Critical: {i.get('issue_type', '').replace('_', ' ').title()} "
            f"in {i.get('table', 'unknown')} affecting {i.get('evidence_rows', 0):,} records."
            for i in high[:3]
        ]
        actions = [
            "Do not publish reports until critical issues are resolved.",
            "Notify the data engineering team immediately.",
            "Review the upstream pipeline for the affected tables.",
        ]
    elif len(high) >= 1:
        health = "fair"
        summary = (
            f"{total} data quality issue(s) detected, including {len(high)} critical. "
            "Reports should be reviewed carefully before distribution."
        )
        risks = [
            f"{i.get('issue_type', '').replace('_', ' ').title()} "
            f"in {i.get('table', 'unknown')}."
            for i in high[:3]
        ]
        actions = [
            "Resolve critical issues before the next reporting cycle.",
            "Flag affected reports as under review.",
        ]
    else:
        health = "good"
        summary = (
            f"{total} minor data quality issue(s) found. "
            "No critical issues detected. Remediation can follow normal scheduling."
        )
        risks = []
        actions = ["Schedule remediation in the next sprint.", "Monitor for recurrence."]

    return {
        "run_id": run_id,
        "executive_summary": summary,
        "top_risks": risks,
        "recommended_actions": actions,
        "overall_data_health": health,
        "total_issues": total,
        "critical_issues": len(high),
        "source": "fallback",
    }
