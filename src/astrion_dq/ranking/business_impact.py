from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from astrion_dq.checks.profiling import QualityIssue


REPORT_WEIGHTS = {
    "daily_sales_summary": 1.0,
    "sales_by_store": 0.8,
    "sales_by_category": 0.8,
    "promotion_performance": 1.0,
    "top_products": 0.6,
}


@dataclass
class RankedIssue:
    issue: QualityIssue
    impact_score: float
    affected_reports: List[str]


def estimate_impact(issue: QualityIssue) -> RankedIssue:
    severity_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(issue.severity, 1.0)

    reports: List[str] = ["daily_sales_summary", "sales_by_store", "sales_by_category", "top_products"]
    if "promo" in " ".join(issue.columns).lower():
        reports.append("promotion_performance")

    if issue.issue_type in {"duplicate_rows", "missing_key_values"}:
        base = 1.0
    elif issue.issue_type in {"numeric_outliers", "high_null_fraction"}:
        base = 0.8
    else:
        base = 0.6

    weight_sum = sum(REPORT_WEIGHTS[r] for r in reports if r in REPORT_WEIGHTS)
    impact_score = float(issue.metric_value * severity_multiplier * (base + weight_sum / 5.0))

    return RankedIssue(issue=issue, impact_score=impact_score, affected_reports=reports)


def rank_issues(issues: List[QualityIssue]) -> List[RankedIssue]:
    ranked = [estimate_impact(i) for i in issues]
    ranked.sort(key=lambda r: r.impact_score, reverse=True)
    return ranked

