from astrion_dq.checks.profiling import QualityIssue
from astrion_dq.ranking.business_impact import estimate_impact


def test_estimate_impact_increases_with_metric():
    issue_low = QualityIssue(
        issue_id="Q1",
        issue_type="duplicate_rows",
        table="fact_sales",
        columns=["transaction_id"],
        severity="medium",
        metric_value=0.01,
        evidence_rows=10,
        description="",
    )
    issue_high = QualityIssue(
        issue_id="Q2",
        issue_type="duplicate_rows",
        table="fact_sales",
        columns=["transaction_id"],
        severity="medium",
        metric_value=0.1,
        evidence_rows=100,
        description="",
    )

    impact_low = estimate_impact(issue_low).impact_score
    impact_high = estimate_impact(issue_high).impact_score
    assert impact_high > impact_low

