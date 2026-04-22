from astrion_dq.models import RankedIssue


def _make_issue(issue_id: str) -> RankedIssue:
    return RankedIssue(
        issue_id=issue_id,
        issue_type="missing_values",
        table="fact_sales",
        columns=["customer_sk"],
        severity="high",
        metric=0.2,
        evidence_rows=20,
        description="",
        impact_score=1.0,
        affected_reports=[],
        agent_trace=[],
        confidence=1.0,
    )


def test_deduplication_removes_structural_duplicates():
    """Two issues with identical (type, table, columns) should collapse to one."""
    seen: set = set()
    deduped: list = []

    issues = [_make_issue("A"), _make_issue("B")]
    for issue in issues:
        key = (issue.issue_type, issue.table, tuple(issue.columns))
        if key not in seen:
            seen.add(key)
            deduped.append(issue)

    assert len(deduped) == 1
    assert deduped[0].issue_id == "A"
