from astrion_dq.workflows.mvp import RankedIssue, _dedupe_ranked


def test_supervisor_deduplicates_structural_duplicates():
    ranked = [
        RankedIssue(
            issue_id="A",
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
        ),
        RankedIssue(
            issue_id="B",
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
        ),
    ]
    deduped, removed = _dedupe_ranked(ranked)
    assert len(deduped) == 1
    assert removed == 1
