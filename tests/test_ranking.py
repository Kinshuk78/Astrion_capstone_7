from astrion_dq.models import RankedIssue
from astrion_dq.ranking.impact import score_issue_v2
from astrion_dq.config import REPORT_MAPPING


def test_bis_increases_with_evidence_rows():
    """Higher evidence row count should yield a higher V2 BIS score."""
    def _make(evidence_rows: int) -> RankedIssue:
        return RankedIssue(
            issue_id=f"Q{evidence_rows}",
            issue_type="duplicate_rows",
            table="fact_sales",
            columns=["transaction_id"],
            severity="medium",
            metric=evidence_rows / 1000,
            evidence_rows=evidence_rows,
            description="",
            impact_score=0.0,
            affected_reports=REPORT_MAPPING.get("duplicate_rows", []),
            agent_trace=[],
            confidence=1.0,
        )

    low = _make(10)
    high = _make(100)

    score_low = score_issue_v2(low, table_sizes={"fact_sales": 1000}).final_score
    score_high = score_issue_v2(high, table_sizes={"fact_sales": 1000}).final_score
    assert score_high > score_low
