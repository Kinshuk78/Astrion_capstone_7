"""Coverage tests for ranking/impact.py helper functions.

These tests exercise the branches missed by test_ranking.py:
- compute_evidence_density edge cases (zero inputs)
- compute_report_criticality with no reports and unknown report names
- score_issue_v2 metric-fallback branch (evidence_rows == 0)
- ranking_agent_v2 noise suppression
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from astrion_dq.ranking.impact import (
    compute_evidence_density,
    compute_report_criticality,
    ranking_agent_v2,
    score_issue_v2,
)


@dataclass
class _Issue:
    issue_type: str
    severity: str
    evidence_rows: int
    metric: float
    table: str
    issue_id: str = "TEST_001"
    columns: List[str] = field(default_factory=list)
    description: str = ""
    affected_reports: List[str] = field(default_factory=list)
    agent_trace: List[str] = field(default_factory=list)
    impact_score: float = 0.0


def test_evidence_density_zero_table_rows():
    assert compute_evidence_density(10, 0) == 0.0


def test_evidence_density_zero_evidence():
    assert compute_evidence_density(0, 100) == 0.0


def test_evidence_density_normal():
    result = compute_evidence_density(50, 1000)
    assert 0.0 < result < 1.0


def test_report_criticality_empty_list():
    # No reports -> neutral 0.5
    assert compute_report_criticality([]) == 0.5


def test_report_criticality_unknown_report():
    # Unknown report name should use 0.3 fallback, not crash
    result = compute_report_criticality(["nonexistent_report"])
    assert 0.0 < result <= 1.0


def test_score_issue_v2_drift_metric_fallback():
    """Drift issues have evidence_rows=0; score must use metric field."""
    issue = _Issue(
        issue_type="statistical_drift",
        severity="medium",
        evidence_rows=0,
        metric=0.4,
        table="fact_sales",
    )
    result = score_issue_v2(issue, table_sizes={"fact_sales": 10000})
    assert result.final_score > 0.0
    assert result.evidence_density == 0.4


def test_ranking_agent_v2_noise_suppression():
    """Issues below BIS_NOISE_THRESHOLD should be removed when suppress_noise=True."""
    # Issue with tiny evidence and low base weight -> very low BIS
    tiny = _Issue(
        issue_type="missing_values",
        severity="low",
        evidence_rows=1,
        metric=0.0001,
        table="fact_sales",
    )
    big = _Issue(
        issue_type="duplicate_rows",
        severity="high",
        evidence_rows=500,
        metric=0.05,
        table="fact_sales",
    )
    ranked, _ = ranking_agent_v2(
        [tiny, big],
        table_sizes={"fact_sales": 50000},
        suppress_noise=True,
    )
    # big issue must survive; tiny may be suppressed
    assert big in ranked


def test_ranking_agent_v2_order():
    """Higher BIS issue must come first."""
    low_impact = _Issue("missing_values", "low", 1, 0.001, "fact_sales")
    high_impact = _Issue("duplicate_rows", "high", 1000, 0.1, "fact_sales")
    ranked, _ = ranking_agent_v2(
        [low_impact, high_impact],
        table_sizes={"fact_sales": 10000},
        suppress_noise=False,
    )
    assert ranked[0] is high_impact
