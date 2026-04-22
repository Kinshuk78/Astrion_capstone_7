"""V2 Business Impact Scoring for ranked data quality issues.

Formula: BIS = base_weight × severity_weight × evidence_density × report_criticality

Where:
  base_weight       = issue type priority from config.ISSUE_TYPE_BASE_WEIGHTS
  severity_weight   = {high: 3.0, medium: 2.0, low: 1.0}
  evidence_density  = log(1 + evidence_rows) / log(1 + table_total_rows)
                      Log-normalisation (borrowed from TF-IDF) captures diminishing
                      marginal impact: 0→1,000 bad rows is more impactful than
                      50,000→51,000 bad rows, even at the same row fraction.
  report_criticality = sum(affected_report_scores) / max_possible_score
                      Reflects that a daily CEO dashboard has higher business impact
                      than a weekly marketing report.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from astrion_dq.config import (
    BIS_NOISE_THRESHOLD,
    ISSUE_TYPE_BASE_WEIGHTS,
    REPORT_CRITICALITY_SCORES,
    SEVERITY_WEIGHTS,
)

logger = logging.getLogger(__name__)

_MAX_REPORT_SCORE = sum(REPORT_CRITICALITY_SCORES.values())


@dataclass
class ImpactScoreV2:
    """Per-component breakdown of a V2 Business Impact Score.

    All four components are exposed so engineers can audit why one issue ranked
    higher than another — important for trust in automated triage systems.
    """
    issue_id: str
    base_weight: float
    severity_weight: float
    evidence_density: float
    report_criticality: float
    final_score: float
    score_breakdown: str


def compute_evidence_density(evidence_rows: int, table_rows: int) -> float:
    """Log-normalised evidence density: log(1 + e) / log(1 + N).

    Returns a value in (0, 1]. Returns 0.0 for empty or zero-evidence inputs.
    """
    if table_rows <= 0 or evidence_rows <= 0:
        return 0.0
    return math.log1p(evidence_rows) / math.log1p(max(table_rows, evidence_rows))


def compute_report_criticality(affected_reports: List[str]) -> float:
    """Weighted fraction of maximum possible downstream report impact.

    Returns 0.5 when no report mapping is available (neutral default).
    Unknown report names receive a score of 0.3.
    """
    if not affected_reports or _MAX_REPORT_SCORE <= 0:
        return 0.5
    score = sum(REPORT_CRITICALITY_SCORES.get(r, 0.3) for r in affected_reports)
    return min(score / _MAX_REPORT_SCORE, 1.0)


def score_issue_v2(
    issue,
    table_sizes: Optional[Dict[str, int]] = None,
) -> ImpactScoreV2:
    """Compute the V2 Business Impact Score for a single issue.

    Args:
        issue: Any dataclass with issue_type, severity, evidence_rows, metric,
               table, and (optionally) affected_reports attributes.
        table_sizes: Dict mapping table name → row count. Used for log-normalisation.
    """
    base_weight = ISSUE_TYPE_BASE_WEIGHTS.get(issue.issue_type, 1.5)
    severity_weight = SEVERITY_WEIGHTS.get(issue.severity, 1.0)

    evidence_rows = getattr(issue, "evidence_rows", 0)
    table_rows = (table_sizes or {}).get(issue.table, max(evidence_rows * 10, 1))
    evidence_density = compute_evidence_density(evidence_rows, table_rows)

    # Drift and schema issues have evidence_rows=0; fall back to the metric value.
    if evidence_rows == 0:
        evidence_density = min(float(getattr(issue, "metric", 0.0)), 1.0)

    affected_reports = getattr(issue, "affected_reports", [])
    report_crit = compute_report_criticality(affected_reports)

    final_score = base_weight * severity_weight * evidence_density * report_crit

    breakdown = (
        f"BIS = {base_weight:.2f}(base) × {severity_weight:.2f}(sev) × "
        f"{evidence_density:.4f}(density) × {report_crit:.4f}(criticality) "
        f"= {final_score:.4f}"
    )

    return ImpactScoreV2(
        issue_id=issue.issue_id,
        base_weight=base_weight,
        severity_weight=severity_weight,
        evidence_density=evidence_density,
        report_criticality=report_crit,
        final_score=round(final_score, 6),
        score_breakdown=breakdown,
    )


def ranking_agent_v2(
    issues: List,
    table_sizes: Optional[Dict[str, int]] = None,
    suppress_noise: bool = True,
) -> Tuple[List, List[ImpactScoreV2]]:
    """Rank issues by V2 Business Impact Score, descending.

    Args:
        issues: List of issue objects (RankedIssue, QualityIssue, or VerifiedIssue).
        table_sizes: Table row counts for log-normalised evidence density.
        suppress_noise: If True, drop issues with BIS < BIS_NOISE_THRESHOLD.

    Returns:
        Tuple of (ranked_issues, score_details). The ranked_issues list is sorted
        descending by BIS; score_details provides the per-component breakdown for
        reporting. The impact_score attribute of each issue is updated in place.
    """
    scored: List[Tuple[float, object, ImpactScoreV2]] = []

    for issue in issues:
        s = score_issue_v2(issue, table_sizes)
        try:
            issue.impact_score = s.final_score
        except AttributeError:
            pass
        scored.append((s.final_score, issue, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [p[1] for p in scored]
    details = [p[2] for p in scored]

    if suppress_noise:
        before = len(ranked)
        pairs = [(r, d) for r, d in zip(ranked, details) if d.final_score >= BIS_NOISE_THRESHOLD]
        ranked = [p[0] for p in pairs]
        details = [p[1] for p in pairs]
        if before - len(ranked):
            logger.info(
                "Noise suppression (BIS < %.3f): removed %d issues.",
                BIS_NOISE_THRESHOLD, before - len(ranked),
            )

    logger.info("V2 ranking complete: %d issues.", len(ranked))
    return ranked, details


def compare_v1_v2_scores(issues: List, table_sizes: Optional[Dict[str, int]] = None) -> str:
    """Generate a markdown table comparing V1 (linear) and V2 (log-normalised) scores.

    V1 formula: severity_weight × issue_type_weight × metric (linear metric fraction).
    V2 formula: base × severity × log-density × report_criticality.

    The comparison demonstrates that V2 changes issue ordering: referential integrity
    breaks rank higher relative to null issues because log-normalised density plus
    report criticality better captures the impact of join failures on downstream reports.
    """
    v1_sev = {"high": 3.0, "medium": 2.0, "low": 1.0}
    v1_type = {
        "referential_integrity_break": 3.0,
        "duplicate_rows": 2.8,
        "numeric_outliers": 2.6,
        "missing_values": 2.3,
        "invalid_future_dates": 2.1,
    }

    v1_pairs = [
        (issue, v1_sev.get(issue.severity, 1.0)
                * v1_type.get(issue.issue_type, 1.0)
                * max(getattr(issue, "metric", 0.001), 0.001))
        for issue in issues
    ]

    v1_rank = {
        issue.issue_id: rank
        for rank, (issue, _) in enumerate(
            sorted(v1_pairs, key=lambda x: x[1], reverse=True), start=1
        )
    }

    _, v2_details = ranking_agent_v2(list(issues), table_sizes, suppress_noise=False)
    v2_rank = {s.issue_id: rank for rank, s in enumerate(v2_details, start=1)}

    lines = [
        "| Issue ID | Type | V1 Score | V2 Score | Rank Change |",
        "|----------|------|----------|----------|-------------|",
    ]
    for issue, v1_score in v1_pairs:
        v2_score = next((s.final_score for s in v2_details if s.issue_id == issue.issue_id), 0.0)
        delta = v1_rank.get(issue.issue_id, 0) - v2_rank.get(issue.issue_id, 0)
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        lines.append(
            f"| {issue.issue_id} | {issue.issue_type[:20]} | "
            f"{v1_score:.4f} | {v2_score:.4f} | {arrow}{abs(delta)} |"
        )

    return "\n".join(lines)
