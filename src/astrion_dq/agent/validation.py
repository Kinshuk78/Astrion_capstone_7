"""Input validation and LLM response scoring for the agent layer.

Two responsibilities:
    1. validate_input()  — checks issues and schema are well-formed BEFORE the
       LLM call so we never send malformed payloads.
    2. validate_response() — scores the LLM response on four dimensions AFTER
       the call so hallucinated or low-quality outputs are rejected.

Scoring dimensions
------------------
factual_consistency_score
    Fraction of issue_type strings mentioned in the response that actually
    exist in the input issue list. Detects invented issue categories.

schema_compliance_score
    Fraction of quoted identifiers (column / table names) in the response
    that exist in the allowed columns set built from the input schema.
    A response that mentions no specific names scores 1.0 by convention.

completeness_score
    Fraction of the top-5 input issues whose table or column is mentioned
    anywhere in the response. Measures whether the LLM ignored key issues.

confidence_score
    Average confidence value of the input issues. Reflects data quality
    of the detection step, not the LLM output.

overall_score
    Weighted average: 0.30 * factual + 0.40 * schema + 0.20 * complete + 0.10 * confidence

Rejection
---------
Responses with overall_score below AGENT_RESPONSE_MIN_SCORE (default 0.70)
are rejected and the caller falls back to deterministic output.

Environment variables:
    AGENT_RESPONSE_MIN_SCORE    Minimum overall score to accept (default 0.70)
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MIN_SCORE = float(os.getenv("AGENT_RESPONSE_MIN_SCORE", "0.70"))

_KNOWN_ISSUE_TYPES = frozenset({
    "missing_values",
    "duplicate_rows",
    "numeric_outliers",
    "invalid_future_dates",
    "referential_integrity_break",
    "empty_table",
    "statistical_drift",
})

# Regex to find quoted identifiers that are NOT JSON object keys.
# Negative lookahead (?!\s*:) excludes patterns like "column_name": which
# are JSON structural keys defined by our prompt contract, not schema references.
_QUOTED_IDENT_RE = re.compile(r'[`"\']([a-zA-Z_][a-zA-Z0-9_]*)[`"\'](?!\s*:)')


@dataclass
class ValidationResult:
    factual_consistency_score: float
    schema_compliance_score: float
    completeness_score: float
    confidence_score: float
    overall_score: float
    passed: bool
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Pre-call: input validation
# ---------------------------------------------------------------------------

def validate_input(issues: list[dict], schema: list[dict]) -> None:
    """Raise ValueError if the input is invalid before sending to the LLM.

    Checks:
        - issues is non-empty
        - schema is non-empty
        - every issue has issue_type in _KNOWN_ISSUE_TYPES
    """
    if not issues:
        raise ValueError("issues list is empty — nothing to send to the LLM.")
    if not schema:
        raise ValueError("schema list is empty — cannot validate LLM output without schema.")
    for issue in issues:
        if "issue_type" not in issue:
            raise ValueError(f"Issue missing 'issue_type' field: {issue}")
        if issue["issue_type"] not in _KNOWN_ISSUE_TYPES:
            raise ValueError(
                f"Unknown issue_type '{issue['issue_type']}'. "
                f"Allowed: {sorted(_KNOWN_ISSUE_TYPES)}"
            )


def build_allowed_columns(schema: list[dict]) -> set[str]:
    """Return a lowercase set of all valid table and column names from schema."""
    allowed: set[str] = set()
    for entry in schema:
        tbl = entry.get("table", "")
        if tbl:
            allowed.add(tbl.lower())
        for col in entry.get("columns", []):
            if col:
                allowed.add(col.lower())
    return allowed


# ---------------------------------------------------------------------------
# Post-call: response scoring
# ---------------------------------------------------------------------------

def validate_response(
    response: str,
    issues: list[dict],
    allowed_columns: set[str],
    top_k: int = 5,
) -> ValidationResult:
    """Score an LLM response and return a ValidationResult.

    A response is accepted (passed=True) when overall_score >= _MIN_SCORE.
    """
    if not response or not response.strip():
        return ValidationResult(
            factual_consistency_score=0.0,
            schema_compliance_score=0.0,
            completeness_score=0.0,
            confidence_score=0.0,
            overall_score=0.0,
            passed=False,
            failure_reason="LLM returned an empty response.",
        )

    resp_lower = response.lower()

    # 1. Factual consistency — check issue_types
    mentioned_types = {t for t in _KNOWN_ISSUE_TYPES if t in resp_lower}
    input_types = {i.get("issue_type", "") for i in issues}
    if mentioned_types:
        valid_types = mentioned_types & input_types
        factual = len(valid_types) / len(mentioned_types)
    else:
        factual = 1.0  # no type claims made

    # 2. Schema compliance — check quoted identifiers
    mentioned_idents = {m.lower() for m in _QUOTED_IDENT_RE.findall(response)}
    if mentioned_idents and allowed_columns:
        valid_idents = mentioned_idents & allowed_columns
        schema_c = len(valid_idents) / len(mentioned_idents)
    else:
        schema_c = 1.0  # no identifiers mentioned, or no schema to check against

    # 3. Completeness — top-k issues addressed
    top_issues = issues[:top_k]
    addressed = 0
    for issue in top_issues:
        tbl = issue.get("table", "").lower()
        cols = [c.lower() for c in issue.get("columns", [])]
        if tbl and tbl in resp_lower:
            addressed += 1
            continue
        if any(col in resp_lower for col in cols):
            addressed += 1
    completeness = addressed / len(top_issues) if top_issues else 1.0

    # 4. Confidence — average of input issue confidence values
    conf_vals = [float(i.get("confidence", 1.0)) for i in issues]
    confidence = sum(conf_vals) / len(conf_vals) if conf_vals else 1.0

    overall = (
        factual * 0.30
        + schema_c * 0.40
        + completeness * 0.20
        + confidence * 0.10
    )
    overall = round(min(overall, 1.0), 4)

    passed = overall >= _MIN_SCORE
    reason = "" if passed else (
        f"Overall score {overall:.2f} is below threshold {_MIN_SCORE:.2f}. "
        f"[factual={factual:.2f} schema={schema_c:.2f} complete={completeness:.2f} conf={confidence:.2f}]"
    )

    return ValidationResult(
        factual_consistency_score=round(factual, 4),
        schema_compliance_score=round(schema_c, 4),
        completeness_score=round(completeness, 4),
        confidence_score=round(confidence, 4),
        overall_score=overall,
        passed=passed,
        failure_reason=reason,
    )
