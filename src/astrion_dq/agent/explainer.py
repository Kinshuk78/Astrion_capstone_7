"""LLM-powered business explanation of data quality issues.

Sends a structured JSON payload (NEVER raw data) to the OpenAI API and asks
for plain-English business explanations. Falls back to deterministic output
when the LLM is unavailable or when the response fails validation.

Cost controls:
    - Only top AGENT_MAX_ISSUES issues sent (default 10)
    - Descriptions truncated to AGENT_MAX_DESC_CHARS (default 200)
    - Responses cached in memory for AGENT_CACHE_TTL seconds (default 3600)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from .fallback import fallback_explain
from .key_manager import APIKeyManager, AllKeysExhausted
from .retry_manager import call_with_retry
from .validation import build_allowed_columns, validate_input, validate_response

logger = logging.getLogger(__name__)

_MAX_ISSUES = int(os.getenv("AGENT_MAX_ISSUES", "10"))
_MAX_DESC_CHARS = int(os.getenv("AGENT_MAX_DESC_CHARS", "200"))
_CACHE_TTL = int(os.getenv("AGENT_CACHE_TTL", "3600"))
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_cache: dict[str, tuple[list, float]] = {}

_SYSTEM_PROMPT = """You are a senior data quality analyst explaining technical issues to business stakeholders who are not technical.

STRICT RULES:
1. Only reference tables and columns listed in the provided schema. Never invent names.
2. Do not use SQL terms, database jargon, or programming concepts.
3. Write in plain English. Be direct and concise.
4. If you are uncertain, write: "Based on available data, ..."
5. If insufficient data exists, set business_explanation to: "Insufficient data to determine business impact."

OUTPUT FORMAT: Return a JSON array. Each object must have exactly these fields:
  issue_id, business_explanation, risk_summary, recommended_action

Return only the JSON array — no preamble, no markdown fences."""


def _cache_key(issues: list[dict]) -> str:
    ids = tuple(sorted(i.get("issue_id", str(idx)) for idx, i in enumerate(issues)))
    return f"explain:{ids}"


def _cache_get(key: str) -> list | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0]
    _cache.pop(key, None)
    return None


def _cache_set(key: str, value: list) -> None:
    _cache[key] = (value, time.time())


def _summarise_issues(issues: list[dict]) -> list[dict]:
    return [
        {
            "issue_id": i.get("issue_id", ""),
            "issue_type": i.get("issue_type", ""),
            "table": i.get("table", ""),
            "columns": i.get("columns", []),
            "severity": i.get("severity", ""),
            "metric": round(float(i.get("metric", 0)), 4),
            "evidence_rows": int(i.get("evidence_rows", 0)),
            "impact_score": round(float(i.get("impact_score", 0)), 4),
            "confidence": round(float(i.get("confidence", 1.0)), 4),
            "description": str(i.get("description", ""))[:_MAX_DESC_CHARS],
        }
        for i in issues[:_MAX_ISSUES]
    ]


def _build_schema(issues: list[dict], table_sizes: dict | None = None) -> list[dict]:
    table_cols: dict[str, set[str]] = {}
    for issue in issues:
        tbl = issue.get("table", "")
        if tbl:
            table_cols.setdefault(tbl, set()).update(issue.get("columns", []))
    return [
        {
            "table": tbl,
            "columns": sorted(cols),
            "row_count": (table_sizes or {}).get(tbl, "unknown"),
        }
        for tbl, cols in table_cols.items()
    ]


def _call_openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1800,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content.strip()


def _parse_response(raw: str, issues: list[dict], score) -> list[dict]:
    """Try to extract a list from the JSON response; attach metadata."""
    try:
        parsed = json.loads(raw)
        items = parsed if isinstance(parsed, list) else parsed.get("issues") or parsed.get("explanations") or []
        if isinstance(items, list) and items:
            for item, issue in zip(items, issues):
                item.setdefault("issue_id", issue.get("issue_id", ""))
                item["validation_score"] = score.overall_score
                item["source"] = "llm"
            return items
    except (json.JSONDecodeError, TypeError):
        pass

    logger.warning("Could not parse LLM response as JSON list; using fallback.")
    return []


def explain_issues(
    issues: list[dict],
    table_sizes: dict | None = None,
    key_manager: Optional[APIKeyManager] = None,
) -> list[dict]:
    """Return business-language explanations for each issue.

    Tries the LLM first; falls back to deterministic text on any failure.
    Results are cached for AGENT_CACHE_TTL seconds.
    """
    if key_manager is None:
        key_manager = APIKeyManager.from_env()

    ck = _cache_key(issues)
    cached = _cache_get(ck)
    if cached is not None:
        logger.debug("explain_issues: cache hit for %d issue(s)", len(issues))
        return cached

    schema = _build_schema(issues, table_sizes)

    try:
        validate_input(issues, schema)
    except ValueError as exc:
        logger.warning("explain_issues: input validation failed (%s) — using fallback.", exc)
        return fallback_explain(issues)

    allowed = build_allowed_columns(schema)
    summarised = _summarise_issues(issues)

    payload = json.dumps({
        "schema": schema,
        "issues": summarised,
        "instruction": (
            "Explain each issue in plain business English. "
            "Return a JSON array with one object per issue."
        ),
    }, indent=2)

    try:
        raw = call_with_retry(_call_openai, key_manager, payload)
    except AllKeysExhausted as exc:
        logger.warning("explain_issues: all API keys exhausted (%s) — using fallback.", exc)
        return fallback_explain(issues)

    score = validate_response(raw, issues, allowed)
    if not score.passed:
        logger.warning("explain_issues: response failed validation (%s) — using fallback.", score.failure_reason)
        return fallback_explain(issues)

    result = _parse_response(raw, issues, score)
    if not result:
        return fallback_explain(issues)

    _cache_set(ck, result)
    return result
