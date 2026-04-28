"""LLM-powered executive report generation.

Sends a structured summary of ranked issues to the OpenAI API and returns a
JSON object containing an executive summary, top risks, recommended actions,
and an overall data health rating. Falls back to a deterministic report when
the LLM is unavailable or the response fails validation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from .fallback import fallback_report
from .key_manager import APIKeyManager, AllKeysExhausted
from .retry_manager import call_with_retry
from .validation import build_allowed_columns, validate_response
from .explainer import _build_schema, _summarise_issues

logger = logging.getLogger(__name__)

_CACHE_TTL = int(os.getenv("AGENT_CACHE_TTL", "3600"))
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_cache: dict[str, tuple[dict, float]] = {}

_SYSTEM_PROMPT = """You are an executive data quality advisor writing for a CEO or General Manager.

STRICT RULES:
1. No technical jargon. Write for a senior business audience.
2. Only reference tables and columns from the provided schema.
3. Focus on business risk and recommended actions.
4. If data is insufficient, set executive_summary to: "Insufficient data to generate executive summary."

OUTPUT FORMAT: Return a JSON object with exactly these fields:
  executive_summary  (2-3 plain sentences)
  top_risks          (array of up to 3 plain-language risk strings)
  recommended_actions (array of up to 5 action strings)
  overall_data_health (one of: poor / fair / good / excellent)

Return only the JSON object — no preamble, no markdown fences."""


def _cache_key(issues: list[dict], run_id: str) -> str:
    ids = tuple(sorted(i.get("issue_id", str(i)) for i in issues))
    return f"report:{ids}:{run_id}"


def _cache_get(key: str) -> dict | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0]
    _cache.pop(key, None)
    return None


def _cache_set(key: str, value: dict) -> None:
    _cache[key] = (value, time.time())


def _call_openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=700,
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content.strip()


def generate_report(
    issues: list[dict],
    run_id: str = "",
    table_sizes: dict | None = None,
    key_manager: Optional[APIKeyManager] = None,
) -> dict:
    """Generate an executive business report for the supplied ranked issues.

    Args:
        issues:      Ranked issue dicts from the triage pipeline.
        run_id:      Optional run identifier for traceability.
        table_sizes: Optional dict of {table: row_count} for richer context.
        key_manager: Optional pre-built key manager (built from env if None).

    Returns:
        Dict with: run_id, executive_summary, top_risks, recommended_actions,
                   overall_data_health, total_issues, critical_issues, source
    """
    if key_manager is None:
        key_manager = APIKeyManager.from_env()

    if not issues:
        return fallback_report([], run_id)

    ck = _cache_key(issues, run_id)
    cached = _cache_get(ck)
    if cached is not None:
        logger.debug("generate_report: cache hit")
        return cached

    schema = _build_schema(issues, table_sizes)
    allowed = build_allowed_columns(schema)
    summarised = _summarise_issues(issues)

    high_count = sum(1 for i in issues if i.get("severity") == "high")
    payload = json.dumps({
        "schema": schema,
        "summary_statistics": {
            "total_issues": len(issues),
            "critical_issues": high_count,
            "run_id": run_id,
        },
        "top_issues": summarised[:5],
        "instruction": "Write an executive data quality report based on the issues above.",
    }, indent=2)

    try:
        raw = call_with_retry(_call_openai, key_manager, payload)
    except AllKeysExhausted as exc:
        logger.warning("generate_report: all keys exhausted (%s) — using fallback.", exc)
        return fallback_report(issues, run_id)

    score = validate_response(raw, issues, allowed)
    if not score.passed:
        logger.warning("generate_report: response failed validation (%s) — using fallback.", score.failure_reason)
        return fallback_report(issues, run_id)

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("generate_report: JSON parse failed — using fallback.")
        return fallback_report(issues, run_id)

    required = {"executive_summary", "top_risks", "recommended_actions", "overall_data_health"}
    if not required.issubset(parsed.keys()):
        logger.warning("generate_report: missing required fields — using fallback.")
        return fallback_report(issues, run_id)

    result = {
        "run_id": run_id,
        "executive_summary": parsed["executive_summary"],
        "top_risks": parsed["top_risks"][:3],
        "recommended_actions": parsed["recommended_actions"][:5],
        "overall_data_health": parsed["overall_data_health"],
        "total_issues": len(issues),
        "critical_issues": high_count,
        "validation_score": score.overall_score,
        "source": "llm",
    }
    _cache_set(ck, result)
    return result
