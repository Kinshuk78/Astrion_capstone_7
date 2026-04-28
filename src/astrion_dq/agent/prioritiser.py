"""LLM-powered issue prioritisation with business justification.

Asks the LLM to rank issues by business impact and provide a human-readable
justification for each rank. Falls back to BIS-sorted deterministic output
when the LLM is unavailable or the response fails validation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from .fallback import fallback_prioritise
from .key_manager import APIKeyManager, AllKeysExhausted
from .retry_manager import call_with_retry
from .validation import build_allowed_columns, validate_input, validate_response
from .explainer import _build_schema, _summarise_issues

logger = logging.getLogger(__name__)

_MAX_ISSUES = int(os.getenv("AGENT_MAX_ISSUES", "10"))
_CACHE_TTL = int(os.getenv("AGENT_CACHE_TTL", "3600"))
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

_cache: dict[str, tuple[list, float]] = {}

_SYSTEM_PROMPT = """You are a data operations manager deciding which data quality issues to fix first.

STRICT RULES:
1. Only reference tables and columns from the provided schema.
2. Prioritise based on severity, number of affected records, and downstream report impact.
3. Justify each rank in 1-2 plain sentences. No jargon.
4. If you cannot determine priority, order by impact_score descending.

OUTPUT FORMAT: Return a JSON array. Each object must have:
  issue_id, priority_rank (integer starting at 1), urgency (critical/high/medium/low), priority_justification

Return only the JSON array — no preamble, no markdown fences."""


def _cache_key(issues: list[dict]) -> str:
    ids = tuple(sorted(i.get("issue_id", str(i)) for i in issues))
    return f"prioritise:{ids}"


def _cache_get(key: str) -> list | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0]
    _cache.pop(key, None)
    return None


def _cache_set(key: str, value: list) -> None:
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
        max_tokens=1200,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content.strip()


def _merge_priority_into_issues(
    issues: list[dict],
    priority_items: list[dict],
) -> list[dict]:
    """Merge priority data from the LLM back into the original issue dicts."""
    by_id = {p.get("issue_id", ""): p for p in priority_items}
    merged = []
    for issue in issues:
        pid = issue.get("issue_id", "")
        priority = by_id.get(pid, {})
        merged.append({
            **issue,
            "priority_rank": priority.get("priority_rank", 999),
            "urgency": priority.get("urgency", "medium"),
            "priority_justification": priority.get("priority_justification", ""),
            "source": "llm",
        })
    return sorted(merged, key=lambda x: x.get("priority_rank", 999))


def prioritise_issues(
    issues: list[dict],
    table_sizes: dict | None = None,
    key_manager: Optional[APIKeyManager] = None,
) -> list[dict]:
    """Return issues enriched with AI-generated priority ranks and justifications."""
    if key_manager is None:
        key_manager = APIKeyManager.from_env()

    ck = _cache_key(issues)
    cached = _cache_get(ck)
    if cached is not None:
        logger.debug("prioritise_issues: cache hit")
        return cached

    schema = _build_schema(issues, table_sizes)

    try:
        validate_input(issues, schema)
    except ValueError as exc:
        logger.warning("prioritise_issues: input validation failed (%s) — using fallback.", exc)
        return fallback_prioritise(issues)

    allowed = build_allowed_columns(schema)
    summarised = _summarise_issues(issues)

    payload = json.dumps({
        "schema": schema,
        "issues": summarised,
        "instruction": (
            "Rank these issues by the urgency of fixing them. "
            "Return a JSON array with one object per issue."
        ),
    }, indent=2)

    try:
        raw = call_with_retry(_call_openai, key_manager, payload)
    except AllKeysExhausted as exc:
        logger.warning("prioritise_issues: all keys exhausted (%s) — using fallback.", exc)
        return fallback_prioritise(issues)

    score = validate_response(raw, issues, allowed)
    if not score.passed:
        logger.warning("prioritise_issues: response failed validation (%s) — using fallback.", score.failure_reason)
        return fallback_prioritise(issues)

    try:
        parsed = json.loads(raw)
        items = parsed if isinstance(parsed, list) else (
            parsed.get("issues") or parsed.get("priorities") or []
        )
        if not items:
            raise ValueError("Empty items list")
        result = _merge_priority_into_issues(issues, items)
        _cache_set(ck, result)
        return result
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("prioritise_issues: JSON parse error (%s) — using fallback.", exc)
        return fallback_prioritise(issues)
