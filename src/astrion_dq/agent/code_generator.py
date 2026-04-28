"""LLM-powered fix code generation (SQL + Python pandas).

For each data quality issue, generates syntactically valid SQL and pandas
code that addresses the problem using ONLY the columns and tables in the
provided schema. Column references in the generated code are validated
post-generation to catch hallucinated names.

Falls back to deterministic templates when:
    - LLM is unavailable
    - Response fails schema compliance validation
    - Generated code references columns not in the schema
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

from .fallback import fallback_generate_fix
from .key_manager import APIKeyManager, AllKeysExhausted
from .retry_manager import call_with_retry
from .validation import build_allowed_columns

logger = logging.getLogger(__name__)

_CACHE_TTL = int(os.getenv("AGENT_CACHE_TTL", "3600"))
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_cache: dict[str, tuple[dict, float]] = {}

# Patterns to extract column/table identifiers from generated code
_QUOTED_SQL_IDENT = re.compile(r'["`]([a-zA-Z_][a-zA-Z0-9_]*)["`]')
_PANDAS_COL = re.compile(r'df\[[\'"]([a-zA-Z_][a-zA-Z0-9_]*)[\'"]\]')
_SQL_AFTER_KEYWORD = re.compile(
    r'\b(?:FROM|JOIN|UPDATE|TABLE|INTO)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    re.IGNORECASE,
)

_SYSTEM_PROMPT = """You are a senior data engineer generating fix code for data quality issues.

STRICT RULES:
1. ONLY use column names and table names from the provided schema. Do not invent any names.
2. SQL must be compatible with DuckDB / PostgreSQL syntax.
3. Python must use pandas and handle null values explicitly.
4. If you cannot safely generate code due to insufficient schema data, use the fallback strings below.
5. Do not add explanatory prose — return only the JSON object.

FALLBACK STRINGS (use these exactly when unsure):
  sql_fix:    "-- Insufficient schema data to generate SQL fix"
  python_fix: "# Insufficient schema data to generate Python fix"

OUTPUT FORMAT: Return a JSON object with exactly two string fields: sql_fix, python_fix"""


def _cache_key(issue: dict) -> str:
    return f"fix:{issue.get('issue_id', '')}:{issue.get('issue_type', '')}"


def _cache_get(key: str) -> dict | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0]
    _cache.pop(key, None)
    return None


def _cache_set(key: str, value: dict) -> None:
    _cache[key] = (value, time.time())


def _build_issue_schema(issue: dict) -> list[dict]:
    tbl = issue.get("table", "")
    cols = issue.get("columns", [])
    schema = []
    if tbl:
        schema.append({"table": tbl, "columns": cols})
    dim_table = issue.get("dim_table", "")
    dim_pk = issue.get("dim_pk", "")
    if dim_table and dim_table not in (tbl, ""):
        schema.append({"table": dim_table, "columns": [dim_pk] if dim_pk else []})
    return schema


def _call_openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content.strip()


def _extract_code_identifiers(sql: str, python: str) -> set[str]:
    """Extract all column/table identifiers referenced in generated code."""
    idents: set[str] = set()
    idents.update(m.lower() for m in _QUOTED_SQL_IDENT.findall(sql))
    idents.update(m.lower() for m in _SQL_AFTER_KEYWORD.findall(sql))
    idents.update(m.lower() for m in _PANDAS_COL.findall(python))
    return idents


def _validate_code(sql: str, python: str, allowed: set[str]) -> bool:
    """Return True if the generated code references only known identifiers.

    We only fail validation if there are references we can definitely identify
    as not in the schema. Unquoted SQL keywords and identifiers we cannot
    extract reliably are passed through to avoid over-rejection.
    """
    if not allowed:
        return True  # no schema basis to reject on

    extracted = _extract_code_identifiers(sql, python)
    if not extracted:
        return True  # no identifiers we could parse — pass through

    unknown = extracted - allowed
    # Allow common SQL/DuckDB built-ins that may appear in generated code
    sql_builtins = {
        "rowid", "null", "true", "false", "current_date", "current_timestamp",
        "avg", "sum", "count", "min", "max", "stddev_pop", "stddev",
        "information_schema", "tables",
    }
    unknown -= sql_builtins

    if unknown:
        logger.warning(
            "Code generator produced unknown identifiers: %s (allowed: %s)",
            sorted(unknown),
            sorted(allowed),
        )
        return False
    return True


def generate_fix_code(
    issue: dict,
    extra_schema: list[dict] | None = None,
    key_manager: Optional[APIKeyManager] = None,
) -> dict:
    """Generate SQL and Python fix code for a single issue.

    Args:
        issue:        A ranked issue dict (must contain issue_type, table, columns).
        extra_schema: Additional schema entries beyond what is inferred from the issue.
        key_manager:  Optional pre-built key manager (built from env if None).

    Returns:
        Dict with: issue_id, issue_type, sql_fix, python_fix, source, validation_score
    """
    if key_manager is None:
        key_manager = APIKeyManager.from_env()

    ck = _cache_key(issue)
    cached = _cache_get(ck)
    if cached is not None:
        logger.debug("generate_fix_code: cache hit for %s", issue.get("issue_id"))
        return cached

    schema = _build_issue_schema(issue)
    if extra_schema:
        schema = schema + extra_schema

    allowed = build_allowed_columns(schema)

    payload = json.dumps({
        "schema": schema,
        "issue": {
            "issue_id": issue.get("issue_id", ""),
            "issue_type": issue.get("issue_type", ""),
            "table": issue.get("table", ""),
            "columns": issue.get("columns", []),
            "severity": issue.get("severity", ""),
            "description": str(issue.get("description", ""))[:200],
            "dim_table": issue.get("dim_table", ""),
            "dim_pk": issue.get("dim_pk", ""),
        },
        "instruction": "Generate SQL and Python fix code using ONLY the schema provided.",
    }, indent=2)

    try:
        raw = call_with_retry(_call_openai, key_manager, payload)
    except AllKeysExhausted as exc:
        logger.warning("generate_fix_code: all keys exhausted (%s) — using fallback.", exc)
        return fallback_generate_fix(issue)

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("generate_fix_code: JSON parse failed — using fallback.")
        return fallback_generate_fix(issue)

    sql_fix = parsed.get("sql_fix", "")
    python_fix = parsed.get("python_fix", "")

    if not sql_fix or not python_fix:
        logger.warning("generate_fix_code: incomplete response — using fallback.")
        return fallback_generate_fix(issue)

    if not _validate_code(sql_fix, python_fix, allowed):
        logger.warning("generate_fix_code: schema compliance failed — using fallback.")
        return fallback_generate_fix(issue)

    result = {
        "issue_id": issue.get("issue_id", ""),
        "issue_type": issue.get("issue_type", ""),
        "sql_fix": sql_fix,
        "python_fix": python_fix,
        "source": "llm",
        "validation_passed": True,
    }
    _cache_set(ck, result)
    return result
