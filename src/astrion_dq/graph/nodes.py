from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import asdict
from typing import Dict, List

from astrion_dq.checks.detect import infer_metadata, run_all_checks_parallel
from astrion_dq.checks.drift import detect_drift
from astrion_dq.config import CONFIDENCE_THRESHOLD, DB_PATH, REPORT_MAPPING
from astrion_dq.models import QualityIssue, RankedIssue, TableMeta
from astrion_dq.ranking.impact import ranking_agent_v2
from astrion_dq.warehouse.loader import load_retail_tables, load_tables_to_duckdb
from .debugger import IssueVerifier
from .state import TriageState

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.perf_counter()


def _meta_from_state(state: TriageState) -> Dict[str, TableMeta]:
    """Reconstruct TableMeta objects from the state dict.

    Foreign key values are stored as JSON lists after serialisation; they must
    be restored to tuples to match the TableMeta field type.
    """
    return {
        name: TableMeta(
            role=m["role"],
            primary_keys=m["primary_keys"],
            foreign_keys={k: tuple(v) for k, v in m["foreign_keys"].items()},
            date_cols=m["date_cols"],
            numeric_cols=m["numeric_cols"],
            promo_cols=m["promo_cols"],
        )
        for name, m in state["metadata"].items()
    }


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def data_loader_node(state: TriageState) -> dict:
    """Load retail CSV tables and register them in DuckDB."""
    t0 = _now()
    try:
        tables = load_retail_tables(source=state["source"])
        load_tables_to_duckdb(tables)
        table_sizes = {name: len(df) for name, df in tables.items()}
    except Exception as exc:
        logger.exception("data_loader_node failed: %s", exc)
        return {
            "error": str(exc),
            "agent_trace": state["agent_trace"] + ["data_loader"],
        }
    return {
        "tables": tables,
        "table_sizes": table_sizes,
        "db_path": str(DB_PATH),
        "data_loaded": True,
        "agent_trace": state["agent_trace"] + ["data_loader"],
        "timing": {**state["timing"], "data_loader": round(_now() - t0, 3)},
    }


def profiler_node(state: TriageState) -> dict:
    """Infer table metadata: roles, primary keys, foreign keys, column types."""
    t0 = _now()
    meta = infer_metadata(state["tables"])
    return {
        "metadata": {name: asdict(m) for name, m in meta.items()},
        "metadata_ready": True,
        "agent_trace": state["agent_trace"] + ["profiler"],
        "timing": {**state["timing"], "profiler": round(_now() - t0, 3)},
    }


def detector_node(state: TriageState) -> dict:
    """Run all five quality checks in parallel (nulls, dups, outliers, dates, RI)."""
    t0 = _now()
    meta = _meta_from_state(state)
    issues = run_all_checks_parallel(state["tables"], meta, state["sensitivity"])
    return {
        "raw_issues": [asdict(i) for i in issues],
        "detection_done": True,
        "agent_trace": state["agent_trace"] + ["detector"],
        "timing": {**state["timing"], "detector": round(_now() - t0, 3)},
    }


def drift_detector_node(state: TriageState) -> dict:
    """Detect statistical drift via PSI and the KS test against a saved snapshot."""
    t0 = _now()
    drift = detect_drift(state["tables"], _meta_from_state(state))
    return {
        "drift_issues": [asdict(i) for i in drift],
        "all_issues": state["raw_issues"] + [asdict(i) for i in drift],
        "drift_done": True,
        "agent_trace": state["agent_trace"] + ["drift_detector"],
        "timing": {**state["timing"], "drift_detector": round(_now() - t0, 3)},
    }


def debugger_node(state: TriageState) -> dict:
    """SQL cross-validation: verify each detected issue with a DuckDB query."""
    t0 = _now()
    raw_dicts = state.get("all_issues") or state.get("raw_issues") or []
    issues = [QualityIssue(**d) for d in raw_dicts]
    verifier = IssueVerifier(sensitivity=state["sensitivity"])
    verified = verifier.verify_all(issues)
    needs_review = any(v.confidence < CONFIDENCE_THRESHOLD for v in verified)
    return {
        "verified_issues": [asdict(v) for v in verified],
        "needs_human_review": needs_review,
        "debug_done": True,
        "agent_trace": state["agent_trace"] + ["debugger"],
        "timing": {**state["timing"], "debugger": round(_now() - t0, 3)},
    }


def _apply_decision(verified: List[dict], decision: str) -> List[dict]:
    """Filter the verified issue list according to the analyst's decision string."""
    if not decision or decision.strip().lower() == "approve all":
        return verified
    reject_ids = {
        p.strip()
        for p in decision.lower().replace("reject", "").strip().split(",")
        if p.strip()
    }
    return [i for i in verified if i["issue_id"] not in reject_ids]


def human_review_node(state: TriageState) -> dict:
    """Pause the graph and prompt an analyst to review low-confidence issues.

    Uses langgraph.types.interrupt() to pause execution and surface the issue
    list to the calling application. The graph resumes when the application
    calls graph.invoke(Command(resume="..."), config=config).

    Set ASTRION_AUTO_APPROVE=1 to bypass the interrupt in automated runs
    (CI/CD, evaluation scripts, unit tests).
    """
    t0 = _now()
    if os.getenv("ASTRION_AUTO_APPROVE") == "1":
        return {
            "review_done": True,
            "human_decision": "approve all",
            "agent_trace": state["agent_trace"] + ["human_review(auto)"],
            "timing": {**state["timing"], "human_review": round(_now() - t0, 3)},
        }

    from langgraph.types import interrupt

    low_conf = [i for i in state["verified_issues"] if i["confidence"] < CONFIDENCE_THRESHOLD]
    lines = [
        f"{len(low_conf)} issue(s) have confidence < {CONFIDENCE_THRESHOLD} and require review:",
        "",
    ]
    for item in low_conf:
        lines.append(
            f"  [{item['issue_id']}] {item['issue_type']} in {item['table']}"
            f" - confidence={item['confidence']:.2f}, evidence_rows={item['evidence_rows']}"
        )
    lines += ["", "Reply: 'approve all'  or  'reject ID1,ID2,...'"]

    decision: str = interrupt("\n".join(lines))
    verified = _apply_decision(state["verified_issues"], decision)
    return {
        "verified_issues": verified,
        "review_done": True,
        "human_decision": decision,
        "agent_trace": state["agent_trace"] + ["human_review"],
        "timing": {**state["timing"], "human_review": round(_now() - t0, 3)},
    }


def ranker_node(state: TriageState) -> dict:
    """Rank issues by V2 Business Impact Score (BIS) in descending order."""
    t0 = _now()
    source_dicts = (
        state.get("verified_issues")
        or state.get("all_issues")
        or state.get("raw_issues")
        or []
    )

    ranked_input: List[RankedIssue] = []
    for d in source_dicts:
        ranked_input.append(RankedIssue(
            issue_id=d["issue_id"],
            issue_type=d["issue_type"],
            table=d["table"],
            columns=d["columns"],
            severity=d["severity"],
            metric=d["metric"],
            evidence_rows=d["evidence_rows"],
            description=d["description"],
            impact_score=0.0,
            affected_reports=REPORT_MAPPING.get(d["issue_type"], []),
            agent_trace=state["agent_trace"] + ["ranker"],
            confidence=d.get("confidence", 1.0),
            dim_table=d.get("dim_table", ""),
            dim_pk=d.get("dim_pk", ""),
        ))

    ranked, _ = ranking_agent_v2(ranked_input, state["table_sizes"])
    return {
        "ranked_issues": [asdict(r) for r in ranked],
        "ranking_done": True,
        "agent_trace": state["agent_trace"] + ["ranker"],
        "timing": {**state["timing"], "ranker": round(_now() - t0, 3)},
    }


def _llm_executive_summary(ranked: list, source: str) -> str:
    """Call OpenRouter to generate an executive narrative for the top-N issues.

    Returns an empty string when OPENROUTER_API_KEY is not set or the call fails.
    The rest of the summariser always runs regardless of this return value.
    """
    from astrion_dq.config import LLM_TOP_N
    from astrion_dq.llm.client import LLMUnavailable, chat

    top = ranked[:LLM_TOP_N]
    if not top:
        return ""

    issue_lines = []
    for i, r in enumerate(top, 1):
        cols = ", ".join(r.get("columns") or []) or "n/a"
        reports = ", ".join(r.get("affected_reports") or []) or "none"
        issue_lines.append(
            f"{i}. {r['issue_type']} in {r['table']} (columns: {cols}, "
            f"severity: {r['severity']}, evidence_rows: {r['evidence_rows']}, "
            f"BIS: {r.get('impact_score', 0):.3f}, affects: {reports})"
        )

    prompt = (
        "You are a senior data quality analyst reviewing a retail data warehouse.\n"
        f"Data source: {source!r}.\n\n"
        "The automated triage pipeline detected these top issues ranked by business impact:\n"
        + "\n".join(issue_lines)
        + "\n\n"
        "Write a concise executive summary (3-5 sentences) for the data engineering lead. "
        "Explain the business risk, which downstream reports are affected, and the most "
        "urgent corrective action. Synthesise the pattern — do not repeat raw numbers. "
        "Then add a short 'Recommended Actions' section with at most 3 bullet points, "
        "one sentence each. Use plain text only, no markdown headers or bold."
    )

    try:
        return chat(prompt, max_tokens=450)
    except LLMUnavailable:
        logger.debug("OpenRouter not configured — using template summary.")
        return ""
    except Exception as exc:
        logger.warning("OpenRouter call failed (%s) — using template summary.", exc)
        return ""


def _resolution_advice(issue: dict, rank: int) -> List[str]:
    """Return deterministic resolution SQL and explanation lines for a ranked issue.

    This function is LLM-free: it generates fix SQL from templates based on the
    issue_type. The output is valid DuckDB SQL. Every template handles the case
    where optional fields (columns, dim_table, dim_pk) may be missing.
    """
    issue_type = issue.get("issue_type", "unknown")
    table = issue.get("table", "your_table")
    cols = issue.get("columns") or []
    col = cols[0] if cols else "your_column"
    desc = issue.get("description", "")
    reports = ", ".join(issue.get("affected_reports") or []) or "none"
    bis = issue.get("impact_score", 0.0)
    confidence = issue.get("confidence", 1.0)
    evidence = issue.get("evidence_rows", 0)

    # For referential_integrity_break, read dim_table/dim_pk from the issue dict
    # (populated by detector → verifier pipeline). Fall back to description parsing
    # for legacy issues that pre-date the RankedIssue dim_table/dim_pk fields.
    dim_table = issue.get("dim_table", "") or ""
    dim_pk = issue.get("dim_pk", "") or ""
    if issue_type == "referential_integrity_break" and not (dim_table and dim_pk):
        m = re.search(r"not found in (\w+)\.(\w+)", desc)
        if m:
            dim_table, dim_pk = m.group(1), m.group(2)
    if not dim_table:
        dim_table = "dimension_table"
    if not dim_pk:
        dim_pk = "pk_column"

    lines: List[str] = [
        f"---",
        f"",
        f"### Issue #{rank} — `{issue_type}` in `{table}`",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| Priority | **#{rank}** |",
        f"| Severity | `{issue.get('severity', 'medium')}` |",
        f"| Business Impact Score | `{bis:.4f}` |",
        f"| Confidence | `{confidence:.2f}` |",
        f"| Evidence Rows | `{evidence:,}` |",
        f"| Affected Reports | {reports} |",
        f"| Column(s) | `{', '.join(cols) if cols else 'n/a'}` |",
        f"",
        f"**What's wrong**: {desc}",
        f"",
    ]

    # ── Per-type resolution SQL ────────────────────────────────────────────────

    if issue_type == "referential_integrity_break":
        lines += [
            "**Why it matters**: Foreign key violations cause silent row drops in JOIN "
            "operations. Downstream aggregations will silently under-count without any "
            "error message, producing reports that look correct but are wrong.",
            "",
            "#### Resolution",
            "",
            "**Step 1 — Identify the orphaned keys**",
            "```sql",
            f"-- How many distinct invalid FK values exist, and how many rows they affect",
            f"SELECT",
            f"    f.{col} AS invalid_fk_value,",
            f"    COUNT(*) AS affected_rows",
            f"FROM {table} f",
            f"LEFT JOIN {dim_table} d ON f.{col} = d.{dim_pk}",
            f"WHERE d.{dim_pk} IS NULL",
            f"GROUP BY f.{col}",
            f"ORDER BY affected_rows DESC",
            f"LIMIT 20;",
            "```",
            "",
            "**Step 2 — Choose a resolution strategy**",
            "",
            "*Option A — Delete orphaned fact rows (use when the fact rows are bad data)*",
            "```sql",
            f"DELETE FROM {table}",
            f"WHERE {col} NOT IN (",
            f"    SELECT {dim_pk} FROM {dim_table}",
            f");",
            "```",
            "",
            "*Option B — Insert placeholder dimension records (use when dimension is incomplete)*",
            "```sql",
            f"INSERT INTO {dim_table} ({dim_pk})",
            f"SELECT DISTINCT f.{col}",
            f"FROM {table} f",
            f"WHERE f.{col} NOT IN (SELECT {dim_pk} FROM {dim_table});",
            "```",
            "",
            "*Option C — NULL out the invalid FK (preserve fact rows, break the join cleanly)*",
            "```sql",
            f"UPDATE {table}",
            f"SET {col} = NULL",
            f"WHERE {col} NOT IN (",
            f"    SELECT {dim_pk} FROM {dim_table}",
            f");",
            "```",
        ]

    elif issue_type == "duplicate_rows":
        lines += [
            "**Why it matters**: Duplicate rows inflate every metric — revenue totals, "
            "transaction counts, customer activity figures — by an unknown multiplier. "
            "Reports built on this table will overstate results until deduplication runs.",
            "",
            "#### Resolution",
            "",
            "**Step 1 — Measure the duplication rate**",
            "```sql",
            f"SELECT",
            f"    COUNT(*) AS total_rows,",
            f"    COUNT(*) - COUNT(DISTINCT *) AS duplicate_rows,",
            f"    ROUND(100.0 * (COUNT(*) - COUNT(DISTINCT *)) / COUNT(*), 2) AS dup_pct",
            f"FROM {table};",
            "```",
            "",
            "**Step 2 — Preview duplicate groups**",
            "```sql",
            f"SELECT *, COUNT(*) OVER (PARTITION BY *) AS occurrences",
            f"FROM {table}",
            f"WHERE occurrences > 1",
            f"ORDER BY occurrences DESC",
            f"LIMIT 20;",
            "```",
            "",
            "**Step 3 — Deduplicate (keep one row per group)**",
            "```sql",
            f"-- DuckDB: use rowid to keep the first occurrence of each duplicate",
            f"DELETE FROM {table}",
            f"WHERE rowid NOT IN (",
            f"    SELECT MIN(rowid)",
            f"    FROM {table}",
            f"    GROUP BY *",
            f");",
            "```",
            "",
            "**ETL prevention**: Add a `UNIQUE` constraint or a `SELECT DISTINCT` in the "
            "ETL load step to prevent re-insertion of duplicates at source.",
        ]

    elif issue_type == "numeric_outliers":
        lines += [
            f"**Why it matters**: Outliers in `{col}` will skew aggregated metrics "
            "(averages, totals) and trigger false alerts in downstream monitoring dashboards. "
            "Revenue or quantity columns with outliers can distort P&L reporting.",
            "",
            "#### Resolution",
            "",
            "**Step 1 — Find the outlier bounds (IQR method)**",
            "```sql",
            f"SELECT",
            f"    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) AS q1,",
            f"    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) AS q3,",
            f"    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col})",
            f"        - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) AS iqr,",
            f"    MIN({col}) AS min_val,",
            f"    MAX({col}) AS max_val,",
            f"    AVG({col}) AS mean_val",
            f"FROM {table};",
            "```",
            "",
            "**Step 2 — Inspect the extreme values**",
            "```sql",
            f"WITH stats AS (",
            f"    SELECT",
            f"        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) AS q1,",
            f"        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) AS q3",
            f"    FROM {table}",
            f")",
            f"SELECT {col}",
            f"FROM {table}, stats",
            f"WHERE {col} < q1 - 1.5 * (q3 - q1)",
            f"   OR {col} > q3 + 1.5 * (q3 - q1)",
            f"ORDER BY {col}",
            f"LIMIT 20;",
            "```",
            "",
            "**Step 3 — Resolution options**",
            "",
            "*Option A — Winsorise (cap at IQR bounds, keeps row count)*",
            "```sql",
            f"WITH stats AS (",
            f"    SELECT",
            f"        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) AS q1,",
            f"        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) AS q3",
            f"    FROM {table}",
            f")",
            f"UPDATE {table}",
            f"SET {col} = CASE",
            f"    WHEN {col} < q1 - 1.5*(q3-q1) THEN q1 - 1.5*(q3-q1)",
            f"    WHEN {col} > q3 + 1.5*(q3-q1) THEN q3 + 1.5*(q3-q1)",
            f"    ELSE {col}",
            f"END",
            f"FROM stats;",
            "```",
            "",
            "*Option B — Remove outlier rows (use when they are data entry errors)*",
            "```sql",
            f"WITH stats AS (",
            f"    SELECT",
            f"        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) AS q1,",
            f"        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) AS q3",
            f"    FROM {table}",
            f")",
            f"DELETE FROM {table}",
            f"USING stats",
            f"WHERE {col} < q1 - 1.5*(q3-q1)",
            f"   OR {col} > q3 + 1.5*(q3-q1);",
            "```",
        ]

    elif issue_type == "missing_values":
        lines += [
            f"**Why it matters**: Nulls in `{col}` propagate silently through GROUP BY and "
            "JOIN operations — null keys will never match, null metrics are excluded from "
            "SUM/AVG without warning. Reports built on this column will silently under-count.",
            "",
            "#### Resolution",
            "",
            "**Step 1 — Measure the null rate per column**",
            "```sql",
            f"SELECT",
            f"    COUNT(*) AS total_rows,",
            f"    SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,",
            f"    ROUND(100.0 * SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END)",
            f"          / COUNT(*), 2) AS null_pct",
            f"FROM {table};",
            "```",
            "",
            "**Step 2 — Resolution options**",
            "",
            "*Option A — Fill with column mean (for numeric columns)*",
            "```sql",
            f"UPDATE {table}",
            f"SET {col} = (SELECT AVG({col}) FROM {table} WHERE {col} IS NOT NULL)",
            f"WHERE {col} IS NULL;",
            "```",
            "",
            "*Option B — Fill with a domain-specific sentinel*",
            "```sql",
            f"UPDATE {table}",
            f"SET {col} = 'UNKNOWN'  -- replace with your domain default",
            f"WHERE {col} IS NULL;",
            "```",
            "",
            "*Option C — Remove rows where this column is null*",
            "```sql",
            f"DELETE FROM {table} WHERE {col} IS NULL;",
            "```",
            "",
            "*Option D — Flag for upstream investigation (non-destructive)*",
            "```sql",
            f"-- Add an audit flag column instead of modifying data",
            f"ALTER TABLE {table} ADD COLUMN {col}_is_null BOOLEAN DEFAULT FALSE;",
            f"UPDATE {table} SET {col}_is_null = TRUE WHERE {col} IS NULL;",
            "```",
        ]

    elif issue_type == "invalid_future_dates":
        lines += [
            f"**Why it matters**: Dates in the far future (e.g., `2050-01-01`) in `{col}` "
            "are sentinel placeholders left by the ETL. They cause 'open-ended' records to "
            "appear in date-range filters, inflating active-record counts and time-based metrics.",
            "",
            "#### Resolution",
            "",
            "**Step 1 — Identify all future date values and their frequency**",
            "```sql",
            f"SELECT",
            f"    {col},",
            f"    COUNT(*) AS row_count",
            f"FROM {table}",
            f"WHERE TRY_CAST({col} AS DATE) > CURRENT_DATE",
            f"GROUP BY {col}",
            f"ORDER BY {col};",
            "```",
            "",
            "**Step 2 — Resolution options**",
            "",
            "*Option A — Replace sentinel with NULL (open-ended convention)*",
            "```sql",
            f"UPDATE {table}",
            f"SET {col} = NULL",
            f"WHERE TRY_CAST({col} AS DATE) > CURRENT_DATE;",
            "```",
            "",
            "*Option B — Replace with ISO open-ended convention date*",
            "```sql",
            f"UPDATE {table}",
            f"SET {col} = '9999-12-31'",
            f"WHERE TRY_CAST({col} AS DATE) > CURRENT_DATE",
            f"  AND TRY_CAST({col} AS DATE) < DATE '9999-01-01';",
            "```",
            "",
            "*Option C — Remove records with future dates (if they are truly invalid)*",
            "```sql",
            f"DELETE FROM {table}",
            f"WHERE TRY_CAST({col} AS DATE) > CURRENT_DATE;",
            "```",
        ]

    elif issue_type == "statistical_drift":
        lines += [
            f"**Why it matters**: The distribution of `{col}` has shifted significantly "
            "compared to the baseline snapshot. This is invisible to rule-based checks — "
            "every individual value may be valid, but the population has changed. "
            "Common causes: ETL bug (wrong exchange rate, wrong filter), schema change, "
            "or a genuine business shift that needs a new baseline.",
            "",
            "#### Resolution",
            "",
            "**Step 1 — Compare current statistics against the baseline**",
            "```sql",
            f"SELECT",
            f"    COUNT(*) AS row_count,",
            f"    ROUND(MIN({col}), 4) AS min_val,",
            f"    ROUND(MAX({col}), 4) AS max_val,",
            f"    ROUND(AVG({col}), 4) AS mean_val,",
            f"    ROUND(STDDEV({col}), 4) AS std_val,",
            f"    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP",
            f"          (ORDER BY {col}), 4) AS median_val",
            f"FROM {table};",
            "```",
            "",
            "**Step 2 — Sample recent rows to spot ETL anomalies**",
            "```sql",
            f"SELECT *",
            f"FROM {table}",
            f"ORDER BY rowid DESC",
            f"LIMIT 20;",
            "```",
            "",
            "**Step 3 — Decision tree**",
            "",
            "- **ETL bug confirmed** → fix the pipeline, reload the data, re-run triage",
            "- **Legitimate business shift** → run `astrion-dq snapshot` to update the baseline",
            "- **Schema change** → update the profiler configuration and save a new snapshot",
        ]

    elif issue_type == "empty_table":
        lines += [
            f"**Why it matters**: An empty `{table}` means every downstream report "
            "that JOINs against it will return zero rows. This is a complete data loss "
            "scenario for any report dependent on this table.",
            "",
            "#### Resolution",
            "",
            "**Step 1 — Confirm the table is empty**",
            "```sql",
            f"SELECT COUNT(*) AS row_count FROM {table};",
            "```",
            "",
            "**Step 2 — Check the ETL source**",
            "```sql",
            f"-- After re-loading: verify the row count is as expected",
            f"SELECT COUNT(*) FROM {table};",
            "```",
            "",
            "**Step 3 — Investigation checklist**",
            "- Verify the source file / API response is non-empty",
            "- Check ETL job logs for silent failures or filtered-out records",
            "- Confirm file permissions and source path are correct",
            "- Re-trigger the ETL load job for this table",
        ]

    else:
        lines += [
            f"**Why it matters**: Unexpected issue type `{issue_type}` — investigate directly.",
            "",
            "#### Resolution",
            "",
            "**Investigate the affected rows**",
            "```sql",
            f"SELECT * FROM {table} LIMIT 20;",
            "```",
        ]

    lines.append("")
    return lines


def summariser_node(state: TriageState) -> dict:
    """Produce a structured markdown triage report with per-issue resolution SQL.

    Structure:
      1. Header (source, totals)
      2. Executive summary (LLM-generated when OpenRouter is configured)
      3. Per-issue sections: what's wrong, why it matters, resolution SQL options
      4. Agent trace + timing

    The resolution SQL is generated deterministically from templates — it never
    requires the LLM and works in offline / no-API-key environments.
    """
    t0 = _now()
    ranked = state.get("ranked_issues") or []

    lines = [
        "# Astrion Data Quality Triage Report",
        "",
        f"**Source**: `{state['source']}` | **Sensitivity**: `high`",
        f"**Total issues detected**: {len(ranked)}",
        "",
    ]

    if not ranked:
        lines += [
            "No issues detected — data quality checks passed.",
            "",
        ]
    else:
        # LLM executive summary (bonus section when OpenRouter is configured)
        exec_summary = _llm_executive_summary(ranked, state["source"])
        if exec_summary:
            lines += [
                "## Executive Summary",
                "",
                exec_summary,
                "",
            ]

        lines += [
            "## Issues & Resolution Guide",
            "",
            f"> {len(ranked)} issue(s) detected and ranked by Business Impact Score (BIS). "
            "Each section below includes the root cause, SQL queries to investigate, "
            "and multiple resolution options.",
            "",
        ]

        for rank, issue in enumerate(ranked, 1):
            lines += _resolution_advice(issue, rank)

    lines += [
        "## Agent Trace",
        "",
        " → ".join(state.get("agent_trace") or []),
        "",
        "## Timing",
        "",
    ]
    for node, elapsed in (state.get("timing") or {}).items():
        lines.append(f"- `{node}`: {elapsed:.3f}s")

    return {
        "report_md": "\n".join(lines),
        "agent_trace": state["agent_trace"] + ["summariser"],
        "timing": {**state["timing"], "summariser": round(_now() - t0, 3)},
    }
