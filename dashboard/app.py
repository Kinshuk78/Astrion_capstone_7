"""Astrion DQ: Data Quality Triage Dashboard

Read-only view over outputs produced by the CLI pipeline, plus two interactive
features:
  • SQL Assistant — LLM-powered chat agent for data engineers
  • Upload & Analyze — triage any CSV dataset without manual injection

Files consumed (all under outputs/):
  ranked_issues_{source}.json  -- produced by 'astrion-dq triage --source <source>'
  evaluation_comparison.json   -- produced by 'astrion-dq evaluate'
  triage_report_{source}.md    -- produced by 'astrion-dq triage --source <source>'

Pipeline triggers in the sidebar invoke astrion_dq.cli as a subprocess.
The dashboard never re-implements pipeline logic for the retail dataset.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Astrion DQ: Data Quality Triage",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Password gate
# When ASTRION_API_TOKEN is set the dashboard requires the same token.
# When the env var is unset the gate is skipped (dev / local mode).
# ---------------------------------------------------------------------------
_expected_token = os.environ.get("ASTRION_API_TOKEN", "")
if _expected_token:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("Astrion DQ: Login")
        entered = st.text_input("Access token", type="password")
        if st.button("Login"):
            if entered == _expected_token:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid token.")
        st.stop()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_ranked_issues(source: str = "injected") -> list:
    path = OUTPUTS_DIR / f"ranked_issues_{source}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=30)
def load_evaluation() -> list:
    path = OUTPUTS_DIR / "evaluation_comparison.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=30)
def load_report_md(source: str = "injected") -> Optional[str]:
    path = OUTPUTS_DIR / f"triage_report_{source}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


@st.cache_data(ttl=10)
def load_run_log() -> list:
    """Load all entries from outputs/run_log.jsonl, most-recent first."""
    path = OUTPUTS_DIR / "run_log.jsonl"
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return list(reversed(entries))


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_cli(cmd: list) -> tuple[bool, str]:
    """Run an astrion_dq.cli command and return (success, combined_output)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "astrion_dq.cli"] + cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 300 seconds."
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Retail session DB helpers
# ---------------------------------------------------------------------------

def _ensure_retail_live_connection(source: str) -> Optional[duckdb.DuckDBPyConnection]:
    """Load the selected retail dataset into this Streamlit process.

    Sidebar pipeline buttons run CLI subprocesses, so their DuckDB singleton
    lives in a different process. The SQL Assistant needs a connection inside
    the Streamlit process, so we lazily materialise the selected retail source
    here when needed.
    """
    try:
        from astrion_dq.warehouse.loader import load_retail_tables, load_tables_to_duckdb

        tables = load_retail_tables(source=source)
        conn = load_tables_to_duckdb(tables)
        st.session_state["_retail_conn_source"] = source
        return conn
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SQL Agent helpers
# ---------------------------------------------------------------------------

def _get_schema_context() -> tuple[Optional[duckdb.DuckDBPyConnection], str]:
    """Return (connection, schema_description) from whatever is loaded.

    Priority: upload DuckDB (session_state) → retail DuckDB singleton /
    lazily-loaded selected retail dataset → None.
    """
    conn = None
    schema_lines = []
    source = None

    # Prefer uploaded data when present; Upload & Analyze explicitly advertises
    # that the SQL Assistant connects to the uploaded dataset.
    upload_conn = st.session_state.get("_upload_conn")
    if upload_conn is not None:
        try:
            upload_conn.execute("SELECT 1")  # test connection still alive
            conn = upload_conn
            source = "upload"
        except Exception:
            st.session_state.pop("_upload_conn", None)
            conn = None

    if conn is None:
        desired_source = st.session_state.get("src", "injected")
        try:
            from astrion_dq.warehouse.loader import get_connection

            conn = get_connection()
            if st.session_state.get("_retail_conn_source") != desired_source:
                conn = _ensure_retail_live_connection(desired_source)
            source = "retail" if conn is not None else None
        except Exception:
            conn = _ensure_retail_live_connection(desired_source)
            source = "retail" if conn is not None else None

    if conn is None:
        return None, ""

    # Build schema description
    try:
        schema_prefix = "dq_retail." if source == "retail" else ""
        tables_q = conn.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE'"
        ).fetchall()
        for schema, tname in tables_q:
            full_name = f"{schema}.{tname}" if schema else tname
            cols_q = conn.execute(
                f"SELECT column_name, data_type "
                f"FROM information_schema.columns "
                f"WHERE table_name = '{tname}'"
            ).fetchall()
            col_parts = ", ".join(f"{c} {t}" for c, t in cols_q)
            schema_lines.append(f"  {full_name} ({col_parts})")
    except Exception:
        schema_lines = ["  (schema introspection failed)"]

    schema_desc = "\n".join(schema_lines) if schema_lines else "  (no tables found)"
    return conn, schema_desc


def _build_agent_system_prompt(schema_desc: str, issues_json: str) -> str:
    return f"""You are an expert data engineer assistant specialised in DuckDB and retail data warehouse quality.

You have access to a live DuckDB database with these tables:
{schema_desc}

Current data quality issues (ranked by Business Impact Score):
{issues_json}

Your role:
1. Help the engineer understand what each issue means and why it exists
2. Write correct DuckDB SQL to investigate and fix issues
3. Explain SQL errors the engineer pastes — identify root cause and show the corrected query
4. Suggest ETL improvements and data validation patterns
5. Answer schema and data questions

Rules:
- Always use fully-qualified table names (e.g. dq_retail.fact_sales_normalized) for retail data
- Format all SQL in ```sql ... ``` code blocks so it can be auto-executed
- When multiple fix options exist, show each as a separate labelled SQL block
- The DuckDB instance is in-memory — modifications are safe and non-persistent
- Be concise but complete; prefer showing working SQL over lengthy prose"""


def _sql_agent_respond(user_message: str) -> str:
    """Call the LLM with full conversation history and return the reply."""
    from astrion_dq.llm.client import LLMUnavailable, chat_with_history

    conn, schema_desc = _get_schema_context()

    # Build issues context from the last ranked_issues file loaded
    issues = (
        st.session_state.get("_upload_results")
        or load_ranked_issues(st.session_state.get("src", "injected"))
    )
    top_issues = []
    for r in issues[:10]:
        top_issues.append(
            f"  [{r.get('issue_id','')}] {r.get('issue_type','')} "
            f"in {r.get('table','')} — BIS={r.get('impact_score',0):.3f}, "
            f"severity={r.get('severity','')}, "
            f"evidence_rows={r.get('evidence_rows',0)}"
        )
    issues_json = "\n".join(top_issues) if top_issues else "  (no issues loaded yet)"

    system = _build_agent_system_prompt(schema_desc or "  (no database loaded)", issues_json)

    # Trim history by estimated token count (≈4 chars per token).
    # Budget: 6,000 tokens (24,000 chars) — safe across all OpenRouter models.
    # Counting from the most recent message backward so the newest context is
    # always included; oldest messages are dropped first when budget is exceeded.
    _TOKEN_BUDGET_CHARS = 24_000
    history = st.session_state.get("_agent_messages", [])
    _used_chars = 0
    trimmed: list = []
    for msg in reversed(history):
        _msg_chars = len(msg.get("content", ""))
        if _used_chars + _msg_chars > _TOKEN_BUDGET_CHARS:
            break
        trimmed.insert(0, msg)
        _used_chars += _msg_chars
    if not trimmed and history:
        trimmed = [history[-1]]  # always keep at least the last message

    messages = trimmed + [{"role": "user", "content": user_message}]

    try:
        return chat_with_history(messages, system=system, max_tokens=1200)
    except LLMUnavailable:
        return (
            "**OPENROUTER_API_KEY is not configured.**\n\n"
            "Add it to `config/.env`:\n"
            "```\nOPENROUTER_API_KEY=sk-or-...\n```\n\n"
            "The rest of the dashboard (triage, evaluation, reports) works without it."
        )
    except Exception as exc:
        if "402" in str(exc) and ("credits" in str(exc).lower() or "can only afford" in str(exc).lower()):
            return (
                f"**LLM call failed**: {exc}\n\n"
                "OpenRouter accepted the API key but rejected the request because the "
                "remaining credit/token budget is too low for this response. "
                "Add credits or ask a shorter question."
            )
        return f"**LLM call failed**: {exc}\n\nCheck your API key and network connection."


def _execute_sql_blocks(response: str, conn: duckdb.DuckDBPyConnection) -> list[tuple[str, object]]:
    """Extract ```sql blocks from response and execute each against conn.

    Returns list of (sql, result_df_or_error_str) tuples.
    """
    results = []
    blocks = re.findall(r"```sql\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    for sql in blocks:
        sql = sql.strip()
        if not sql:
            continue
        try:
            df = conn.execute(sql).df()
            results.append((sql, df))
        except Exception as exc:
            results.append((sql, f"Error: {exc}"))
    return results


# ---------------------------------------------------------------------------
# Upload & Analyze helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_upload_triage(
    tables: dict[str, pd.DataFrame],
    baseline_tables: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    """Run a verified upload triage on arbitrary DataFrames.

    Flow:
      profiler -> detector -> [optional drift] -> debugger(SQL cross-validation) -> ranker

    Decorated with @st.cache_data so results survive browser refreshes within
    the same Streamlit server process. Streamlit hashes DataFrames using their
    content, so re-uploading identical files reuses the cached result.

    Drift detection runs only when *baseline_tables* is provided. This keeps
    ad hoc single-dataset uploads reliable while still enabling full workflow
    comparisons when the user supplies a trusted baseline upload.
    Returns ranked issue dicts with confidence scores preserved from the
    verifier so low-confidence issues can be reviewed in the UI.
    """
    from dataclasses import asdict

    from astrion_dq.checks.detect import infer_metadata, run_all_checks_parallel
    from astrion_dq.checks.drift import build_snapshot, detect_drift
    from astrion_dq.config import REPORT_MAPPING
    from astrion_dq.graph.debugger import IssueVerifier
    from astrion_dq.models import RankedIssue
    from astrion_dq.ranking.impact import ranking_agent_v2

    meta = infer_metadata(tables)
    issues = run_all_checks_parallel(tables, meta, sensitivity="high")
    if baseline_tables:
        # Use the same snapshot-style drift path as CLI triage so uploaded
        # baseline comparisons and saved-snapshot comparisons produce
        # consistent signals.
        issues += detect_drift(
            tables,
            meta,
            reference_snapshot=build_snapshot(baseline_tables),
        )
    table_sizes = {name: len(df) for name, df in tables.items()}
    conn = _build_upload_conn(tables)

    try:
        verified = IssueVerifier(sensitivity="high", connection=conn).verify_all(issues)
    finally:
        conn.close()

    ranked_input = [
        RankedIssue(
            issue_id=i.issue_id,
            issue_type=i.issue_type,
            table=i.table,
            columns=i.columns,
            severity=i.severity,
            metric=i.metric,
            evidence_rows=i.evidence_rows,
            description=i.description,
            impact_score=0.0,
            affected_reports=REPORT_MAPPING.get(i.issue_type, []),
            agent_trace=[],
            confidence=i.confidence,
            dim_table=i.dim_table,
            dim_pk=i.dim_pk,
        )
        for i in verified
    ]

    ranked, _ = ranking_agent_v2(ranked_input, table_sizes)
    return [asdict(r) for r in ranked]


def _build_upload_conn(tables: dict[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    """Load uploaded DataFrames into a fresh in-memory DuckDB.

    Tables are written under the standard ``dq_retail`` schema so the shared
    IssueVerifier can run unchanged, and mirrored as top-level views so the
    SQL Assistant can still reference simple table names when needed.
    """
    from astrion_dq.config import DUCKDB_SCHEMA

    conn = duckdb.connect()
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{DUCKDB_SCHEMA}"')
    for name, df in tables.items():
        conn.register("_tmp_upload", df)
        conn.execute(
            f'CREATE OR REPLACE TABLE "{DUCKDB_SCHEMA}"."{name}" AS SELECT * FROM _tmp_upload'
        )
        conn.execute(
            f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM "{DUCKDB_SCHEMA}"."{name}"'
        )
        conn.unregister("_tmp_upload")
    return conn


def _load_uploaded_tables(uploaded_files) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Parse uploaded CSVs into table-name -> DataFrame plus any load warnings."""
    tables: dict[str, pd.DataFrame] = {}
    load_errors: list[str] = []

    for f in uploaded_files:
        raw_name = Path(f.name).stem.lower().replace(" ", "_").replace("-", "_")
        table_name = re.sub(r"[^\w]", "_", raw_name)

        try:
            try:
                df = pd.read_csv(f)
            except UnicodeDecodeError:
                f.seek(0)
                df = pd.read_csv(f, encoding="latin-1")

            if df.empty:
                load_errors.append(f"`{f.name}` is empty — skipped.")
                continue
            if df.shape[1] == 0:
                load_errors.append(f"`{f.name}` has no columns — skipped.")
                continue

            tables[table_name] = df
        except Exception as exc:
            load_errors.append(f"`{f.name}` failed to load: {exc}")

    return tables, load_errors


def _generate_upload_report(ranked: list[dict]) -> str:
    """Generate a resolution-focused markdown report for uploaded data."""
    from astrion_dq.graph.nodes import _resolution_advice

    lines = [
        "# Upload Data Quality Report",
        "",
        f"**Total issues detected**: {len(ranked)}",
        "",
    ]

    if not ranked:
        lines += ["No issues detected — uploaded data passed all quality checks.", ""]
    else:
        lines += [
            "## Issues & Resolution Guide",
            "",
            f"> {len(ranked)} issue(s) ranked by Business Impact Score.",
            "",
        ]
        for rank, issue in enumerate(ranked, 1):
            lines += _resolution_advice(issue, rank)

    return "\n".join(lines)


def _close_upload_conn() -> None:
    """Close the uploaded-data DuckDB connection when it exists."""
    conn = st.session_state.pop("_upload_conn", None)
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass


def _reset_upload_session() -> None:
    """Clear uploaded files, upload results, and SQL Assistant upload context."""
    _close_upload_conn()
    for key in ("_upload_results", "_upload_report_md"):
        st.session_state.pop(key, None)
    st.session_state.pop("_agent_messages", None)
    st.session_state["_upload_uploader_version"] = (
        st.session_state.get("_upload_uploader_version", 0) + 1
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Astrion DQ")
    st.caption("Retail Data Quality Triage")
    st.divider()

    st.subheader("Pipeline Controls")
    data_source = st.selectbox("Data source", ["injected", "clean"], key="src")
    # Sensitivity is always "high" — maximises issue detection.
    sensitivity = "high"

    _running = st.session_state.get("_running", False)

    if st.button("Inject Issues", use_container_width=True, disabled=_running):
        st.session_state["_running"] = True
        with st.spinner("Injecting synthetic issues..."):
            ok, out = run_cli(["inject"])
        st.session_state["_running"] = False
        if ok:
            st.success("Issues injected.")
        else:
            st.error(f"Failed:\n{out[:400]}")

    if st.button("Save Snapshot", use_container_width=True, disabled=_running):
        st.session_state["_running"] = True
        with st.spinner("Saving drift snapshot..."):
            ok, out = run_cli(["snapshot"])
        st.session_state["_running"] = False
        if ok:
            st.success("Snapshot saved.")
        else:
            st.error(f"Failed:\n{out[:400]}")

    if st.button("Run Triage", use_container_width=True, type="primary", disabled=_running):
        st.session_state["_running"] = True
        with st.spinner(f"Running triage on {data_source!r} (sensitivity=high)..."):
            ok, out = run_cli(["triage", "--source", data_source, "--sensitivity", "high"])
        st.session_state["_running"] = False
        if ok:
            _ensure_retail_live_connection(data_source)
            st.success("Triage complete.")
            st.cache_data.clear()
        else:
            st.error(f"Failed:\n{out[:400]}")

    if st.button("Evaluate Strategies", use_container_width=True, disabled=_running):
        st.session_state["_running"] = True
        with st.spinner("Evaluating A / B / C strategies..."):
            ok, out = run_cli(["evaluate", "--source", "injected"])
        st.session_state["_running"] = False
        if ok:
            st.success("Evaluation complete.")
            st.cache_data.clear()
        else:
            st.error(f"Failed:\n{out[:400]}")

    if st.button("Generate PDF Report", use_container_width=True, disabled=_running):
        st.session_state["_running"] = True
        with st.spinner("Generating PDF..."):
            ok, out = run_cli(["report", "--source", data_source])
        st.session_state["_running"] = False
        if ok:
            st.success("PDF report generated.")
        else:
            st.error(f"Failed:\n{out[:400]}")

    st.divider()
    st.caption("v0.6.0 - Astrion Capstone 7")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

(
    tab_triage, tab_compare, tab_report,
    tab_history, tab_arch, tab_sql, tab_upload,
) = st.tabs([
    "Triage Issues",
    "Strategy Comparison",
    "Markdown Report",
    "Run History",
    "Architecture",
    "SQL Assistant",
    "Upload & Analyze",
])

# ============================================================================
# TAB 1: TRIAGE ISSUES
# ============================================================================
with tab_triage:
    st.header("Triage Issues")

    issues = load_ranked_issues(data_source)

    if not issues:
        st.info(
            "No ranked issues found. Run the pipeline:\n\n"
            "1. Click **Inject Issues** in the sidebar.\n"
            "2. Click **Run Triage**."
        )
    else:
        total = len(issues)
        high_count = sum(1 for i in issues if i.get("severity") == "high")
        med_count = sum(1 for i in issues if i.get("severity") == "medium")
        avg_score = sum(i.get("impact_score", 0.0) for i in issues) / max(total, 1)
        avg_conf = sum(i.get("confidence", 1.0) for i in issues) / max(total, 1)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Issues", total)
        c2.metric("HIGH Severity", high_count)
        c3.metric("MEDIUM Severity", med_count)
        c4.metric("Avg BIS", f"{avg_score:.4f}")
        c5.metric("Avg Confidence", f"{avg_conf:.2f}")

        st.divider()

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sev_filter = st.multiselect(
                "Severity",
                ["high", "medium", "low"],
                default=["high", "medium"],
                key="sev_f",
            )
        with col_f2:
            types_available = sorted({i.get("issue_type", "") for i in issues})
            type_filter = st.multiselect(
                "Issue type",
                types_available,
                default=[],
                key="type_f",
            )

        filtered = [
            i for i in issues
            if i.get("severity") in sev_filter
            and (not type_filter or i.get("issue_type") in type_filter)
        ]

        st.subheader(f"Issues ({len(filtered)} shown)")
        if filtered:
            rows = []
            for rank, i in enumerate(filtered[:50], start=1):
                rows.append({
                    "Rank": rank,
                    "Issue ID": i.get("issue_id", ""),
                    "Issue Type": i.get("issue_type", ""),
                    "Table": i.get("table", ""),
                    "Columns": ", ".join(i.get("columns") or []) or "n/a",
                    "Severity": i.get("severity", "low").upper(),
                    "BIS": round(i.get("impact_score", 0.0), 6),
                    "Confidence": round(i.get("confidence", 1.0), 2),
                    "Evidence Rows": i.get("evidence_rows", 0),
                    "Description": i.get("description", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Issue Distribution")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.write("Count by issue type")
            type_counts: dict = {}
            for i in issues:
                t = i.get("issue_type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
            if type_counts:
                df_tc = pd.DataFrame(
                    list(type_counts.items()), columns=["Issue Type", "Count"]
                ).sort_values("Count", ascending=False)
                st.altair_chart(
                    alt.Chart(df_tc).mark_bar().encode(
                        x=alt.X("Issue Type:N", sort="-y", title="Issue Type"),
                        y=alt.Y("Count:Q", title="Count"),
                        tooltip=["Issue Type", "Count"],
                    ).properties(height=250),
                    use_container_width=True,
                )

        with chart_col2:
            st.write("Total BIS by table")
            table_scores: dict = {}
            for i in issues:
                t = i.get("table", "unknown")
                table_scores[t] = table_scores.get(t, 0.0) + i.get("impact_score", 0.0)
            if table_scores:
                df_ts = pd.DataFrame(
                    list(table_scores.items()), columns=["Table", "Total BIS"]
                ).sort_values("Total BIS", ascending=False)
                st.altair_chart(
                    alt.Chart(df_ts).mark_bar().encode(
                        x=alt.X("Table:N", sort="-y", title="Table"),
                        y=alt.Y("Total BIS:Q", title="Total BIS"),
                        tooltip=["Table", "Total BIS"],
                    ).properties(height=250),
                    use_container_width=True,
                )

# ============================================================================
# TAB 2: STRATEGY COMPARISON
# ============================================================================
with tab_compare:
    st.header("Strategy Comparison")
    st.caption(
        "Precision, recall, and noise rate for the three evaluation strategies "
        "(A Baseline, B Supervisor, C Full) run against injected ground truth."
    )

    eval_data = load_evaluation()

    if not eval_data:
        st.info(
            "No evaluation data found.\n\n"
            "Run: astrion-dq inject, then astrion-dq evaluate."
        )
    else:
        metric_keys = [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
            ("top_5_recall", "Top-5 Recall"),
            ("noise_rate", "Noise Rate"),
            ("summary_accuracy", "Summary Accuracy"),
            ("wall_seconds", "Wall Time (s)"),
        ]

        by_strategy = {r["strategy"]: r for r in eval_data}

        rows = []
        for key, label in metric_keys:
            row: dict = {"Metric": label}
            for strat_key, col_label in [
                ("A_baseline", "A Baseline"),
                ("B_supervisor", "B Supervisor"),
                ("C_full", "C Full"),
            ]:
                val = by_strategy.get(strat_key, {}).get(key)
                if val is None:
                    row[col_label] = "n/a"
                elif isinstance(val, float):
                    row[col_label] = f"{val:.4f}"
                else:
                    row[col_label] = str(val)
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for r in eval_data:
            trace = r.get("agent_trace") or []
            if trace:
                with st.expander(f"Agent trace: {r.get('strategy', '')}"):
                    st.code(" -> ".join(trace))

# ============================================================================
# TAB 3: MARKDOWN REPORT
# ============================================================================
with tab_report:
    st.header("Triage Report")

    report_text = load_report_md(data_source)
    if report_text is None:
        st.info(
            "No triage report found.\n\n"
            "Run 'astrion-dq triage' to generate one."
        )
    else:
        st.markdown(report_text)

# ============================================================================
# TAB 4: RUN HISTORY
# ============================================================================
with tab_history:
    st.header("Run History")
    st.caption("Every triage run appends one record to outputs/run_log.jsonl.")

    if st.button("Refresh", key="refresh_history"):
        st.cache_data.clear()

    run_entries = load_run_log()

    if not run_entries:
        st.info("No runs recorded yet. Run 'astrion-dq triage' or POST /triage to create an entry.")
    else:
        rows = []
        for e in run_entries:
            rows.append({
                "Run ID": e.get("run_id", ""),
                "Source": e.get("source", ""),
                "Sensitivity": e.get("sensitivity", ""),
                "Timestamp (UTC)": e.get("timestamp", ""),
                "Issues Ranked": e.get("issue_count", 0),
                "Agent Trace": " -> ".join(e.get("agent_trace") or []),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(rows)} run(s), most recent first.")

# ============================================================================
# TAB 5: ARCHITECTURE
# ============================================================================
with tab_arch:
    st.header("System Architecture")
    st.markdown(
        """
Astrion DQ is a retail data quality triage system built around a LangGraph
workflow with deterministic supervisor routing.

**What it does**

Loads a retail star schema (six dimension tables, one fact table) from CSV into
an in-process DuckDB database. Runs five rule-based quality checks plus
statistical drift detection. Cross-validates every issue with an independent SQL
query. Ranks surviving issues by a V2 Business Impact Score. Produces a markdown
triage report and JSON output files.

**Graph nodes**

```
data_loader    -- load CSVs, register tables in DuckDB
profiler       -- infer table roles, PKs, FKs, column types
detector       -- five parallel checks: nulls, duplicates, outliers, dates, RI
drift_detector -- PSI + KS test against a saved baseline snapshot
debugger       -- SQL cross-validation, per-issue confidence score
human_review   -- interrupt for analyst input (auto-approved in evaluation runs)
ranker         -- V2 Business Impact Score, descending sort
summariser     -- structured markdown report with per-issue resolution SQL
```

**Routing**

A single deterministic ``_route()`` function in ``workflow.py`` reads completion
flags from the state dict. No LLM routing.

**Three evaluation strategies**

```
A Baseline  : data_loader -> profiler -> detector -> ranker
B Supervisor: A + debugger + human_review
C Full      : B + drift_detector
```

**V2 Business Impact Score**

```
BIS = base_weight × severity_weight × evidence_density × report_criticality
```

**CLI commands**

```
astrion-dq snapshot            Save baseline drift snapshot
astrion-dq inject              Inject synthetic issues (ground truth)
astrion-dq triage              Run full triage workflow
astrion-dq evaluate            Compare strategies A, B, C
astrion-dq report              Generate PDF triage report
astrion-dq dashboard           Launch this dashboard
astrion-dq serve               Start REST API server (port 8000)
```

**REST API (astrion-dq serve)**

```
GET  /health                   Liveness probe (no auth required)
POST /triage                   Run pipeline, returns ranked issues + run_id
GET  /runs/{run_id}            Look up a past run from run_log.jsonl
```

Set ``ASTRION_API_TOKEN`` to require a Bearer token on API and dashboard requests.
        """
    )

# ============================================================================
# TAB 6: SQL ASSISTANT
# ============================================================================
with tab_sql:
    st.header("SQL Assistant")
    st.caption(
        "An LLM-powered data engineer agent. Ask it to explain issues, "
        "write investigation queries, fix SQL errors, or generate remediation scripts. "
        "SQL blocks in responses are auto-executed against the live DuckDB."
    )

    # Connection status banner
    conn, schema_desc = _get_schema_context()
    if conn is None:
        st.warning(
            "No database loaded. Run **Triage** (sidebar) to load the retail dataset, "
            "or upload CSVs in the **Upload & Analyze** tab. "
            "The agent will still answer questions without a live connection."
        )
    else:
        with st.expander("Connected schema", expanded=False):
            st.code(schema_desc or "(empty)", language="sql")

    # Conversation controls
    col_ctrl1, col_ctrl2 = st.columns([6, 1])
    with col_ctrl2:
        if st.button("Clear chat", key="clear_agent"):
            st.session_state["_agent_messages"] = []
            st.rerun()

    # Initialise conversation history
    if "_agent_messages" not in st.session_state:
        st.session_state["_agent_messages"] = []

    # Render existing messages
    for msg in st.session_state["_agent_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-render SQL results stored alongside the message
            for sql, result in msg.get("sql_results", []):
                st.caption(f"Executed: `{sql[:80]}{'...' if len(sql) > 80 else ''}`")
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result.head(50), use_container_width=True)
                else:
                    st.error(result)

    # New user input
    if user_input := st.chat_input(
        "Ask about your data... e.g. 'Why does customer_sk fail FK checks?' "
        "or 'Show me the top 10 outlier rows in total_amount'"
    ):
        # Display user turn
        with st.chat_message("user"):
            st.markdown(user_input)

        # Append to history
        st.session_state["_agent_messages"].append(
            {"role": "user", "content": user_input, "sql_results": []}
        )

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = _sql_agent_respond(user_input)

            st.markdown(response)

            # Auto-execute SQL blocks if connection is available
            sql_results = []
            if conn is not None:
                results = _execute_sql_blocks(response, conn)
                for sql, result in results:
                    st.caption(
                        f"Executed: `{sql[:80]}{'...' if len(sql) > 80 else ''}`"
                    )
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result.head(50), use_container_width=True)
                        sql_results.append((sql, result))
                    else:
                        st.error(result)
                        sql_results.append((sql, result))

        st.session_state["_agent_messages"].append(
            {"role": "assistant", "content": response, "sql_results": sql_results}
        )

# ============================================================================
# TAB 7: UPLOAD & ANALYZE
# ============================================================================
with tab_upload:
    st.header("Upload & Analyze")
    st.caption(
        "Upload your own CSV files. The default path runs a verified analysis: "
        "profiler -> detector -> SQL cross-validation -> ranker. "
        "Each file becomes one table, and the SQL Assistant connects to the uploaded data."
    )
    st.caption(
        "You can also enable drift detection by uploading a matching baseline dataset. "
        "Without a baseline upload, the workflow skips drift and runs the verified path only."
    )

    uploader_version = st.session_state.setdefault("_upload_uploader_version", 0)
    reset_col, help_col = st.columns([1, 2])
    with reset_col:
        if st.button("Clear Uploaded Files", key="clear_upload_session", use_container_width=True):
            _reset_upload_session()
            st.rerun()
    with help_col:
        st.caption(
            "Use this if you uploaded the wrong file or need to reset an oversize upload. "
            "It clears the uploader, upload results, and uploaded SQL Assistant context."
        )

    uploaded_files = st.file_uploader(
        "Current dataset CSVs (one file = one table, filename = table name)",
        type=["csv"],
        accept_multiple_files=True,
        key=f"csv_uploader_{uploader_version}",
    )

    enable_upload_drift = st.toggle(
        "Enable drift detection using a baseline upload",
        key="enable_upload_drift",
        help="Upload a second dataset with matching table and column names to compare "
             "the current upload against a trusted baseline.",
    )

    baseline_files = []
    if enable_upload_drift:
        baseline_files = st.file_uploader(
            "Baseline dataset CSVs for drift comparison",
            type=["csv"],
            accept_multiple_files=True,
            key=f"csv_baseline_uploader_{uploader_version}",
        )

    if uploaded_files:
        # ── Load and validate ──────────────────────────────────────────────
        tables, load_errors = _load_uploaded_tables(uploaded_files)
        baseline_tables: dict[str, pd.DataFrame] = {}
        baseline_errors: list[str] = []

        if enable_upload_drift and baseline_files:
            baseline_tables, baseline_errors = _load_uploaded_tables(baseline_files)

        for err in load_errors:
            st.warning(err)
        for err in baseline_errors:
            st.warning(f"Baseline: {err}")

        if not tables:
            st.error("No valid CSV files loaded. Check the warnings above.")
        else:
            # ── Preview ───────────────────────────────────────────────────
            st.subheader(f"{len(tables)} table(s) loaded")
            for tname, df in tables.items():
                with st.expander(f"`{tname}` — {len(df):,} rows × {df.shape[1]} columns"):
                    st.dataframe(df.head(10), use_container_width=True)

            st.divider()

            baseline_ready = bool(baseline_tables)
            drift_enabled = enable_upload_drift and baseline_ready

            if enable_upload_drift and not baseline_ready:
                st.info(
                    "Upload baseline CSVs to enable drift detection. "
                    "Otherwise the analysis will run without drift."
                )
            elif drift_enabled:
                matching_tables = sorted(set(tables) & set(baseline_tables))
                st.success(
                    f"Drift detection enabled across {len(matching_tables)} matching table(s): "
                    f"{', '.join(matching_tables) if matching_tables else 'none'}."
                )
                if not matching_tables:
                    st.warning(
                        "No table names match between current and baseline uploads, so drift "
                        "checks will not produce signals until names align."
                    )

            # ── Run Analysis button ───────────────────────────────────────
            run_label = "Run Full Verified Analysis" if drift_enabled else "Run Verified Analysis"
            spinner_label = (
                "Running verified data quality + drift checks on uploaded data..."
                if drift_enabled else
                "Running verified data quality checks on uploaded data..."
            )
            if st.button(run_label, type="primary", key="run_upload_triage"):
                with st.spinner(spinner_label):
                    try:
                        ranked = _run_upload_triage(
                            tables,
                            baseline_tables=baseline_tables if drift_enabled else None,
                        )
                        st.session_state["_upload_results"] = ranked
                        # Build DuckDB connection for SQL Agent
                        _close_upload_conn()
                        st.session_state["_upload_conn"] = _build_upload_conn(tables)
                        st.session_state["_upload_report_md"] = _generate_upload_report(ranked)
                        if drift_enabled:
                            st.success(
                                f"Full verified analysis complete — {len(ranked)} issue(s) found."
                            )
                        else:
                            st.success(
                                f"Verified analysis complete — {len(ranked)} issue(s) found."
                            )
                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")

            # ── Results (shown after analysis runs) ───────────────────────
            ranked = st.session_state.get("_upload_results")
            if ranked is not None:
                st.subheader(f"Issues ({len(ranked)} found)")

                if not ranked:
                    st.success("No data quality issues detected in the uploaded files.")
                else:
                    # Metrics row
                    from astrion_dq.config import CONFIDENCE_THRESHOLD

                    total = len(ranked)
                    high_c = sum(1 for r in ranked if r.get("severity") == "high")
                    med_c = sum(1 for r in ranked if r.get("severity") == "medium")
                    avg_bis = sum(r.get("impact_score", 0) for r in ranked) / max(total, 1)
                    avg_conf = sum(r.get("confidence", 1.0) for r in ranked) / max(total, 1)
                    needs_review = sum(
                        1 for r in ranked
                        if r.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
                    )

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total Issues", total)
                    m2.metric("HIGH", high_c)
                    m3.metric("MEDIUM", med_c)
                    m4.metric("Avg BIS", f"{avg_bis:.4f}")
                    m5.metric("Avg Confidence", f"{avg_conf:.2f}")

                    if needs_review:
                        st.warning(
                            f"{needs_review} issue(s) have confidence below "
                            f"{CONFIDENCE_THRESHOLD:.2f} and should be reviewed manually."
                        )

                    # Issues table
                    rows_out = []
                    for rank, r in enumerate(ranked[:50], 1):
                        rows_out.append({
                            "Rank": rank,
                            "Issue ID": r.get("issue_id", ""),
                            "Issue Type": r.get("issue_type", ""),
                            "Table": r.get("table", ""),
                            "Columns": ", ".join(r.get("columns") or []) or "n/a",
                            "Severity": r.get("severity", "").upper(),
                            "BIS": round(r.get("impact_score", 0.0), 4),
                            "Confidence": round(r.get("confidence", 1.0), 2),
                            "Evidence Rows": r.get("evidence_rows", 0),
                            "Description": r.get("description", ""),
                        })
                    st.dataframe(
                        pd.DataFrame(rows_out),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Distribution chart
                    type_counts: dict = {}
                    for r in ranked:
                        k = r.get("issue_type", "unknown")
                        type_counts[k] = type_counts.get(k, 0) + 1
                    df_tc = pd.DataFrame(
                        list(type_counts.items()), columns=["Issue Type", "Count"]
                    ).sort_values("Count", ascending=False)
                    st.altair_chart(
                        alt.Chart(df_tc).mark_bar().encode(
                            x=alt.X("Issue Type:N", sort="-y"),
                            y=alt.Y("Count:Q"),
                            tooltip=["Issue Type", "Count"],
                        ).properties(height=220, title="Issues by type"),
                        use_container_width=True,
                    )

                # ── Resolution Report ────────────────────────────────────
                st.divider()
                st.subheader("Resolution Report")
                report_md = st.session_state.get("_upload_report_md", "")
                if report_md:
                    st.markdown(report_md)
                    st.download_button(
                        label="Download report (.md)",
                        data=report_md.encode("utf-8"),
                        file_name="upload_triage_report.md",
                        mime="text/markdown",
                    )

                st.info(
                    "The uploaded tables are now available in the **SQL Assistant** tab. "
                    "Ask the agent to investigate or fix any of the issues above."
                )
