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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

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


def _get_setting(name: str, default: str = "") -> str:
    """Return an env var or Streamlit secret without hard-failing when absent."""
    value = os.environ.get(name)
    if value:
        return value

    try:
        secret_value = st.secrets.get(name, "")
    except Exception:
        secret_value = ""

    return str(secret_value or default)


def _normalise_api_base_url(raw: str) -> str:
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    if raw.endswith(".onrender.com") or raw.endswith(".streamlit.app"):
        return f"https://{raw}"
    return f"http://{raw}"


_API_BASE_URL = _normalise_api_base_url(
    _get_setting("ASTRION_API_URL") or _get_setting("ASTRION_API_BASE_URL")
)
_API_AUTH_TOKEN = _get_setting("ASTRION_API_TOKEN", "")
_DASHBOARD_TOKEN = _get_setting("ASTRION_DASHBOARD_TOKEN", "")


def _api_json_request(
    path: str,
    payload: Optional[dict[str, Any]] = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Call the configured Render API and parse the JSON response."""
    if not _API_BASE_URL:
        raise RuntimeError("ASTRION_API_URL is not configured.")

    url = f"{_API_BASE_URL}{path}"
    headers = {"Accept": "application/json"}
    data = None
    method = "GET"

    if _API_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {_API_AUTH_TOKEN}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
        method = "POST"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{exc.code} {exc.reason}: {detail or 'request failed'}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc

    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}.") from exc


def _api_bytes_request(path: str, timeout: int = 120) -> bytes:
    """Fetch raw bytes from the configured Render API."""
    if not _API_BASE_URL:
        raise RuntimeError("ASTRION_API_URL is not configured.")

    url = f"{_API_BASE_URL}{path}"
    headers = {}
    if _API_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {_API_AUTH_TOKEN}"

    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{exc.code} {exc.reason}: {detail or 'request failed'}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc


def _poll_remote_job(job_id: str, timeout_seconds: int = 180) -> dict[str, Any]:
    """Poll a remote triage job until it completes or times out."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = _api_json_request(f"/jobs/{job_id}", timeout=60)
        if result.get("status") != "running":
            return result
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for remote job {job_id}.")


def _run_remote_triage(source: str) -> tuple[bool, str]:
    submit = _api_json_request("/triage", {"source": source}, timeout=60)
    job_id = str(submit.get("job_id") or "")
    if not job_id:
        return False, "Remote triage did not return a job_id."

    result = _poll_remote_job(job_id)
    if result.get("status") == "done":
        return True, f"Assessment complete (job {job_id})."
    return False, str(result.get("error") or f"Remote job {job_id} failed.")


def _run_remote_operation(path: str, payload: Optional[dict[str, Any]] = None) -> tuple[bool, str]:
    try:
        result = _api_json_request(path, payload or {}, timeout=180)
        return True, str(result.get("status") or "ok")
    except Exception as exc:
        return False, str(exc)


@st.cache_data(ttl=20, show_spinner=False)
def load_backend_status(source: str = "injected") -> dict[str, Any]:
    """Return API connectivity plus remote artifact readiness flags."""
    if not _API_BASE_URL:
        return {"api_connected": False, "mode": "local"}

    try:
        health = _api_json_request("/health", timeout=20)
        status = _api_json_request(f"/outputs/status?source={source}", timeout=20)
        return {
            "mode": "remote",
            "api_connected": health.get("status") == "ok",
            **status,
        }
    except Exception as exc:
        return {
            "mode": "remote_error",
            "api_connected": False,
            "error": str(exc),
        }


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_remote_summary(source: str, issues_payload: str) -> dict[str, Any]:
    """Fetch an LLM summary from the Render API using only ranked issue metadata."""
    issues = json.loads(issues_payload)
    try:
        return _api_json_request(
            "/assistant/summary",
            {"source": source, "issues": issues},
            timeout=90,
        )
    except Exception as exc:
        return {"summary": "", "error": str(exc), "used_fallback": True}

# ---------------------------------------------------------------------------
# Password gate
# Use ASTRION_DASHBOARD_TOKEN for a UI login gate.
# For backwards compatibility, ASTRION_API_TOKEN still gates local-only runs.
# ---------------------------------------------------------------------------
_expected_token = _DASHBOARD_TOKEN or ("" if _API_BASE_URL else _API_AUTH_TOKEN)
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
    if _API_BASE_URL:
        try:
            return _api_json_request(f"/outputs/ranked-issues?source={source}", timeout=30).get("issues", [])
        except Exception:
            return []
    path = OUTPUTS_DIR / f"ranked_issues_{source}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=30)
def load_evaluation() -> list:
    if _API_BASE_URL:
        try:
            return _api_json_request("/outputs/evaluation", timeout=30).get("results", [])
        except Exception:
            return []
    path = OUTPUTS_DIR / "evaluation_comparison.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=30)
def load_report_md(source: str = "injected") -> Optional[str]:
    if _API_BASE_URL:
        try:
            return _api_json_request(f"/outputs/report?source={source}", timeout=30).get("content") or None
        except Exception:
            return None
    path = OUTPUTS_DIR / f"triage_report_{source}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


@st.cache_data(ttl=10)
def load_run_log() -> list:
    """Load all entries from outputs/run_log.jsonl, most-recent first."""
    if _API_BASE_URL:
        try:
            return _api_json_request("/outputs/run-log?limit=100", timeout=30).get("entries", [])
        except Exception:
            return []
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
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        src_path = str(PROJECT_ROOT / "src")
        env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
        result = subprocess.run(
            [sys.executable, "-m", "astrion_dq.cli"] + cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300,
            env=env,
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


def _trim_agent_history(history: list[dict], budget_chars: int = 24_000) -> list[dict]:
    """Keep the newest chat turns that fit within the configured char budget."""
    used_chars = 0
    trimmed: list[dict] = []
    for msg in reversed(history):
        msg_chars = len(msg.get("content", ""))
        if used_chars + msg_chars > budget_chars:
            break
        trimmed.insert(0, msg)
        used_chars += msg_chars
    if not trimmed and history:
        trimmed = [history[-1]]
    return trimmed


def _sql_agent_respond(user_message: str) -> str:
    """Call the LLM with full conversation history and return the reply."""
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

    history = st.session_state.get("_agent_messages", [])
    trimmed = _trim_agent_history(history)

    if _API_BASE_URL:
        try:
            result = _api_json_request(
                "/assistant/chat",
                {
                    "message": user_message,
                    "history": [
                        {"role": msg.get("role", ""), "content": msg.get("content", "")}
                        for msg in trimmed
                    ],
                    "schema_desc": schema_desc or "  (no database loaded)",
                    "issues": issues[:25],
                    "max_tokens": 1200,
                },
                timeout=90,
            )
            response = (result.get("response") or "").strip()
            return response or "The remote assistant returned an empty response."
        except Exception as exc:
            return (
                f"**Remote assistant call failed**: {exc}\n\n"
                "Check `ASTRION_API_URL`, `ASTRION_API_TOKEN`, and the Render API deployment."
            )

    from astrion_dq.llm.client import LLMUnavailable, chat_with_history

    system = _build_agent_system_prompt(schema_desc or "  (no database loaded)", issues_json)

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
# Dashboard presentation helpers
# ---------------------------------------------------------------------------

def _inject_dashboard_styles() -> None:
    st.markdown(
        """
<style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.4rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(17, 24, 39, 0.85));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 0.35rem 0.8rem;
        box-shadow: 0 18px 40px rgba(2, 6, 23, 0.18);
    }
    .astrion-hero {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 34%),
            linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.92));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 24px;
        padding: 1.35rem 1.5rem 1.1rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 24px 60px rgba(2, 6, 23, 0.22);
    }
    .astrion-kicker {
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 0.75rem;
        color: #67e8f9;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .astrion-title {
        font-size: 2.35rem;
        line-height: 1.05;
        font-weight: 800;
        color: #f8fafc;
        margin: 0 0 0.55rem 0;
    }
    .astrion-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 58rem;
        margin-bottom: 0.9rem;
    }
    .astrion-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }
    .astrion-pill {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(125, 211, 252, 0.18);
        background: rgba(30, 41, 59, 0.72);
        color: #e2e8f0;
        font-size: 0.84rem;
        font-weight: 600;
    }
    .astrion-section-note {
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin: 0 0 1rem 0;
        color: #cbd5e1;
    }
    .astrion-sidebar-note {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 18px;
        padding: 0.85rem 0.9rem;
        margin: 0.5rem 0 1rem 0;
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .astrion-sidebar-note strong {
        color: #f8fafc;
    }
    div.stButton > button {
        border-radius: 999px;
        min-height: 2.9rem;
        font-weight: 650;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _format_run_timestamp(ts: str) -> str:
    if not ts:
        return "Not available"
    return ts.replace("T", " ").replace("+00:00", " UTC")


def _issue_metrics(issues: list[dict]) -> dict[str, float | int]:
    from astrion_dq.config import CONFIDENCE_THRESHOLD

    total = len(issues)
    high_count = sum(1 for issue in issues if issue.get("severity") == "high")
    medium_count = sum(1 for issue in issues if issue.get("severity") == "medium")
    avg_score = sum(issue.get("impact_score", 0.0) for issue in issues) / max(total, 1)
    avg_conf = sum(issue.get("confidence", 1.0) for issue in issues) / max(total, 1)
    impacted_reports = len(
        {
            report
            for issue in issues
            for report in (issue.get("affected_reports") or [])
        }
    )
    needs_review = sum(
        1 for issue in issues
        if issue.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
    )
    return {
        "total": total,
        "high": high_count,
        "medium": medium_count,
        "avg_score": avg_score,
        "avg_conf": avg_conf,
        "impacted_reports": impacted_reports,
        "needs_review": needs_review,
    }


def _issue_dataframe(issues: list[dict], limit: int = 50) -> pd.DataFrame:
    rows = []
    for rank, issue in enumerate(issues[:limit], start=1):
        rows.append({
            "Rank": rank,
            "Issue ID": issue.get("issue_id", ""),
            "Issue Type": issue.get("issue_type", ""),
            "Table": issue.get("table", ""),
            "Columns": ", ".join(issue.get("columns") or []) or "n/a",
            "Severity": issue.get("severity", "low").upper(),
            "BIS": round(issue.get("impact_score", 0.0), 4),
            "Confidence": round(issue.get("confidence", 1.0), 2),
            "Evidence Rows": issue.get("evidence_rows", 0),
            "Description": issue.get("description", ""),
        })
    return pd.DataFrame(rows)


def _render_issue_distribution_charts(issues: list[dict], key_prefix: str) -> None:
    chart_col1, chart_col2 = st.columns(2)

    type_counts: dict[str, int] = {}
    table_scores: dict[str, float] = {}
    for issue in issues:
        issue_type = issue.get("issue_type", "unknown")
        issue_table = issue.get("table", "unknown")
        type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        table_scores[issue_table] = table_scores.get(issue_table, 0.0) + issue.get("impact_score", 0.0)

    with chart_col1:
        st.caption("Issue volume by failure type")
        if type_counts:
            df_type = pd.DataFrame(
                list(type_counts.items()), columns=["Issue Type", "Count"]
            ).sort_values("Count", ascending=False)
            st.altair_chart(
                alt.Chart(df_type).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("Issue Type:N", sort="-y", title=None),
                    y=alt.Y("Count:Q", title="Count"),
                    color=alt.value("#38bdf8"),
                    tooltip=["Issue Type", "Count"],
                ).properties(height=260),
                use_container_width=True,
            )
        else:
            st.info("No issue distribution available yet.")

    with chart_col2:
        st.caption("Business impact concentration by table")
        if table_scores:
            df_table = pd.DataFrame(
                list(table_scores.items()), columns=["Table", "Total BIS"]
            ).sort_values("Total BIS", ascending=False)
            st.altair_chart(
                alt.Chart(df_table).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("Table:N", sort="-y", title=None),
                    y=alt.Y("Total BIS:Q", title="Total BIS"),
                    color=alt.value("#f59e0b"),
                    tooltip=["Table", "Total BIS"],
                ).properties(height=260),
                use_container_width=True,
            )
        else:
            st.info("No table concentration data available yet.")


def _render_strategy_benchmark(eval_data: list[dict]) -> None:
    st.caption(
        "Benchmark the three workflow variants against injected ground truth. "
        "Keep this for technical reviews, not for daily analyst operations."
    )

    if not eval_data:
        st.info("No benchmark output found. Use `Benchmark Engine` in Admin & Demo Tools.")
        return

    metric_keys = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("top_5_recall", "Top-5 Recall"),
        ("noise_rate", "Noise Rate"),
        ("summary_accuracy", "Summary Accuracy"),
        ("wall_seconds", "Wall Time (s)"),
    ]
    by_strategy = {row["strategy"]: row for row in eval_data}

    rows = []
    for key, label in metric_keys:
        row: dict[str, str] = {"Metric": label}
        for strat_key, col_label in [
            ("A_baseline", "Baseline"),
            ("B_supervisor", "Supervisor"),
            ("C_full", "Full"),
        ]:
            value = by_strategy.get(strat_key, {}).get(key)
            if value is None:
                row[col_label] = "n/a"
            elif isinstance(value, float):
                row[col_label] = f"{value:.4f}"
            else:
                row[col_label] = str(value)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for row in eval_data:
        trace = row.get("agent_trace") or []
        if trace:
            with st.expander(f"Agent trace: {row.get('strategy', '')}"):
                st.code(" -> ".join(trace))


def _render_architecture_reference() -> None:
    st.markdown(
        """
Astrion DQ is a retail data quality triage system built around a LangGraph workflow with deterministic supervisor routing.

**What it does**

Loads a retail star schema from CSV into DuckDB, runs rule-based quality checks plus drift detection, cross-validates every issue with SQL, ranks the surviving issues by Business Impact Score, and generates reports for business and engineering teams.

**Workflow**

```
data_loader    -> load CSVs and register tables
profiler       -> infer PKs, FKs, roles, and types
detector       -> nulls, duplicates, outliers, dates, referential integrity
drift_detector -> PSI + KS test against the saved baseline
debugger       -> SQL cross-validation and confidence scoring
human_review   -> optional analyst approval step
ranker         -> Business Impact Score ordering
summariser     -> markdown report and fix guidance
```

**Evaluation variants**

```
Baseline   : data_loader -> profiler -> detector -> ranker
Supervisor : baseline + debugger + human_review
Full       : supervisor + drift_detector
```

**Operations**

```
Run Assessment  -> astrion-dq triage
Set Baseline    -> astrion-dq snapshot
Load Demo Data  -> astrion-dq inject
Benchmark       -> astrion-dq evaluate
Export PDF      -> astrion-dq report
```

Set ``ASTRION_API_TOKEN`` to protect the API. Set ``ASTRION_DASHBOARD_TOKEN`` separately if you also want a login prompt on the dashboard.
        """
    )


def _render_investigation_assistant() -> None:
    st.header("Investigation Assistant")
    st.caption(
        "Use the assistant to explain issues, generate DuckDB investigation SQL, "
        "or rewrite failing queries. SQL blocks run automatically against the active dataset."
    )
    if _API_BASE_URL:
        st.caption(
            f"LLM responses are proxied through `{_API_BASE_URL}`. "
            "This Streamlit app does not store the OpenRouter key."
        )

    conn, schema_desc = _get_schema_context()
    if conn is None:
        st.warning(
            "No database is loaded yet. Run **Run Assessment** for the retail warehouse or use "
            "**Ad Hoc Analysis** in Overview to upload CSVs. The assistant can still answer at a high level."
        )
    else:
        with st.expander("Connected schema", expanded=False):
            st.code(schema_desc or "(empty)", language="sql")

    cue_col1, cue_col2, cue_col3 = st.columns(3)
    cue_col1.markdown(
        "<div class='astrion-section-note'><strong>Ask for evidence</strong><br/>"
        "“Show me the rows behind the top-ranked referential-integrity break.”</div>",
        unsafe_allow_html=True,
    )
    cue_col2.markdown(
        "<div class='astrion-section-note'><strong>Ask for remediation</strong><br/>"
        "“Draft DuckDB SQL to quarantine duplicate order rows.”</div>",
        unsafe_allow_html=True,
    )
    cue_col3.markdown(
        "<div class='astrion-section-note'><strong>Ask for explanation</strong><br/>"
        "“Explain why this issue impacts downstream sales reporting.”</div>",
        unsafe_allow_html=True,
    )

    control_col1, control_col2 = st.columns([6, 1])
    with control_col2:
        if st.button("Clear Chat", key="clear_agent"):
            st.session_state["_agent_messages"] = []
            st.rerun()

    if "_agent_messages" not in st.session_state:
        st.session_state["_agent_messages"] = []

    for msg in st.session_state["_agent_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for sql, result in msg.get("sql_results", []):
                st.caption(f"Executed: `{sql[:80]}{'...' if len(sql) > 80 else ''}`")
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result.head(50), use_container_width=True)
                else:
                    st.error(result)

    if user_input := st.chat_input(
        "Ask the assistant to explain an issue, generate DuckDB SQL, or fix a query"
    ):
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state["_agent_messages"].append(
            {"role": "user", "content": user_input, "sql_results": []}
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = _sql_agent_respond(user_input)

            st.markdown(response)

            sql_results = []
            if conn is not None:
                results = _execute_sql_blocks(response, conn)
                for sql, result in results:
                    st.caption(f"Executed: `{sql[:80]}{'...' if len(sql) > 80 else ''}`")
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result.head(50), use_container_width=True)
                    else:
                        st.error(result)
                    sql_results.append((sql, result))

        st.session_state["_agent_messages"].append(
            {"role": "assistant", "content": response, "sql_results": sql_results}
        )


def _render_upload_workspace() -> None:
    st.subheader("Ad Hoc Analysis Workspace")
    st.caption(
        "Upload CSVs for one-off investigations. The workspace runs a verified path "
        "through profiling, detection, SQL cross-validation, ranking, and optional drift checks."
    )

    uploader_version = st.session_state.setdefault("_upload_uploader_version", 0)
    reset_col, help_col = st.columns([1, 2])
    with reset_col:
        if st.button("Clear Upload", key="clear_upload_session", use_container_width=True):
            _reset_upload_session()
            st.rerun()
    with help_col:
        st.caption(
            "Use this when you want to replace the uploaded files or clear the uploaded SQL Assistant context."
        )

    uploaded_files = st.file_uploader(
        "Current dataset CSVs (one file = one table, filename = table name)",
        type=["csv"],
        accept_multiple_files=True,
        key=f"csv_uploader_{uploader_version}",
    )

    enable_upload_drift = st.toggle(
        "Enable drift comparison with a baseline upload",
        key="enable_upload_drift",
        help="Upload a second dataset with matching tables and columns to compare against a trusted baseline.",
    )

    baseline_files = []
    if enable_upload_drift:
        baseline_files = st.file_uploader(
            "Baseline dataset CSVs for drift comparison",
            type=["csv"],
            accept_multiple_files=True,
            key=f"csv_baseline_uploader_{uploader_version}",
        )

    if not uploaded_files:
        if st.session_state.get("_upload_results") is not None:
            st.info("Uploaded analysis results remain available below until you clear the upload session.")
        return

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
        return

    st.markdown(
        f"<div class='astrion-section-note'><strong>{len(tables)} table(s) loaded.</strong> "
        "Review the previews below, then run the ad hoc assessment.</div>",
        unsafe_allow_html=True,
    )

    for table_name, df in tables.items():
        with st.expander(f"`{table_name}` — {len(df):,} rows × {df.shape[1]} columns"):
            st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    baseline_ready = bool(baseline_tables)
    drift_enabled = enable_upload_drift and baseline_ready

    if enable_upload_drift and not baseline_ready:
        st.info("Upload baseline CSVs to enable drift detection. Otherwise the analysis runs without drift.")
    elif drift_enabled:
        matching_tables = sorted(set(tables) & set(baseline_tables))
        st.success(
            f"Drift comparison enabled across {len(matching_tables)} matching table(s): "
            f"{', '.join(matching_tables) if matching_tables else 'none'}."
        )
        if not matching_tables:
            st.warning(
                "No table names match between the current and baseline uploads, so drift signals will stay empty until the names align."
            )

    run_label = "Analyze Upload with Drift Checks" if drift_enabled else "Analyze Upload"
    spinner_label = (
        "Running verified data quality and drift checks on uploaded data..."
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
                _close_upload_conn()
                st.session_state["_upload_conn"] = _build_upload_conn(tables)
                st.session_state["_upload_report_md"] = _generate_upload_report(ranked)
                st.success(f"Ad hoc analysis complete — {len(ranked)} issue(s) found.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    ranked = st.session_state.get("_upload_results")
    if ranked is None:
        return

    st.subheader(f"Uploaded Issues ({len(ranked)} found)")
    if not ranked:
        st.success("No data quality issues detected in the uploaded files.")
    else:
        metrics = _issue_metrics(ranked)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Issues", metrics["total"])
        m2.metric("Critical", metrics["high"])
        m3.metric("Medium", metrics["medium"])
        m4.metric("Avg BIS", f"{metrics['avg_score']:.4f}")
        m5.metric("Needs Review", metrics["needs_review"])

        st.dataframe(
            _issue_dataframe(ranked),
            use_container_width=True,
            hide_index=True,
        )
        _render_issue_distribution_charts(ranked, "upload")

    st.divider()
    st.subheader("Upload Resolution Report")
    report_md = st.session_state.get("_upload_report_md", "")
    if report_md:
        st.markdown(report_md)
        st.download_button(
            label="Download upload report (.md)",
            data=report_md.encode("utf-8"),
            file_name="upload_triage_report.md",
            mime="text/markdown",
            key="download_upload_report",
        )

    st.info(
        "The uploaded tables are now available in **Investigation**. Use the assistant to inspect or remediate the uploaded issues."
    )


def _render_no_assessment_message() -> None:
    st.info(
        "No assessment output is available yet. Use **Run Assessment** in the sidebar. "
        "If you need seeded demo issues first, open **Admin & Demo Tools** and run **Load Demo Data**."
    )


_inject_dashboard_styles()

from astrion_dq.config import SNAPSHOTS_DIR

snapshot_path = SNAPSHOTS_DIR / "snapshot_baseline.json"
source_labels = {
    "injected": "Injected Scenario",
    "clean": "Clean Baseline",
}

with st.sidebar:
    run_entries = load_run_log()
    latest_run = run_entries[0] if run_entries else {}
    latest_run_label = _format_run_timestamp(latest_run.get("timestamp", ""))
    backend_status = load_backend_status(st.session_state.get("src", "injected"))
    baseline_ready = bool(
        backend_status.get("baseline_snapshot", False)
        if _API_BASE_URL else
        snapshot_path.exists()
    )
    api_mode_label = "Connected" if backend_status.get("api_connected") else ("Local" if not _API_BASE_URL else "Unavailable")

    st.title("Astrion DQ")
    st.caption("Enterprise Data Quality Command Center")
    st.markdown(
        "<div class='astrion-sidebar-note'>"
        "<strong>Focus mode</strong><br/>"
        "Daily operators should run the assessment, review priority issues, investigate root causes, and export reports. "
        "Demo and benchmark actions are grouped separately below."
        "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Operational Context")
    data_source = st.selectbox("Active dataset", ["injected", "clean"], key="src")
    backend_status = load_backend_status(data_source)
    baseline_ready = bool(
        backend_status.get("baseline_snapshot", False)
        if _API_BASE_URL else
        snapshot_path.exists()
    )
    api_mode_label = "Connected" if backend_status.get("api_connected") else ("Local" if not _API_BASE_URL else "Unavailable")
    st.caption(f"Last recorded run: {latest_run_label}")
    st.caption(f"Baseline snapshot: {'Ready' if baseline_ready else 'Not set'}")
    st.caption(f"API mode: {api_mode_label}")
    if _API_BASE_URL and not backend_status.get("api_connected"):
        st.caption(f"API error: {backend_status.get('error', 'Unreachable')}")

    _running = st.session_state.get("_running", False)
    if st.button("Run Assessment", use_container_width=True, type="primary", disabled=_running):
        st.session_state["_running"] = True
        with st.spinner(f"Running assessment on {data_source!r}..."):
            if _API_BASE_URL:
                ok, out = _run_remote_triage(data_source)
            else:
                ok, out = run_cli(["triage", "--source", data_source, "--sensitivity", "high"])
        st.session_state["_running"] = False
        if ok:
            _ensure_retail_live_connection(data_source)
            st.success("Assessment complete.")
            st.cache_data.clear()
        else:
            st.error(f"Failed:\n{out[:400]}")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Refresh Data", use_container_width=True, disabled=_running, key="refresh_dashboard"):
            st.cache_data.clear()
            st.rerun()
    with action_col2:
        if st.button("Export PDF", use_container_width=True, disabled=_running, key="export_dashboard_pdf"):
            if _API_BASE_URL:
                try:
                    pdf_bytes = _api_bytes_request(f"/triage/report.pdf?source={data_source}", timeout=180)
                    st.session_state["_report_pdf_bytes"] = pdf_bytes
                    st.session_state["_report_pdf_name"] = f"triage_report_{data_source}.pdf"
                    st.success("PDF report is ready for download below.")
                except Exception as exc:
                    st.error(f"Failed:\n{str(exc)[:400]}")
            else:
                st.session_state["_running"] = True
                with st.spinner("Preparing PDF report..."):
                    ok, out = run_cli(["report", "--source", data_source])
                st.session_state["_running"] = False
                if ok:
                    st.success("PDF report prepared.")
                    st.cache_data.clear()
                else:
                    st.error(f"Failed:\n{out[:400]}")

    if st.session_state.get("_report_pdf_bytes"):
        st.download_button(
            "Download Current PDF",
            data=st.session_state["_report_pdf_bytes"],
            file_name=st.session_state.get("_report_pdf_name", f"triage_report_{data_source}.pdf"),
            mime="application/pdf",
            use_container_width=True,
            key="sidebar_download_pdf",
        )

    with st.expander("Admin & Demo Tools", expanded=False):
        st.caption(
            "Use these actions when preparing a demonstration, resetting the drift baseline, "
            "or comparing workflow variants."
        )
        if st.button("Set Baseline", use_container_width=True, disabled=_running, key="set_baseline"):
            st.session_state["_running"] = True
            with st.spinner("Saving baseline snapshot..."):
                if _API_BASE_URL:
                    ok, out = _run_remote_operation("/snapshot", {"tag": "baseline"})
                else:
                    ok, out = run_cli(["snapshot"])
            st.session_state["_running"] = False
            if ok:
                st.success("Baseline updated.")
                st.cache_data.clear()
            else:
                st.error(f"Failed:\n{out[:400]}")

        if st.button("Load Demo Data", use_container_width=True, disabled=_running, key="load_demo_data"):
            st.session_state["_running"] = True
            with st.spinner("Injecting synthetic demo issues..."):
                if _API_BASE_URL:
                    ok, out = _run_remote_operation("/inject", {"seed": 42})
                else:
                    ok, out = run_cli(["inject"])
            st.session_state["_running"] = False
            if ok:
                st.success("Demo scenario loaded.")
                st.cache_data.clear()
            else:
                st.error(f"Failed:\n{out[:400]}")

        if st.button("Benchmark Engine", use_container_width=True, disabled=_running, key="benchmark_engine"):
            st.session_state["_running"] = True
            with st.spinner("Evaluating workflow strategies..."):
                if _API_BASE_URL:
                    ok, out = _run_remote_operation("/evaluate", {"source": "injected"})
                else:
                    ok, out = run_cli(["evaluate", "--source", "injected"])
            st.session_state["_running"] = False
            if ok:
                st.success("Benchmark complete.")
                st.cache_data.clear()
            else:
                st.error(f"Failed:\n{out[:400]}")

    st.divider()
    st.caption("v0.6.0 · Astrion Capstone 7")


issues = load_ranked_issues(data_source)
eval_data = load_evaluation()
report_text = load_report_md(data_source)
run_entries = load_run_log()
latest_run_for_source = next((entry for entry in run_entries if entry.get("source") == data_source), {})
latest_run_label = _format_run_timestamp(latest_run_for_source.get("timestamp", ""))
metrics = _issue_metrics(issues) if issues else None
backend_status = load_backend_status(data_source)
baseline_ready = bool(
    backend_status.get("baseline_snapshot", False)
    if _API_BASE_URL else
    snapshot_path.exists()
)
api_mode_label = "Connected" if backend_status.get("api_connected") else ("Local" if not _API_BASE_URL else "Unavailable")
summary_text = ""
summary_error = ""
if _API_BASE_URL and issues:
    summary_result = _fetch_remote_summary(
        data_source,
        json.dumps(issues[:25], sort_keys=True),
    )
    summary_text = str(summary_result.get("summary") or "").strip()
    summary_error = str(summary_result.get("error") or "").strip()

status_pills = [
    f"Dataset: {source_labels.get(data_source, data_source.title())}",
    f"Baseline: {'Ready' if baseline_ready else 'Not Set'}",
    f"API: {api_mode_label}",
    f"Last Run: {latest_run_label}",
]
if st.session_state.get("_upload_results") is not None:
    status_pills.append("Ad Hoc Upload: Active")

st.markdown(
    (
        "<section class='astrion-hero'>"
        "<div class='astrion-kicker'>Operational Data Quality</div>"
        "<h1 class='astrion-title'>Enterprise Retail Data Health Workspace</h1>"
        "<p class='astrion-subtitle'>Monitor business-critical data quality signals, "
        "prioritise remediation by impact, investigate failures with SQL, and export a review-ready report without exposing the underlying engineering controls by default.</p>"
        "<div class='astrion-pill-row'>"
        + "".join(f"<span class='astrion-pill'>{pill}</span>" for pill in status_pills)
        + "</div></section>"
    ),
    unsafe_allow_html=True,
)

(
    tab_overview,
    tab_issues,
    tab_investigation,
    tab_reports,
    tab_audit,
) = st.tabs([
    "Overview",
    "Issues",
    "Investigation",
    "Reports",
    "Audit",
])

with tab_overview:
    st.header("Operational Overview")
    st.caption(
        "Start here to assess overall data health, current priority queues, and whether the baseline and reporting outputs are ready."
    )

    if not issues:
        _render_no_assessment_message()
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Issues", metrics["total"])
        m2.metric("Critical Issues", metrics["high"])
        m3.metric("Impacted Reports", metrics["impacted_reports"])
        m4.metric("Needs Review", metrics["needs_review"])
        m5.metric("Average Confidence", f"{metrics['avg_conf']:.2f}")

        st.markdown(
            "<div class='astrion-section-note'><strong>Priority Queue</strong><br/>"
            "Review the highest-ranked issues first. Critical referential-integrity breaks and high-BIS null or duplication issues are the fastest route to business risk reduction."
            "</div>",
            unsafe_allow_html=True,
        )

        top_issue_cols = st.columns([1.4, 1])
        with top_issue_cols[0]:
            st.subheader("Top Priority Issues")
            st.dataframe(
                _issue_dataframe(issues, limit=10),
                use_container_width=True,
                hide_index=True,
            )
        with top_issue_cols[1]:
            st.subheader("Current Operating Notes")
            st.markdown(
                "\n".join(
                    [
                        f"- Active dataset: `{source_labels.get(data_source, data_source.title())}`",
                        f"- Last run completed: `{latest_run_label}`",
                        f"- Baseline status: `{'Ready' if snapshot_path.exists() else 'Not Set'}`",
                        f"- Executive summary: `{'Available' if summary_text else 'Pending'}`",
                        f"- Upload workspace: `{'Active' if st.session_state.get('_upload_results') is not None else 'Idle'}`",
                    ]
                )
            )

        st.divider()
        st.subheader("Impact Distribution")
        _render_issue_distribution_charts(issues, "overview")

    with st.expander(
        "Ad Hoc Analysis Workspace",
        expanded=bool(st.session_state.get("_upload_results")),
    ):
        _render_upload_workspace()

with tab_issues:
    st.header("Priority Issues")
    st.caption(
        "Filter the current issue queue by severity, failure mode, and source table. Use this view for analyst triage and remediation planning."
    )

    if not issues:
        _render_no_assessment_message()
    else:
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            severity_filter = st.multiselect(
                "Severity",
                ["high", "medium", "low"],
                default=["high", "medium"],
                key="issues_severity_filter",
            )
        with filter_col2:
            type_filter = st.multiselect(
                "Failure mode",
                sorted({issue.get("issue_type", "") for issue in issues}),
                default=[],
                key="issues_type_filter",
            )
        with filter_col3:
            table_filter = st.multiselect(
                "Table",
                sorted({issue.get("table", "") for issue in issues}),
                default=[],
                key="issues_table_filter",
            )

        filtered = [
            issue for issue in issues
            if issue.get("severity") in severity_filter
            and (not type_filter or issue.get("issue_type") in type_filter)
            and (not table_filter or issue.get("table") in table_filter)
        ]

        st.dataframe(
            _issue_dataframe(filtered),
            use_container_width=True,
            hide_index=True,
        )

        detail_col1, detail_col2 = st.columns([1.4, 1])
        with detail_col1:
            low_conf = [issue for issue in filtered if issue.get("confidence", 1.0) < 0.70]
            st.subheader("Review Queue")
            if low_conf:
                st.dataframe(
                    _issue_dataframe(low_conf, limit=15),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("No current issues fall below the manual-review confidence threshold.")
        with detail_col2:
            st.subheader("Impacted Business Reports")
            impacted_rows = []
            for issue in filtered[:20]:
                for report in issue.get("affected_reports") or []:
                    impacted_rows.append({
                        "Issue ID": issue.get("issue_id", ""),
                        "Report": report,
                        "Table": issue.get("table", ""),
                        "Severity": issue.get("severity", ""),
                    })
            if impacted_rows:
                st.dataframe(pd.DataFrame(impacted_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No impacted-report mapping is available for the current filter set.")

with tab_investigation:
    _render_investigation_assistant()

with tab_reports:
    st.header("Executive Reports")
    st.caption(
        "Use this view for business communication, exported documents, and model benchmark outputs used in technical reviews."
    )

    download_col1, download_col2 = st.columns([1, 1])
    pdf_path = OUTPUTS_DIR / f"triage_report_{data_source}.pdf"
    if report_text:
        with download_col1:
            st.download_button(
                "Download Markdown",
                data=report_text.encode("utf-8"),
                file_name=f"triage_report_{data_source}.md",
                mime="text/markdown",
                key="download_retail_report_md",
            )
    if pdf_path.exists():
        with download_col2:
            st.download_button(
                "Download PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                key="download_retail_report_pdf",
            )

    if summary_text:
        st.subheader("AI Executive Summary")
        st.markdown(summary_text)
        st.divider()
    elif summary_error:
        st.caption(f"Remote AI summary unavailable: {summary_error}")

    if report_text is None and not summary_text:
        st.info("No retail report is available yet. Run **Run Assessment** and then **Export PDF**.")
    elif report_text is not None:
        st.subheader("Detailed Report")
        st.markdown(report_text)

    with st.expander("Benchmark Engine", expanded=False):
        _render_strategy_benchmark(eval_data)

with tab_audit:
    st.header("Audit Trail")
    st.caption(
        "Review recent operational runs, their agent traces, and the platform reference notes used during technical reviews."
    )

    if not run_entries:
        st.info("No runs recorded yet. Run an assessment to seed the audit trail.")
    else:
        audit_rows = []
        for entry in run_entries:
            audit_rows.append({
                "Run ID": entry.get("run_id", ""),
                "Source": entry.get("source", ""),
                "Timestamp (UTC)": entry.get("timestamp", ""),
                "Issues Ranked": entry.get("issue_count", 0),
                "Agent Trace": " -> ".join(entry.get("agent_trace") or []),
            })
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(audit_rows)} run(s), most recent first.")

    with st.expander("Platform Architecture", expanded=False):
        _render_architecture_reference()
