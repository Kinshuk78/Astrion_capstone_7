"""Astrion DQ -- FastAPI REST layer.

Endpoints:
  GET  /health                -- unauthenticated liveness probe
  POST /triage                -- submit async triage job, returns job_id
  GET  /jobs/{job_id}         -- poll job status and result
  GET  /runs/{run_id}         -- look up a past run from outputs/run_log.jsonl
  GET  /triage/report.pdf     -- generate and download a PDF report
  POST /assistant/chat        -- SQL assistant reply via the API-held LLM key
  POST /assistant/summary     -- executive summary via the API-held LLM key

Authentication:
  Bearer token read from ASTRION_API_TOKEN env var.
  When ASTRION_API_TOKEN is unset or empty, auth is disabled (dev mode).
  Wrong or missing token returns HTTP 401.

Rate limiting:
  /triage is limited to 10 requests per minute per IP via slowapi.
  Override by setting RATE_LIMIT env var (e.g. "20/minute").

Async triage:
  POST /triage returns immediately with {"job_id": "...", "status": "running"}.
  Poll GET /jobs/{job_id} until "status" is "done" or "error".
  Only one triage job runs at a time (DuckDB singleton safety via threading.Lock).
  Wall time is typically 5-20 seconds.

Outputs persistence:
  Run logs are written to outputs/run_log.jsonl. On Render's ephemeral filesystem
  the write is wrapped in a try/except — a warning is logged and the response is
  still returned so the caller is not affected.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[_RATE_LIMIT])

app = FastAPI(
    title="Astrion DQ API",
    description=(
        "Agentic retail data quality triage via LangGraph. "
        "POST /triage submits an async job; poll GET /jobs/{job_id} for results. "
        "Interactive docs: /docs — OpenAPI spec: /openapi.json"
    ),
    version="0.6.1",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def _require_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> None:
    """Dependency that enforces Bearer token auth.

    When ASTRION_API_TOKEN is unset the check is skipped (dev / test mode).
    """
    expected = os.getenv("ASTRION_API_TOKEN", "")
    if not expected:
        return
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Job store + thread-safety
# ---------------------------------------------------------------------------

# Only one triage job runs at a time — DuckDB uses a module-level singleton
# connection that is not safe to share across concurrent runs.
_triage_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="triage")

# In-memory job store: job_id -> {status, result, error, submitted_at, completed_at}
# On Render's free tier this is ephemeral (resets on restart); good enough for demo.
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TriageRequest(BaseModel):
    source: str = "injected"
    # sensitivity is always "high" — maximises issue detection coverage.
    # The field is retained for backwards compatibility but has no effect.

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if v not in ("clean", "injected"):
            raise ValueError("source must be 'clean' or 'injected'")
        return v


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str


class TriageResult(BaseModel):
    run_id: str
    source: str
    issue_count: int
    ranked_issues: List[dict]
    agent_trace: List[str]


class AssistantMessage(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class AssistantChatRequest(BaseModel):
    message: str
    history: List[AssistantMessage] = Field(default_factory=list)
    schema_desc: str = ""
    issues: List[dict] = Field(default_factory=list)
    max_tokens: int = 1200

    @field_validator("message")
    @classmethod
    def _validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("message must not be empty")
        return v.strip()


class AssistantSummaryRequest(BaseModel):
    issues: List[dict] = Field(default_factory=list)
    source: str = "injected"


# ---------------------------------------------------------------------------
# Background triage worker
# ---------------------------------------------------------------------------

def _write_run_log(entry: dict) -> None:
    """Append run entry to run_log.jsonl. Silently skips on filesystem errors
    (e.g., Render ephemeral disk full or read-only after restart)."""
    try:
        from astrion_dq.config import OUTPUTS_DIR
        log_path = OUTPUTS_DIR / "run_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Could not write run_log.jsonl (ephemeral filesystem?): %s", exc)


def _execute_triage(source: str) -> TriageResult:
    """Run the full LangGraph pipeline. Called inside a background thread."""
    from astrion_dq.graph.state import initial_state
    from astrion_dq.graph.workflow import build_graph
    from astrion_dq.warehouse.loader import close_connection

    os.environ["ASTRION_AUTO_APPROVE"] = "1"
    graph = build_graph()
    state = initial_state(source=source, sensitivity="high")
    config = {"configurable": {"thread_id": f"api_{uuid.uuid4().hex[:8]}"}}

    try:
        result = graph.invoke(state, config=config)
    finally:
        os.environ.pop("ASTRION_AUTO_APPROVE", None)
        close_connection()

    ranked = result.get("ranked_issues") or []
    agent_trace = result.get("agent_trace") or []
    run_id = uuid.uuid4().hex[:12]

    _write_run_log({
        "run_id": run_id,
        "source": source,
        "sensitivity": "high",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "issue_count": len(ranked),
        "agent_trace": agent_trace,
    })

    return TriageResult(
        run_id=run_id,
        source=source,
        issue_count=len(ranked),
        ranked_issues=ranked,
        agent_trace=agent_trace,
    )


def _run_triage_job(job_id: str, source: str) -> None:
    """Background worker: acquires the DuckDB lock, runs triage, stores result."""
    with _triage_lock:
        try:
            result = _execute_triage(source)
            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "done",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "result": result.model_dump(),
                })
        except Exception as exc:
            logger.exception("Triage job %s failed: %s", job_id, exc)
            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "error",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Unauthenticated liveness probe — used by Render health checks."""
    return {"status": "ok"}


@app.post("/triage", response_model=JobSubmitResponse, status_code=202)
@limiter.limit(_RATE_LIMIT)
def triage_submit(
    request: Request,
    req: TriageRequest,
    _auth: None = Depends(_require_token),
) -> JobSubmitResponse:
    """Submit a triage job.

    Returns immediately with a ``job_id``. Poll ``GET /jobs/{job_id}`` until
    ``status`` is ``"done"`` or ``"error"``. Typical completion: 5-20 seconds.

    Only one job runs at a time (DuckDB singleton safety). Subsequent submissions
    queue behind the running job.
    """
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "source": req.source,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
    _executor.submit(_run_triage_job, job_id, req.source)
    return JobSubmitResponse(
        job_id=job_id,
        status="running",
        poll_url=f"/jobs/{job_id}",
    )


@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    _auth: None = Depends(_require_token),
) -> dict:
    """Poll a triage job.

    Returns the job dict with ``status`` = ``"running"`` | ``"done"`` | ``"error"``.
    When ``"done"``, the ``result`` key contains the full triage result.
    When ``"error"``, the ``error`` key contains the exception message.
    """
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/runs/{run_id}")
def get_run(
    run_id: str,
    _auth: None = Depends(_require_token),
) -> dict:
    """Return the log entry for a past triage run from run_log.jsonl.

    Returns 404 if the run_id is not recorded or if the log file does not exist
    (e.g., first run on an ephemeral filesystem).
    """
    from astrion_dq.config import OUTPUTS_DIR

    log_path = OUTPUTS_DIR / "run_log.jsonl"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("run_id") == run_id:
                return entry

    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")


# ---------------------------------------------------------------------------
# LLM assistant helpers
# ---------------------------------------------------------------------------

def _assistant_issue_context(issues: List[dict], limit: int = 10) -> str:
    lines: list[str] = []
    for issue in issues[:limit]:
        lines.append(
            f"  [{issue.get('issue_id', '')}] {issue.get('issue_type', '')} "
            f"in {issue.get('table', '')} — BIS={issue.get('impact_score', 0):.3f}, "
            f"severity={issue.get('severity', '')}, "
            f"evidence_rows={issue.get('evidence_rows', 0)}"
        )
    return "\n".join(lines) if lines else "  (no issues loaded yet)"


def _assistant_system_prompt(schema_desc: str, issues_text: str) -> str:
    return f"""You are an expert data engineer assistant specialised in DuckDB and retail data warehouse quality.

You have access to a live DuckDB database with these tables:
{schema_desc}

Current data quality issues (ranked by Business Impact Score):
{issues_text}

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


def _assistant_fallback_response(req: AssistantChatRequest) -> str:
    top_issues = req.issues[:3]
    issue_lines = [
        f"- `{issue.get('issue_type', 'unknown')}` in `{issue.get('table', 'unknown')}` "
        f"(severity={issue.get('severity', 'unknown')}, BIS={issue.get('impact_score', 0):.3f})"
        for issue in top_issues
    ]
    lines = [
        "The remote assistant is running without an OpenRouter key, so this reply is template-only.",
        "",
        "You can still inspect the current context:",
        f"- Schema provided: {'yes' if req.schema_desc.strip() else 'no'}",
        f"- Ranked issues provided: {len(req.issues)}",
    ]
    if issue_lines:
        lines.extend(["", "Top issues in context:"])
        lines.extend(issue_lines)
    lines.extend(
        [
            "",
            "Ask a narrower question such as:",
            "- `Show me DuckDB SQL to inspect the rows behind the top ranked issue.`",
            "- `Explain why this foreign-key break is happening.`",
            "- `Rewrite this failing DuckDB query: ...`",
        ]
    )
    return "\n".join(lines)


def _assistant_summary_fallback(issues: List[dict], source: str) -> str:
    if not issues:
        return f"No ranked issues were supplied for the {source} run, so there is nothing to summarise."

    high_count = sum(1 for issue in issues if issue.get("severity") == "high")
    top = issues[:3]
    top_tables = ", ".join(
        dict.fromkeys(issue.get("table", "unknown") for issue in top if issue.get("table"))
    ) or "the loaded tables"
    top_types = ", ".join(issue.get("issue_type", "unknown") for issue in top) or "the ranked issues"

    lines = [
        (
            f"The {source} run surfaced {len(issues)} ranked data quality issues, "
            f"including {high_count} high-severity items. The highest-risk findings are "
            f"clustered in {top_tables}, with the top problems driven by {top_types}."
        ),
        (
            "The immediate business risk is inaccurate downstream reporting and slower analyst "
            "triage while root causes remain unresolved."
        ),
        "",
        "Recommended Actions",
        "- Investigate the highest-ranked issue first and verify the affected upstream ETL step.",
        "- Re-run the relevant validation query after the fix to confirm row-level recovery.",
        "- Add a targeted quality check for the same failure mode to catch future regressions earlier.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent Layer endpoints
# POST /analyze          -- triage + agent explanations + report (async job)
# POST /explain          -- business explanations for pre-computed issues
# POST /prioritise       -- AI-ranked priority list with justifications
# POST /generate-fix     -- SQL + Python fix code for a single issue
# POST /report           -- executive business summary report
# POST /assistant/chat   -- SQL assistant response via OpenRouter
# POST /assistant/summary -- executive summary via OpenRouter
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    issues: List[dict]
    table_sizes: dict = {}


class PrioritiseRequest(BaseModel):
    issues: List[dict]
    table_sizes: dict = {}


class GenerateFixRequest(BaseModel):
    issue: dict
    table_schema: List[dict] = []


class ReportRequest(BaseModel):
    issues: List[dict]
    run_id: str = ""
    table_sizes: dict = {}


class AnalyzeRequest(BaseModel):
    source: str = "injected"

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if v not in ("clean", "injected"):
            raise ValueError("source must be 'clean' or 'injected'")
        return v


@app.post("/assistant/chat")
def assistant_chat(
    req: AssistantChatRequest,
    _auth: None = Depends(_require_token),
) -> dict:
    """Return a SQL-assistant chat reply using the API-held OpenRouter key."""
    from astrion_dq.llm.client import LLMUnavailable, chat_with_history

    history = [msg.model_dump() for msg in req.history]
    messages = history + [{"role": "user", "content": req.message}]
    system = _assistant_system_prompt(
        req.schema_desc or "  (no database loaded)",
        _assistant_issue_context(req.issues),
    )

    try:
        response = chat_with_history(
            messages,
            system=system,
            max_tokens=max(128, min(req.max_tokens, 1600)),
        )
        return {"response": response, "used_fallback": False}
    except LLMUnavailable:
        return {"response": _assistant_fallback_response(req), "used_fallback": True}
    except Exception as exc:
        text = str(exc)
        if "402" in text and (
            "credits" in text.lower() or "can only afford" in text.lower()
        ):
            return {
                "response": (
                    f"**LLM call failed**: {exc}\n\n"
                    "OpenRouter accepted the API key but rejected the request because the "
                    "remaining credit/token budget is too low for this response. "
                    "Add credits or ask a shorter question."
                ),
                "used_fallback": True,
            }
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc


@app.post("/assistant/summary")
def assistant_summary(
    req: AssistantSummaryRequest,
    _auth: None = Depends(_require_token),
) -> dict:
    """Return an executive summary using the API-held OpenRouter key."""
    from astrion_dq.graph.nodes import _llm_executive_summary

    summary = _llm_executive_summary(req.issues, req.source)
    if summary:
        return {"summary": summary, "used_fallback": False}
    return {
        "summary": _assistant_summary_fallback(req.issues, req.source),
        "used_fallback": True,
    }


def _run_analyze_job(job_id: str, source: str) -> None:
    """Background worker: run triage then agent explain + report."""
    with _triage_lock:
        try:
            triage_result = _execute_triage(source)
            issues = triage_result.ranked_issues

            from astrion_dq.agent import explain_issues, generate_report
            from astrion_dq.agent.key_manager import APIKeyManager

            key_manager = APIKeyManager.from_env()
            explanations = explain_issues(issues, key_manager=key_manager)
            report = generate_report(
                issues,
                run_id=triage_result.run_id,
                key_manager=key_manager,
            )

            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "done",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "result": {
                        **triage_result.model_dump(),
                        "explanations": explanations,
                        "report": report,
                    },
                })
        except Exception as exc:
            logger.exception("Analyze job %s failed: %s", job_id, exc)
            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "error",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                })


@app.post("/analyze", response_model=JobSubmitResponse, status_code=202)
@limiter.limit(_RATE_LIMIT)
def analyze_submit(
    request: Request,
    req: AnalyzeRequest,
    _auth: None = Depends(_require_token),
) -> JobSubmitResponse:
    """Submit a full analysis job (triage + AI explanations + executive report).

    Returns 202 with a job_id immediately. Poll ``GET /jobs/{job_id}`` for
    completion. The result includes ranked_issues, explanations, and report.
    Falls back to deterministic output when no OpenAI keys are configured.
    """
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "source": req.source,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
    _executor.submit(_run_analyze_job, job_id, req.source)
    return JobSubmitResponse(
        job_id=job_id,
        status="running",
        poll_url=f"/jobs/{job_id}",
    )


@app.post("/explain")
def explain(
    req: ExplainRequest,
    _auth: None = Depends(_require_token),
) -> dict:
    """Return business-language explanations for a pre-computed issue list.

    Sends structured summaries (not raw data) to the OpenAI API.
    Falls back to deterministic text when no API keys are configured.
    All outputs are validated for schema compliance before being returned.
    """
    if not req.issues:
        raise HTTPException(status_code=422, detail="issues list must not be empty.")

    from astrion_dq.agent import explain_issues
    from astrion_dq.agent.key_manager import APIKeyManager

    explanations = explain_issues(
        req.issues,
        table_sizes=req.table_sizes or None,
        key_manager=APIKeyManager.from_env(),
    )
    used_fallback = any(e.get("source") == "fallback" for e in explanations)
    return {"explanations": explanations, "used_fallback": used_fallback}


@app.post("/prioritise")
def prioritise(
    req: PrioritiseRequest,
    _auth: None = Depends(_require_token),
) -> dict:
    """Return a priority-ranked issue list with AI-generated justifications.

    Falls back to BIS-sorted deterministic ranking when no API keys are
    configured or the LLM response fails validation.
    """
    if not req.issues:
        raise HTTPException(status_code=422, detail="issues list must not be empty.")

    from astrion_dq.agent import prioritise_issues
    from astrion_dq.agent.key_manager import APIKeyManager

    prioritised = prioritise_issues(
        req.issues,
        table_sizes=req.table_sizes or None,
        key_manager=APIKeyManager.from_env(),
    )
    used_fallback = any(p.get("source") == "fallback" for p in prioritised)
    return {"prioritised_issues": prioritised, "used_fallback": used_fallback}


@app.post("/generate-fix")
def generate_fix(
    req: GenerateFixRequest,
    _auth: None = Depends(_require_token),
) -> dict:
    """Generate SQL and Python fix code for a single data quality issue.

    Uses only columns and tables listed in the provided schema or inferred
    from the issue dict. Post-generation validation rejects code that
    references columns not in the schema and falls back to templates.
    """
    if not req.issue:
        raise HTTPException(status_code=422, detail="issue must not be empty.")

    from astrion_dq.agent import generate_fix_code
    from astrion_dq.agent.key_manager import APIKeyManager

    fix = generate_fix_code(
        req.issue,
        extra_schema=req.table_schema or None,
        key_manager=APIKeyManager.from_env(),
    )
    return fix


@app.post("/report")
def report(
    req: ReportRequest,
    _auth: None = Depends(_require_token),
) -> dict:
    """Generate an executive business summary report for a set of ranked issues.

    The report covers: overall data health, top risks, and recommended actions.
    Fallback returns a deterministic report when the LLM is unavailable.
    """
    if not req.issues:
        from astrion_dq.agent.fallback import fallback_report
        return fallback_report([], req.run_id)

    from astrion_dq.agent import generate_report
    from astrion_dq.agent.key_manager import APIKeyManager

    return generate_report(
        req.issues,
        run_id=req.run_id,
        table_sizes=req.table_sizes or None,
        key_manager=APIKeyManager.from_env(),
    )


@app.get("/triage/report.pdf")
@limiter.limit("5/minute")
def triage_report_pdf(
    request: Request,
    source: str = "injected",
    _auth: None = Depends(_require_token),
) -> StreamingResponse:
    """Generate and stream a PDF triage report.

    Reads the latest ``ranked_issues_{source}.json`` and
    ``evaluation_comparison.json`` from outputs/, generates the PDF in memory
    (no disk write required), and returns it as an inline download.

    Query param:
        source: ``clean`` | ``injected`` (default: injected)
    """
    from astrion_dq.config import OUTPUTS_DIR
    from astrion_dq.report.pdf import generate_triage_report_bytes

    if source not in ("clean", "injected"):
        raise HTTPException(status_code=422, detail="source must be 'clean' or 'injected'")

    ranked: list = []
    ranked_path = OUTPUTS_DIR / f"ranked_issues_{source}.json"
    if ranked_path.exists():
        try:
            ranked = json.loads(ranked_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    metrics: list = []
    eval_path = OUTPUTS_DIR / "evaluation_comparison.json"
    if eval_path.exists():
        try:
            metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    pdf_bytes = generate_triage_report_bytes(ranked, metrics_list=metrics)

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="triage_report_{source}.pdf"'},
    )
