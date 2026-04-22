"""Astrion DQ -- FastAPI REST layer.

Endpoints:
  GET  /health                -- unauthenticated liveness probe
  POST /triage                -- submit async triage job, returns job_id
  GET  /jobs/{job_id}         -- poll job status and result
  GET  /runs/{run_id}         -- look up a past run from outputs/run_log.jsonl
  GET  /triage/report.pdf     -- generate and download a PDF report

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
from pydantic import BaseModel, field_validator
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
    version="0.6.0",
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
