"""Astrion DQ — Command-Line Interface

Commands:
  inject     — inject synthetic quality issues into the retail dataset (ground truth)
  snapshot   — save baseline drift statistics snapshot
  triage     — run the full LangGraph triage workflow
  evaluate   — compare strategies A, B, C against ground truth
  report     — generate PDF report from latest outputs
  dashboard  — launch the Streamlit analytics dashboard
  serve      — launch the FastAPI REST server
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import typer


class Source(str, Enum):
    clean = "clean"
    injected = "injected"


class Sensitivity(str, Enum):
    normal = "normal"
    high = "high"

app = typer.Typer(
    help="Astrion DQ — Enterprise agentic data quality triage.",
    add_completion=False,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def inject(
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible injection."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Inject synthetic data quality issues into the retail dataset (creates ground truth)."""
    _setup_logging(verbose)
    from astrion_dq.injectors.retail_issues import inject_retail_issues
    from astrion_dq.warehouse.loader import load_retail_tables

    tables = load_retail_tables(source="clean")
    _, issues = inject_retail_issues(tables, seed=seed)
    typer.echo(f"Injected {len(issues)} synthetic issues (seed={seed}).")
    typer.echo("Output: data/injected/retail/, outputs/retail_injected_issues.json")


@app.command()
def snapshot(
    tag: str = typer.Option("baseline", "--tag", help="Snapshot tag name."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Save a baseline drift statistics snapshot for future drift detection."""
    _setup_logging(verbose)
    from astrion_dq.checks.drift import save_snapshot
    from astrion_dq.warehouse.loader import load_retail_tables

    tables = load_retail_tables(source="clean")
    path = save_snapshot(tables, tag=tag)
    typer.echo(f"Drift snapshot saved: {path}")


@app.command()
def triage(
    source: Source = typer.Option(Source.injected, "--source", "-s", help="Data source: clean | injected"),
    sensitivity: Sensitivity = typer.Option(Sensitivity.high, "--sensitivity", help="Sensitivity: normal | high"),
    auto_approve: bool = typer.Option(
        True, "--auto-approve/--interactive",
        help="Auto-approve low-confidence issues without prompting.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the full LangGraph triage workflow (data_loader → profiler → ... → summariser)."""
    _setup_logging(verbose)

    if auto_approve:
        os.environ["ASTRION_AUTO_APPROVE"] = "1"

    from astrion_dq.config import OUTPUTS_DIR
    from astrion_dq.graph.state import initial_state
    from astrion_dq.graph.workflow import build_graph
    from astrion_dq.warehouse.loader import close_connection

    src = source.value if hasattr(source, "value") else source
    typer.echo(f"Running triage on {src!r} data (sensitivity={sensitivity.value if hasattr(sensitivity, 'value') else sensitivity}) ...")
    graph = build_graph()
    state = initial_state(source=src, sensitivity=sensitivity.value if hasattr(sensitivity, "value") else sensitivity)
    config = {"configurable": {"thread_id": "triage"}}
    result = graph.invoke(state, config=config)
    close_connection()

    ranked = result.get("ranked_issues") or []
    typer.echo(f"\nTriage complete: {len(ranked)} issue(s) ranked.")
    typer.echo(f"Agent trace: {' -> '.join(result.get('agent_trace') or [])}")

    if result.get("report_md"):
        out = OUTPUTS_DIR / f"triage_report_{src}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result["report_md"], encoding="utf-8")
        typer.echo(f"Report (markdown): {out}")

    ranked_out = OUTPUTS_DIR / f"ranked_issues_{src}.json"
    ranked_out.parent.mkdir(parents=True, exist_ok=True)
    ranked_out.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    typer.echo(f"Ranked issues ({len(ranked)}): {ranked_out}")

    run_id = uuid.uuid4().hex[:12]
    sens = sensitivity.value if hasattr(sensitivity, "value") else sensitivity
    log_entry = {
        "run_id": run_id,
        "source": src,
        "sensitivity": sens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "issue_count": len(ranked),
        "agent_trace": result.get("agent_trace") or [],
    }
    log_path = OUTPUTS_DIR / "run_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry) + "\n")
    typer.echo(f"Run logged: {run_id}")


@app.command()
def evaluate(
    source: Source = typer.Option(Source.injected, "--source", "-s",
                                   help="Data source (use 'injected' for ground-truth comparison)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Evaluate strategies A (baseline), B (supervisor), C (full) against injected ground truth."""
    _setup_logging(verbose)
    os.environ["ASTRION_AUTO_APPROVE"] = "1"

    from astrion_dq.evaluation.metrics import evaluate_all

    src = source.value if hasattr(source, "value") else source
    typer.echo(f"Evaluating strategies A, B, C on {src!r} data ...")
    results = evaluate_all(source=src, save=True)

    header = (
        f"\n{'Strategy':<15} {'Precision':>10} {'Recall':>8} {'F1':>8} "
        f"{'Top-5':>8} {'Noise':>8} {'SumAcc':>8} {'Wall(s)':>9}"
    )
    separator = "-" * len(header.lstrip("\n"))
    typer.echo(header)
    typer.echo(separator)

    for m in results:
        if "error" in m:
            typer.echo(f"{m['strategy']:<15} ERROR: {m['error']}")
        else:
            typer.echo(
                f"{m['strategy']:<15} "
                f"{m['precision']:>10.3f} "
                f"{m['recall']:>8.3f} "
                f"{m['f1']:>8.3f} "
                f"{m['top_5_recall']:>8.3f} "
                f"{m['noise_rate']:>8.3f} "
                f"{m['summary_accuracy']:>8.3f} "
                f"{m['wall_seconds']:>9.1f}"
            )

    typer.echo("\nOutput: outputs/evaluation_comparison.json")


@app.command()
def report(
    source: Source = typer.Option(Source.injected, "--source", "-s", help="Data source: clean | injected"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate a PDF triage report from the latest triage and evaluation outputs."""
    _setup_logging(verbose)
    from astrion_dq.config import OUTPUTS_DIR
    from astrion_dq.report.pdf import generate_triage_report

    src = source.value if hasattr(source, "value") else source

    ranked: list = []
    ranked_path = OUTPUTS_DIR / f"ranked_issues_{src}.json"
    if ranked_path.exists():
        ranked = json.loads(ranked_path.read_text(encoding="utf-8"))

    metrics: list = []
    eval_path = OUTPUTS_DIR / "evaluation_comparison.json"
    if eval_path.exists():
        metrics = json.loads(eval_path.read_text(encoding="utf-8"))

    trace: list = []
    report_md_path = OUTPUTS_DIR / f"triage_report_{src}.md"
    if report_md_path.exists():
        for line in report_md_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(" → ") or ("→" in line and "data_loader" in line):
                trace = line.strip().split(" → ")
                break

    out = generate_triage_report(ranked, metrics_list=metrics, agent_trace=trace)
    typer.echo(f"PDF report generated: {out}")


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Port for the Streamlit server."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Launch the Streamlit analytics dashboard."""
    _setup_logging(verbose)
    import subprocess

    dashboard_path = Path(__file__).resolve().parents[3] / "dashboard" / "app.py"
    if not dashboard_path.exists():
        typer.echo(f"Dashboard not found at {dashboard_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Launching dashboard on http://localhost:{port}")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(port),
            "--server.headless", "false",
        ],
        check=True,
    )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Port for the API server."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Launch the FastAPI REST server (astrion-dq serve).

    Set ASTRION_API_TOKEN to enable Bearer token authentication.
    Leave it unset for unauthenticated dev mode.
    """
    _setup_logging(verbose)
    import subprocess

    typer.echo(f"Starting Astrion DQ API on http://{host}:{port}")
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "astrion_dq.api.app:app",
            "--host", host,
            "--port", str(port),
        ],
        check=True,
    )


if __name__ == "__main__":
    app()
