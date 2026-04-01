from __future__ import annotations

import typer

from astrion_dq.workflows.mvp import (
    baseline_workflow,
    evaluate_workflow,
    inject_retail,
    supervisor_workflow,
)

app = typer.Typer(help="Retail data quality triage workflows.")


@app.command()
def profile():
    """
    Run deterministic baseline profiling workflow on clean retail dataset.
    """
    ranked = baseline_workflow(source="clean", save_outputs=True)
    typer.echo(f"Detected {len(ranked)} issues in baseline workflow.")


@app.command()
def inject(seed: int = typer.Option(42, "--seed", help="Random seed for reproducible issue injection.")):
    """
    Inject synthetic data quality issues into the retail dataset.
    """
    issues = inject_retail(seed=seed)
    typer.echo(f"Injected {len(issues)} synthetic issues into retail dataset.")


@app.command("run-workflow")
def run_workflow(
    name: str = typer.Argument(..., help="baseline or supervisor"),
    source: str = typer.Option("clean", "--source", help="clean or injected"),
):
    """
    Run a single workflow.
    """
    if source not in {"clean", "injected"}:
        raise typer.BadParameter("source must be 'clean' or 'injected'")

    if name == "baseline":
        ranked = baseline_workflow(source=source, save_outputs=True)
    elif name == "supervisor":
        ranked = supervisor_workflow(source=source, save_outputs=True)
    else:
        raise typer.BadParameter("Unknown workflow name. Use baseline or supervisor.")

    typer.echo(f"Workflow {name} on {source} data produced {len(ranked)} ranked issues.")


@app.command()
def evaluate(name: str = typer.Argument(..., help="baseline or supervisor")):
    """
    Evaluate a workflow against injected ground-truth issues.
    """
    if name not in {"baseline", "supervisor"}:
        raise typer.BadParameter("Use baseline or supervisor.")
    result = evaluate_workflow(name)
    typer.echo(f"Evaluation complete for {name}: precision={result['precision']}, recall={result['recall']}, f1={result['f1']}")


if __name__ == "__main__":
    app()
