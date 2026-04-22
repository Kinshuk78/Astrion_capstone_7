"""Evaluation framework for comparing three agentic triage strategies.

Three strategies are compared to isolate the value of each architectural layer:

  A — Baseline:   data_loader → profiler → detector → ranker
                  No drift detection, no SQL verification.

  B — Supervisor: A + debugger + human_review (analyst gate)
                  Adds SQL cross-validation; surfaces low-confidence issues.

  C — Full:       B + drift_detector
                  Adds PSI + KS statistical drift detection to strategy B.

Metrics:
  precision       — fraction of predicted issues that are true positives
  recall          — fraction of injected issues that were correctly detected
  f1              — harmonic mean of precision and recall
  top_5_recall    — fraction of the top-5 ranked issues that are true positives
  noise_rate      — fraction of predicted issues that are false positives
  summary_accuracy — precision within the top-k ranked results presented to the analyst;
                     answers "how often does the ranked list mislead the analyst?"
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from astrion_dq.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(path: Optional[Path] = None) -> List[dict]:
    """Load injected issue ground truth produced by 'astrion-dq inject'."""
    if path is None:
        path = OUTPUTS_DIR / "retail_injected_issues.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found at {path}. Run 'astrion-dq inject' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _match(pred: dict, gt: dict) -> bool:
    """Return True if a predicted issue corresponds to a ground-truth injected issue.

    Three conditions must all hold:
      1. ``gt.issue_type`` belongs to the equivalence set for ``pred.issue_type``.
      2. ``pred.table == gt.table``.
      3. If both sides have non-empty column lists they must share at least one column.

    Equivalence map (pred_type -> set of matching gt_types):
      - ``promotion_drift`` appears only under ``statistical_drift``.  It was
        previously also in the ``numeric_outliers`` set, which caused one
        injected promotion_drift issue to satisfy two different prediction types
        simultaneously — inflating top-k recall.
    """
    equivalence: Dict[str, set] = {
        "missing_values":              {"missing_key_values", "dimension_missing_values"},
        "duplicate_rows":              {"duplicate_transactions"},
        "numeric_outliers":            {"numeric_outliers"},
        "invalid_future_dates":        {"invalid_future_dates"},
        "referential_integrity_break": {"referential_integrity_break"},
        "statistical_drift":           {"promotion_drift"},
    }
    pred_type = pred.get("issue_type", "")
    gt_type = gt.get("issue_type", "")
    gt_set = equivalence.get(pred_type, {pred_type})

    if gt_type not in gt_set:
        return False
    if pred.get("table") != gt.get("table"):
        return False

    pred_cols = pred.get("columns") or []
    gt_cols = gt.get("columns") or []
    if pred_cols and gt_cols:
        return bool(set(pred_cols) & set(gt_cols))

    return True


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(predicted: List[dict], ground_truth: List[dict]) -> dict:
    """Compute all evaluation metrics for a single strategy run.

    Args:
        predicted: Ranked issue dicts from ranker_node output (ranked_issues).
        ground_truth: Injected issue dicts loaded from retail_injected_issues.json.

    Returns:
        Dict with keys: true_positives, false_positives, false_negatives,
        precision, recall, f1, top_5_recall, noise_rate, summary_accuracy.
    """
    matched_gt: set = set()
    tp = 0
    for pred in predicted:
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            if _match(pred, gt):
                tp += 1
                matched_gt.add(i)
                break

    fp = max(0, len(predicted) - tp)
    fn = max(0, len(ground_truth) - tp)

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    noise_rate = fp / len(predicted) if predicted else 0.0

    # Top-k one-to-one matching: each ground-truth issue can satisfy at most one
    # prediction. Without a matched set, a single repeated prediction type can
    # claim credit against the same gt issue k times — inflating summary_accuracy.
    top_k = min(5, len(predicted))
    matched_top: set = set()
    top_k_hits = 0
    for pred in predicted[:top_k]:
        for j, gt in enumerate(ground_truth):
            if j in matched_top:
                continue
            if _match(pred, gt):
                top_k_hits += 1
                matched_top.add(j)
                break

    top_5_recall = top_k_hits / min(5, len(ground_truth)) if ground_truth else 0.0

    # summary_accuracy: precision within the top-k ranked results.
    # A value < 1.0 means the analyst's prioritised list contains misleading items.
    summary_accuracy = top_k_hits / top_k if top_k > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "top_5_recall": round(top_5_recall, 4),
        "noise_rate": round(noise_rate, 4),
        "summary_accuracy": round(summary_accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def _prepare_data(source: str) -> dict:
    """Load retail tables and register them in DuckDB once for all strategies.

    Returns a partial TriageState dict that can be merged into initial_state()
    before graph.invoke(). Setting data_loaded=True causes the router to skip
    data_loader_node, so the three strategy runs share one load.
    """
    from astrion_dq.config import DB_PATH
    from astrion_dq.warehouse.loader import load_retail_tables, load_tables_to_duckdb

    tables = load_retail_tables(source=source)
    load_tables_to_duckdb(tables)
    return {
        "tables": tables,
        "table_sizes": {name: len(df) for name, df in tables.items()},
        "db_path": str(DB_PATH),
        "data_loaded": True,
    }


def _run_graph(
    source: str,
    skip_drift: bool = False,
    skip_debug: bool = False,
    shared_data: Optional[dict] = None,
    close_after: bool = True,
) -> dict:
    """Run the LangGraph triage graph and return the final state dict.

    skip_drift=True pre-sets drift_done=True so drift_detector_node is bypassed.
    skip_debug=True pre-sets debug_done=True and review_done=True so the debugger
    and human_review_node are bypassed.

    shared_data: optional pre-loaded state dict from _prepare_data(). When
        provided, tables/table_sizes/db_path/data_loaded are pre-seeded so the
        router skips data_loader_node. This allows all three strategies to share
        one data load when called from evaluate_all().

    close_after: if False, skip close_connection() after graph invocation.
        Used by evaluate_all() which calls close_connection() once at the end.

    ASTRION_AUTO_APPROVE=1 is always set to prevent interrupt() from blocking.
    """
    os.environ["ASTRION_AUTO_APPROVE"] = "1"

    from astrion_dq.graph.state import initial_state
    from astrion_dq.graph.workflow import build_graph
    from astrion_dq.warehouse.loader import close_connection

    graph = build_graph()
    state = initial_state(source=source)

    if shared_data:
        state["tables"] = shared_data["tables"]
        state["table_sizes"] = shared_data["table_sizes"]
        state["db_path"] = shared_data["db_path"]
        state["data_loaded"] = shared_data["data_loaded"]

    if skip_drift:
        state["drift_done"] = True
        state["drift_issues"] = []

    if skip_debug:
        state["debug_done"] = True
        state["review_done"] = True

    import uuid
    config = {"configurable": {"thread_id": f"eval_{source}_{uuid.uuid4().hex[:8]}"}}
    result = graph.invoke(state, config=config)
    if close_after:
        close_connection()
    return result


def run_strategy_a(source: str = "injected", shared_data: Optional[dict] = None) -> dict:
    """Strategy A (Baseline): data_loader → profiler → detector → ranker.

    No drift detection, no SQL verification. Measures baseline detection
    capability without cross-validation or distributional checks.
    """
    t0 = time.perf_counter()
    result = _run_graph(
        source=source, skip_drift=True, skip_debug=True,
        shared_data=shared_data, close_after=shared_data is None,
    )
    return {"result": result, "wall_seconds": time.perf_counter() - t0, "strategy": "A_baseline"}


def run_strategy_b(source: str = "injected", shared_data: Optional[dict] = None) -> dict:
    """Strategy B (Supervisor): Strategy A + debugger + human_review.

    Adds SQL cross-validation to measure detection confidence and surfaces
    low-confidence issues through the analyst gate.
    """
    t0 = time.perf_counter()
    result = _run_graph(
        source=source, skip_drift=True, skip_debug=False,
        shared_data=shared_data, close_after=shared_data is None,
    )
    return {"result": result, "wall_seconds": time.perf_counter() - t0, "strategy": "B_supervisor"}


def run_strategy_c(source: str = "injected", shared_data: Optional[dict] = None) -> dict:
    """Strategy C (Full): Strategy B + drift_detector.

    Adds PSI + KS statistical drift detection. The most complete pipeline.
    Requires a saved baseline snapshot ('astrion-dq snapshot').
    """
    t0 = time.perf_counter()
    result = _run_graph(
        source=source, skip_drift=False, skip_debug=False,
        shared_data=shared_data, close_after=shared_data is None,
    )
    return {"result": result, "wall_seconds": time.perf_counter() - t0, "strategy": "C_full"}


# ---------------------------------------------------------------------------
# Combined evaluator
# ---------------------------------------------------------------------------

def evaluate_all(source: str = "injected", save: bool = True) -> List[dict]:
    """Run all three strategies and compute evaluation metrics against ground truth.

    Data is loaded once via _prepare_data() and shared across all three strategy
    runs. close_connection() is called once after all runs complete.

    Args:
        source: Data source -- "injected" for meaningful ground-truth comparison.
        save: If True, write results to outputs/evaluation_comparison.json.

    Returns:
        List of metric dicts, one per strategy, in order [A, B, C].
    """
    from astrion_dq.warehouse.loader import close_connection

    gt = load_ground_truth()
    shared = _prepare_data(source)
    results = []

    runners = [
        (run_strategy_a, "A_baseline"),
        (run_strategy_b, "B_supervisor"),
        (run_strategy_c, "C_full"),
    ]

    try:
        for runner, name in runners:
            logger.info("Running strategy %s ...", name)
            try:
                run_out = runner(source=source, shared_data=shared)
                ranked = run_out["result"].get("ranked_issues") or []
                metrics = compute_metrics(ranked, gt)
                metrics.update({
                    "strategy": name,
                    "predicted_issues": len(ranked),
                    "ground_truth_issues": len(gt),
                    "wall_seconds": round(run_out["wall_seconds"], 3),
                    "agent_trace": run_out["result"].get("agent_trace") or [],
                })
            except Exception as exc:
                logger.exception("Strategy %s failed: %s", name, exc)
                metrics = {"strategy": name, "error": str(exc)}
            results.append(metrics)
    finally:
        close_connection()

    if save:
        out = OUTPUTS_DIR / "evaluation_comparison.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Evaluation results saved → %s", out)

    return results
