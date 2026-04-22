"""Regression tests for evaluation matching correctness.

Covers three bugs that were present in the original metrics.py:

  Bug 1 (equivalence map): promotion_drift appeared under both numeric_outliers
    and statistical_drift.  One injected promotion_drift could satisfy two
    prediction types simultaneously.

  Bug 2 (top-k one-to-one): the top-k loop used any(_match(pred, gt) ...)
    without a matched set, so the same gt issue could satisfy multiple preds.
    This inflated summary_accuracy.

  Bug 3 (column-blind matching): _match ignored columns entirely, so a
    wrong-column prediction could match the right-table, right-type gt issue.
"""
from __future__ import annotations

import pytest

from astrion_dq.evaluation.metrics import _match, compute_metrics


# ---------------------------------------------------------------------------
# _match unit tests
# ---------------------------------------------------------------------------

def test_match_type_equivalence():
    """duplicate_rows matches duplicate_transactions on the same table."""
    pred = {"issue_type": "duplicate_rows", "table": "fact_sales", "columns": []}
    gt = {"issue_type": "duplicate_transactions", "table": "fact_sales", "columns": []}
    assert _match(pred, gt)


def test_match_table_mismatch_fails():
    """Same type equivalence but different tables must not match."""
    pred = {"issue_type": "duplicate_rows", "table": "fact_sales", "columns": []}
    gt = {"issue_type": "duplicate_transactions", "table": "dim_customers", "columns": []}
    assert not _match(pred, gt)


def test_promotion_drift_only_matches_statistical_drift():
    """promotion_drift must NOT match a numeric_outliers prediction (old double-count bug)."""
    pred_numeric = {"issue_type": "numeric_outliers", "table": "fact_sales", "columns": ["amount"]}
    gt_promo = {"issue_type": "promotion_drift", "table": "fact_sales", "columns": ["campaign_sk"]}
    assert not _match(pred_numeric, gt_promo), (
        "numeric_outliers should not match promotion_drift (promotion_drift is "
        "distributional drift, not a row-level outlier)"
    )

    pred_drift = {"issue_type": "statistical_drift", "table": "fact_sales", "columns": ["campaign_sk"]}
    assert _match(pred_drift, gt_promo), (
        "statistical_drift should match promotion_drift"
    )


def test_column_disagreement_fails():
    """When both sides have columns and they do not overlap, match must return False."""
    pred = {"issue_type": "referential_integrity_break", "table": "fact_sales", "columns": ["wrong_col"]}
    gt = {"issue_type": "referential_integrity_break", "table": "fact_sales", "columns": ["customer_sk"]}
    assert not _match(pred, gt)


def test_column_agreement_passes():
    """When both sides share at least one column, match must return True."""
    pred = {"issue_type": "referential_integrity_break", "table": "fact_sales", "columns": ["customer_sk"]}
    gt = {"issue_type": "referential_integrity_break", "table": "fact_sales", "columns": ["customer_sk"]}
    assert _match(pred, gt)


def test_empty_pred_columns_ignores_column_check():
    """When pred has no columns, the column check is skipped (pred does not narrow scope)."""
    pred = {"issue_type": "duplicate_rows", "table": "fact_sales", "columns": []}
    gt = {"issue_type": "duplicate_transactions", "table": "fact_sales", "columns": []}
    assert _match(pred, gt)


# ---------------------------------------------------------------------------
# compute_metrics: one-to-one top-k matching
# ---------------------------------------------------------------------------

def _gt(issue_type, table, columns):
    return {"issue_type": issue_type, "table": table, "columns": columns}


def _pred(issue_type, table, columns, impact_score=1.0):
    return {
        "issue_type": issue_type,
        "table": table,
        "columns": columns,
        "impact_score": impact_score,
    }


def test_summary_accuracy_one_to_one():
    """Five identical duplicate_rows predictions should claim only ONE gt match.

    The old any()-based top-k loop would report 5/5 = 1.0 (overcounting the
    same gt issue five times). The fixed loop uses a matched set and must report
    1/5 = 0.2.
    """
    ground_truth = [
        _gt("missing_key_values", "fact_sales", ["customer_sk"]),
        _gt("duplicate_transactions", "fact_sales", []),
        _gt("promotion_drift", "fact_sales", ["campaign_sk"]),
    ]

    # Five predictions of the same duplicate_rows type (matching gt[1] only).
    predicted = [_pred("duplicate_rows", "fact_sales", []) for _ in range(5)]

    metrics = compute_metrics(predicted, ground_truth)

    assert metrics["summary_accuracy"] == pytest.approx(1 / 5, abs=1e-4), (
        f"Expected summary_accuracy=0.2 (one-to-one), got {metrics['summary_accuracy']}"
    )
    assert metrics["top_5_recall"] == pytest.approx(1 / 3, abs=1e-4), (
        f"Expected top_5_recall=0.333 (1 of 3 gt found), got {metrics['top_5_recall']}"
    )


def test_precision_recall_correctness():
    """Sanity check on precision and recall with a well-defined set."""
    ground_truth = [
        _gt("missing_key_values", "fact_sales", ["customer_sk"]),
        _gt("duplicate_transactions", "fact_sales", []),
    ]
    predicted = [
        _pred("missing_values", "fact_sales", ["customer_sk"]),  # TP
        _pred("duplicate_rows", "fact_sales", []),               # TP
        _pred("numeric_outliers", "fact_sales", ["amount"]),     # FP
    ]

    metrics = compute_metrics(predicted, ground_truth)

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["recall"] == pytest.approx(1.0, abs=1e-4)

