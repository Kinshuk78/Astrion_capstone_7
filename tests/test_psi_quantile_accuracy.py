"""P3-B: PSI quantile CDF reconstruction accuracy (test-first for F-12).

Test design:
  - Reference distribution: exponential(scale=1.0), n=1000
  - Current distribution:   exponential(scale=3.0), n=1000 (3x tail shift)
  - Oracle PSI: computed from full reference array (ground truth)
  - Histogram PSI: reconstructed from midpoints (pre-fix behaviour)
  - Quantile PSI: reconstructed via np.interp CDF using saved q_probs/q_values

Pre-fix expected failures:
  test_save_snapshot_stores_quantile_tails  -- q_probs/q_values not stored pre-fix
  test_quantile_reconstruction_more_accurate_than_histogram  -- KeyError on q_probs pre-fix
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from unittest.mock import patch

from astrion_dq.checks.drift import _safe_psi_bins, save_snapshot


def _exp_tables(mu: float, n: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    return {"fact_sales": pd.DataFrame({"amount": rng.exponential(scale=mu, size=n)})}


def test_save_snapshot_stores_quantile_tails(tmp_path):
    """save_snapshot must persist q_probs/q_values including p1 (0.01) and p99 (0.99)."""
    import astrion_dq.checks.drift as drift_mod

    ref_tables = _exp_tables(mu=1.0)
    with patch.object(drift_mod, "SNAPSHOTS_DIR", tmp_path):
        path = save_snapshot(ref_tables, tag="quant_test")

    snap = json.loads(path.read_text(encoding="utf-8"))
    entry = snap["fact_sales"]["amount"]

    assert "q_probs" in entry, "save_snapshot must store q_probs list"
    assert "q_values" in entry, "save_snapshot must store q_values list"
    assert 0.01 in entry["q_probs"], "q_probs must include p1 (0.01)"
    assert 0.99 in entry["q_probs"], "q_probs must include p99 (0.99)"
    assert 0.10 in entry["q_probs"], "q_probs must include p10 (0.10)"
    assert 0.90 in entry["q_probs"], "q_probs must include p90 (0.90)"
    assert len(entry["q_values"]) == len(entry["q_probs"]), (
        "q_values length must match q_probs length"
    )


def test_quantile_reconstruction_more_accurate_than_histogram(tmp_path):
    """np.interp quantile PSI error must be less than histogram midpoint PSI error.

    On an exponential distribution with a 3x tail shift, histogram midpoints
    lose tail precision; quantile CDF reconstruction preserves it.
    """
    import astrion_dq.checks.drift as drift_mod

    rng = np.random.default_rng(42)
    n = 1000
    ref_arr = rng.exponential(scale=1.0, size=n)
    cur_arr = rng.exponential(scale=3.0, size=n)

    # Oracle PSI: use full reference array
    oracle_psi = _safe_psi_bins(ref_arr, cur_arr)

    ref_tables = {"fact_sales": pd.DataFrame({"amount": ref_arr})}
    with patch.object(drift_mod, "SNAPSHOTS_DIR", tmp_path):
        path = save_snapshot(ref_tables, tag="acc_test")

    snap = json.loads(path.read_text(encoding="utf-8"))
    entry = snap["fact_sales"]["amount"]

    # Histogram midpoint reconstruction (pre-fix behaviour)
    counts = np.array(entry["hist_counts"], dtype=float)
    edges = np.array(entry["hist_edges"])
    midpoints = (edges[:-1] + edges[1:]) / 2
    scale = max(1, int(1000 / counts.sum())) if counts.sum() > 0 else 1
    hist_ref = np.repeat(midpoints, (counts * scale).astype(int))
    hist_psi = _safe_psi_bins(hist_ref, cur_arr) if len(hist_ref) >= 10 else float("nan")

    # Quantile CDF reconstruction (post-fix behaviour)
    q_probs_full = np.array([0.0] + list(entry["q_probs"]) + [1.0])
    q_values_full = np.array(
        [entry["min"]] + list(entry["q_values"]) + [entry["max"]]
    )
    quant_ref = np.interp(np.linspace(0, 1, 1000), q_probs_full, q_values_full)
    quant_psi = _safe_psi_bins(quant_ref, cur_arr)

    hist_error = abs(hist_psi - oracle_psi)
    quant_error = abs(quant_psi - oracle_psi)

    assert quant_error < hist_error, (
        f"Quantile PSI error ({quant_error:.4f}) must be less than "
        f"histogram PSI error ({hist_error:.4f}). "
        f"Oracle={oracle_psi:.4f}, hist={hist_psi:.4f}, quant={quant_psi:.4f}"
    )
    assert quant_error < 0.05, (
        f"Quantile reconstruction error {quant_error:.4f} must be < 0.05"
    )
