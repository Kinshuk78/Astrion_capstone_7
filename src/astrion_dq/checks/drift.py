from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore

from astrion_dq.config import KS_ALPHA, MIN_ROWS_FOR_STATS, PSI_AMBER, PSI_RED, SNAPSHOTS_DIR
from astrion_dq.models import QualityIssue, TableMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PSI core
# ---------------------------------------------------------------------------

def _safe_psi_bins(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index using quantile-based bins from the reference distribution.

    PSI = Σ (cur_i − ref_i) × ln(cur_i / ref_i)

    Quantile bins are derived from *reference* only — this is the industry
    standard that prevents PSI from firing on legitimate distributional shape
    differences confined to a single linear bin.
    """
    if len(reference) < n_bins or len(current) < 2:
        return 0.0
    breakpoints = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(breakpoints) < 2:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / max(ref_counts.sum(), 1))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / max(cur_counts.sum(), 1))
    return max(float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))), 0.0)


def _psi_severity(psi: float) -> str:
    if psi < PSI_AMBER:
        return "low"   # stable — not reported
    if psi < PSI_RED:
        return "medium"
    return "high"


def _ks_severity(p_value: float) -> str:
    if p_value >= KS_ALPHA:
        return "low"   # stable — not reported
    if p_value >= 0.01:
        return "medium"
    return "high"


def _direction(reference: np.ndarray, current: np.ndarray) -> str:
    if len(reference) == 0 or len(current) == 0:
        return "unknown"
    ref_mean, cur_mean = float(np.mean(reference)), float(np.mean(current))
    if abs(ref_mean) < 1e-9:
        return "shift"
    ratio = cur_mean / ref_mean
    if ratio > 1.1:
        return "increase"
    if ratio < 0.9:
        return "decrease"
    return "shift"


# ---------------------------------------------------------------------------
# Snapshot management
# ---------------------------------------------------------------------------

def build_snapshot(tables: Dict[str, pd.DataFrame]) -> dict:
    """Build an in-memory drift snapshot from DataFrames.

    The shape matches the JSON persisted by ``save_snapshot`` so callers can
    reuse the same snapshot-style drift path without writing temporary files.
    """
    snapshot: dict = {}

    for table, df in tables.items():
        snapshot[table] = {}
        for col in df.columns:
            series = df[col].dropna()
            entry: dict = {
                "dtype": str(df[col].dtype),
                "n_rows": len(df),
                "n_null": int(df[col].isna().sum()),
            }
            if pd.api.types.is_numeric_dtype(series) and len(series) > 0:
                arr = series.to_numpy(dtype=float)
                entry["mean"] = float(arr.mean())
                entry["std"] = float(arr.std())
                entry["min"] = float(arr.min())
                entry["max"] = float(arr.max())
                entry["p25"] = float(np.percentile(arr, 25))
                entry["p50"] = float(np.percentile(arr, 50))
                entry["p75"] = float(np.percentile(arr, 75))
                entry["p95"] = float(np.percentile(arr, 95))
                _q_probs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
                entry["q_probs"] = _q_probs
                entry["q_values"] = [float(np.percentile(arr, p * 100)) for p in _q_probs]
                hist, edges = np.histogram(arr, bins=min(20, len(arr)))
                entry["hist_counts"] = hist.tolist()
                entry["hist_edges"] = edges.tolist()
            else:
                counts = series.value_counts(normalize=True).head(50).to_dict()
                entry["top_value_fracs"] = {str(k): float(v) for k, v in counts.items()}
            snapshot[table][col] = entry
    return snapshot


def save_snapshot(tables: Dict[str, pd.DataFrame], tag: str = "baseline") -> Path:
    """Persist column-level distribution statistics as a JSON snapshot.

    Stores mean, std, percentiles, and histogram bins for numeric columns;
    top-value fractions for categorical columns. Raw row data is never stored.
    """
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot(tables)

    path = SNAPSHOTS_DIR / f"snapshot_{tag}.json"
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    logger.info("Drift snapshot saved → %s", path)
    return path


def load_snapshot(tag: str = "baseline") -> Optional[dict]:
    """Load a previously saved snapshot. Returns None if not found."""
    path = SNAPSHOTS_DIR / f"snapshot_{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main drift detector
# ---------------------------------------------------------------------------

def detect_drift(
    current_tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    reference_tables: Optional[Dict[str, pd.DataFrame]] = None,
    reference_snapshot: Optional[dict] = None,
    snapshot_tag: str = "baseline",
) -> List[QualityIssue]:
    """Detect statistical drift in numeric columns using PSI and the KS test.

    PSI (Population Stability Index) detects gradual distributional shift.
    The two-sample KS test detects abrupt discontinuities. When both tests
    fire on the same (table, col) pair, only the more severe result is kept.

    Columns are gated via ``meta[table].numeric_cols``, which excludes surrogate
    key columns and date columns. This prevents PSI/KS from firing on
    value-range shifts in ``sales_sk``, ``customer_sk``, etc.

    PSI thresholds: stable < PSI_AMBER; medium in [PSI_AMBER, PSI_RED); high >= PSI_RED.
    KS threshold: p >= KS_ALPHA is stable.

    Args:
        current_tables: DataFrames from the current pipeline run.
        meta: TableMeta mapping produced by ``infer_metadata``. Used to gate
            drift scanning to genuine numeric columns only.
        reference_tables: Optional baseline DataFrames. If None, a saved
            snapshot identified by *snapshot_tag* is used instead unless
            *reference_snapshot* is provided.
        reference_snapshot: Optional in-memory snapshot in the same structure
            produced by ``save_snapshot`` / ``build_snapshot``. When present,
            this takes precedence over *reference_tables* so callers can reuse
            the same snapshot-style drift path as CLI triage without writing to disk.
        snapshot_tag: Tag for the snapshot file (default "baseline").

    Returns:
        List of QualityIssue objects with issue_type="statistical_drift".
        Stable columns produce no issues.
    """
    snapshot: Optional[dict] = reference_snapshot
    if snapshot is None and reference_tables is None:
        snapshot = load_snapshot(snapshot_tag)
        if snapshot is None:
            logger.warning(
                "No reference_tables and no snapshot '%s' found. "
                "Run 'astrion-dq snapshot' first. Skipping drift detection.",
                snapshot_tag,
            )
            return []

    # best[(table, col)] = (severity, description, metric_val)
    # metric_val: normalised statistic in [0, 1] used as evidence_density proxy in BIS scoring.
    _sev_rank = {"low": 0, "medium": 1, "high": 2}
    best: Dict[tuple, tuple] = {}

    for table, cur_df in current_tables.items():
        m = meta.get(table)
        # Drift detection is meaningful only for fact table business metrics.
        # Dimension tables are reference data; their structure is validated by
        # the RI checker and schema comparison, not by PSI/KS distribution tests.
        # Running PSI on calendar attributes (year, month, quarter) or small
        # dimension tables (n < 100) produces statistically unreliable signals.
        if m is not None and m.role != "fact":
            continue
        for col in cur_df.columns:
            if m is not None and col not in m.numeric_cols:
                continue
            cur_series = cur_df[col].dropna()
            if not pd.api.types.is_numeric_dtype(cur_series):
                continue
            if len(cur_series) < MIN_ROWS_FOR_STATS:
                continue
            cur_arr = cur_series.to_numpy(dtype=float)

            # Resolve reference array
            if snapshot is not None:
                if table not in snapshot or col not in snapshot[table]:
                    continue
                snap_col = snapshot[table][col]
                if "q_probs" in snap_col and "q_values" in snap_col:
                    # Quantile CDF reconstruction via np.interp (post-P3-B path).
                    # Prepend min(0.0) and append max(1.0) to cover the full range.
                    q_probs_full = np.array(
                        [0.0] + list(snap_col["q_probs"]) + [1.0]
                    )
                    q_values_full = np.array(
                        [snap_col.get("min", snap_col["q_values"][0])]
                        + list(snap_col["q_values"])
                        + [snap_col.get("max", snap_col["q_values"][-1])]
                    )
                    ref_arr = np.interp(np.linspace(0, 1, 1000), q_probs_full, q_values_full)
                elif "hist_counts" in snap_col:
                    # Legacy histogram midpoint fallback (backward compatible).
                    counts = np.array(snap_col["hist_counts"], dtype=float)
                    edges = np.array(snap_col["hist_edges"])
                    if len(edges) < 2:
                        continue
                    midpoints = (edges[:-1] + edges[1:]) / 2
                    scale = max(1, int(1000 / counts.sum())) if counts.sum() > 0 else 1
                    ref_arr = np.repeat(midpoints, (counts * scale).astype(int))
                    if len(ref_arr) < 10:
                        continue
                else:
                    continue
            elif reference_tables is not None:
                if table not in reference_tables or col not in reference_tables[table].columns:
                    continue
                ref_arr = reference_tables[table][col].dropna().to_numpy(dtype=float)
            else:
                continue

            if len(ref_arr) < 10:
                continue

            key = (table, col)
            drift_dir = _direction(ref_arr, cur_arr)

            # PSI
            psi = _safe_psi_bins(ref_arr, cur_arr)
            sev = _psi_severity(psi)
            if sev != "low":
                # Normalise PSI to [0, 1]: cap at PSI_RED (major threshold)
                metric_val = round(min(psi / PSI_RED, 1.0), 4)
                desc = (
                    f"PSI={psi:.4f} on {table}.{col}: {sev} distribution shift ({drift_dir}). "
                    f"Thresholds: medium≥{PSI_AMBER}, high≥{PSI_RED}."
                )
                if key not in best or _sev_rank[sev] > _sev_rank[best[key][0]]:
                    best[key] = (sev, desc, metric_val)

            # KS test
            try:
                ks_stat, p_val = stats.ks_2samp(ref_arr, cur_arr)
            except Exception:
                continue
            sev = _ks_severity(p_val)
            if sev != "low":
                # KS D-statistic is in [0, 1] — use directly as metric
                metric_val = round(float(ks_stat), 4)
                desc = (
                    f"KS D={ks_stat:.4f}, p={p_val:.4f} on {table}.{col}: "
                    f"statistically significant distribution change (α={KS_ALPHA})."
                )
                if key not in best or _sev_rank[sev] > _sev_rank[best[key][0]]:
                    best[key] = (sev, desc, metric_val)

    issues: List[QualityIssue] = []
    for i, ((table, col), (sev, desc, metric_val)) in enumerate(best.items(), start=1):
        issues.append(QualityIssue(
            issue_id=f"DRIFT_{i:04d}",
            issue_type="statistical_drift",
            table=table,
            columns=[col],
            severity=sev,
            metric=metric_val,
            evidence_rows=0,
            description=desc,
        ))

    logger.info("Drift detection: %d signal(s) found.", len(issues))
    return issues
