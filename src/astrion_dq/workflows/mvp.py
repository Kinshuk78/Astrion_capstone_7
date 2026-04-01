from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

from astrion_dq.injectors.retail_issues import InjectedIssue, inject_retail_issues
from astrion_dq.warehouse.duckdb_loader import load_tables_to_duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_RETAIL_DIR = PROJECT_ROOT / "data" / "raw" / "retail"
RAW_ROOT_DIR = PROJECT_ROOT / "data" / "raw"
INJECTED_RETAIL_DIR = PROJECT_ROOT / "data" / "injected" / "retail"
LEGACY_INJECTED_DIR = PROJECT_ROOT / "data" / "injected"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

KEY_SUFFIXES = ("_id", "_sk", "_key")


@dataclass
class TableMeta:
    role: str
    primary_keys: List[str]
    foreign_keys: Dict[str, Tuple[str, str]]
    date_cols: List[str]
    numeric_cols: List[str]
    promo_cols: List[str]


@dataclass
class QualityIssue:
    issue_id: str
    issue_type: str
    table: str
    columns: List[str]
    severity: str
    metric: float
    evidence_rows: int
    description: str


@dataclass
class RankedIssue:
    issue_id: str
    issue_type: str
    table: str
    columns: List[str]
    severity: str
    metric: float
    evidence_rows: int
    description: str
    impact_score: float
    affected_reports: List[str]
    agent_trace: List[str]


def _ensure_clean_retail_dir() -> None:
    RAW_RETAIL_DIR.mkdir(parents=True, exist_ok=True)
    retail_files = list(RAW_RETAIL_DIR.glob("*.csv"))
    if retail_files:
        return

    for path in RAW_ROOT_DIR.glob("*.csv"):
        name = path.name.lower()
        if "denormalized" in name:
            continue
        shutil.copy2(path, RAW_RETAIL_DIR / path.name)


def _load_csvs(folder: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for path in sorted(folder.glob("*.csv")):
        if path.stat().st_size == 0:
            continue
        name = path.stem
        if name.endswith("_injected"):
            name = name[:-9]
        if "denormalized" in name.lower():
            continue
        tables[name] = pd.read_csv(path)
    return tables


def load_retail_tables(source: str = "clean") -> Dict[str, pd.DataFrame]:
    _ensure_clean_retail_dir()

    if source == "clean":
        return _load_csvs(RAW_RETAIL_DIR)

    if source == "injected":
        if INJECTED_RETAIL_DIR.exists() and list(INJECTED_RETAIL_DIR.glob("*.csv")):
            return _load_csvs(INJECTED_RETAIL_DIR)

        legacy = {}
        for path in sorted(LEGACY_INJECTED_DIR.glob("*_injected.csv")):
            if path.stat().st_size == 0:
                continue
            name = path.stem.replace("_injected", "")
            if "denormalized" in name.lower():
                continue
            legacy[name] = pd.read_csv(path)
        if legacy:
            return legacy

        raise FileNotFoundError("No injected retail tables found. Run inject first.")

    raise ValueError("source must be 'clean' or 'injected'")


def _norm(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _strip_key_suffix(col: str) -> tuple[str, str] | None:
    c = col.strip()
    lower = c.lower()
    for suffix in KEY_SUFFIXES:
        if lower.endswith(suffix):
            return c[: -len(suffix)], suffix
    return None


def _singularize(entity: str) -> str:
    e = _norm(entity)
    if not e:
        return e

    # Keep "sales" as-is so sales_id never collapses into "sale"
    if e == "sales":
        return e

    if e.endswith("ies") and len(e) > 3:
        return e[:-3] + "y"
    if e.endswith("sses") and len(e) > 4:
        return e[:-2]
    if e.endswith("ses") and len(e) > 3:
        return e[:-1]
    if e.endswith("s") and not e.endswith("ss") and len(e) > 1:
        return e[:-1]
    return e


def _table_entity_base(table_name: str) -> str:
    name = table_name.lower().strip()
    for prefix in ("dim_", "fact_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    tokens = [t for t in name.split("_") if t and t not in {"normalized", "denormalized", "injected"}]
    if not tokens:
        return _singularize(name)
    return _singularize(tokens[0])


def _key_entity_base(col: str) -> str | None:
    stripped = _strip_key_suffix(col)
    if not stripped:
        return None
    stem, _ = stripped
    stem_norm = _norm(stem)
    if not stem_norm:
        return None
    return _singularize(stem_norm)


def _is_key_col(col: str) -> bool:
    return _strip_key_suffix(col) is not None


def infer_metadata(tables: Dict[str, pd.DataFrame]) -> Dict[str, TableMeta]:
    meta: Dict[str, TableMeta] = {}
    dim_primary_keys_by_entity: Dict[str, Tuple[str, str]] = {}

    # First pass: identify table roles, PKs, and dimension PK lookup
    for name, df in tables.items():
        lower_name = name.lower()
        role = "fact" if "fact" in lower_name else "dimension"
        entity_base = _table_entity_base(name)

        date_cols = [c for c in df.columns if "date" in c.lower() or "day" in c.lower()]
        numeric_cols = [
            c for c in df.columns
            if is_numeric_dtype(df[c]) and not _is_key_col(c) and c not in date_cols
        ]
        promo_cols = [c for c in df.columns if any(tok in c.lower() for tok in ["promo", "campaign", "discount"])]

        key_like = [c for c in df.columns if _is_key_col(c)]
        primary_keys: List[str] = []

        if role == "dimension":
            # Prefer a PK whose base entity matches the dimension entity exactly
            preferred = [c for c in key_like if _key_entity_base(c) == entity_base]
            candidates = preferred if preferred else key_like

            for c in candidates:
                if df[c].nunique(dropna=True) == len(df):
                    primary_keys = [c]
                    break

            if not primary_keys and candidates:
                primary_keys = [candidates[0]]

            if primary_keys:
                dim_primary_keys_by_entity[entity_base] = (name, primary_keys[0])

        else:
            # Fact PK: prefer exact entity match, e.g. fact_sales_normalized -> sales_id / sales_sk
            preferred = [c for c in key_like if _key_entity_base(c) == entity_base]
            candidates = preferred if preferred else key_like

            for c in candidates:
                if df[c].nunique(dropna=True) == len(df):
                    primary_keys = [c]
                    break

        meta[name] = TableMeta(
            role=role,
            primary_keys=primary_keys[:1],
            foreign_keys={},
            date_cols=date_cols,
            numeric_cols=numeric_cols,
            promo_cols=promo_cols,
        )

    # Second pass: infer FKs for fact tables using exact base-entity matching only
    for name, df in tables.items():
        if meta[name].role != "fact":
            continue

        fact_entity = _table_entity_base(name)
        fact_primary = set(meta[name].primary_keys)
        foreign_keys: Dict[str, Tuple[str, str]] = {}

        for c in df.columns:
            if c in fact_primary:
                continue
            if not _is_key_col(c):
                continue

            col_entity = _key_entity_base(c)
            if not col_entity:
                continue

            # Do not allow same-entity fact PK/FK confusion
            if col_entity == fact_entity:
                continue

            dim_match = dim_primary_keys_by_entity.get(col_entity)
            if not dim_match:
                continue

            dim_name, dim_pk = dim_match
            foreign_keys[c] = (dim_name, dim_pk)

        meta[name].foreign_keys = foreign_keys

    return meta


# ---------------------------
# Detector agent
# ---------------------------

def _severity_from_metric(metric: float, high: float = 0.05, medium: float = 0.01) -> str:
    if metric >= high:
        return "high"
    if metric >= medium:
        return "medium"
    return "low"


def detect_nulls(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    issue_counter = 1
    base_threshold = 0.01 if sensitivity == "high" else 0.02

    for table, df in tables.items():
        important_cols = set(meta[table].primary_keys + list(meta[table].foreign_keys.keys()) + meta[table].date_cols)
        for col in df.columns:
            frac = float(df[col].isna().mean())
            threshold = base_threshold if col in important_cols else 0.05
            if frac >= threshold:
                issues.append(
                    QualityIssue(
                        issue_id=f"DNULL_{issue_counter:04d}",
                        issue_type="missing_values",
                        table=table,
                        columns=[col],
                        severity=_severity_from_metric(frac),
                        metric=frac,
                        evidence_rows=int(df[col].isna().sum()),
                        description=f"Column {col} has {frac:.2%} missing values.",
                    )
                )
                issue_counter += 1
    return issues


def detect_duplicates(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    issue_counter = 1

    for table, df in tables.items():
        subset = meta[table].primary_keys if meta[table].primary_keys else None
        dup_mask = df.duplicated(subset=subset, keep=False)
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            frac = dup_count / max(1, len(df))
            threshold = 0.005 if sensitivity == "high" else 0.01
            if frac >= threshold or dup_count >= 2:
                issues.append(
                    QualityIssue(
                        issue_id=f"DDUP_{issue_counter:04d}",
                        issue_type="duplicate_rows",
                        table=table,
                        columns=subset or [],
                        severity=_severity_from_metric(frac),
                        metric=frac,
                        evidence_rows=dup_count,
                        description=f"Detected {dup_count} duplicated rows in {table}.",
                    )
                )
                issue_counter += 1
    return issues


def detect_outliers(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    issue_counter = 1
    iqr_mult = 1.5 if sensitivity == "high" else 3.0

    for table, df in tables.items():
        if meta[table].role != "fact":
            continue
        for col in meta[table].numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 20:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - iqr_mult * iqr
            upper = q3 + iqr_mult * iqr
            out_count = int(((series < lower) | (series > upper)).sum())
            frac = out_count / max(1, len(series))
            threshold = 0.005 if sensitivity == "high" else 0.01
            if frac >= threshold:
                issues.append(
                    QualityIssue(
                        issue_id=f"DOUT_{issue_counter:04d}",
                        issue_type="numeric_outliers",
                        table=table,
                        columns=[col],
                        severity=_severity_from_metric(frac),
                        metric=frac,
                        evidence_rows=out_count,
                        description=f"Column {col} has {out_count} outlier rows.",
                    )
                )
                issue_counter += 1
    return issues


def detect_future_dates(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    issue_counter = 1

    for table, df in tables.items():
        for col in meta[table].date_cols:
            s = df[col]
            if is_integer_dtype(s):
                bad_mask = pd.to_numeric(s, errors="coerce") > 20500101
            elif is_datetime64_any_dtype(s):
                bad_mask = pd.to_datetime(s, errors="coerce") > pd.Timestamp("2050-01-01")
            else:
                parsed = pd.to_datetime(s, errors="coerce")
                bad_mask = parsed > pd.Timestamp("2050-01-01")

            bad_count = int(bad_mask.fillna(False).sum())
            if bad_count > 0:
                frac = bad_count / max(1, len(df))
                issues.append(
                    QualityIssue(
                        issue_id=f"DDATE_{issue_counter:04d}",
                        issue_type="invalid_future_dates",
                        table=table,
                        columns=[col],
                        severity=_severity_from_metric(frac),
                        metric=frac,
                        evidence_rows=bad_count,
                        description=f"Column {col} contains {bad_count} future or invalid date values.",
                    )
                )
                issue_counter += 1
    return issues


def detect_referential_breaks(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    issue_counter = 1

    for table, df in tables.items():
        if meta[table].role != "fact":
            continue

        for fk_col, (dim_table, dim_pk) in meta[table].foreign_keys.items():
            if dim_table not in tables or dim_pk not in tables[dim_table].columns or fk_col not in df.columns:
                continue

            valid_values = set(tables[dim_table][dim_pk].dropna().unique().tolist())
            mask = ~df[fk_col].isna() & ~df[fk_col].isin(valid_values)
            bad_count = int(mask.sum())

            if bad_count > 0:
                frac = bad_count / max(1, len(df))
                issues.append(
                    QualityIssue(
                        issue_id=f"DFK_{issue_counter:04d}",
                        issue_type="referential_integrity_break",
                        table=table,
                        columns=[fk_col],
                        severity=_severity_from_metric(frac),
                        metric=frac,
                        evidence_rows=bad_count,
                        description=f"Column {fk_col} contains {bad_count} values not found in {dim_table}.{dim_pk}.",
                    )
                )
                issue_counter += 1

    return issues


def detector_agent(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    sensitivity: str = "normal",
) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    issues.extend(detect_nulls(tables, meta, sensitivity=sensitivity))
    issues.extend(detect_duplicates(tables, meta, sensitivity=sensitivity))
    issues.extend(detect_outliers(tables, meta, sensitivity=sensitivity))
    issues.extend(detect_future_dates(tables, meta, sensitivity=sensitivity))
    issues.extend(detect_referential_breaks(tables, meta, sensitivity=sensitivity))
    return issues


# ---------------------------
# Ranking agent
# ---------------------------

def _report_mapping(issue_type: str) -> List[str]:
    mapping = {
        "missing_values": ["sales by store", "sales by product category"],
        "duplicate_rows": ["daily sales summary", "top products"],
        "numeric_outliers": ["daily sales summary", "top products"],
        "invalid_future_dates": ["daily sales summary"],
        "referential_integrity_break": ["sales by store", "sales by product category"],
    }
    return mapping.get(issue_type, ["daily sales summary"])


def ranking_agent(issues: List[QualityIssue]) -> List[RankedIssue]:
    sev_weight = {"high": 3.0, "medium": 2.0, "low": 1.0}
    issue_weight = {
        "referential_integrity_break": 3.0,
        "duplicate_rows": 2.8,
        "numeric_outliers": 2.6,
        "missing_values": 2.3,
        "invalid_future_dates": 2.1,
    }

    ranked: List[RankedIssue] = []
    for issue in issues:
        score = sev_weight.get(issue.severity, 1.0) * issue_weight.get(issue.issue_type, 1.0) * max(issue.metric, 0.001)
        score += min(issue.evidence_rows / 1000.0, 1.0)

        ranked.append(
            RankedIssue(
                issue_id=issue.issue_id,
                issue_type=issue.issue_type,
                table=issue.table,
                columns=issue.columns,
                severity=issue.severity,
                metric=issue.metric,
                evidence_rows=issue.evidence_rows,
                description=issue.description,
                impact_score=round(score, 4),
                affected_reports=_report_mapping(issue.issue_type),
                agent_trace=["detector_agent", "ranking_agent"],
            )
        )

    ranked.sort(key=lambda x: x.impact_score, reverse=True)
    return ranked


# ---------------------------
# Summary agent
# ---------------------------

def summary_agent(ranked: List[RankedIssue], workflow_name: str, rerun_used: bool = False) -> str:
    title = "Baseline data quality triage summary" if workflow_name == "baseline" else "Supervisor triage summary"
    lines = [f"## {title}", ""]
    lines.append(f"- Total issues detected: **{len(ranked)}**")
    lines.append(f"- Fallback rerun used: **{'Yes' if rerun_used else 'No'}**")
    lines.append("")

    if not ranked:
        lines.append("No issues were detected for this dataset and threshold setting.")
        return "\n".join(lines) + "\n"

    lines.append("### Top issues")
    lines.append("")

    for i, issue in enumerate(ranked[:10], start=1):
        cols = ", ".join(issue.columns) if issue.columns else "n/a"
        reports = ", ".join(issue.affected_reports)

        lines.append(f"{i}. **{issue.issue_type}** in `{issue.table}`")
        lines.append(f"   - Severity: `{issue.severity}`")
        lines.append(f"   - Columns: `{cols}`")
        lines.append(f"   - Evidence rows: `{issue.evidence_rows}`")
        lines.append(f"   - Impact score: `{issue.impact_score}`")
        lines.append(f"   - Affected reports: `{reports}`")
        lines.append(f"   - Summary: {issue.description}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


# ---------------------------
# Fallback / rebound agent
# ---------------------------

def fallback_agent(
    tables: Dict[str, pd.DataFrame],
    meta: Dict[str, TableMeta],
    initial_ranked: List[RankedIssue],
    source: str,
) -> Tuple[List[RankedIssue], bool]:
    """
    If injected data produced too few issues, rerun detector with higher sensitivity.
    """
    if source != "injected":
        return initial_ranked, False

    if len(initial_ranked) >= 2:
        return initial_ranked, False

    rerun_issues = detector_agent(tables, meta, sensitivity="high")
    rerun_ranked = ranking_agent(rerun_issues)
    return rerun_ranked, True


def _dedupe_ranked(ranked: List[RankedIssue]) -> Tuple[List[RankedIssue], int]:
    seen = set()
    deduped = []
    removed = 0

    for issue in ranked:
        key = (issue.issue_type, issue.table, tuple(issue.columns))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(issue)

    return deduped, removed


# ---------------------------
# IO helpers
# ---------------------------

def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------
# Public workflows
# ---------------------------

def baseline_workflow(source: str = "clean", save_outputs: bool = True) -> List[RankedIssue]:
    tables = load_retail_tables(source=source)
    load_tables_to_duckdb(tables, schema="dq_retail", overwrite=True)

    meta = infer_metadata(tables)
    issues = detector_agent(tables, meta, sensitivity="normal")
    ranked = ranking_agent(issues)

    if save_outputs:
        _write_json(OUTPUTS_DIR / "baseline_issues.json", [asdict(x) for x in ranked])
        _write_text(OUTPUTS_DIR / "baseline_summary.md", summary_agent(ranked, "baseline", rerun_used=False))

    return ranked


def supervisor_workflow(source: str = "clean", save_outputs: bool = True) -> List[RankedIssue]:
    tables = load_retail_tables(source=source)
    load_tables_to_duckdb(tables, schema="dq_retail", overwrite=True)

    meta = infer_metadata(tables)

    initial_issues = detector_agent(tables, meta, sensitivity="normal")
    initial_ranked = ranking_agent(initial_issues)
    reranked, rerun_used = fallback_agent(tables, meta, initial_ranked, source=source)
    deduped, removed = _dedupe_ranked(reranked)
    summary = summary_agent(deduped, "supervisor", rerun_used=rerun_used)

    if save_outputs:
        payload = {
            "duplicates_removed": removed,
            "rerun_used": rerun_used,
            "issues": [asdict(x) for x in deduped],
        }
        _write_json(OUTPUTS_DIR / "supervisor_issues.json", payload)
        _write_text(OUTPUTS_DIR / "supervisor_summary.md", summary)

    return deduped


def inject_retail(seed: int = 42) -> List[InjectedIssue]:
    tables = load_retail_tables(source="clean")
    _, issues = inject_retail_issues(tables, seed=seed)
    return issues


# ---------------------------
# Evaluation
# ---------------------------

def _load_ground_truth() -> List[dict]:
    path = OUTPUTS_DIR / "retail_injected_issues.json"
    if not path.exists():
        raise FileNotFoundError("Ground-truth issues not found. Run inject first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _match(pred: RankedIssue, gt: dict) -> bool:
    pred_type = pred.issue_type
    gt_type = gt["issue_type"]

    equivalent = {
        "missing_values": {"missing_key_values", "dimension_missing_values"},
        "duplicate_rows": {"duplicate_transactions"},
        "numeric_outliers": {"numeric_outliers", "promotion_drift"},
        "invalid_future_dates": {"invalid_future_dates"},
        "referential_integrity_break": {"referential_integrity_break"},
    }

    for pred_key, gt_set in equivalent.items():
        if pred_type == pred_key and gt_type in gt_set and pred.table == gt["table"]:
            return True
    return False


def evaluate_workflow(name: str) -> dict:
    if name == "baseline":
        ranked = baseline_workflow(source="injected", save_outputs=True)
        output_path = OUTPUTS_DIR / "evaluation_baseline.json"
    elif name == "supervisor":
        ranked = supervisor_workflow(source="injected", save_outputs=True)
        output_path = OUTPUTS_DIR / "evaluation_supervisor.json"
    else:
        raise ValueError("name must be baseline or supervisor")

    gt = _load_ground_truth()

    matched_gt = set()
    tp = 0
    for pred in ranked:
        for i, truth in enumerate(gt):
            if i in matched_gt:
                continue
            if _match(pred, truth):
                tp += 1
                matched_gt.add(i)
                break

    fp = max(0, len(ranked) - tp)
    fn = max(0, len(gt) - tp)

    precision = tp / len(ranked) if ranked else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    noise_rate = fp / len(ranked) if ranked else 0.0

    top_k = min(5, len(ranked))
    top_k_hits = 0
    for pred in ranked[:top_k]:
        if any(_match(pred, truth) for truth in gt):
            top_k_hits += 1
    top_k_recall = top_k_hits / min(5, len(gt)) if gt else 0.0

    duplicate_rate = 0.0
    if ranked:
        keys = [(x.issue_type, x.table, tuple(x.columns)) for x in ranked]
        duplicate_rate = 1 - (len(set(keys)) / len(keys))

    result = {
        "workflow": name,
        "predicted_issues": len(ranked),
        "ground_truth_issues": len(gt),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "top_5_recall": round(top_k_recall, 4),
        "noise_rate": round(noise_rate, 4),
        "duplicate_rate": round(duplicate_rate, 4),
    }

    _write_json(output_path, result)
    return result