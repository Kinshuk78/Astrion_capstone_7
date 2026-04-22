# Astrion DQ: System Architecture

## Overview

Astrion DQ is a retail data quality triage system. It uses LangGraph to
orchestrate a pipeline of specialised nodes under a single deterministic
supervisor routing function. The pipeline loads a retail star schema, detects
quality issues via rule-based checks and statistical drift analysis, cross-
validates every issue with SQL, ranks issues by a V2 Business Impact Score,
and produces a triage report.

There is no LLM runtime path in this codebase.

---

## Package Layout

```
src/astrion_dq/
    checks/
        __init__.py
        detect.py          Rule-based detectors; re-exports infer_metadata
        drift.py           PSI + KS drift detection; snapshot management
    evaluation/
        __init__.py
        metrics.py         Three-strategy evaluation framework
    graph/
        __init__.py
        debugger.py        IssueVerifier: SQL cross-validation
        nodes.py           LangGraph node functions
        state.py           TriageState TypedDict + initial_state()
        workflow.py        build_graph(), _route() supervisor
    injectors/
        __init__.py
        retail_issues.py   Seven synthetic issue type injectors
    ranking/
        __init__.py
        impact.py          V2 Business Impact Score (BIS)
    report/
        __init__.py
        pdf.py             ReportLab PDF generator
    warehouse/
        __init__.py
        loader.py          DuckDB singleton connection; CSV loading
    metadata.py            Schema inference: infer_metadata, is_key_col, etc.
    config.py              All thresholds, weights, and path constants
    models.py              QualityIssue, VerifiedIssue, RankedIssue, TableMeta
    cli.py                 Typer CLI entry point
```

---

## Module Descriptions

### `metadata.py`

Provides schema inference helpers used by the detector, injector, and
evaluation layers.

**Key functions**

- `infer_metadata(tables)` - two-pass algorithm that classifies tables as fact
  or dimension, detects primary keys by uniqueness, and resolves foreign key
  relationships by matching column entity bases to dimension primary keys.
  Returns a `Dict[str, TableMeta]`.
- `is_key_col(col)` - returns True when a column name ends with `_id`, `_sk`,
  or `_key`.
- `table_entity_base(name)`, `key_entity_base(col)` - normalise and singularise
  table/column name tokens for entity-level matching.

**Consumed by:** `checks/detect.py` (re-exports `infer_metadata`), `injectors/retail_issues.py`.

### `config.py`

Single source of truth for all numeric thresholds and weights. No logic.

**Key constants**

- `NULL_THRESHOLD_IMPORTANT` (0.01) - null fraction threshold for key columns.
- `IQR_MULT_NORMAL` (3.0), `IQR_MULT_HIGH` (1.5) - outlier IQR multipliers.
- `ISSUE_TYPE_BASE_WEIGHTS` - per-type base weight for the BIS formula.
- `REPORT_CRITICALITY_SCORES` - downstream report weights for BIS.
- `DB_PATH`, `OUTPUTS_DIR`, `INJECTED_DIR`, `SNAPSHOTS_DIR` - filesystem paths.

### `models.py`

Four dataclasses that form the data contract between pipeline stages:

- `TableMeta` - inferred schema: role, PKs, FKs, date/numeric/promo columns.
- `QualityIssue` - raw detector output: type, table, columns, severity, metric, evidence_rows.
- `VerifiedIssue` - `QualityIssue` + sql_count, pd_count, confidence.
- `RankedIssue` - `VerifiedIssue` + impact_score, affected_reports, agent_trace.

### `checks/detect.py`

Five detector functions plus a parallel runner.

**Detectors**

- `detect_nulls` - flags columns where null fraction exceeds the sensitivity-
  adjusted threshold. Key columns (PKs, FKs, date cols) use
  `NULL_THRESHOLD_IMPORTANT` in normal mode.
- `detect_duplicates` - flags duplicate rows scoped to primary key columns.
- `detect_outliers` - IQR method on fact table numeric columns; multiplier
  selected by sensitivity.
- `detect_future_dates` - flags date column values beyond year 2050.
- `detect_referential_breaks` - flags fact FK values absent from the dim PK.

**`run_all_checks_parallel`** submits all five detectors to a `ThreadPoolExecutor`.
Empty tables are flagged before the pool runs. Exceptions per check are logged
and do not abort the run.

**Consumed by:** `graph/nodes.py::detector_node`.

### `checks/drift.py`

Statistical drift detection using PSI and the two-sample KS test.

- `save_snapshot(tables, tag)` - persists column-level distribution statistics
  (mean, std, percentiles, histogram bins) as JSON. Raw row data is never stored.
- `load_snapshot(tag)` - loads a saved snapshot.
- `detect_drift(current_tables, ...)` - computes PSI and KS for each numeric
  column against the reference. Reports the more severe of the two signals per
  column. Returns `List[QualityIssue]` with `issue_type="statistical_drift"`.

**Consumed by:** `graph/nodes.py::drift_detector_node`.

### `graph/debugger.py`

`IssueVerifier` cross-validates each `QualityIssue` against DuckDB SQL.

**`__init__` parameters:** `schema` (default `dq_retail`), `sensitivity`
(default `"normal"` — must match the detector sensitivity used upstream).

The `sensitivity` parameter controls which IQR multiplier the SQL outlier query
uses. Using the wrong value produces systematically wrong confidence scores.

**Per-type SQL logic**

- `missing_values` - `COUNT WHERE col IS NULL`.
- `duplicate_rows` - `COUNT(*) - COUNT(DISTINCT pk_cols)`; fallback to pandas
  count when no PK columns are stored.
- `numeric_outliers` - `PERCENTILE_CONT` for Q1/Q3, then outlier count with the
  sensitivity-appropriate IQR multiplier.
- `invalid_future_dates` - integer-typed columns use the integer sentinel
  (20500101); others use `TRY_CAST ... AS DATE`.
- `referential_integrity_break` - `LEFT JOIN` on FK/dim-PK, count unmatched rows.
- `statistical_drift`, `empty_table` - bypass SQL; return fixed confidence values.

**Consumed by:** `graph/nodes.py::debugger_node`.

### `graph/nodes.py`

One function per graph node. Each function accepts the full `TriageState` dict
and returns a partial dict with only the keys it modifies.

**`data_loader_node`** - calls `load_retail_tables(source)` and
`load_tables_to_duckdb(tables)`. Sets `data_loaded=True` and `db_path`.

**`profiler_node`** - calls `infer_metadata(tables)`, serialises `TableMeta`
objects to plain dicts (JSON-serialisable), sets `metadata_ready=True`.

**`detector_node`** - reconstructs `TableMeta` from state, calls
`run_all_checks_parallel`. Sets `detection_done=True`.

**`drift_detector_node`** - calls `detect_drift`. Merges drift issues with
raw issues into `all_issues`. Sets `drift_done=True`.

**`debugger_node`** - constructs `IssueVerifier(sensitivity=state["sensitivity"])`,
calls `verify_all`. Sets `debug_done=True` and `needs_human_review`.

**`human_review_node`** - auto-approves when `ASTRION_AUTO_APPROVE=1`; otherwise
calls `langgraph.types.interrupt()`. Records timing in both branches.

**`ranker_node`** - calls `ranking_agent_v2`, produces `ranked_issues`.

**`summariser_node`** - produces `report_md` (markdown string).

### `graph/workflow.py`

`build_graph(checkpointer, interactive)` constructs and compiles the
`StateGraph`. Attaches `_route` as a conditional edge from START and from every
worker node except `summariser`.

`_route(state)` is the supervisor: a pure function returning the next node name.
No LLM, no randomness. Routing is determined entirely by the completion flags in
state.

### `ranking/impact.py`

`ranking_agent_v2(issues, table_sizes, suppress_noise)` scores, sorts, and
optionally noise-suppresses a list of issues. Issues with BIS below
`BIS_NOISE_THRESHOLD` (0.05) are dropped.

`score_issue_v2(issue, table_sizes)` computes the four-component BIS and returns
an `ImpactScoreV2` breakdown for audit purposes.

### `evaluation/metrics.py`

**Three strategies**

- `run_strategy_a` - skip drift and debug nodes.
- `run_strategy_b` - skip drift only.
- `run_strategy_c` - full pipeline.

All strategies set `ASTRION_AUTO_APPROVE=1` to bypass human review interrupts.

**`_match(pred, gt)`** - three-condition match: type equivalence, same table,
column overlap when both sides have non-empty column lists.

**`compute_metrics`** - one-to-one matching for both overall precision/recall and
top-k summary accuracy. Each ground-truth issue can satisfy at most one
prediction.

**`evaluate_all`** - runs all three strategies, computes metrics against the
injected ground truth, and writes `outputs/evaluation_comparison.json`.

### `injectors/retail_issues.py`

Injects seven controlled issue types into a copy of the retail tables.

Uses `infer_metadata` to select columns by role rather than by suffix pattern.
This handles the actual retail schema which uses `_sk` for surrogate FK columns
and `_id` for natural keys.

Column selection:
- `missing_key_values` uses the fact's detected primary key.
- `referential_integrity_break` uses the first FK column from `meta.foreign_keys`.
- Other injections use `meta.date_cols`, `meta.numeric_cols`, `meta.promo_cols`.

### `warehouse/loader.py`

Manages a module-level DuckDB singleton connection (`_CONN`). All tables are
registered under the `dq_retail` schema so `IssueVerifier` can reference them
with fully-qualified names.

- `load_tables_to_duckdb(tables)` - registers DataFrames, stores connection.
- `get_connection()` - returns `_CONN`; raises if not initialised.
- `close_connection()` - closes and clears `_CONN`.
- `load_retail_tables(source)` - loads CSVs from `data/raw/retail/` (clean) or
  `data/injected/retail/` (injected).

---

## Data Flow: `astrion-dq triage --source injected`

1. CLI parses arguments, sets `ASTRION_AUTO_APPROVE=1`, calls `build_graph()`.
2. `initial_state(source="injected")` creates a zeroed `TriageState`.
3. `graph.invoke(state)` enters the routing loop.
4. `_route` returns `"data_loader"`. `data_loader_node` loads CSVs, registers
   tables in DuckDB, returns `data_loaded=True`.
5. `_route` returns `"profiler"`. `profiler_node` infers metadata, returns
   `metadata_ready=True`.
6. `_route` returns `"detector"`. `detector_node` runs five parallel checks,
   returns `raw_issues` list and `detection_done=True`.
7. `_route` returns `"drift_detector"`. `drift_detector_node` computes PSI + KS,
   merges into `all_issues`, returns `drift_done=True`. If no snapshot exists,
   drift returns an empty list and logs a warning.
8. `_route` returns `"debugger"`. `debugger_node` runs `IssueVerifier` on all
   issues, returns `verified_issues` with confidence scores and `debug_done=True`.
9. `needs_human_review` is True if any confidence is below the threshold.
   `_route` returns `"human_review"`. `ASTRION_AUTO_APPROVE=1` short-circuits
   the interrupt, returns `review_done=True`.
10. `_route` returns `"ranker"`. `ranker_node` scores and sorts issues, returns
    `ranked_issues`.
11. `_route` returns `"summariser"`. `summariser_node` writes `report_md`.
12. `summariser` edge points unconditionally to END.
13. CLI writes `ranked_issues.json` and `triage_report.md` to `outputs/`.

---

## Evaluation Framework

The three strategies are implemented via skip flags on the initial state:

```python
# Strategy A: skip drift and debug
state["drift_done"] = True
state["debug_done"] = True
state["review_done"] = True

# Strategy B: skip drift only
state["drift_done"] = True

# Strategy C: full pipeline (no flags pre-set)
```

`_route` reads these flags and bypasses nodes that are already marked done.
This reuses the same compiled graph for all three strategies without
duplicating any node logic.

---

## V2 Business Impact Score

```
BIS = base_weight x severity_weight x evidence_density x report_criticality

evidence_density = log(1 + evidence_rows) / log(1 + table_total_rows)
```

The log-normalisation is borrowed from TF-IDF: the marginal cost of each
additional bad row decreases as the total row count grows. This prevents a
10M-row fact table from always dominating a 50K-row dimension table even when
the dimension issue has higher business impact.

---

## Extension Points

**Adding a new check**

1. Add a function `detect_<name>(tables, meta, sensitivity)` in `checks/detect.py`.
2. Add a corresponding SQL verification branch in `graph/debugger.py::_dispatch`.
3. Add the check to the `checks` list in `run_all_checks_parallel`.

**Adding a new drift metric**

1. Extend the `detect_drift` function in `checks/drift.py`.
2. The snapshot format (`save_snapshot`) may need new keys; existing snapshots
   will ignore unknown keys gracefully.

**Adding a new evaluation strategy**

1. Add a `run_strategy_<x>` function in `evaluation/metrics.py` that pre-sets
   the appropriate skip flags on the initial state.
2. Add the runner to the `runners` list in `evaluate_all`.
