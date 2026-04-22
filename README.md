# Astrion DQ

**Capstone Project 7 - Evaluating Agentic Workflows for Data Quality Triage**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/jaideepgarlyal/Astrion_capstone_7/actions/workflows/ci.yml/badge.svg)](https://github.com/jaideepgarlyal/Astrion_capstone_7/actions/workflows/ci.yml)

---

## What is Astrion DQ?

Astrion DQ is a retail data quality triage system built as an academic capstone. It loads
a retail star schema into an in-process DuckDB database, injects synthetic quality issues
as ground truth, detects issues via five rule-based checks plus statistical drift detection,
cross-validates every issue against an independent DuckDB SQL query, ranks issues by a
V2 Business Impact Score, and evaluates three ablation strategies against the injected
ground truth.

The pipeline is exposed as a CLI, a Streamlit dashboard, and a FastAPI REST server.

---

## Architecture

### LangGraph workflow with deterministic supervisor routing

The pipeline is a LangGraph `StateGraph`. A single deterministic `_route()` function reads
completion flags from shared state and selects the next node. No LLM routing is used.

**Nodes**

```
data_loader    load CSVs, register tables in DuckDB
profiler       infer table roles, PKs, FKs, column types
detector       five parallel checks: nulls, duplicates, outliers, dates, RI
drift_detector PSI + KS statistical drift against a saved baseline snapshot
debugger       SQL cross-validation, per-issue confidence score
human_review   interrupt for analyst input (auto-approved in evaluation runs)
ranker         V2 Business Impact Score, descending sort
summariser     markdown triage report + optional LLM executive summary
```

**Graph topology**

```
START -> _route -> data_loader -> _route -> profiler -> _route -> detector
      -> _route -> drift_detector -> _route -> debugger -> _route
      -> human_review -> _route -> ranker -> _route -> summariser -> END
```

### Three evaluation strategies

```
A Baseline  : data_loader -> profiler -> detector -> ranker
B Supervisor: A + debugger + human_review
C Full      : B + drift_detector
```

`human_review` is auto-approved in evaluation runs via the
`ASTRION_AUTO_APPROVE=1` environment variable.

---

## Quick Start

```bash
pip install -e ".[dev]"

# 1. Save a baseline drift snapshot (run once on clean data)
astrion-dq snapshot

# 2. Inject synthetic quality issues (creates ground truth)
astrion-dq inject --seed 42

# 3. Run the full triage workflow
ASTRION_AUTO_APPROVE=1 astrion-dq triage --source injected

# 4. Evaluate all three strategies against injected ground truth
astrion-dq evaluate --source injected

# 5. Generate a PDF triage report
astrion-dq report --source injected

# 6. Launch the Streamlit dashboard
astrion-dq dashboard

# 7. Start the REST API server
astrion-dq serve --port 8000
```

---

## CLI Reference

```
astrion-dq snapshot  [--tag TAG]                     Save baseline drift snapshot
astrion-dq inject    [--seed N]                      Inject synthetic issues (ground truth)
astrion-dq triage    [--source clean|injected]       Run triage workflow
                     [--sensitivity normal|high]
                     [--auto-approve|--interactive]
astrion-dq evaluate  [--source clean|injected]       Compare strategies A, B, C
astrion-dq report    [--source clean|injected]       Generate PDF triage report
astrion-dq dashboard [--port N]                      Launch Streamlit dashboard
astrion-dq serve     [--host HOST] [--port N]        Start FastAPI REST server
```

Invalid values for `--source` or `--sensitivity` are rejected at the CLI boundary with
a clean error (exit 2). Every triage run appends one record to `outputs/run_log.jsonl`.

---

## REST API

Start the server:

```bash
# Development (no auth)
astrion-dq serve

# Production (token required)
ASTRION_API_TOKEN=my-secret-token astrion-dq serve
```

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | None | Liveness probe |
| POST | `/triage` | Bearer | Run full pipeline, return ranked issues |
| GET | `/runs/{run_id}` | Bearer | Look up a past run from run_log.jsonl |

### POST /triage

Request body:

```json
{
  "source": "injected",
  "sensitivity": "normal"
}
```

Response:

```json
{
  "run_id": "a3f1c9b20d44",
  "source": "injected",
  "sensitivity": "normal",
  "issue_count": 3,
  "ranked_issues": [...],
  "agent_trace": ["data_loader", "profiler", "detector", "debugger", "ranker", "summariser"]
}
```

### Authentication

Set `ASTRION_API_TOKEN` to enable Bearer token auth on the API and the Streamlit dashboard.
When the variable is unset, both services run in unauthenticated dev mode.

```bash
# API call with auth
curl -X POST http://localhost:8000/triage \
  -H "Authorization: Bearer my-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"source": "injected"}'
```

---

## Evaluation Results

Command: `astrion-dq evaluate --source injected`
Run date: 2026-04-22, version 0.5.0.
Ground truth: 4 issues (missing_key_values, duplicate_transactions,
invalid_future_dates, numeric_outliers).

| Strategy | Predicted | TP | FP | FN | Precision | Recall | F1 | Noise | SumAcc | Wall(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| A_baseline | 3 | 3 | 0 | 1 | 1.000 | 0.750 | 0.857 | 0.000 | 1.000 | 14.9 |
| B_supervisor | 3 | 3 | 0 | 1 | 1.000 | 0.750 | 0.857 | 0.000 | 1.000 | 13.7 |
| C_full | 3 | 3 | 0 | 1 | 1.000 | 0.750 | 0.857 | 0.000 | 1.000 | 9.0 |

DoD targets: F1 > 0.80 for A/B (met: 0.857), F1 > 0.85 for C (met: 0.857).

Precision = 1.000 and Noise = 0.000 across all strategies. The 1 FN is
`numeric_outliers`: the IQR detector does not fire on the current injected dataset
at normal sensitivity. Switching to `--sensitivity high` detects it at the cost of
additional false positives on other columns.

---

## V2 Business Impact Score

```
BIS = base_weight x severity_weight x evidence_density x report_criticality

base_weight
    referential_integrity_break   4.0
    duplicate_rows                3.5
    empty_table                   3.0
    numeric_outliers              2.8
    statistical_drift             2.5
    missing_values                2.3
    invalid_future_dates          2.1

severity_weight
    high: 3.0   medium: 2.0   low: 1.0

evidence_density
    log(1 + evidence_rows) / log(1 + table_total_rows)
    Log-normalisation prevents large tables from always dominating smaller ones.

report_criticality
    sum(downstream_report_weights) / max_possible_score
    daily_sales_summary:       1.00
    promotion_performance:     0.90
    sales_by_store:            0.85
    sales_by_product_category: 0.85
    inventory_replenishment:   0.80
    top_products:              0.70
    customer_segmentation:     0.65
```

---

## Project Structure

```
src/astrion_dq/
    api/
        app.py             FastAPI REST server (POST /triage, GET /runs/{id})
    checks/
        detect.py          Five rule-based detectors (nulls, dups, outliers, dates, RI)
        drift.py           Statistical drift: PSI + KS test
    evaluation/
        metrics.py         Three-strategy evaluation framework (A / B / C)
    graph/
        debugger.py        IssueVerifier: SQL cross-validation
        nodes.py           LangGraph node functions
        state.py           TriageState TypedDict
        workflow.py        build_graph(), _route() supervisor function
    injectors/
        retail_issues.py   Seven synthetic issue types (ground truth)
    llm/
        client.py          OpenRouter client (optional LLM executive summary)
    ranking/
        impact.py          V2 Business Impact Score
    report/
        pdf.py             ReportLab PDF generator
    warehouse/
        loader.py          DuckDB warehouse loader, CSV ingestion
    metadata.py            Schema inference helpers (infer_metadata)
    config.py              All thresholds, weights, paths
    models.py              QualityIssue, VerifiedIssue, RankedIssue, TableMeta
    cli.py                 Typer CLI (inject/snapshot/triage/evaluate/report/dashboard/serve)

dashboard/
    app.py                 Streamlit dashboard (5 tabs: Issues, Comparison, Report, History, Architecture)

data/raw/retail/           Six dimension CSVs + fact_sales_normalized.csv
outputs/                   Pipeline outputs (JSON, markdown, PDF, snapshots, run_log.jsonl)
tests/                     62 pytest tests, 70% coverage gate
```

---

## Dataset

Retail star schema. Six dimension tables and one fact table committed to `data/raw/retail/`:

- `dim_customers.csv`
- `dim_products.csv`
- `dim_stores.csv`
- `dim_dates.csv`
- `dim_campaigns.csv`
- `dim_salespersons.csv`
- `fact_sales_normalized.csv`

---

## Injected Issue Types

| Issue type | Table | Fraction | Severity |
|---|---|---|---|
| `missing_key_values` | fact | 2% | high |
| `duplicate_transactions` | fact | 2% | high |
| `invalid_future_dates` | fact | 2% | medium |
| `referential_integrity_break` | fact | 2% | high |
| `numeric_outliers` | fact | 2% | high |
| `promotion_drift` | fact | 2% | medium |
| `dimension_missing_values` | dim_customers | 3% | medium |

---

## Configuration

All thresholds are in `src/astrion_dq/config.py`. Environment variables loaded from
`config/.env` (gitignored) and `config/.env.example` (template, safe to commit):

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | (empty) | OpenRouter API key. When unset, LLM summary is skipped. |
| `OPENROUTER_MODEL` | `anthropic/claude-sonnet-4-6` | Model for executive summary |
| `ASTRION_API_TOKEN` | (empty) | Bearer token for API and dashboard. Unset = dev mode. |
| `ASTRION_AUTO_APPROVE` | `0` | Set to `1` to auto-approve human_review_node |

---

## Docker

```bash
# Build image
docker build -t astrion-dq .

# Run triage
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -e ASTRION_AUTO_APPROVE=1 \
  astrion-dq \
  astrion-dq triage --source injected

# Launch dashboard on port 8503
docker run -p 8503:8503 \
  -v $(pwd)/outputs:/app/outputs \
  astrion-dq \
  streamlit run dashboard/app.py --server.port 8503

# Full stack (dashboard + pipeline service)
docker-compose up dashboard

# Start REST API
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -e ASTRION_API_TOKEN=my-secret-token \
  astrion-dq \
  astrion-dq serve --host 0.0.0.0 --port 8000
```

---

## Development

```bash
pip install -e ".[dev]"

pytest tests/ -v                          # run all 62 tests
ruff check src/                           # lint
mypy src/astrion_dq/ --ignore-missing-imports --no-strict-optional  # type check
```

Coverage gate: 70%. CI runs on every push and PR via `.github/workflows/ci.yml`.

---

## Migration Notes

**v0.5.0:** FastAPI REST server (`astrion-dq serve`), Bearer token auth
(`ASTRION_API_TOKEN`), run audit log (`outputs/run_log.jsonl`), dashboard password gate,
run history tab, CI coverage gate raised to 70%.

**v0.4.0:** `triage` and `report` commands now write/read `ranked_issues_{source}.json`
and `triage_report_{source}.md`. The legacy unsuffixed files are no longer written.
The `report` command gains a required `--source` flag (default: `injected`).

**v0.3.0:** `duplicate_rows.evidence_rows` now counts excess copies only (was: all rows
in duplicate groups). For N duplicate pairs the value is now N instead of 2N. Update
any downstream threshold or display logic that reads this field.

---

*Astrion DQ v0.5.0 - Capstone 7 - Dr. William So, Synogize*
