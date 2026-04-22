# Changelog

## 0.5.0 (2026-04-22) -- REST API, Auth, Audit Log, Dashboard

### New features

- **REST API** (`src/astrion_dq/api/app.py`): FastAPI server with three endpoints.
  - `GET /health` -- unauthenticated liveness probe.
  - `POST /triage` -- synchronous pipeline run; returns `run_id`, `ranked_issues`,
    `agent_trace`, `issue_count`. Accepts `source` and `sensitivity` in request body.
    Invalid values return HTTP 422.
  - `GET /runs/{run_id}` -- look up a past run from `run_log.jsonl`; 404 if not found.
  - Start with: `astrion-dq serve [--host HOST] [--port N]` (new CLI command).
  - Runtime deps added: `fastapi>=0.110.0`, `uvicorn[standard]>=0.29.0`.
  - Dev dep added: `httpx>=0.27.0` (for `fastapi.testclient.TestClient`).

- **Bearer token auth** (P6c): set `ASTRION_API_TOKEN` env var to require a token on
  all API endpoints except `/health`. When unset, auth is disabled (dev mode).

- **Streamlit password gate**: when `ASTRION_API_TOKEN` is set, the dashboard renders
  a token input form before showing any content. `st.stop()` enforces the gate.

- **Run audit log** (P6a): every `astrion-dq triage` invocation appends one JSON line
  to `outputs/run_log.jsonl` containing `run_id` (12-char hex), `source`, `sensitivity`,
  `timestamp` (ISO 8601 UTC), `issue_count`, `agent_trace`.
  `POST /triage` also appends to the same log.

- **Dashboard: Run History tab**: fifth tab reads `run_log.jsonl` and renders a table
  of all past runs (run_id, source, sensitivity, timestamp, issue_count, agent_trace).

- **Dashboard: debounce** (P6a): all sidebar pipeline buttons are disabled while a
  subprocess is running via `st.session_state["_running"]`.

### Hardening (P5)

- `--source` and `--sensitivity` CLI options now use `str, Enum` types (`Source`,
  `Sensitivity`). Invalid values are rejected at the CLI boundary with exit code 2.
  Fixed Python 3.11 `f"{enum}"` formatting: f-strings now use `.value`.
- `build_graph()` raises `RuntimeError` immediately when `checkpointer is not None`
  or `interactive=True`. Removes misleading MemorySaver suggestion from the docstring.
  `MemorySaver` import removed from `workflow.py`.
- `QualityIssue` dataclass gains `dim_table: str = ""` and `dim_pk: str = ""` fields.
  `detect_referential_breaks` populates them. `IssueVerifier._verify_ri_break` reads
  them directly -- eliminates fragile regex parsing from issue descriptions.
- `evaluation/metrics.py`: replaced `state.update(shared_data)` with explicit key
  assignments to fix mypy TypedDict error. mypy is now fully blocking in CI.
- `LICENSE` file added (MIT).

### Quality gates

- CI coverage gate raised from 60% to 70% (actual: 70%).
- mypy `|| true` removed -- type check is now a hard CI gate.
- 9 new API tests in `tests/test_api.py`; 3 new audit log tests in
  `tests/test_run_audit_log.py`. Total: 62 tests.

### Evaluation baseline (2026-04-22, v0.5.0)

| Strategy | Precision | Recall | F1 | Noise | SumAcc | Wall(s) |
|---|---|---|---|---|---|---|
| A_baseline | 1.000 | 0.750 | 0.857 | 0.000 | 1.000 | 14.9 |
| B_supervisor | 1.000 | 0.750 | 0.857 | 0.000 | 1.000 | 13.7 |
| C_full | 1.000 | 0.750 | 0.857 | 0.000 | 1.000 | 9.0 |

DoD targets met (F1 > 0.80 for A/B, F1 > 0.85 for C). One FN remains
(`numeric_outliers` -- IQR detector does not fire at normal sensitivity on
current injected data). Accepted as known limitation.

---

## 0.4.0 (2026-04-21) -- BREAKING, OpenRouter LLM, CI

### Breaking changes

- `triage` command now writes `ranked_issues_{source}.json` and
  `triage_report_{source}.md`. The legacy unsuffixed `ranked_issues.json`
  and `triage_report.md` are no longer written. Update any downstream scripts
  or integrations that read these paths directly.
- `report` command gains a required `--source` flag (default `injected`).
  Callers that previously relied on `astrion-dq report` reading a single
  `ranked_issues.json` must now pass `--source injected` or `--source clean`.

### Improvements

- F-03: `dashboard/app.py` `load_ranked_issues` and `load_report_md` accept a
  `source` parameter; the sidebar data-source selector now correctly gates which
  files are displayed.
- F-12: `save_snapshot` stores `q_probs`/`q_values` (p1, p5, p10, p25, p50, p75,
  p90, p95, p99) alongside histogram bins. `detect_drift` uses `np.interp` CDF
  reconstruction from these quantiles when present, falling back to histogram
  midpoints for legacy snapshots. Eliminates systematic underestimation of drift
  on heavy-tailed columns (total_amount, discount, refund).
  Note: existing `outputs/drift_snapshots/` files must be regenerated.
- F-14: `evaluate_all` calls `_prepare_data` once and shares the loaded tables
  and DuckDB connection across all three strategy runs. `close_connection` is
  called once at the end instead of three times. Reduces CSV I/O by ~2/3 for
  large datasets.

### OpenRouter LLM integration

- `src/astrion_dq/llm/client.py`: thin OpenRouter client using the OpenAI SDK
  wire format. Raises `LLMUnavailable` when `OPENROUTER_API_KEY` is unset --
  all callers fall back gracefully.
- `summariser_node`: when `OPENROUTER_API_KEY` is configured, an LLM-generated
  executive summary (3-5 sentences + recommended actions) is prepended to the
  markdown report under `## Executive Summary`. Deterministic template always
  runs regardless of LLM availability.
- `config/.env.example` updated with `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`,
  `LLM_TOP_N` settings. `config.py` auto-loads `config/.env` via python-dotenv.
- `openai>=1.0.0` added as a runtime dependency.

### CI/CD (P4, F-06)

- `.github/workflows/ci.yml`: two jobs -- `test` (ruff + mypy + pip-audit +
  pytest --cov) on every push/PR; `docker` (image build + CLI smoke test) on
  main pushes only.
- `pyproject.toml` dev extras: `pytest-cov>=4.0`, `mypy>=1.0`, `pip-audit>=2.0`.
- Coverage gate: 60% (67% actual). Raise to 70% in P5 once report/pdf.py and
  loader tests are added. mypy and pip-audit are non-blocking until P5.
- ruff config migrated from deprecated top-level `[tool.ruff]` to
  `[tool.ruff.lint]` -- eliminates the deprecation warning on every run.

---

## 0.3.0 (2026-04-20) -- BREAKING

### Breaking changes

- `duplicate_rows.evidence_rows` now counts excess copies only (the N records
  that should not exist). Previously counted all rows in duplicate groups
  (2 * N for N pairs). Downstream consumers that threshold or display
  `evidence_rows` for `duplicate_rows` issues must account for this halving.
  See migration note in README.

### Bug fixes

- F-01: `detect_drift` now accepts a required `meta` argument and gates column
  scanning to `meta[table].numeric_cols`. Eliminates 16 false-positive drift
  signals on surrogate key columns (sales_sk, customer_sk, etc.).
- F-16: `detect_nulls` augments the `important` threshold set with all
  `is_key_col(c)` columns regardless of PK inference outcome. Fixes silenced
  2% null detection on `sales_sk` when injected duplicates break PK uniqueness.
- F-17: `detect_future_dates` passes `format='ISO8601'` to `pd.to_datetime`.
  Fixes silent NaT coercion on ArrowStringArray columns with mixed datetime
  and date-only ISO 8601 strings (pandas 2.x regression).

### Infrastructure

- Dockerfile HEALTHCHECK updated to `astrion-dq --help` (was: non-existent
  `astrion-dq profile`). Default CMD updated to `triage --source clean`.
- docker-compose pipeline command updated to `triage --source injected`.
