# Astrion DQ

**Automatic data quality checker for retail databases — built as an academic capstone project.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-112%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What does this project actually do?

Imagine you work at a retail company and your team loads sales data into a database every day. Sometimes that data has problems — missing values, duplicate records, broken relationships between tables, numbers that are way too high or too low. These problems are hard to catch manually and they silently corrupt every report built on top of the data.

**Astrion DQ automatically finds those problems, ranks them by how much business damage they cause, and tells you exactly how to fix them with ready-to-run SQL.**

It works in three steps:

1. **Detect** — Scans your data using 7 types of checks (nulls, duplicates, outliers, broken foreign keys, future dates, empty tables, statistical drift)
2. **Rank** — Scores every issue by how much it would break your downstream reports (using a Business Impact Score)
3. **Report** — Produces a plain-English markdown report with SQL fix code for each issue

You interact with it through a command-line tool, a web dashboard, or a REST API.

---

## Who is this for?

- **Data engineers** who want to validate data before loading it into a warehouse
- **Analytics engineers** who need to trust the data their dashboards are built on
- **Students / researchers** learning about data quality, LangGraph, or agentic AI systems
- **Developers** who want a working example of a LangGraph pipeline with a FastAPI + Streamlit front end

No machine learning background is required. The detection logic is rule-based and fully deterministic — results are reproducible every time.

---

## Prerequisites

You need these installed on your machine before anything else:

| Tool | Why | Install |
|---|---|---|
| Python 3.11 or higher | The project runs on Python | [python.org](https://www.python.org/downloads/) |
| pip or uv | To install Python packages | Comes with Python |
| Git | To clone the repo | [git-scm.com](https://git-scm.com) |

**Optional but unlocks AI features:**
- An [OpenRouter API key](https://openrouter.ai/keys) — adds an LLM executive summary to reports and powers the SQL Assistant chat tab in the dashboard. The system works completely without it.

---

## Getting started (5 minutes)

### 1. Clone and install

```bash
git clone https://github.com/Kinshuk78/Astrion_capstone_7.git
cd Astrion_capstone_7

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows

# Install the package and all dependencies
pip install -e ".[dev]"
```

### 2. Set up your API key (optional)

```bash
cp config/.env.example config/.env
# Open config/.env in any text editor and paste your OpenRouter key:
# OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

If you skip this step, everything still works — you just won't get the LLM-powered summary or SQL Agent chat.

### 3. Run your first triage

```bash
# Step 1: Save a baseline snapshot of your clean data (run once)
astrion-dq snapshot

# Step 2: Inject fake problems into a copy of the data (creates test ground truth)
astrion-dq inject

# Step 3: Run the full triage
astrion-dq triage --source injected

# Step 4: Open the dashboard to see results
astrion-dq dashboard
# Visit http://localhost:8501 in your browser
```

That's it. You should see ranked issues, a resolution report, and strategy comparison charts.

---

## Project structure explained

```
Astrion_capstone_7/
│
├── src/astrion_dq/          <- All the Python source code
│   ├── api/app.py           <- REST API (FastAPI)
│   ├── checks/
│   │   ├── detect.py        <- The 7 issue detectors (this is where detection happens)
│   │   └── drift.py         <- Statistical drift detection (PSI + KS test)
│   ├── graph/
│   │   ├── nodes.py         <- Each step of the pipeline as a function
│   │   ├── workflow.py      <- Wires the steps together using LangGraph
│   │   ├── debugger.py      <- Double-checks detections using SQL queries
│   │   └── state.py         <- The data container passed between pipeline steps
│   ├── ranking/impact.py    <- Business Impact Score calculation
│   ├── report/pdf.py        <- PDF report generator
│   ├── llm/client.py        <- OpenRouter LLM connection (optional)
│   ├── models.py            <- Data structures (QualityIssue, RankedIssue, etc.)
│   ├── config.py            <- All settings, thresholds, and weights in one place
│   └── cli.py               <- The command-line interface (astrion-dq ...)
│
├── dashboard/app.py         <- Streamlit web dashboard (7 tabs)
├── data/raw/retail/         <- The sample retail dataset (7 CSV files)
├── outputs/                 <- Where results are written (JSON, markdown, PDF)
├── tests/                   <- 112 automated tests
├── Dockerfile               <- Container build file
├── docker-compose.yml       <- Run everything with Docker
└── render.yaml              <- One-click cloud deployment on Render.com
```

---

## How the pipeline works

The pipeline is built with **LangGraph** — a library that lets you define a workflow as a graph of steps. Think of it like an assembly line where each station does one specific job and passes results to the next.

```
Your CSV data
     │
     ▼
[data_loader]   ── Reads CSV files, loads them into an in-memory database
     │
     ▼
[profiler]      ── Figures out which columns are primary keys, foreign keys, dates, numbers
     │
     ▼
[detector]      ── Runs 7 checks in parallel to find issues
     │
     ▼
[drift_detector]── Compares distributions against a saved baseline (catches gradual changes)
     │
     ▼
[debugger]      ── Re-checks every issue with an independent SQL query (reduces false positives)
     │
     ▼
[human_review]  ── Flags low-confidence issues for analyst approval (auto-skipped in batch mode)
     │
     ▼
[ranker]        ── Scores every issue by business impact, sorts highest first
     │
     ▼
[summariser]    ── Writes the markdown report with SQL fix code
     │
     ▼
outputs/triage_report_injected.md   ← Your report
outputs/ranked_issues_injected.json ← Machine-readable results
```

**Important:** There is no AI making routing decisions. The pipeline uses a plain Python `if/else` function to decide which step runs next. The word "agentic" refers to the multi-step autonomous workflow, not to LLM decision-making.

---

## The 7 issue types it detects

| Issue | What it means | Example |
|---|---|---|
| `missing_values` | A column has NULL / empty values | `customer_id` is blank in 5% of rows |
| `duplicate_rows` | The same record appears more than once | Transaction `#1234` recorded twice |
| `numeric_outliers` | A number is impossibly large or small | A sale for $999,999 when average is $50 |
| `invalid_future_dates` | A date is set far in the future | Order date is `2099-01-01` |
| `referential_integrity_break` | A foreign key points to a record that doesn't exist | `store_id = 999` but store 999 doesn't exist in `dim_stores` |
| `statistical_drift` | Column distributions have shifted significantly compared to baseline | Average order value jumped 40% overnight |
| `empty_table` | A table has zero rows | `dim_products` is completely empty |

---

## Business Impact Score (BIS)

Not all issues are equal. A null in a rarely-used column is less urgent than a broken foreign key that breaks your daily CEO dashboard. The BIS formula prioritises issues by actual business damage:

```
BIS = base_weight × severity_weight × evidence_density × report_criticality

base_weight        How dangerous is this issue type?
                   referential_integrity_break = 4.0 (highest)
                   duplicate_rows = 3.5
                   missing_values = 2.3
                   ...

severity_weight    How bad is it in this specific case?
                   high = 3.0, medium = 2.0, low = 1.0

evidence_density   How many rows are affected? (log-scaled so 1 bad row
                   in 100 is treated differently to 1 bad row in 10,000,000)

report_criticality Which downstream reports does this break?
                   daily_sales_summary = 1.00 (breaks the most important report)
                   customer_segmentation = 0.65 (breaks a less critical report)
```

Issues are always shown in order from highest BIS to lowest.

---

## CLI commands

```bash
astrion-dq snapshot              # Save baseline statistics for drift comparison
astrion-dq inject                # Inject synthetic issues into a copy of the data
astrion-dq triage                # Run the full detection pipeline (default: injected data)
astrion-dq triage --source clean # Run on the original unmodified data
astrion-dq evaluate              # Compare all 3 strategies (A/B/C) and print F1 scores
astrion-dq report                # Generate a PDF report from the last triage run
astrion-dq dashboard             # Open the Streamlit web dashboard
astrion-dq serve                 # Start the REST API server (port 8000)
```

Every triage run logs a record to `outputs/run_log.jsonl` so you can audit what ran and when.

---

## Dashboard

Launch with `astrion-dq dashboard` then open [http://localhost:8501](http://localhost:8501).

The dashboard has 7 tabs:

| Tab | What you see |
|---|---|
| **Triage Issues** | All detected issues ranked by BIS, with charts |
| **Strategy Comparison** | F1/precision/recall table for strategies A, B, C |
| **Markdown Report** | The full triage report with SQL fix code |
| **Run History** | Every previous triage run with timestamps |
| **Architecture** | Diagram of the pipeline and how it works |
| **SQL Assistant** | Chat with an LLM that knows your database schema. Ask it to explain issues, write investigation queries, or fix SQL errors. SQL blocks in responses are auto-executed and results shown inline. Requires `OPENROUTER_API_KEY`. |
| **Upload & Analyze** | Upload any CSV file(s). The system runs a full triage automatically — no injection step needed. Results show in the same format. |

---

## REST API

Start the server:

```bash
astrion-dq serve                             # No auth (dev mode)
ASTRION_API_TOKEN=mysecret astrion-dq serve  # Require Bearer token
```

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Liveness check — always returns `{"status": "ok"}` |
| `/triage` | POST | Submit a triage job, returns `job_id` immediately (async) |
| `/jobs/{job_id}` | GET | Poll for results — `status` is `"running"` then `"done"` |
| `/runs/{run_id}` | GET | Look up a past run from the audit log |
| `/triage/report.pdf` | GET | Download a PDF report for the last triage |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

### Example: submit a job and poll for results

```bash
# Submit triage job
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"source": "injected"}'
# Returns: {"job_id": "abc123", "status": "running", "poll_url": "/jobs/abc123"}

# Poll until done
curl http://localhost:8000/jobs/abc123
# Returns: {"status": "done", "result": {"issue_count": 4, "ranked_issues": [...], ...}}

# Download PDF
curl http://localhost:8000/triage/report.pdf -o report.pdf
```

Rate limit: 10 requests per minute on `/triage`. Change via `RATE_LIMIT` environment variable.

---

## Evaluation strategies

When you run `astrion-dq evaluate`, it runs three versions of the pipeline and compares them against the known ground truth (the issues injected by `astrion-dq inject`):

| Strategy | What it includes | Best for |
|---|---|---|
| **A Baseline** | Detection + ranking only | Speed, simple use cases |
| **B Supervisor** | A + SQL cross-validation + analyst review gate | Higher confidence, fewer false positives |
| **C Full** | B + statistical drift detection | Catching gradual data corruption |

Current results (v0.6.0, injected data, high sensitivity):

| Strategy | Precision | Recall | F1 | Noise Rate |
|---|---|---|---|---|
| A Baseline | 1.000 | 0.750 | 0.857 | 0.000 |
| B Supervisor | 1.000 | 0.750 | 0.857 | 0.000 |
| C Full | 1.000 | 0.750 | 0.857 | 0.000 |

Precision = 1.0 means zero false positives. The 0.25 recall gap is one missed issue (`numeric_outliers`) that falls just below the detection threshold on the synthetic dataset.

---

## Docker (run without installing Python)

```bash
# Build the image once
docker build -t astrion-dq .

# Run triage
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -e ASTRION_AUTO_APPROVE=1 \
  astrion-dq \
  astrion-dq triage --source injected

# Launch dashboard on port 8501
docker run -p 8501:8501 \
  -v $(pwd)/outputs:/app/outputs \
  astrion-dq \
  streamlit run dashboard/app.py --server.port 8501 --server.headless true

# Run everything at once (pipeline + dashboard + API)
docker-compose up
```

---

## Deploy to Render.com (free cloud hosting)

Render.com can host both the dashboard and the API for free. The `render.yaml` file in this repo is a blueprint that sets everything up in one click.

**Steps:**

1. Fork this repo to your own GitHub account
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**
3. Connect your GitHub repo — Render detects `render.yaml` automatically
4. Click **Apply** — it creates two services: `astrion-dq-api` and `astrion-dq-dashboard`
5. In the Render dashboard, set these secret environment variables for both services:
   - `OPENROUTER_API_KEY` — your OpenRouter key (optional but recommended)
   - `ASTRION_API_TOKEN` — any password you choose, or leave blank for open access
6. Wait ~3 minutes for the build to finish

Your live URLs will be:
- Dashboard: `https://astrion-dq-dashboard.onrender.com`
- API docs: `https://astrion-dq-api.onrender.com/docs`

> **Free tier note:** Render's free tier pauses services after 15 minutes of no traffic. The first visit after a pause takes 30-60 seconds to wake up. This is normal. Upgrade to Starter ($7/month) for always-on.

---

## Configuration reference

All settings live in `src/astrion_dq/config.py`. You can override most of them with environment variables in `config/.env`.

| Variable | Default | What it does |
|---|---|---|
| `OPENROUTER_API_KEY` | (empty) | LLM API key. Without it, AI features are disabled but everything else works. |
| `OPENROUTER_MODEL` | `anthropic/claude-sonnet-4-6` | Which AI model to use for summaries and the SQL Agent |
| `ASTRION_API_TOKEN` | (empty) | Password for the API and dashboard. Leave blank for open access. |
| `RATE_LIMIT` | `10/minute` | How many API triage requests are allowed per IP per minute |
| `DETECTION_SENSITIVITY` | `high` | `high` = catch more issues (tighter thresholds). `normal` = fewer, more certain catches. |
| `LOG_LEVEL` | `INFO` | Set to `DEBUG` to see detailed pipeline logs |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all 112 tests
pytest tests/ -v

# Run just the new feature tests
pytest tests/test_resolution_advice.py tests/test_upload_triage.py tests/test_sql_agent.py -v

# Lint
ruff check src/

# Type check
mypy src/astrion_dq/ --ignore-missing-imports --no-strict-optional
```

---

## Using gstack (AI-assisted development tool)

If you want to contribute to this project using AI coding tools (Claude Code, etc.), the project was originally set up with [gstack](https://github.com/garrytan/gstack) — a toolkit that gives Claude Code extra skills like `/qa`, `/ship`, `/investigate`, `/browse`, and `/review`.

**You do not need gstack to use or develop this project.** It is purely an optional developer productivity tool.

**Install gstack if you want it:**

```bash
git clone --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
cd ~/.claude/skills/gstack && ./setup --team
```

**What it gives you:**

| Command | What it does |
|---|---|
| `/qa` | Runs a full quality assurance pass on the code in a browser |
| `/review` | Reviews open pull requests |
| `/ship` | End-to-end: write code, test, commit, and push |
| `/investigate` | Analyses a bug or performance issue |
| `/browse` | Opens a browser for web research during coding |

After installing gstack, these slash commands are available inside Claude Code sessions. The project has no hard dependency on gstack — all code, tests, and deployment work without it.

---

## Current limitations

These are real limitations of the current version, not things to hide:

**1. Single-user only**
The DuckDB database runs in memory and is shared by the whole process. If two people run a triage at the same time through the API, only one will succeed. The API queues them with a lock, but it's not designed for multi-user production use.

**2. Data is not saved between restarts**
When deployed on Render.com (or any ephemeral server), output files like `run_log.jsonl` and ranked issues JSON are lost when the server restarts. For persistent storage you would need to add an external database or object storage (S3, Supabase, etc.).

**3. Upload triage has no SQL verification step**
When you upload your own CSV in the Upload & Analyze tab, the pipeline skips the SQL cross-validation step (debugger node). Detection results are based on pandas checks only, which means slightly higher false positive rates compared to the full retail pipeline.

**4. Retail dataset only (for the CLI pipeline)**
The CLI commands (`triage`, `evaluate`, etc.) are wired to the 7 retail CSVs in `data/raw/retail/`. There is no built-in way to point the CLI at a different database schema without code changes. The Upload & Analyze tab is the workaround for arbitrary datasets.

**5. LLM summary is fire-and-forget**
If the OpenRouter API call fails mid-run (network timeout, rate limit, quota exceeded), the report is generated without the LLM section. There is no retry logic. The report still contains the full deterministic SQL section.

**6. Drift detection requires a baseline snapshot**
Statistical drift detection only works if you have previously run `astrion-dq snapshot` on clean data. Without a baseline, the drift detector is skipped silently.

**7. No persistent user accounts or roles**
Authentication is a single shared Bearer token. There is no per-user access control, audit trail of who made which API call, or role-based permissions.

---

## Future plans

These are the most valuable next improvements, roughly in priority order:

**Short term (next month)**
- [ ] **Async background triage in dashboard** — The Streamlit sidebar currently blocks while triage runs. Switch to polling the `/jobs/{id}` API endpoint so the UI stays responsive
- [ ] **Persistent storage on Render** — Add Render Disk or a Supabase connection so output files survive server restarts
- [ ] **Multi-schema support in CLI** — Let users point `astrion-dq triage` at any CSV folder or database connection string, not just the hardcoded retail schema

**Medium term (next quarter)**
- [ ] **Real database connectors** — Connect to PostgreSQL, Snowflake, BigQuery, or DuckDB files directly instead of CSV-only. This makes the tool usable on real production data warehouses
- [ ] **Scheduled triage** — Add a cron endpoint so triage runs automatically at midnight without manual triggering
- [ ] **Slack / email alerts** — Send a notification when a high-severity issue is detected, instead of waiting for someone to check the dashboard
- [ ] **Custom issue rules** — Let users define their own SQL-based checks (e.g., "no order can have quantity > 10,000") that run alongside the built-in detectors
- [ ] **Per-user API keys** — Replace the single shared token with proper user management

**Long term (future versions)**
- [ ] **LLM-assisted root cause analysis** — Use the SQL Agent to automatically investigate detected issues and write a "likely root cause" explanation before the human engineer looks at it
- [ ] **Historical trend charts** — Track issue counts and BIS scores over time so teams can see if data quality is improving or getting worse
- [ ] **dbt integration** — Read dbt schema YAML files to automatically understand table relationships and column semantics, replacing the current inference-based profiler
- [ ] **Self-healing suggestions that auto-apply** — Go from "here is the fix SQL" to "I applied the fix, here are the before/after row counts, approve or reject"

---

## Dataset

The sample dataset is a synthetic retail star schema with 7 CSV files in `data/raw/retail/`:

- `dim_customers.csv` — customer demographics
- `dim_products.csv` — product catalogue
- `dim_stores.csv` — store locations
- `dim_dates.csv` — calendar dimension
- `dim_campaigns.csv` — marketing campaigns
- `dim_salespersons.csv` — sales staff
- `fact_sales_normalized.csv` — individual sales transactions (the main fact table)

The data is synthetic (not real customer data). It was designed to have a realistic star schema structure so foreign key checks and drift detection work meaningfully.

---

## Academic context

This project is Capstone 7 for a data engineering / AI course (Dr. William So, Synogize). The research question it addresses is:

> *Does adding SQL cross-validation (Strategy B) and statistical drift detection (Strategy C) to a rule-based data quality pipeline meaningfully improve F1 score, and at what cost in wall-clock time?*

The answer from the evaluation: all three strategies reach the same F1 (0.857) on the current synthetic dataset, with Strategy C being paradoxically fastest (9s vs 15s for A) because drift detection short-circuits some redundant checks. The more interesting finding is that precision = 1.0 across all strategies — zero false positives — which suggests the detection thresholds are well-calibrated for this type of retail data.

---

## Contributing

Pull requests are welcome. Please:

1. Run `pytest tests/` — all 112 tests must pass
2. Run `ruff check src/` — zero lint errors
3. Add tests for any new detection logic or API endpoints
4. Keep `config.py` as the single source of truth for all thresholds — do not hardcode numbers in detector or ranker code

---

*Astrion DQ v0.6.0 — Capstone 7 — Dr. William So, Synogize*
