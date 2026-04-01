## Evaluating Agentic Workflows for Data Quality Triage (Local MVP)

**Project #07 (Dr William So / Synogize)** — a fully **local-first**, **deterministic**, **no-paid-API** MVP for a university research demo.

**Version 1 scope (non-negotiable)**:
- **Retail Store Star Schema Dataset (Kaggle)** only
- **No OpenAI / Anthropic / external LLM inference**
- Two workflows in the main path:
  - **Baseline** (rule-based checks + impact ranking)
  - **Supervisor** (deterministic controller that consolidates, de-dupes, groups, and summarises)

### Stack

- **Language**: Python 3.11+
- **Engine**: DuckDB (local file)
- **Dataframes**: pandas
- **CLI**: typer
- **Config**: python-dotenv
- **Tests**: pytest
- **Plots**: matplotlib (for future extensions)

### MVP repo structure (what the CLI uses)

- **`src/astrion_dq`**: main MVP package
  - **`adapters/retail.py`**: retail dataset loader + metadata inference
  - **`warehouse/duckdb_loader.py`**: loads all retail tables into DuckDB (`data/processed/retail.duckdb`)
  - **`injectors/retail_issues.py`**: reproducible synthetic issue injection + ground truth export
  - **`checks/profiling.py`**: deterministic checks (nulls, duplicates, outliers)
  - **`ranking/business_impact.py`**: heuristic business impact scoring (local)
  - **`workflows/mvp.py`**: **baseline** and **supervisor** workflows, DuckDB load, injection, and **evaluate** (precision/recall/F1; deterministic)
  - **`cli.py`**: Typer-based CLI
- **`data/`**
  - **`raw/retail/`**: Retail Star Schema CSVs (required)
  - **`processed/`**: DuckDB database and cleaned tables
  - **`injected/retail/`**: CSVs with injected issues
- **`outputs/`**: JSON/CSV/Markdown reports and evaluation artefacts
- **`scripts/`**: convenience entrypoints
- **`tests/`**: lightweight tests for injection and ranking

### Setup

1. **Clone and install**

```bash
git clone https://github.com/Kinshuk78/Astrion_capstone_7.git
cd Astrion_capstone_7
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

2. **Configure environment (optional)**

```bash
cp config/.env.example .env
```

Edit `.env` if you want to override `DUCKDB_PATH`. LLM keys are optional and not required for the main flows.

3. **Place retail dataset**

Download the Retail Store Star Schema dataset from Kaggle and place the CSVs under:

- `data/raw/retail/*.csv`

The large fact table `fact_sales_normalized.csv` is listed in `.gitignore` (GitHub size limits); add it locally after clone. Dimension CSVs are versioned here. Run `python -m astrion_dq.cli inject` to regenerate `data/injected/retail/` (also ignored).

Table names are inferred from filenames.

### Core commands

All commands assume you are in the repo root with the virtual environment activated.

- **Inject synthetic issues into retail data**

```bash
python -m astrion_dq.cli inject --seed 42
```

This:
- reads `data/raw/retail/*.csv`
- injects a controlled set of issues
- writes injected CSVs to `data/injected/retail/`
- writes ground-truth metadata to `outputs/retail_injected_issues.json`

- **Run deterministic baseline profiling workflow**

```bash
python -m astrion_dq.cli profile
```

Produces ranked issues and saves:

- `outputs/baseline_issues.json`
- `outputs/baseline_summary.md`

- **Run a specific workflow (baseline or supervisor)**

```bash
python -m astrion_dq.cli run-workflow baseline
python -m astrion_dq.cli run-workflow supervisor
```

- **Evaluate a workflow against injected ground truth**

```bash
python -m astrion_dq.cli evaluate baseline
python -m astrion_dq.cli evaluate supervisor
```

Writes:
- `outputs/evaluation_baseline.json`
- `outputs/evaluation_supervisor.json`

and prints precision/recall/F1 and related rates from `evaluate_workflow`.

### Workflows

- **Baseline**: loads retail CSVs, materialises DuckDB, runs deterministic checks (nulls, duplicates, outliers, dates, referential integrity using inferred keys), ranks issues.
- **Supervisor**: same detection pipeline, optional higher-sensitivity rerun on injected data, de-duplicates ranked issues, writes supervisor JSON + summary.

All workflows are deterministic and do not require any external LLM.

### Evaluation

`python -m astrion_dq.cli evaluate …` compares workflow output to `outputs/retail_injected_issues.json` and writes:

- `outputs/evaluation_baseline.json`
- `outputs/evaluation_supervisor.json`

### Testing

Run the tests with:

```bash
pytest
```

Tests cover:

- basic injection properties
- ranking monotonicity with respect to metric values

### Limitations and future work

- Optional LLM-based layers can be added without changing the default local CLI path.
- Place large fact tables locally under `data/raw/retail/` (see `.gitignore`); dimension tables ship small enough to version if desired.

