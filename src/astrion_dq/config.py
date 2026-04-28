from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------

PROJECT_ROOT     = Path(__file__).resolve().parents[2]

# Load config/.env when present (never required -- env vars take precedence)
_env_file = PROJECT_ROOT / "config" / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file, override=False)
    except ImportError:
        pass  # python-dotenv optional at import time

# ---------------------------------------------------------------------------
# OpenRouter LLM (optional -- system is fully functional without it)
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6")
LLM_TOP_N           = int(os.getenv("LLM_TOP_N", "5"))

# ---------------------------------------------------------------------------
# OpenAI Agent Layer (hackathon-openai-agent branch)
# Powers the /explain, /prioritise, /generate-fix, /report, /analyze endpoints.
# Set OPENAI_API_KEYS (comma-separated) or OPENAI_API_KEY (single key).
# When unset, all agent endpoints fall back gracefully to deterministic output.
# ---------------------------------------------------------------------------

OPENAI_MODEL                    = os.getenv("OPENAI_MODEL", "gpt-4.1")
AGENT_MAX_RETRIES               = int(os.getenv("AGENT_MAX_RETRIES", "3"))
AGENT_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("AGENT_CIRCUIT_BREAKER_THRESHOLD", "3"))
AGENT_RESPONSE_MIN_SCORE        = float(os.getenv("AGENT_RESPONSE_MIN_SCORE", "0.70"))
AGENT_MAX_ISSUES                = int(os.getenv("AGENT_MAX_ISSUES", "10"))
AGENT_MAX_DESC_CHARS            = int(os.getenv("AGENT_MAX_DESC_CHARS", "200"))
AGENT_CACHE_TTL                 = int(os.getenv("AGENT_CACHE_TTL", "3600"))

RAW_RETAIL_DIR   = PROJECT_ROOT / "data" / "raw" / "retail"
RAW_ROOT_DIR     = PROJECT_ROOT / "data" / "raw"
INJECTED_DIR     = PROJECT_ROOT / "data" / "injected" / "retail"
OUTPUTS_DIR      = PROJECT_ROOT / "outputs"
SNAPSHOTS_DIR    = OUTPUTS_DIR / "drift_snapshots"
SCHEMA_SNAPS_DIR = OUTPUTS_DIR / "schema_snapshots"
DB_PATH          = PROJECT_ROOT / "data" / "processed" / "retail.duckdb"

DUCKDB_SCHEMA = "dq_retail"

# ---------------------------------------------------------------------------
# Null detection thresholds
# ---------------------------------------------------------------------------

# Important columns (PKs, FKs, dates) get a stricter threshold.
NULL_THRESHOLD_IMPORTANT = 0.01   # flag if > 1% null for key columns
NULL_THRESHOLD_OTHER     = 0.05   # flag if > 5% null for non-key columns
NULL_THRESHOLD_HIGH_SENS = 0.01   # uniform threshold in high-sensitivity mode

# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

DUP_MIN_FRACTION_NORMAL   = 0.01   # ignore tiny dup fractions in normal mode
DUP_MIN_FRACTION_HIGH     = 0.005

# ---------------------------------------------------------------------------
# Outlier detection (IQR method)
# ---------------------------------------------------------------------------

IQR_MULT_NORMAL   = 3.0    # bounds = Q1 - 3*IQR, Q3 + 3*IQR
IQR_MULT_HIGH     = 1.5
OUTLIER_MIN_FRAC  = 0.01   # minimum fraction to report in normal mode
OUTLIER_MIN_FRAC_HIGH = 0.005
MIN_ROWS_FOR_STATS = 20    # skip statistical checks on very small tables

# ---------------------------------------------------------------------------
# Date sentinel values (used when injecting and detecting future dates)
# ---------------------------------------------------------------------------

FUTURE_DATE_SENTINEL_INT = 20500101   # integer YYYYMMDD format
FUTURE_DATE_SENTINEL_TS  = "2050-01-01"

# ---------------------------------------------------------------------------
# Severity banding
# ---------------------------------------------------------------------------

SEVERITY_HIGH_THRESHOLD   = 0.05   # metric >= 5%  → high
SEVERITY_MEDIUM_THRESHOLD = 0.01   # metric >= 1%  → medium
                                   # metric <  1%  → low

# ---------------------------------------------------------------------------
# Statistical drift (PSI + KS)
# ---------------------------------------------------------------------------

PSI_AMBER = 0.10   # PSI in [0.10, 0.25) → moderate drift
PSI_RED   = 0.25   # PSI >= 0.25         → major drift
KS_ALPHA  = 0.05   # significance level for the two-sample KS test

# ---------------------------------------------------------------------------
# IssueVerifier (SQL cross-validation)
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD    = 0.70   # below this → flag for analyst review
DRIFT_DEFAULT_CONFIDENCE = 0.80  # statistical drift cannot be verified via SQL
SQL_FALLBACK_CONFIDENCE  = 0.50  # DuckDB error → fall back to pandas count

# ---------------------------------------------------------------------------
# V2 Business Impact Score weights
# ---------------------------------------------------------------------------
# These encode the relative severity of each issue type.
# Referential integrity breaks cause silent join failures — the most dangerous
# category because reports appear to run but produce wrong answers.

ISSUE_TYPE_BASE_WEIGHTS: Dict[str, float] = {
    "referential_integrity_break":   4.0,
    "duplicate_rows":                3.5,
    "numeric_outliers":              2.8,
    "statistical_drift":             2.5,
    "missing_values":                2.3,
    "invalid_future_dates":          2.1,
    "empty_table":                   3.0,
}

SEVERITY_WEIGHTS: Dict[str, float] = {
    "high":   3.0,
    "medium": 2.0,
    "low":    1.0,
}

# Downstream report criticality scores.
# A daily sales summary goes to senior leadership; a customer segmentation
# report is reviewed weekly. Adjust these to match actual SLAs.
REPORT_CRITICALITY_SCORES: Dict[str, float] = {
    "daily sales summary":        1.00,
    "daily_sales_summary":        1.00,
    "promotion performance":      0.90,
    "sales by store":             0.85,
    "sales by product category":  0.85,
    "inventory replenishment":    0.80,
    "top products":               0.70,
    "customer segmentation":      0.65,
}

# Minimum BIS below which an issue is considered noise and filtered out.
BIS_NOISE_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Report impact mapping  (issue_type -> affected downstream reports)
# ---------------------------------------------------------------------------

REPORT_MAPPING: Dict[str, List[str]] = {
    "missing_values":              ["sales by store", "sales by product category"],
    "duplicate_rows":              ["daily sales summary", "top products"],
    "numeric_outliers":            ["daily sales summary", "promotion performance"],
    "invalid_future_dates":        ["daily sales summary"],
    "referential_integrity_break": ["sales by store", "sales by product category"],
    "statistical_drift":           ["daily sales summary", "promotion performance"],
    "empty_table":                 ["daily sales summary"],
}

# ---------------------------------------------------------------------------
# Metadata inference helpers
# ---------------------------------------------------------------------------

KEY_SUFFIXES = ("_id", "_sk", "_key")
