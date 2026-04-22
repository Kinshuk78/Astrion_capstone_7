from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

import duckdb
import pandas as pd

from astrion_dq.config import (
    DUCKDB_SCHEMA,
    INJECTED_DIR,
    RAW_RETAIL_DIR,
    RAW_ROOT_DIR,
)

# ---------------------------------------------------------------------------
# DuckDB singleton connection
# ---------------------------------------------------------------------------
# Nodes in the LangGraph graph share this module-level connection.
# load_tables_to_duckdb() sets it; get_connection() retrieves it.
# This avoids passing a live connection through the serialised state dict.

_CONN: Optional[duckdb.DuckDBPyConnection] = None


def load_tables_to_duckdb(
    tables: Dict[str, pd.DataFrame],
    schema: str = DUCKDB_SCHEMA,
    overwrite: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Load DataFrames into an in-memory DuckDB and store the connection as a module singleton.

    Uses an in-memory database so each triage run starts with a completely
    fresh state — no cross-run contamination between 'injected' and 'clean'
    runs regardless of the order in which they are executed.

    All tables are registered under *schema* so the IssueVerifier can reference
    them with fully-qualified names like ``"dq_retail"."fact_sales_normalized"``.

    Returns the open connection so callers can reuse it immediately.
    """
    global _CONN
    # Close any existing connection before creating a new in-memory instance.
    if _CONN is not None:
        try:
            _CONN.close()
        except duckdb.Error:
            pass
        _CONN = None
    _CONN = duckdb.connect()  # in-memory: fresh DB every run, no file bleed
    _CONN.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

    for name, df in tables.items():
        _CONN.register("_tmp_frame", df)
        action = "CREATE OR REPLACE TABLE" if overwrite else "CREATE TABLE IF NOT EXISTS"
        _CONN.execute(f'{action} "{schema}"."{name}" AS SELECT * FROM _tmp_frame')
        _CONN.unregister("_tmp_frame")

    return _CONN


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the current open DuckDB connection.

    Raises RuntimeError if load_tables_to_duckdb() has not been called yet.
    """
    if _CONN is None:
        raise RuntimeError(
            "No DuckDB connection available. "
            "Call load_tables_to_duckdb() (via data_loader_node) first."
        )
    return _CONN


def close_connection() -> None:
    """Close and clear the cached connection."""
    global _CONN
    if _CONN is not None:
        _CONN.close()
        _CONN = None


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _ensure_clean_retail_dir() -> None:
    """Copy root-level CSVs into data/raw/retail/ if that directory is empty."""
    RAW_RETAIL_DIR.mkdir(parents=True, exist_ok=True)
    if list(RAW_RETAIL_DIR.glob("*.csv")):
        return
    for path in RAW_ROOT_DIR.glob("*.csv"):
        if "denormalized" in path.name.lower():
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
    """Load retail CSVs from the clean or injected directory.

    Args:
        source: ``"clean"`` loads from ``data/raw/retail/``;
                ``"injected"`` loads from ``data/injected/retail/``.

    Raises:
        FileNotFoundError: if injected data does not exist (run ``inject`` first).
        ValueError: if *source* is not ``"clean"`` or ``"injected"``.
    """
    _ensure_clean_retail_dir()

    if source == "clean":
        return _load_csvs(RAW_RETAIL_DIR)

    if source == "injected":
        if INJECTED_DIR.exists() and list(INJECTED_DIR.glob("*.csv")):
            return _load_csvs(INJECTED_DIR)
        raise FileNotFoundError(
            "No injected retail data found. Run 'python -m astrion_dq.cli inject' first."
        )

    raise ValueError(f"source must be 'clean' or 'injected', got {source!r}")
