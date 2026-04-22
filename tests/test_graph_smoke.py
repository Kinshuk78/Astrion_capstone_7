"""Smoke test for the LangGraph node functions.

Bypasses CSV loading by monkeypatching load_retail_tables.
Uses a tmp DuckDB for load_tables_to_duckdb so no on-disk DB is created.

Asserts:
  - metadata_ready is True after profiler_node
  - raw_issues is populated (at least one issue) after detector_node
  - timing keys are present for each node
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

import astrion_dq.warehouse.loader as warehouse_loader
from astrion_dq.graph.nodes import (
    data_loader_node,
    detector_node,
    profiler_node,
)
from astrion_dq.graph.state import initial_state


@pytest.fixture()
def fake_tables():
    """A minimal fact table with a detectable null issue (20% nulls on a column)."""
    return {
        "fact_sales": pd.DataFrame({
            "sales_sk": [1, 2, 3, 4, 5],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "quantity": [1, None, 3, None, 5],  # 40% nulls -> fires detect_nulls
        })
    }


@pytest.fixture()
def patched_nodes(monkeypatch, tmp_path, fake_tables):
    """Patch CSV loading and DuckDB path for isolated node execution."""

    # Replace CSV loading with our fake tables.
    monkeypatch.setattr(
        "astrion_dq.graph.nodes.load_retail_tables",
        lambda source: fake_tables,
    )

    # Replace the DuckDB loader to use a tmp file so no global state persists.
    def fake_load_duckdb(tables, **_kwargs):
        from astrion_dq.config import DUCKDB_SCHEMA
        conn = duckdb.connect(str(tmp_path / "smoke.duckdb"))
        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{DUCKDB_SCHEMA}"')
        for name, df in tables.items():
            conn.register("_tmp", df)
            conn.execute(
                f'CREATE OR REPLACE TABLE "{DUCKDB_SCHEMA}"."{name}" '
                f"AS SELECT * FROM _tmp"
            )
            conn.unregister("_tmp")
        warehouse_loader._CONN = conn
        return conn

    monkeypatch.setattr(
        "astrion_dq.graph.nodes.load_tables_to_duckdb",
        fake_load_duckdb,
    )

    yield

    warehouse_loader._CONN = None


def test_data_loader_sets_tables(patched_nodes, fake_tables):
    """data_loader_node should populate tables and set data_loaded."""
    state = initial_state(source="clean")
    result = data_loader_node(state)

    assert result.get("data_loaded") is True
    assert set(result["tables"].keys()) == set(fake_tables.keys())
    assert result.get("db_path"), "db_path must be set by data_loader_node"
    assert "data_loader" in result["timing"]


def test_profiler_sets_metadata_ready(patched_nodes, fake_tables):
    """profiler_node should infer metadata and set metadata_ready."""
    state = initial_state(source="clean")
    state.update(data_loader_node(state))
    result = profiler_node(state)

    assert result.get("metadata_ready") is True
    assert "fact_sales" in result["metadata"]
    assert "profiler" in result["timing"]


def test_detector_produces_raw_issues(patched_nodes, fake_tables):
    """detector_node should detect at least one issue in the fake tables."""
    state = initial_state(source="clean")
    state.update(data_loader_node(state))
    state.update(profiler_node(state))
    result = detector_node(state)

    assert result.get("detection_done") is True
    assert len(result["raw_issues"]) >= 1, (
        "Expected at least one issue (40% nulls on 'quantity' should fire detect_nulls)"
    )
    assert "detector" in result["timing"]


def test_interactive_build_graph_raises(patched_nodes):
    """P5-C: build_graph(interactive=True) must raise RuntimeError immediately.

    DataFrames in state['tables'] are not msgpack-serialisable. MemorySaver
    would crash at invoke time with a cryptic error. Better to fail loudly at
    build time with a clear message pointing to ASTRION_AUTO_APPROVE=1.
    """
    from astrion_dq.graph.workflow import build_graph
    with pytest.raises(RuntimeError, match="ASTRION_AUTO_APPROVE"):
        build_graph(interactive=True)


def test_checkpointer_raises(patched_nodes):
    """build_graph(checkpointer=anything) must also raise RuntimeError."""
    from langgraph.checkpoint.memory import MemorySaver
    from astrion_dq.graph.workflow import build_graph
    with pytest.raises(RuntimeError, match="ASTRION_AUTO_APPROVE"):
        build_graph(checkpointer=MemorySaver())


def test_full_node_chain(patched_nodes):
    """End-to-end smoke: run all three nodes in sequence."""
    state = initial_state(source="clean")

    state.update(data_loader_node(state))
    assert state["data_loaded"]

    state.update(profiler_node(state))
    assert state["metadata_ready"]

    state.update(detector_node(state))
    assert state["detection_done"]
    assert isinstance(state["raw_issues"], list)
