"""Tests for the SQL Agent helpers (LLM client + SQL block extraction).

Tests are split into two groups:
  1. LLM-free: SQL block extraction regex, schema introspection, token trim logic
  2. LLM-mocked: chat_with_history invocation path using a patched client

No real API calls are made.
"""
from __future__ import annotations

import re

import duckdb
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 1. SQL block extraction
# ---------------------------------------------------------------------------

def _extract_sql_blocks(text: str) -> list[str]:
    """Replicate the dashboard's _execute_sql_blocks regex."""
    blocks = re.findall(r"```sql\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return [b.strip() for b in blocks if b.strip()]


def test_single_sql_block_extracted():
    response = "Here is a query:\n```sql\nSELECT 1;\n```"
    blocks = _extract_sql_blocks(response)
    assert blocks == ["SELECT 1;"]


def test_multiple_sql_blocks_extracted():
    response = (
        "Step 1:\n```sql\nSELECT COUNT(*) FROM t;\n```\n"
        "Step 2:\n```sql\nDELETE FROM t WHERE id = 1;\n```"
    )
    blocks = _extract_sql_blocks(response)
    assert len(blocks) == 2
    assert "COUNT(*)" in blocks[0]
    assert "DELETE" in blocks[1]


def test_case_insensitive_sql_tag():
    response = "```SQL\nSELECT 1;\n```"
    blocks = _extract_sql_blocks(response)
    assert blocks == ["SELECT 1;"]


def test_no_sql_blocks_returns_empty():
    response = "Just some plain text without any code blocks."
    assert _extract_sql_blocks(response) == []


def test_empty_sql_block_skipped():
    response = "```sql\n\n```"
    assert _extract_sql_blocks(response) == []


# ---------------------------------------------------------------------------
# 2. SQL execution against in-memory DuckDB
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    df = pd.DataFrame({"id": [1, 2, 3], "val": [10.0, 20.0, 30.0]})
    conn.register("_tmp", df)
    conn.execute('CREATE TABLE "sample" AS SELECT * FROM _tmp')
    conn.unregister("_tmp")
    return conn


def test_valid_sql_returns_dataframe(sample_conn):
    sql = 'SELECT COUNT(*) AS n FROM "sample"'
    result_df = sample_conn.execute(sql).df()
    assert result_df["n"][0] == 3


def test_invalid_sql_raises_duckdb_error(sample_conn):
    with pytest.raises(duckdb.Error):
        sample_conn.execute("SELECT * FROM nonexistent_table_xyz")


def test_schema_introspection_works(sample_conn):
    tables = sample_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "sample" in table_names


# ---------------------------------------------------------------------------
# 3. Token-based history trim logic
# ---------------------------------------------------------------------------

def _trim_history(history: list[dict], budget_chars: int = 24_000) -> list[dict]:
    """Replicate the dashboard's token-based trim."""
    used = 0
    trimmed: list = []
    for msg in reversed(history):
        msg_chars = len(msg.get("content", ""))
        if used + msg_chars > budget_chars:
            break
        trimmed.insert(0, msg)
        used += msg_chars
    if not trimmed and history:
        trimmed = [history[-1]]
    return trimmed


def test_trim_respects_budget():
    history = [{"role": "user", "content": "x" * 5_000}] * 10  # 50,000 chars total
    trimmed = _trim_history(history, budget_chars=24_000)
    total = sum(len(m["content"]) for m in trimmed)
    assert total <= 24_000


def test_trim_keeps_most_recent_messages():
    messages = [
        {"role": "user", "content": f"msg {i}", "sql_results": []}
        for i in range(100)
    ]
    trimmed = _trim_history(messages, budget_chars=200)
    # Most recent messages should be kept (they're small, at least a few fit)
    assert trimmed[-1]["content"] == "msg 99"


def test_trim_always_keeps_one_message_even_over_budget():
    giant = [{"role": "user", "content": "x" * 100_000}]
    trimmed = _trim_history(giant, budget_chars=1_000)
    assert len(trimmed) == 1


def test_trim_empty_history():
    assert _trim_history([]) == []


# ---------------------------------------------------------------------------
# 4. LLM client: LLMUnavailable when no key configured
# ---------------------------------------------------------------------------

def test_llm_unavailable_when_no_key(monkeypatch):
    """chat_with_history raises LLMUnavailable when key is not set.

    The config module caches the key at import time, so we patch the module
    attribute directly rather than the environment variable.
    """
    import astrion_dq.config as cfg
    import astrion_dq.llm.client as llm_client

    # Patch the module-level constant that _get_client() reads
    monkeypatch.setattr(cfg, "OPENROUTER_API_KEY", "")
    # Reset the singleton so _get_client() re-evaluates the patched constant
    llm_client.reset_client()

    from astrion_dq.llm.client import LLMUnavailable, chat_with_history

    with pytest.raises(LLMUnavailable):
        chat_with_history([{"role": "user", "content": "hello"}])

    # Cleanup: restore singleton so other tests aren't affected
    llm_client.reset_client()


def test_chat_calls_openai_client(monkeypatch):
    """chat_with_history passes messages to the underlying OpenAI client."""
    import astrion_dq.llm.client as llm_client

    class _FakeChoice:
        class message:
            content = "  mocked response  "

    class _FakeResp:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResp()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    llm_client._client = _FakeClient()

    result = llm_client.chat_with_history(
        [{"role": "user", "content": "test"}],
        system="You are helpful",
        max_tokens=100,
    )
    assert result == "mocked response"

    # Cleanup
    llm_client.reset_client()
