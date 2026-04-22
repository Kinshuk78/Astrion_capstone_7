"""Schema-inference helpers shared across the detection, injector, and evaluation layers.

All functions are public (no underscore prefix). ``detect.py`` re-exports
``infer_metadata`` for backwards compatibility with existing import sites.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

from astrion_dq.config import KEY_SUFFIXES
from astrion_dq.models import TableMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

def norm(text: str) -> str:
    """Lowercase alphanumeric characters only — strips punctuation and spaces."""
    return "".join(ch.lower() for ch in text if ch.isalnum())


def strip_key_suffix(col: str) -> Optional[Tuple[str, str]]:
    """Return (stem, suffix) if *col* ends with a recognised key suffix, else None.

    Recognised suffixes come from ``config.KEY_SUFFIXES`` (``_id``, ``_sk``, ``_key``).
    """
    c = col.strip()
    lower = c.lower()
    for suffix in KEY_SUFFIXES:
        if lower.endswith(suffix):
            return c[: -len(suffix)], suffix
    return None


def singularize(entity: str) -> str:
    """Naive English singulariser applied to already-normalised (lowercase alnum) tokens.

    Special cases:
      - "sales" is retained as-is (both singular and plural mean the same table).
      - Handles -ies, -sses, -ses, and plain -s endings.
    """
    e = norm(entity)
    if not e:
        return e
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


def table_entity_base(table_name: str) -> str:
    """Return the singular entity token for a table name.

    Strips ``dim_`` / ``fact_`` prefixes and internal noise tokens
    (``normalized``, ``denormalized``, ``injected``), then singularises.
    """
    name = table_name.lower().strip()
    for prefix in ("dim_", "fact_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    tokens = [
        t for t in name.split("_")
        if t and t not in {"normalized", "denormalized", "injected"}
    ]
    return singularize(tokens[0] if tokens else name)


def key_entity_base(col: str) -> Optional[str]:
    """Return the singular entity token for a key column name.

    Returns None when *col* does not end with a recognised key suffix.
    """
    stripped = strip_key_suffix(col)
    if not stripped:
        return None
    stem, _ = stripped
    stem_norm = norm(stem)
    return singularize(stem_norm) if stem_norm else None


def is_key_col(col: str) -> bool:
    """Return True when *col* ends with a recognised key suffix."""
    return strip_key_suffix(col) is not None


# ---------------------------------------------------------------------------
# Metadata inference
# ---------------------------------------------------------------------------

def infer_metadata(tables: Dict[str, pd.DataFrame]) -> Dict[str, TableMeta]:
    """Infer table roles, primary/foreign keys, and column types from names and data.

    Two-pass algorithm:

    Pass 1 — For each table:
      - Classify as fact (name contains "fact") or dimension.
      - Detect candidate primary key columns by checking uniqueness.
      - Build a ``dim_entity -> (table, pk_col)`` lookup for FK resolution.
      - Collect date, numeric, and promo column lists.

    Pass 2 — For each fact table:
      - Walk key-like columns that are not the fact's own PK.
      - Match the column's entity base against the dimension lookup.
      - Any match becomes a foreign key entry in the fact's ``TableMeta``.

    Args:
        tables: Mapping of table name to its DataFrame.

    Returns:
        Mapping of table name to populated ``TableMeta``.
    """
    meta: Dict[str, TableMeta] = {}
    dim_pk_by_entity: Dict[str, Tuple[str, str]] = {}

    # --- Pass 1: classify, detect PKs, build dim index ---
    for name, df in tables.items():
        role = "fact" if "fact" in name.lower() else "dimension"
        entity_base = table_entity_base(name)

        date_cols = [c for c in df.columns if "date" in c.lower() or "day" in c.lower()]
        numeric_cols = [
            c for c in df.columns
            if is_numeric_dtype(df[c]) and not is_key_col(c) and c not in date_cols
        ]
        promo_cols = [
            c for c in df.columns
            if any(tok in c.lower() for tok in ("promo", "campaign", "discount"))
        ]
        key_like = [c for c in df.columns if is_key_col(c)]
        primary_keys: List[str] = []

        if role == "dimension":
            preferred = [c for c in key_like if key_entity_base(c) == entity_base]
            candidates = preferred or key_like
            for c in candidates:
                if df[c].nunique(dropna=True) == len(df):
                    primary_keys = [c]
                    break
            if not primary_keys and candidates:
                primary_keys = [candidates[0]]
            if primary_keys:
                dim_pk_by_entity[entity_base] = (name, primary_keys[0])
        else:
            # Fact table PK: prefer columns whose entity base matches the table entity
            preferred = [c for c in key_like if key_entity_base(c) == entity_base]
            candidates = preferred or key_like
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

    # --- Pass 2: resolve FK relationships for fact tables ---
    for name, df in tables.items():
        if meta[name].role != "fact":
            continue
        fact_entity = table_entity_base(name)
        fact_pk_set = set(meta[name].primary_keys)
        fks: Dict[str, Tuple[str, str]] = {}

        for c in df.columns:
            if c in fact_pk_set or not is_key_col(c):
                continue
            col_entity = key_entity_base(c)
            if not col_entity or col_entity == fact_entity:
                continue
            match = dim_pk_by_entity.get(col_entity)
            if match:
                fks[c] = match

        meta[name].foreign_keys = fks

    return meta
