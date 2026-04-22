"""Regression test: injector must use schema-aware column selection.

Verifies that on a star schema with _sk surrogate keys:
  1. missing_key_values targets the fact PK (sales_sk), not the natural key (sales_id).
  2. referential_integrity_break targets a _sk FK column (customer_sk or product_sk),
     not the natural key (sales_id).

The old code selected id_cols = [c for c in cols if c.lower().endswith("_id")],
which only found sales_id and produced len(id_cols) < 2, silently skipping the
RI injection entirely.
"""
from __future__ import annotations

import pandas as pd
import pytest

from astrion_dq.injectors.retail_issues import inject_retail_issues


@pytest.fixture()
def star_schema(tmp_path, monkeypatch):
    """Minimal star schema with _sk surrogate keys."""
    dim_customers = pd.DataFrame({"customer_sk": list(range(1, 51))})
    dim_products = pd.DataFrame({"product_sk": list(range(1, 31))})
    fact_sales = pd.DataFrame({
        "sales_sk": list(range(1, 201)),          # PK: surrogate key
        "sales_id": list(range(1001, 1201)),       # natural key (business ID)
        "customer_sk": [i % 50 + 1 for i in range(200)],   # FK -> dim_customers
        "product_sk": [i % 30 + 1 for i in range(200)],    # FK -> dim_products
        "sales_date": ["2023-01-01"] * 200,
        "total_amount": [float(i) for i in range(200)],
    })

    tables = {
        "dim_customers": dim_customers,
        "dim_products": dim_products,
        "fact_sales": fact_sales,
    }

    # Redirect file writes to tmp_path so the test does not touch the project tree.
    import astrion_dq.injectors.retail_issues as inj_mod
    monkeypatch.setattr(inj_mod, "INJECTED_DIR", tmp_path / "injected")
    monkeypatch.setattr(inj_mod, "OUTPUTS_DIR", tmp_path / "outputs")

    return tables


def test_missing_key_values_targets_pk(star_schema):
    """missing_key_values should target the fact PK (sales_sk), not sales_id."""
    _, issues = inject_retail_issues(star_schema, seed=42)

    mkv_issues = [i for i in issues if i.issue_type == "missing_key_values"]
    assert mkv_issues, "Expected at least one missing_key_values issue"

    col = mkv_issues[0].columns[0]
    assert col == "sales_sk", (
        f"missing_key_values should target the fact PK 'sales_sk', got '{col}'"
    )


def test_ri_break_targets_sk_column(star_schema):
    """referential_integrity_break must target a _sk FK column, not sales_id."""
    _, issues = inject_retail_issues(star_schema, seed=42)

    ri_issues = [i for i in issues if i.issue_type == "referential_integrity_break"]
    assert ri_issues, (
        "Expected at least one referential_integrity_break issue. "
        "The injector may have silently skipped it (old _id-only detection)."
    )

    col = ri_issues[0].columns[0]
    assert col.endswith("_sk"), (
        f"referential_integrity_break should target a _sk FK column, got '{col}'"
    )
    assert col != "sales_id", (
        "referential_integrity_break must NOT target the natural key 'sales_id'"
    )


def test_both_issue_types_present(star_schema):
    """Both missing_key_values and referential_integrity_break must appear in ground truth."""
    _, issues = inject_retail_issues(star_schema, seed=42)
    issue_types = {i.issue_type for i in issues}

    assert "missing_key_values" in issue_types, "missing_key_values not in injected issues"
    assert "referential_integrity_break" in issue_types, (
        "referential_integrity_break not in injected issues"
    )
