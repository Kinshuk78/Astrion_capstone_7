from pathlib import Path

import pandas as pd

from astrion_dq.adapters.retail import RetailAdapter


def test_retail_adapter_loads_csvs(tmp_path: Path):
    raw = tmp_path / "raw"
    retail = raw / "retail"
    retail.mkdir(parents=True, exist_ok=True)

    (retail / "fact_sales.csv").write_text("sale_id,amount\n1,10\n2,20\n", encoding="utf-8")
    (retail / "dim_products.csv").write_text("product_id,name\n1,A\n2,B\n", encoding="utf-8")

    adapter = RetailAdapter(raw)
    tables = adapter.load_tables()
    assert set(tables.keys()) == {"fact_sales", "dim_products"}
    assert list(tables["fact_sales"].columns) == ["sale_id", "amount"]

