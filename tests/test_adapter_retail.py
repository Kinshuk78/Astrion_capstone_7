from pathlib import Path

from astrion_dq.warehouse.loader import _load_csvs


def test_retail_loader_loads_csvs(tmp_path: Path):
    retail = tmp_path / "retail"
    retail.mkdir(parents=True)

    (retail / "fact_sales.csv").write_text("sale_id,amount\n1,10\n2,20\n", encoding="utf-8")
    (retail / "dim_products.csv").write_text("product_id,name\n1,A\n2,B\n", encoding="utf-8")

    tables = _load_csvs(retail)
    assert set(tables.keys()) == {"fact_sales", "dim_products"}
    assert list(tables["fact_sales"].columns) == ["sale_id", "amount"]
