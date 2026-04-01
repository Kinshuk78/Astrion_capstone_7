import pandas as pd

from astrion_dq.injectors.retail_issues import inject_retail_issues


def test_inject_retail_issues_basic(tmp_path, monkeypatch):
    # Create a simple table
    df = pd.DataFrame(
        {
            "transaction_id": list(range(100)),
            "sales_amount": [10.0] * 100,
            "transaction_date": ["2020-01-01"] * 100,
        }
    )
    tables = {"fact_sales": df}

    injected, issues = inject_retail_issues(tables, seed=123)
    assert "fact_sales" in injected
    assert len(issues) > 0

