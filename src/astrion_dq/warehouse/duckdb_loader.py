from __future__ import annotations

from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "retail.duckdb"


def load_tables_to_duckdb(
    tables: Dict[str, pd.DataFrame],
    schema: str = "dq_retail",
    overwrite: bool = True,
    db_path: Path | None = None,
) -> Path:
    db_path = Path(db_path or DEFAULT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        for name, df in tables.items():
            safe_name = name.replace("-", "_")
            full_name = f'"{schema}"."{safe_name}"'
            temp_name = f"_tmp_{safe_name}"

            con.register(temp_name, df)
            if overwrite:
                con.execute(f"DROP TABLE IF EXISTS {full_name}")
            con.execute(f"CREATE TABLE {full_name} AS SELECT * FROM {temp_name}")
            con.unregister(temp_name)
    finally:
        con.close()

    return db_path
