from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from astrion_dq.adapters.base import DatasetAdapter, TableMetadata


class RetailAdapter(DatasetAdapter):
    """
    Adapter for the Retail Store Star Schema dataset.

    Assumes CSV files under data/raw/retail/*.csv.
    Uses simple naming and type heuristics to infer roles and metadata.
    """

    def load_tables(self) -> Dict[str, pd.DataFrame]:
        tables: Dict[str, pd.DataFrame] = {}
        retail_dir = self.root_dir / "retail"
        for csv_path in sorted(retail_dir.glob("*.csv")):
            name = csv_path.stem.lower()
            tables[name] = pd.read_csv(csv_path)
        return tables

    def infer_metadata(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, TableMetadata]:
        metadata: Dict[str, TableMetadata] = {}
        for name, df in tables.items():
            cols = list(df.columns)
            lower_cols = [c.lower() for c in cols]

            role = "dimension"
            if "fact" in name or "sales" in name or "transaction" in name:
                role = "fact"

            primary_key: List[str] = []
            for i, col_lower in enumerate(lower_cols):
                col = cols[i]
                if col_lower.endswith("_id") and df[col].is_unique:
                    primary_key.append(col)

            foreign_keys = {}
            for i, col_lower in enumerate(lower_cols):
                col = cols[i]
                if col_lower.endswith("_id") and not df[col].is_unique:
                    fk_table = col_lower.replace("_id", "")
                    foreign_keys[col] = f"{fk_table}.id"

            date_columns = [
                cols[i]
                for i, c in enumerate(lower_cols)
                if "date" in c or c.endswith("_dt") or "day" in c
            ]

            numeric_measures = [
                col
                for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col]) and "id" not in col.lower()
            ]

            promotion_columns = [
                cols[i]
                for i, c in enumerate(lower_cols)
                if "promo" in c or "discount" in c or "coupon" in c
            ]

            metadata[name] = TableMetadata(
                name=name,
                role=role,
                primary_key=primary_key or None,
                foreign_keys=foreign_keys,
                date_columns=date_columns,
                numeric_measures=numeric_measures,
                promotion_columns=promotion_columns,
            )
        return metadata

