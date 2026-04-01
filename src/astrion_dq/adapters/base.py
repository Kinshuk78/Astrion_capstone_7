from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class TableMetadata:
    name: str
    role: str  # "fact" or "dimension" or "unknown"
    primary_key: Optional[List[str]]
    foreign_keys: Dict[str, str]  # local_col -> referenced_table.col
    date_columns: List[str]
    numeric_measures: List[str]
    promotion_columns: List[str]


class DatasetAdapter:
    """Abstract dataset adapter interface."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    def load_tables(self) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def infer_metadata(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, TableMetadata]:
        raise NotImplementedError

