from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TableMeta:
    role: str                                    # "fact" | "dimension"
    primary_keys: List[str]
    foreign_keys: Dict[str, Tuple[str, str]]     # col -> (dim_table, dim_pk_col)
    date_cols: List[str]
    numeric_cols: List[str]
    promo_cols: List[str]


@dataclass
class QualityIssue:
    issue_id: str
    issue_type: str
    table: str
    columns: List[str]
    severity: str        # "high" | "medium" | "low"
    metric: float        # fraction of rows affected (0.0 – 1.0)
    evidence_rows: int   # absolute count of affected rows
    description: str
    dim_table: str = ""  # for referential_integrity_break: the referenced dimension table
    dim_pk: str = ""     # for referential_integrity_break: the PK column in dim_table


@dataclass
class VerifiedIssue:
    """A QualityIssue cross-validated by a DuckDB SQL query.

    confidence = min(sql_count, pd_count) / max(sql_count, pd_count, 1)
    A score below the configured threshold triggers analyst review.
    """
    issue_id: str
    issue_type: str
    table: str
    columns: List[str]
    severity: str
    metric: float
    evidence_rows: int
    description: str
    sql_count: int    # row count from the SQL verification query
    pd_count: int     # row count from the pandas detector (same as evidence_rows)
    confidence: float # agreement ratio in [0.0, 1.0]
    dim_table: str = ""  # preserved from QualityIssue for RI breaks
    dim_pk: str = ""     # preserved from QualityIssue for RI breaks


@dataclass
class RankedIssue:
    """A VerifiedIssue decorated with a V2 Business Impact Score and report mapping."""
    issue_id: str
    issue_type: str
    table: str
    columns: List[str]
    severity: str
    metric: float
    evidence_rows: int
    description: str
    impact_score: float
    affected_reports: List[str]
    agent_trace: List[str]
    confidence: float = 1.0  # preserved from VerifiedIssue; 1.0 if not verified
    dim_table: str = ""      # for referential_integrity_break: referenced dimension table
    dim_pk: str = ""         # for referential_integrity_break: PK column in dim_table
