"""SQL cross-validation of pandas-detected quality issues.

``IssueVerifier`` re-runs each detected issue as a DuckDB SQL query and
computes a confidence score measuring agreement between the SQL count and
the original pandas count.

    confidence = min(sql_count, pd_count) / max(sql_count, pd_count, 1)

A confidence below ``CONFIDENCE_THRESHOLD`` triggers the human-review node.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import duckdb

from astrion_dq.config import (
    CONFIDENCE_THRESHOLD,
    DRIFT_DEFAULT_CONFIDENCE,
    DUCKDB_SCHEMA,
    FUTURE_DATE_SENTINEL_INT,
    FUTURE_DATE_SENTINEL_TS,
    IQR_MULT_HIGH,
    IQR_MULT_NORMAL,
    SQL_FALLBACK_CONFIDENCE,
)
from astrion_dq.models import QualityIssue, VerifiedIssue
from astrion_dq.warehouse.loader import get_connection

logger = logging.getLogger(__name__)


class IssueVerifier:
    """Cross-validates pandas-detected issues against independent DuckDB SQL queries.

    The ``sensitivity`` parameter must match the value passed to the detector.
    When ``sensitivity="high"`` the outlier SQL query uses ``IQR_MULT_HIGH``
    (1.5), which is the same multiplier the pandas detector used. Mismatching
    the sensitivity produces artificially deflated confidence scores.

    Issue-type dispatch:
        statistical_drift      -- SQL verification is not meaningful; uses DRIFT_DEFAULT_CONFIDENCE.
        empty_table            -- count is definitionally 0; confidence is 1.0.
        missing_values         -- COUNT WHERE col IS NULL.
        duplicate_rows         -- COUNT(*) - COUNT(DISTINCT pk_cols); fallback when no PK cols.
        numeric_outliers       -- IQR outlier count via PERCENTILE_CONT, multiplier from sensitivity.
        invalid_future_dates   -- count depends on column data type (integer YYYYMMDD vs DATE).
        referential_integrity_break -- LEFT JOIN fact to dim, count unmatched FK rows.
        unknown                -- SQL_FALLBACK_CONFIDENCE (no crash).

    Any ``duckdb.Error`` is caught; the issue keeps ``pd_count`` as ``sql_count``
    and receives ``SQL_FALLBACK_CONFIDENCE`` to avoid aborting the run.
    """

    def __init__(
        self,
        schema: str = DUCKDB_SCHEMA,
        sensitivity: str = "normal",
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        self.schema = schema
        self.sensitivity = sensitivity
        self._connection = connection

    @property
    def _conn(self) -> duckdb.DuckDBPyConnection:
        if self._connection is not None:
            return self._connection
        return get_connection()

    def _fqt(self, table: str) -> str:
        """Fully-qualified DuckDB table name: ``"schema"."table"``."""
        return f'"{self.schema}"."{table}"'

    def _confidence(self, sql_count: int, pd_count: int) -> float:
        return min(sql_count, pd_count) / max(sql_count, pd_count, 1)

    def verify_all(self, issues: List[QualityIssue]) -> List[VerifiedIssue]:
        """Verify every issue in *issues* and return the annotated list."""
        return [self._verify_one(issue) for issue in issues]

    def _verify_one(self, issue: QualityIssue) -> VerifiedIssue:
        try:
            sql_count, confidence = self._dispatch(issue)
        except duckdb.Error as exc:
            logger.warning("DuckDB error on %s (%s): %s", issue.issue_id, issue.issue_type, exc)
            sql_count = issue.evidence_rows
            confidence = SQL_FALLBACK_CONFIDENCE

        return VerifiedIssue(
            issue_id=issue.issue_id,
            issue_type=issue.issue_type,
            table=issue.table,
            columns=issue.columns,
            severity=issue.severity,
            metric=issue.metric,
            evidence_rows=issue.evidence_rows,
            description=issue.description,
            sql_count=sql_count,
            pd_count=issue.evidence_rows,
            confidence=round(confidence, 4),
            dim_table=issue.dim_table,
            dim_pk=issue.dim_pk,
        )

    def _dispatch(self, issue: QualityIssue) -> Tuple[int, float]:
        t = issue.issue_type
        if t == "statistical_drift":
            return issue.evidence_rows, DRIFT_DEFAULT_CONFIDENCE
        if t == "empty_table":
            return 0, 1.0
        if t == "missing_values":
            return self._verify_nulls(issue)
        if t == "duplicate_rows":
            return self._verify_duplicates(issue)
        if t == "numeric_outliers":
            return self._verify_outliers(issue)
        if t == "invalid_future_dates":
            return self._verify_future_dates(issue)
        if t == "referential_integrity_break":
            return self._verify_ri_break(issue)
        return issue.evidence_rows, SQL_FALLBACK_CONFIDENCE

    # ---- per-type verification -----------------------------------------------

    def _verify_nulls(self, issue: QualityIssue) -> Tuple[int, float]:
        col = issue.columns[0]
        sql = f'SELECT COUNT(*) FROM {self._fqt(issue.table)} WHERE "{col}" IS NULL'
        count = self._conn.execute(sql).fetchone()[0]
        return count, self._confidence(count, issue.evidence_rows)

    def _verify_duplicates(self, issue: QualityIssue) -> Tuple[int, float]:
        if not issue.columns:
            # Full-row deduplication requires all column names, which are not stored.
            return issue.evidence_rows, SQL_FALLBACK_CONFIDENCE

        fqt = self._fqt(issue.table)
        pk_cols = ", ".join(f'"{c}"' for c in issue.columns)
        sql = f"SELECT COUNT(*) - COUNT(DISTINCT ({pk_cols})) FROM {fqt}"
        count = max(0, self._conn.execute(sql).fetchone()[0])
        return count, self._confidence(count, issue.evidence_rows)

    def _verify_outliers(self, issue: QualityIssue) -> Tuple[int, float]:
        """Re-count IQR outliers using the same multiplier the pandas detector used.

        The multiplier is selected from ``self.sensitivity``:
          - "high"   -> ``IQR_MULT_HIGH``  (1.5) — tighter bounds, more outliers
          - "normal" -> ``IQR_MULT_NORMAL`` (3.0) — standard bounds

        Using the wrong multiplier produces a systematically biased confidence
        score: too many SQL hits (normal when high was used) or too few (vice-versa).
        """
        col = issue.columns[0]
        fqt = self._fqt(issue.table)
        mult = IQR_MULT_HIGH if self.sensitivity == "high" else IQR_MULT_NORMAL
        sql = f"""
        WITH stats AS (
            SELECT
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{col}") AS q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{col}") AS q3
            FROM {self._fqt(issue.table)}
            WHERE "{col}" IS NOT NULL
        )
        SELECT COUNT(*)
        FROM {fqt}, stats
        WHERE "{col}" IS NOT NULL
          AND (   "{col}" < stats.q1 - {mult} * (stats.q3 - stats.q1)
               OR "{col}" > stats.q3 + {mult} * (stats.q3 - stats.q1))
        """
        count = self._conn.execute(sql).fetchone()[0]
        return count, self._confidence(count, issue.evidence_rows)

    def _col_is_integer(self, table: str, col: str) -> bool:
        """Return True if DuckDB reports the column as an integer type."""
        sql = f'SELECT typeof("{col}") FROM {self._fqt(table)} LIMIT 1'
        result = self._conn.execute(sql).fetchone()
        if result is None:
            return False
        return "int" in result[0].lower()

    def _verify_future_dates(self, issue: QualityIssue) -> Tuple[int, float]:
        """Verify future-date counts using the same sentinel the pandas detector used.

        The pandas detector has two branches:
          - Integer columns (YYYYMMDD): value > FUTURE_DATE_SENTINEL_INT (20500101)
          - Date / string columns:      parsed_date > Timestamp("2050-01-01")

        Using ``CURRENT_DATE`` as the threshold would be wrong — it would flag
        valid dates between today and 2050.
        """
        col = issue.columns[0]
        fqt = self._fqt(issue.table)

        if self._col_is_integer(issue.table, col):
            sql = f'SELECT COUNT(*) FROM {fqt} WHERE "{col}" > {FUTURE_DATE_SENTINEL_INT}'
        else:
            sql = (
                f"SELECT COUNT(*) FROM {fqt} "
                f"WHERE TRY_CAST(\"{col}\" AS DATE) > DATE '{FUTURE_DATE_SENTINEL_TS}'"
            )

        count = self._conn.execute(sql).fetchone()[0]
        return count, self._confidence(count, issue.evidence_rows)

    def _verify_ri_break(self, issue: QualityIssue) -> Tuple[int, float]:
        fk_col = issue.columns[0] if issue.columns else ""
        dim_table = issue.dim_table
        dim_pk = issue.dim_pk

        if not dim_table or not dim_pk or not fk_col:
            logger.warning(
                "Cannot verify RI break for %s: dim_table=%r dim_pk=%r fk_col=%r",
                issue.issue_id, dim_table, dim_pk, fk_col,
            )
            return issue.evidence_rows, SQL_FALLBACK_CONFIDENCE

        sql = f"""
        SELECT COUNT(*)
        FROM {self._fqt(issue.table)} f
        LEFT JOIN {self._fqt(dim_table)} d ON f."{fk_col}" = d."{dim_pk}"
        WHERE f."{fk_col}" IS NOT NULL
          AND d."{dim_pk}" IS NULL
        """
        count = self._conn.execute(sql).fetchone()[0]
        return count, self._confidence(count, issue.evidence_rows)


# Expose threshold for use in nodes.py without re-importing config there.
__all__ = ["IssueVerifier", "CONFIDENCE_THRESHOLD"]
