# ADR 0001 -- P1 Critical Correctness Fixes

**Date:** 2026-04-20
**Status:** Accepted
**Version bump:** 0.2.0 -> 0.3.0

## Context

Four correctness defects blocked production use of Astrion DQ v0.2.0.

## Decisions

**detect_drift requires meta (F-01).** `meta` is now a required second positional
argument in `detect_drift`. No default. A forgotten call site raises `TypeError`
at import time. The alternative (a `None` default that re-enables the broken
scan-all-columns path) was rejected because F-01 is a BLOCKER.

**duplicate_rows.evidence_rows counts excess copies (F-05 -- BREAKING).**
`keep='first'` matches the SQL verifier's `COUNT(*) - COUNT(DISTINCT pk)` formula.
"Excess copies" is the correct business semantic. This halves the reported count
relative to v0.2.0. Evidence_rows previously counted all rows in duplicate groups.

**detect_nulls augments important set with is_key_col columns (F-16).** PK
inference can fail on corrupted data (injected nulls break the uniqueness check
in metadata.py). Augmenting the important set with is_key_col columns is additive
and robust to this failure mode. Does not alter metadata.py.

**detect_future_dates uses format='ISO8601' (F-17).** Pandas 2.x + ArrowStringArray
with mixed datetime/date-only strings silently coerces date-only strings to NaT
under the inferred format. ISO8601 mode handles both variants correctly.

## Consequences

Post-P1 evaluation (2026-04-20, 4-issue GT):
- A/B: F1=0.857, Precision=1.0, Recall=0.75 (up from F1=0.600, Precision=0.75)
- C_full: F1=0.857 (up from F1=0.308; 16 drift FPs eliminated)
- 1 FN remains: numeric_outliers (not in P1 fix set)
