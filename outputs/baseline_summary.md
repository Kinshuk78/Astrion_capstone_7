## Baseline data quality triage summary

- Total issues detected: **2**
- Fallback rerun used: **No**

### Top issues

1. **duplicate_rows** in `fact_sales_normalized`
   - Severity: `medium`
   - Columns: `n/a`
   - Evidence rows: `35442`
   - Impact score: `1.1946`
   - Affected reports: `daily sales summary, top products`
   - Summary: Detected 35442 duplicated rows in fact_sales_normalized.

2. **referential_integrity_break** in `fact_sales_normalized`
   - Severity: `medium`
   - Columns: `campaign_sk`
   - Evidence rows: `13902`
   - Impact score: `1.0818`
   - Affected reports: `sales by store, sales by product category`
   - Summary: Column campaign_sk contains 13902 values not found in dim_campaigns.campaign_sk.
