# SQL and Analytics Cheatsheet

One page for the last ten minutes before an interview. Queries below ran on DuckDB 1.5.6, and the pandas lines ran on pandas 3.0.6. Dialect names differ by engine, so say which one you assume.

## Logical order and the first checks

FROM and JOIN, WHERE, GROUP BY, HAVING, window functions, SELECT, DISTINCT, ORDER BY, LIMIT.

- Aggregates and windows are illegal in WHERE. Aliases work in ORDER BY, not in WHERE (DuckDB and SQLite are lenient).
- No ORDER BY means no order. Ties under LIMIT need a unique tiebreaker.
- Say the grain: "one row per ...", and check that the key is unique.

## NULL rules

- Comparison with NULL is unknown. WHERE, HAVING and ON keep only true. Use IS NULL and IS DISTINCT FROM.
- COUNT(*) counts rows, COUNT(col) skips NULL, SUM and AVG skip NULL, SUM of nothing is NULL.
- NOT IN with a NULL in the list returns nothing. Use NOT EXISTS.
- GROUP BY, DISTINCT and set operators treat NULLs as equal. Join keys do not match NULL.
- Divide by NULLIF(y, 0). Integer division truncates in SQLite, PostgreSQL and SQL Server.

## Joins and fan out

| Join | Keeps |
| --- | --- |
| INNER | matched pairs |
| LEFT, RIGHT | matched pairs plus unmatched rows of one side |
| FULL | matched pairs plus both unmatched sides |
| CROSS | every pair, m times n |
| semi (EXISTS) | left rows with a match, once each |
| anti (NOT EXISTS) | left rows with no match |

- Inner join rows = sum over keys of count in A times count in B. Keys 1, 1, 2, NULL against 1, 1, 1, 3, NULL give inner 6, left 8, full 10, cross 20.
- Filter the optional table in ON to keep left rows. The same filter in WHERE makes it an inner join.
- Fan out check: row count and total before and after each join, uniqueness of each join key, reconcile with the source. Aggregate the many side to the one side's grain first. SUM(DISTINCT) is not a fix.
- After a LEFT JOIN, COUNT the right primary key, not star.

## Windows and frames

| Need | Write |
| --- | --- |
| one row per group | `ROW_NUMBER() OVER (PARTITION BY k ORDER BY x DESC, id)` |
| shared places, gaps | `RANK()` |
| Nth distinct value | `DENSE_RANK()` |
| previous or next row | `LAG(x)`, `LEAD(x)` |
| running total | `SUM(x) OVER (ORDER BY d ROWS UNBOUNDED PRECEDING)` |
| percent of total | `x / SUM(x) OVER ()` |
| 7 row moving average | `AVG(x) OVER (ORDER BY d ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` |
| calendar window | `RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW` |
| last row of partition | `LAST_VALUE(x) OVER (... ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)` |

- Default frame with ORDER BY is RANGE from the start to the current row, and ties share a value. Without ORDER BY it is the whole partition.
- Frames matter for aggregates, FIRST_VALUE, LAST_VALUE and NTH_VALUE. Ranking and LAG ignore them.
- Filter a window in a subquery or CTE, or QUALIFY (DuckDB, Snowflake, BigQuery).

## Pattern recipes

Top N per group (use DENSE_RANK to keep tied values):

```sql
WITH r AS (
  SELECT name, dept_id, salary,
         ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC, emp_id) AS rn
  FROM employees)
SELECT * FROM r WHERE rn <= 2
```

Latest row per key, and the second highest value (NULL if absent):

```sql
SELECT * FROM contacts
QUALIFY ROW_NUMBER() OVER (PARTITION BY email ORDER BY updated_at DESC, id DESC) = 1
```

```sql
SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees)
```

Gaps and islands (deduplicate first, date minus row number is constant in a run):

```sql
WITH dd AS (SELECT DISTINCT user_id, login_date FROM logins),
n AS (SELECT user_id, login_date,
             ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn FROM dd)
SELECT user_id, MIN(login_date) AS start_day, MAX(login_date) AS end_day, COUNT(*) AS days
FROM n GROUP BY user_id, login_date - CAST(rn AS INTEGER)
```

Sessions (gap over 30 minutes starts a new one):

```sql
WITH g AS (
  SELECT user_id, ts,
         CASE WHEN LAG(ts) OVER w IS NULL OR ts - LAG(ts) OVER w > INTERVAL 30 MINUTE THEN 1 ELSE 0 END AS new_s
  FROM clicks WINDOW w AS (PARTITION BY user_id ORDER BY ts))
SELECT user_id, SUM(new_s) OVER (PARTITION BY user_id ORDER BY ts) AS session_id, ts FROM g
```

Growth, and users who did X but not Y:

```sql
SELECT month, ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
       / NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 1) AS mom_pct
FROM monthly_revenue
```

```sql
SELECT user_id FROM events
GROUP BY user_id
HAVING COUNT(*) FILTER (WHERE event_name = 'view') > 0
   AND COUNT(*) FILTER (WHERE event_name = 'purchase') = 0
```

Day 1 retention (LEFT JOIN, COUNT DISTINCT, NULL for cohorts not yet mature):

```sql
SELECT u.signup_date, COUNT(DISTINCT u.user_id) AS size,
       COUNT(DISTINCT CASE WHEN a.activity_date - u.signup_date = 1 THEN u.user_id END) AS d1
FROM users u LEFT JOIN activity a ON a.user_id = u.user_id
GROUP BY u.signup_date
```

- Funnel: first time per step per user, each step after the previous, optional time window.
- Median: `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)`, or ROW_NUMBER and COUNT with the middle one or two rows.
- Pivot: one `SUM(CASE WHEN ... THEN x ELSE 0 END)` per column. Unpivot: UNION ALL, one SELECT per column.
- Distinct counts and medians do not add across groups. Use exact keys or sketches.

## Recursion and dates

- Recursive CTE: anchor, UNION ALL, recursive part, stop condition. Guard cycles with depth, path or UNION.
- Date spine then LEFT JOIN then COALESCE fills missing days.
- Month filter: `d >= '2024-02-01' AND d < '2024-03-01'`. Never BETWEEN with timestamps.
- datediff counts boundaries, not full periods. Adding a month to January 31 differs by engine.

## Performance

- Index the filter, join and sort columns. It costs writes and space.
- Sargable: keep the column bare. `date(col) = x` scans, a range seeks.
- Composite index: equality columns first, range last, leftmost prefix only.
- Read the plan inside out. Compare estimated and actual rows, and refresh statistics when they differ a lot.
- Nested loop for few outer rows with an index, hash join for big equality joins, merge join for sorted inputs.
- Warehouses prune partitions and columns. Avoid SELECT star and filter on the partition column.

## Pandas map

| SQL | pandas |
| --- | --- |
| GROUP BY, COUNT(*), COUNT(DISTINCT) | `groupby().agg`, `size`, `nunique` |
| JOIN, anti join | `merge(how, validate=)`, `indicator=True` |
| AVG OVER partition | `groupby().transform("mean")` |
| ROW_NUMBER, RANK, DENSE_RANK | `cumcount() + 1`, `rank(method="min")`, `rank(method="dense")` |
| LAG, LEAD, running total | `shift(1)`, `shift(-1)`, `cumsum()` |
| ROWS, RANGE frame | `rolling(3)`, `rolling("3D")` |
| pivot, unpivot | `pivot_table`, `melt` |

- Pandas drops NaN group keys, matches NaN merge keys, and sums all missing values to 0. Pass `dropna=False` and clean keys.

## Modelling and dbt

- Star schema: fact (events, foreign keys, measures) and dimensions (attributes). State the grain and test uniqueness.
- Additive, semi additive (balances, never sum over time), non additive (rebuild from numerator and denominator).
- SCD: type 1 overwrite, type 2 new row with validity dates, type 3 previous column. Join type 2 with half open ranges.
- dbt: models are SELECTs, `ref` builds the graph, tests return failing rows, incremental models need a lookback for late data, snapshots build type 2 history.
- Metric changed after a refactor: totals, then diff by key, then check grain and join keys.

## If asked, say

- "Why does this NOT IN return nothing?" A NULL in the subquery makes every comparison unknown.
- "Revenue doubled after the join." The key is not unique on one side. Aggregate to the grain first.
- "Top 3 with ties?" I would ask whether it means three rows or three values, then pick ROW_NUMBER or DENSE_RANK.
- "My running total repeats." The default RANGE frame includes ties. Add a tiebreaker and use ROWS.
