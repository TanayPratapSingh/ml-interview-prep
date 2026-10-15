# SQL and Analytics for Interviews, Deep Version

This guide covers the SQL and analytics material that data analyst, data scientist and analytics engineering interviews test, with a full explanation and a run query for each idea. Every query below ran on DuckDB 1.5.6 unless it says SQLite 3.45.3, and the pandas code ran on pandas 3.0.6. Outputs are the real outputs of those runs. Statements about PostgreSQL, MySQL, SQL Server, BigQuery, Snowflake and Oracle come from the sources listed at the end or from memory, and the text says which. Dialect support changes, so confirm syntax against the documentation of the engine named in the interview.

## How SQL runs a query

A SELECT statement has a logical processing order that is different from the order you write it. The order is FROM and JOIN, WHERE, GROUP BY, HAVING, window functions, SELECT, DISTINCT, ORDER BY, then OFFSET and LIMIT. Each step receives the output of the previous one and can only use what exists at that point. That single fact explains the common errors: aggregates are illegal in WHERE because groups do not exist yet, window functions are illegal in WHERE and HAVING because they are computed later, and an alias defined in SELECT is legal in ORDER BY but not, under the standard, in WHERE.

Engines that accept an alias in WHERE (DuckDB and SQLite do, and I ran both) offer it as an extension. When an alias shares a name with a real column, WHERE sees the table column and ORDER BY sees the output column, which is a trap.

The physical plan can differ from the logical order. The optimizer pushes filters below joins, reorders joins and drops unneeded steps, but it must produce the rows the logical order would produce. This is also why the result of a query without ORDER BY has no guaranteed order, and why ORDER BY with ties needs a unique tiebreaker before you trust LIMIT.

Here is the trace of one query on DuckDB 1.5.6. Nine rows leave FROM, eight survive the WHERE (Hana earns 90), three groups form (departments 10, 20 and the NULL department), two survive HAVING (Ian is alone in the NULL group), ORDER BY sorts by the alias, and LIMIT keeps two:

```sql
SELECT dept_id, COUNT(*) AS n, AVG(salary) AS avg_salary
FROM employees
WHERE salary >= 100
GROUP BY dept_id
HAVING COUNT(*) >= 2
ORDER BY avg_salary DESC
LIMIT 2
```

Output:

```text
dept_id  n  avg_salary
-------  -  ----------
10       4  155
20       3  116.6667
```

> **Say it like this:** "A query runs FROM first and SELECT late, so WHERE cannot see aggregates or aliases, HAVING can see aggregates, and ORDER BY can see aliases."

> **Interview trap:** using LIMIT without ORDER BY, or ordering by a column with ties, and expecting the same rows every run.

## NULL and three valued logic

SQL has three truth values: true, false and unknown. A comparison with NULL is unknown. AND returns false if either side is false and unknown otherwise when an input is unknown, OR returns true if either side is true, and NOT unknown is unknown. WHERE, HAVING, ON and CASE WHEN treat only true as a pass. A CHECK constraint is the reverse: unknown passes. So `dept_id <> 10` loses the rows where `dept_id` is NULL, and `NOT (dept_id = 10)` loses them too. Use `IS DISTINCT FROM`, or `OR dept_id IS NULL`.

Aggregates ignore NULL inputs. COUNT(*) counts rows, COUNT(col) counts non NULL values and COUNT(DISTINCT col) counts distinct non NULL values. SUM, AVG, MIN and MAX return NULL over an empty or all NULL input, and AVG divides by the number of non NULL values, so `AVG(COALESCE(x, 0))` answers a different question.

```sql
SELECT COUNT(*) AS all_rows, COUNT(dept_id) AS non_null, COUNT(DISTINCT dept_id) AS distinct_non_null FROM employees
```

Output:

```text
all_rows  non_null  distinct_non_null
--------  --------  -----------------
9         8         3
```

Grouping and de duplication treat NULLs as one value: GROUP BY makes one NULL group, and DISTINCT, UNION, INTERSECT and EXCEPT treat NULLs as equal. Joins do not: `a.k = b.k` never matches NULL keys, so use `IS NOT DISTINCT FROM` for a null safe join. ORDER BY places NULLs first in SQLite and last in DuckDB by default, so write NULLS FIRST or NULLS LAST when it matters.

`NOT IN` with a NULL in the list is never true, because `x NOT IN (1, NULL)` expands to `x <> 1 AND x <> NULL`. One order with a NULL customer makes this return nothing, even though one customer has no orders:

```sql
SELECT customer_id, name FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders) ORDER BY customer_id
```

Output:

```text
customer_id  name
-----------  ----
```

The safe form is NOT EXISTS, which only yields true or false:

```sql
SELECT c.customer_id, c.name FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id) ORDER BY c.customer_id
```

Output:

```text
customer_id  name
-----------  ----
4            Di
```

COALESCE returns the first non NULL argument, and NULLIF(a, b) returns NULL when a equals b, which makes `x / NULLIF(y, 0)` a portable division guard. An unguarded division by zero returned `inf` in DuckDB, NULL in SQLite, and raises an error in PostgreSQL (from its documented behaviour, not run).

> **Interview trap:** `WHERE col = NULL`, and `NOT IN` against a nullable subquery column.

## GROUP BY, HAVING and aggregates

GROUP BY collapses rows that share the same grouping values into one output row. The select list may then contain grouping columns, aggregates and expressions of those. A bare ungrouped column is an error in DuckDB and standard SQL, while SQLite accepts it and picks a value from some row of the group. HAVING filters groups after aggregation and may use aggregates that are not selected. WHERE filters rows before grouping, which is cheaper, so keep row conditions in WHERE.

Aggregate functions accept DISTINCT (`COUNT(DISTINCT user_id)`). Conditional aggregation puts a condition inside the aggregate with FILTER or CASE. ROLLUP, CUBE and GROUPING SETS compute several grouping levels in one pass, and the GROUPING function tells a subtotal NULL apart from a real NULL. SQLite does not support them.

An aggregate query without GROUP BY always returns exactly one row, even over no input rows (COUNT is 0, SUM is NULL). A grouped query over no input returns no rows.

The average of group averages weights every group equally. The correct combined mean is total sum over total count. Using customers with orders (amounts 50, 30, 60 for customer 1, 20 and 70 for customer 2, 40 for customer 3):

```sql
WITH per_customer AS (
  SELECT customer_id, COUNT(*) AS n, AVG(amount) AS avg_amount
  FROM orders WHERE customer_id IS NOT NULL GROUP BY customer_id
)
SELECT ROUND(AVG(avg_amount), 2) AS avg_of_avgs,
       ROUND(SUM(avg_amount * n) / SUM(n), 2) AS weighted_avg
FROM per_customer
```

Output:

```text
avg_of_avgs  weighted_avg
-----------  ------------
43.89        45
```

The plain mean of the three group averages is 43.89, the weighted result is 45, and 45 is also what averaging the six raw rows gives. Medians and distinct counts cannot be combined from group results at all.

Finding duplicates is a grouped query: `GROUP BY email HAVING COUNT(*) > 1`. Decide first which columns define a duplicate, because rows that share a key but differ elsewhere are conflicts, not copies.

> **Remember:** carry sums and counts through intermediate steps, and divide at the end.

## Joins

An inner join returns the pairs of rows that satisfy the ON condition. A left join adds the unmatched left rows with NULLs on the right, a right join does the mirror, and a full join adds both. A cross join returns every pair, so m rows times n rows. A self join joins a table to itself under two aliases, as in employees and their managers. For each key value, an inner join returns count in A times count in B, so duplicates multiply, and NULL keys never match.

With keys 1, 1, 2, NULL in A and 1, 1, 1, 3, NULL in B, the id 1 gives 2 times 3 = 6 matches. The other rows add unmatched ones:

```sql
WITH a(id) AS (VALUES (1), (1), (2), (NULL)),
     b(id) AS (VALUES (1), (1), (1), (3), (NULL))
SELECT (SELECT COUNT(*) FROM a JOIN b ON a.id = b.id) AS inner_n,
       (SELECT COUNT(*) FROM a LEFT JOIN b ON a.id = b.id) AS left_n,
       (SELECT COUNT(*) FROM a FULL JOIN b ON a.id = b.id) AS full_n,
       (SELECT COUNT(*) FROM a CROSS JOIN b) AS cross_n
```

Output:

```text
inner_n  left_n  full_n  cross_n
-------  ------  ------  -------
6        8       10      20
```

SQLite returned the same counts. ON and WHERE differ for outer joins. A condition on the optional table in ON limits which rows match and keeps all left rows. The same condition in WHERE removes the unmatched rows and turns the left join into an inner join:

```sql
SELECT d.dept_name, e.name FROM departments d LEFT JOIN employees e ON e.dept_id = d.dept_id AND e.salary > 120 ORDER BY d.dept_id, e.emp_id
```

Output:

```text
dept_name    name
-----------  ----
Engineering  Ada
Engineering  Ben
Engineering  Cleo
Sales        Eli
Marketing    NULL
Legal        NULL
```

```sql
SELECT d.dept_name, e.name FROM departments d LEFT JOIN employees e ON e.dept_id = d.dept_id WHERE e.salary > 120 ORDER BY d.dept_id, e.emp_id
```

Output:

```text
dept_name    name
-----------  ----
Engineering  Ada
Engineering  Ben
Engineering  Cleo
Sales        Eli
```

A semi join (EXISTS, IN) returns left rows that have a match, once each. An anti join (NOT EXISTS, or LEFT JOIN with IS NULL on a right column that cannot be NULL) returns left rows with no match. Both avoid the duplication of an inner join. Set operators combine whole result sets: UNION removes duplicates, UNION ALL keeps them and is cheaper, INTERSECT returns rows in both, and EXCEPT returns rows in the first only.

USING (col) joins on equal same named columns and returns the column once. NATURAL JOIN joins on every shared column name, which is fragile when schemas change. A non equi join uses ranges or inequalities, such as orders against price tiers, and needs non overlapping ranges to avoid fan out.

> **Interview trap:** a WHERE condition on the right table of a LEFT JOIN, and joins on nullable keys.

## Join fan out

Fan out is the multiplication of rows when a join matches one row to several. Any measure that lives on the repeated side gets counted once per match. The orders table sums to 280 in total, but joined to its item rows, the sum changes:

```sql
SELECT SUM(amount) AS all_orders FROM orders
```

Output:

```text
all_orders
----------
280
```

```sql
SELECT SUM(o.amount) AS inflated, COUNT(*) AS joined_rows FROM orders o JOIN order_items i ON i.order_id = o.order_id
```

Output:

```text
inflated  joined_rows
--------  -----------
420       8
```

The eight joined rows sum to 420. Two effects are mixed: orders with several items are repeated (order 101 twice, order 106 three times), and orders with no items disappear in the inner join (orders 103 and 107). The result is wrong with no error.

Detection works from the top: compare row counts and totals before and after each join, test key uniqueness on each side with `GROUP BY key HAVING COUNT(*) > 1`, and reconcile the final total with the source. The fix is to aggregate the many side down to the grain of the one side before joining, so each order meets exactly one item row:

```sql
SELECT SUM(o.amount) AS revenue, SUM(i.units) AS units
FROM orders o
JOIN (SELECT order_id, SUM(qty) AS units FROM order_items GROUP BY order_id) i ON i.order_id = o.order_id
```

Output:

```text
revenue  units
-------  -----
250      12
```

Revenue here covers only orders with items (250), which is what an inner join to items means, and units are 10 across them. Use a LEFT JOIN to keep orders without items.

`SUM(DISTINCT amount)` is not a fix, because it removes equal values from different rows. Two orders of 50 collapse into one. COUNT(DISTINCT key) fixes counts but not sums. Joining two child tables at once multiplies them: an order with 3 items and 2 shipments yields 6 rows, so pre aggregate each child to the order key. After a LEFT JOIN, count the right table's primary key, not `*`, or an entity with no matches reports 1 instead of 0.

> **Say it like this:** "I check the join keys for uniqueness, aggregate the many side to the grain of the one side, and reconcile the total to the source."

> **Interview trap:** summing a header level amount after joining to detail rows.

## CASE, dates and strings

CASE evaluates conditions top to bottom and returns the first match, then ELSE, then NULL if there is no ELSE. Put the narrowest condition first. `WHEN col = NULL` never matches, so use IS NULL. CASE works in SELECT, WHERE, ORDER BY, GROUP BY and inside aggregates, which gives conditional aggregation: `SUM(CASE WHEN status = 'refunded' THEN amount ELSE 0 END)` or `COUNT(*) FILTER (WHERE status = 'paid')`. Averaging a 1 or 0 flag gives a proportion.

Dates need care. Truncate with `date_trunc('month', d)` (DuckDB, PostgreSQL) to group by month and keep the year. EXTRACT returns one part, so extracting only the month merges years. SQLite has no date_trunc, so it uses strftime and returns text. Day differences use boundary counting in many functions: `date_diff('year', 2018-03-01, 2024-01-01)` is 6, while only 5 full years have passed. Adding one month to January 31 gives February 29 in DuckDB and March 2 in SQLite.

BETWEEN includes both ends and a date literal means midnight. Three timestamps (09:00 and 23:59:59.5 on March 1, and midnight starting March 2) show the problem. The goal is the count for March 1, which is 2:

```sql
WITH t(ts) AS (VALUES (TIMESTAMP '2024-03-01 09:00:00'),
                      (TIMESTAMP '2024-03-01 23:59:59.500'),
                      (TIMESTAMP '2024-03-02 00:00:00'))
SELECT COUNT(*) FILTER (WHERE ts BETWEEN '2024-03-01' AND '2024-03-01') AS same_day_literal,
       COUNT(*) FILTER (WHERE ts BETWEEN '2024-03-01' AND '2024-03-01 23:59:59') AS to_23_59_59,
       COUNT(*) FILTER (WHERE ts BETWEEN '2024-03-01' AND '2024-03-02') AS to_next_day,
       COUNT(*) FILTER (WHERE ts >= '2024-03-01' AND ts < '2024-03-02') AS half_open
FROM t
```

Output:

```text
same_day_literal  to_23_59_59  to_next_day  half_open
----------------  -----------  -----------  ---------
0                 1            3            2
```

BETWEEN with equal day literals returns 0, ending at 23:59:59 returns 1, and ending at the next day returns 3. Only the half open range returns 2, and it leaves the column bare so an index can be used.

Integer division truncates in SQLite, PostgreSQL and SQL Server, but not in DuckDB, so a share like 6 of 7 orders comes out differently:

```sql
SELECT COUNT(*) FILTER (WHERE status = 'paid') / COUNT(*) AS paid_share FROM orders
```

Output:

```text
paid_share
----------
0
```

SQLite returns 0. DuckDB returns 0.8571 for the same statement. Multiply by 1.0 or cast first, multiply before dividing, and guard the denominator with NULLIF.

Strings: LIKE uses percent for any run and underscore for one character. LIKE is case insensitive for ASCII in SQLite and case sensitive in DuckDB, which also has ILIKE. Concatenation with double pipes returns NULL if any part is NULL. Text compares character by character, so `'10' < '9'` is true.

## Subqueries and CTEs

A scalar subquery returns one value, an IN or EXISTS subquery returns a set or a boolean, a derived table sits in FROM, and a correlated subquery refers to the outer row. A scalar subquery must return at most one row. DuckDB raises an error on more, while SQLite silently takes the first row. A correlated subquery is logically evaluated once per outer row, though optimizers usually decorrelate it into a join or aggregate. Its NULL behaviour follows from the equality: a NULL correlation key matches nothing.

```sql
SELECT e.name, e.dept_id, e.salary FROM employees e WHERE e.salary > (SELECT AVG(x.salary) FROM employees x WHERE x.dept_id = e.dept_id) ORDER BY e.emp_id
```

Output:

```text
name  dept_id  salary
----  -------  ------
Ada   10       200
Eli   20       130
```

Department 10 averages 155, so only Ada beats it. Department 20 averages 116.67, so only Eli does. Ian has no department, so the inner equality is unknown and he is never returned.

A CTE (WITH name AS) names a step for the single statement that follows. Later CTEs can reference earlier ones. It is a readability tool. A view stores a named query in the database, a temporary table stores rows, and a materialized view stores refreshed rows. According to the PostgreSQL documentation on WITH queries, which I read, a non recursive CTE without side effects is folded into the parent when referenced once and materialized when referenced more than once, and MATERIALIZED or NOT MATERIALIZED overrides that (I recall this behaviour arrived in version 12). Materializing blocks pushdown of filters and index use, folding can repeat work. Other engines decide differently, so check the plan.

EXISTS and IN return each outer row once, a JOIN returns one row per match. Negated, NOT IN fails with NULLs and NOT EXISTS does not. The question "employees earning above their department average" can be written as a correlated subquery, a join to a pre aggregated table, or a window AVG filtered in an outer query. All three return Ada and Eli here. The window version groups NULL departments into one partition, while the join and correlated versions never match them.

> **Say it like this:** "I use a CTE per logical step, and I check the plan rather than assume the CTE is computed once."

> **Interview trap:** repeating the rule that CTEs are always optimization fences, which stopped being true in PostgreSQL 12.

## Recursive CTEs

A recursive CTE has an anchor query, UNION ALL, and a recursive query that reads the CTE. Evaluation, as the PostgreSQL documentation describes it, is iterative: run the anchor into a working table, then repeatedly run the recursive part against the working table, append the new rows to the result and make them the next working table, until a step adds nothing. Each step sees only the previous step's rows, so the recursion must be linear, the order of rows is not guaranteed, and termination is the author's job.

Walking the employee table from the top, with depth and path:

```sql
WITH RECURSIVE chain AS (
  SELECT emp_id, name, manager_id, 0 AS depth, name AS path
  FROM employees WHERE manager_id IS NULL
  UNION ALL
  SELECT e.emp_id, e.name, e.manager_id, c.depth + 1, c.path || ' > ' || e.name
  FROM employees e
  JOIN chain c ON e.manager_id = c.emp_id
)
SELECT name, depth, path FROM chain ORDER BY path
```

Output:

```text
name  depth  path
----  -----  ----------------
Ada   0      Ada
Ben   1      Ada > Ben
Dev   2      Ada > Ben > Dev
Cleo  1      Ada > Cleo
Eli   1      Ada > Eli
Fay   2      Ada > Eli > Fay
Gus   2      Ada > Eli > Gus
Hana  1      Ada > Hana
Ian   2      Ada > Hana > Ian
```

SQLite gave the same rows. Step 0 is Ada, step 1 adds her four direct reports, step 2 adds four more, step 3 finds nobody.

A cycle in the data (say Ada's manager is set to one of her reports) never reaches an empty step. Protect the query three ways: a depth limit in the recursive WHERE, which stops but repeats nodes up to the limit; a visited path that refuses to revisit a node; or UNION instead of UNION ALL, which works only when the rows are otherwise identical, since a changing depth or path makes every revisit a new row. PostgreSQL has a CYCLE clause for this, according to its docs (I recall it arrived in version 14).

Recursive CTEs also generate date spines (start date, plus one day while below the end), which you LEFT JOIN to facts and COALESCE to fill missing days. DuckDB and PostgreSQL also offer `generate_series`.

Trees can be stored four ways. An adjacency list (parent id) is easy to update and needs recursion to read. A materialized path (`/1/5/6/`) reads subtrees with a LIKE prefix and rewrites paths on moves. A closure table stores every ancestor and descendant pair, so reads and moves are easy but rows grow. Nested sets store left and right numbers, which reads fast and updates badly.

> **Interview trap:** forgetting the stopping condition, and trusting row order without ORDER BY.

## Window functions

A window function computes over related rows and keeps every row. The syntax is `function() OVER (PARTITION BY ... ORDER BY ... frame)`. There are three families: aggregates used as windows (SUM, AVG, COUNT, MIN, MAX), ranking (ROW_NUMBER, RANK, DENSE_RANK, NTILE, PERCENT_RANK, CUME_DIST) and value functions (LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE). Windows run after WHERE, GROUP BY and HAVING, so they appear only in SELECT and ORDER BY, and a filter on them needs a subquery, a CTE or QUALIFY (DuckDB, and per their documentation others such as Snowflake and BigQuery).

On ties the ranking functions differ. Ben and Cleo tie at 150, and Fay and Gus tie at 110:

```sql
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC, emp_id) AS rn,
       RANK()       OVER (ORDER BY salary DESC)         AS rnk,
       DENSE_RANK() OVER (ORDER BY salary DESC)         AS dense_rnk
FROM employees
ORDER BY salary DESC, emp_id
```

Output:

```text
name  salary  rn  rnk  dense_rnk
----  ------  --  ---  ---------
Ada   200     1   1    1
Ben   150     2   2    2
Cleo  150     3   2    2
Eli   130     4   4    3
Dev   120     5   5    4
Fay   110     6   6    5
Gus   110     7   6    5
Ian   100     8   8    6
Hana  90      9   9    7
```

ROW_NUMBER is unique and arbitrary on ties, RANK shares and skips, DENSE_RANK shares and does not skip. The numbering of tied rows with ROW_NUMBER depends on physical order: I stored the same rows in reverse order and Ben and Cleo swapped numbers. Add a unique tiebreaker to the window's ORDER BY. The ORDER BY inside OVER defines the window order, while the final ORDER BY only sorts the output.

LAG and LEAD return the previous or next row's value (NULL at the edges, or a default you pass), and mean previous row, not previous calendar period. A running total is `SUM(x) OVER (ORDER BY d)`, a percent of total is `x / SUM(x) OVER ()`, and a window can sit on aggregated rows with `SUM(SUM(x)) OVER ()` because windows run after GROUP BY. NTILE(n) cuts rows into n nearly equal buckets and gives the extra rows to the first buckets, so 9 rows into 4 buckets is 3, 2, 2, 2. A WINDOW clause names a definition for reuse.

Execution: partition, sort by the partition and order keys, then one pass per distinct specification. Cost is dominated by one sort per distinct specification, and a window with no PARTITION BY on a big table is a single large sort. SQLite's plan for two windows with different keys shows two `USE TEMP B-TREE FOR ORDER BY` steps. `COUNT(DISTINCT x) OVER ()` is rejected by SQLite and accepted by DuckDB.

> **Say it like this:** "I use ROW_NUMBER for one row per group, RANK when ties should share a place, and DENSE_RANK for the Nth distinct value, always with a tiebreaker."

> **Interview trap:** writing a window function in WHERE.

## Window frames

A frame is the set of rows around the current row that an aggregate or value function reads. ROWS counts physical rows, RANGE works on the value of the ORDER BY column and includes all peers (rows with equal ordering values), and GROUPS counts peer groups. According to the SQLite documentation, which I read, the default with ORDER BY is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE NO OTHERS`, and without ORDER BY all rows are peers, so the frame is the whole partition. FIRST_VALUE, LAST_VALUE, NTH_VALUE and aggregates honour the frame, while ranking functions and LAG or LEAD ignore it (I ran LAG with a frame, and the result did not change).

Two rows share January 2 (amounts 20 and 5). The default and RANGE frames give both rows the same running total, while ROWS and GROUPS differ:

```sql
SELECT day, amount,
       SUM(amount) OVER (ORDER BY day)                                   AS default_frame,
       SUM(amount) OVER (ORDER BY day, amount ROWS UNBOUNDED PRECEDING)  AS rows_frame,
       SUM(amount) OVER (ORDER BY day RANGE UNBOUNDED PRECEDING)         AS range_frame,
       SUM(amount) OVER (ORDER BY day GROUPS UNBOUNDED PRECEDING)        AS groups_frame,
       SUM(amount) OVER (ORDER BY day GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW) AS last_two_days
FROM sales_by_day
ORDER BY day, amount
```

Output:

```text
day         amount  default_frame  rows_frame  range_frame  groups_frame  last_two_days
----------  ------  -------------  ----------  -----------  ------------  -------------
2024-01-01  10      10             10          10           10            10
2024-01-02  5       35             15          35           35            35
2024-01-02  20      35             35          35           35            35
2024-01-03  30      65             65          65           65            55
2024-01-05  40      105            105         105          105           70
2024-01-06  50      155            155         155          155           90
```

The same statement ran identically on SQLite, which supports GROUPS. `rows_frame` needs `day, amount` as a unique ordering or the order of the tied rows is arbitrary.

A 3 month moving average is `AVG(x) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`. In the first rows the window is shorter, so wrap it in a CASE on `COUNT(*) OVER` the same window if you want NULL until it is full. ROWS counts rows, so a missing month stretches the window across more calendar time. A RANGE frame on a date, `RANGE BETWEEN INTERVAL 2 DAYS PRECEDING AND CURRENT ROW`, follows the calendar. On the sample, with January 4 missing, the average on January 5 is 35 with RANGE (January 3 and 5) and 30 with ROWS (January 2, 3 and 5). If a missing day should count as zero, join a date spine first.

LAST_VALUE with the default frame returns the current row, because the frame stops there. Extend it with `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`, or use FIRST_VALUE with the order reversed. NTH_VALUE has the same issue: NTH_VALUE(x, 2) is NULL on the first row under the default frame.

EXCLUDE CURRENT ROW, EXCLUDE GROUP and EXCLUDE TIES remove rows from the frame. RANGE with an offset needs exactly one numeric or date ORDER BY column. Frames ending at UNBOUNDED FOLLOWING force the engine to read the whole partition before returning a row.

> **Interview trap:** a running total that repeats on tied rows, which is the RANGE default, and LAST_VALUE returning the current row.

## Top N, duplicates and Nth value

Top N per group is rank, then filter. Number rows within each group with ROW_NUMBER (exactly N rows per group), RANK (ties share a place, N places) or DENSE_RANK (N distinct values), then keep the rank at most N in an outer query:

```sql
WITH ranked AS (
  SELECT name, dept_id, salary,
         ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC, emp_id) AS rn
  FROM employees
)
SELECT dept_id, name, salary, rn
FROM ranked
WHERE rn <= 2
ORDER BY dept_id NULLS LAST, rn
```

Output:

```text
dept_id  name  salary  rn
-------  ----  ------  --
10       Ada   200     1
10       Ben   150     2
20       Eli   130     1
20       Fay   110     2
30       Hana  90      1
NULL     Ian   100     1
```

GROUP BY cannot do this because it collapses the rows you need. Department 30 has one employee, and Ian is alone in the NULL department partition. On the department with salaries 200, 150, 150, 120, "top 3" returns Ada, Ben and Cleo with ROW_NUMBER or RANK, and four rows with DENSE_RANK. Ask what top N means.

The second highest salary has several correct forms: the MAX below the MAX, a DENSE_RANK equal to 2, or DISTINCT with LIMIT 1 OFFSET 1 wrapped in a scalar subquery:

```sql
SELECT MAX(salary) AS second_highest FROM employees WHERE salary < (SELECT MAX(salary) FROM employees)
```

Output:

```text
second_highest
--------------
150
```

State the edge cases: ties at the top (values 10, 10, 7 give 10 with OFFSET 1 and no DISTINCT, but the answer is 7), fewer than two distinct values (return NULL, not an empty result) and NULL salaries. The Nth highest generalises to `DENSE_RANK() = N`. The third highest distinct salary here is 130, while the third row is 150.

To keep only the latest row per key, rank within the key, newest first, with a unique tiebreaker, and keep 1:

```sql
SELECT id, email, name, updated_at
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY email ORDER BY updated_at DESC, id DESC) AS rn
  FROM contacts
) AS t
WHERE rn = 1
ORDER BY id
```

Output:

```text
id  email    name       updated_at
--  -------  ---------  ----------
2   a@x.com  Ann B      2024-03-01
3   b@x.com  Bob        2024-02-01
6   c@x.com  Catherine  2024-02-20
```

Ids 2, 3 and 6 survive. SELECT DISTINCT only removes rows that are identical in every column. To remove the losers, first SELECT the rows with a rank above 1, check the result, run the removal on the primary key inside a transaction, and then add a unique index. For big tables, build a new table with the rows to keep and swap it in. The older pattern of NOT IN with MIN(id) is slow and breaks when the id can be NULL.

The latest order per customer three ways behaves differently: ROW_NUMBER keeps one row per customer and keeps the NULL customer, a join to MAX(date) returns every tied row and drops the NULL customer, and DuckDB's `arg_max` aggregate returns one value per group.

> **Interview trap:** OFFSET without DISTINCT for the Nth highest, and SELECT DISTINCT to remove duplicates.

## Gaps, islands and streaks

An island is a run of consecutive values, a gap is the break between runs. Number the distinct rows per key in order and subtract the row number from the value. Inside a run both advance by one, so the difference is constant. A gap of g missing values adds g to the difference. Group by the difference to get one group per island, then take MIN, MAX and COUNT for the start, end and length. Deduplicate first: a repeated day makes the row numbers run ahead of the dates and splits a streak.

```sql
WITH dd AS (SELECT DISTINCT user_id, login_date FROM logins),
numbered AS (
  SELECT user_id, login_date,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM dd
),
islands AS (
  SELECT user_id, login_date - CAST(rn AS INTEGER) AS grp,
         MIN(login_date) AS start_day, MAX(login_date) AS end_day, COUNT(*) AS days
  FROM numbered
  GROUP BY user_id, grp
)
SELECT user_id, start_day, end_day, days
FROM islands
ORDER BY user_id, start_day
```

Output:

```text
user_id  start_day   end_day     days
-------  ----------  ----------  ----
1        2024-01-01  2024-01-03  3
1        2024-01-05  2024-01-06  2
1        2024-01-09  2024-01-09  1
2        2024-01-01  2024-01-01  1
2        2024-01-03  2024-01-07  5
3        2024-01-10  2024-01-10  1
```

User 1 logged in twice on January 2, so the distinct step matters. User 2's longest streak is five days (January 3 to 7), user 1's is three. The longest streak per user is the maximum days over each user's islands. SQLite spells the date minus integer as `date(login_date, '-' || rn || ' days')` and gave the same islands.

For a fixed small run length, LEAD works: with distinct dates, `LEAD(login_date, 2) = login_date + 2` means three consecutive days. Islands scale to any length with `HAVING COUNT(*) >= N`. Missing ids in a sequence come from `LEAD(id) - id > 1`, with the gap running from `id + 1` to `next_id - 1`, and the edge gaps before the first and after the last id need separate handling.

Collapsing a status log into periods uses LAG: flag a change when the status differs from the previous row (or the previous row is NULL), take a running SUM of the flags as the period id, then group. Grouping by status alone would merge separate periods.

A current streak is the island that ends on the as of date or the day before. Tolerating a missed day means starting a new streak only when the gap is above 2 days. Weekly streaks truncate to the week and subtract `7 * row_number`. Definition questions matter: calendar days or active days, whose time zone, whether yesterday still counts, what counts as activity, and how late data is handled.

> **Say it like this:** "I deduplicate the days, number them, and group by the date minus the row number, which is constant inside a run."

> **Interview trap:** forgetting to deduplicate days before numbering.

## Retention, cohorts and funnels

Retention is the share of a cohort that returns. Classic day N counts users active exactly on day N after signup, rolling (unbounded) retention counts users active on day N or later and never rises as N grows, and bracket retention counts activity within a window. The denominator is the full cohort. Define the cohort, what counts as activity, the day boundary and its time zone.

Day 1 and day 7 retention per signup date joins users to activity, computes the day offset, and counts distinct users per offset. Use a LEFT JOIN so non returning users stay in the denominator, and COUNT DISTINCT so several events a day do not push the rate above 100 percent:

```sql
SELECT u.signup_date,
       COUNT(DISTINCT u.user_id) AS cohort_size,
       COUNT(DISTINCT CASE WHEN a.activity_date - u.signup_date = 1 THEN u.user_id END) AS d1,
       COUNT(DISTINCT CASE WHEN a.activity_date - u.signup_date = 7 THEN u.user_id END) AS d7,
       ROUND(100.0 * COUNT(DISTINCT CASE WHEN a.activity_date - u.signup_date = 1 THEN u.user_id END) / COUNT(DISTINCT u.user_id), 1) AS d1_pct,
       ROUND(100.0 * COUNT(DISTINCT CASE WHEN a.activity_date - u.signup_date = 7 THEN u.user_id END) / COUNT(DISTINCT u.user_id), 1) AS d7_pct
FROM users u
LEFT JOIN activity a ON a.user_id = u.user_id
GROUP BY u.signup_date
ORDER BY u.signup_date
```

Output:

```text
signup_date  cohort_size  d1  d7  d1_pct  d7_pct
-----------  -----------  --  --  ------  ------
2024-01-01   3            1   1   33.3    33.3
2024-01-02   2            1   1   50      50
2024-01-08   1            1   0   100     0
```

The January 8 cohort shows 0 percent at day 7, but the data stops on January 9, so day 7 has not happened. That is missing data, not churn, and a cohort is mature for day N only when `signup_date + N` is within the data. A monthly cohort triangle uses the month offset, pivots with conditional aggregation, and shows NULL where the offset has not been observed.

A funnel counts distinct users per step, but counting per event name ignores order and skipped steps. On the sample it gives 6 viewers, 4 carters and 4 purchasers, a suspicious 100 percent cart to purchase rate. A true funnel takes the earliest time per step per user and chains the conditions, so each step requires the previous one, and optionally a time window:

```sql
WITH t AS (
  SELECT user_id,
         MIN(event_ts) FILTER (WHERE event_name = 'view')     AS t_view,
         MIN(event_ts) FILTER (WHERE event_name = 'cart')     AS t_cart,
         MIN(event_ts) FILTER (WHERE event_name = 'purchase') AS t_purchase
  FROM events
  GROUP BY user_id
)
SELECT COUNT(t_view) AS viewed,
       COUNT(*) FILTER (WHERE t_cart > t_view) AS carted_after_view,
       COUNT(*) FILTER (WHERE t_cart > t_view AND t_purchase > t_cart) AS purchased_after_cart,
       COUNT(*) FILTER (WHERE t_cart > t_view AND t_purchase > t_cart
                          AND t_purchase <= t_view + INTERVAL 24 HOUR) AS within_24h
FROM t
```

Output:

```text
viewed  carted_after_view  purchased_after_cart  within_24h
------  -----------------  --------------------  ----------
6       3                  2                     1
```

Ordered, the funnel is 6, 3 and 2, and with a 24 hour window from the first view only 1 user remains. Earliest time per step is a simplification: a user whose first cart preceded the view but who carted again afterwards is excluded.

DAU is the count of distinct users per day. Rolling distinct active users joins a date spine to activity on a date range, because distinct counts do not add across days: three days of DAU counts sum to 8 while 5 distinct users were active. Stickiness is DAU divided by MAU.

> **Interview trap:** reporting 0 percent for an immature cohort, and summing daily distinct counts.

## Growth, percentiles and sessions

Month over month growth is `(revenue - prev) / prev` with `prev = LAG(revenue) OVER (ORDER BY month)`. The first month is NULL, a zero previous month should give NULL through `NULLIF(prev, 0)`, and `100.0 *` avoids integer division. LAG takes the previous row, so a missing month needs a date spine. Year over year uses `LAG(x, 12)` only on complete monthly data.

The median is PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x) or MEDIAN in DuckDB. A portable version numbers the rows with ROW_NUMBER, counts them with COUNT OVER, and averages the rows whose number is between FLOOR((n + 1) / 2.0) and CEIL((n + 1) / 2.0). PERCENTILE_DISC returns an actual value, PERCENTILE_CONT interpolates at position `1 + p * (n - 1)`. For the nine salaries, the 90th percentile is 160 by CONT, which is 0.2 of the way from the 8th value (150) to the 9th (200), and 200 by DISC:

```sql
SELECT percentile_cont(0.5)  WITHIN GROUP (ORDER BY salary) AS p50_cont,
       percentile_disc(0.5)  WITHIN GROUP (ORDER BY salary) AS p50_disc,
       percentile_cont(0.75) WITHIN GROUP (ORDER BY salary) AS p75_cont,
       percentile_disc(0.75) WITHIN GROUP (ORDER BY salary) AS p75_disc,
       percentile_cont(0.9)  WITHIN GROUP (ORDER BY salary) AS p90_cont,
       percentile_disc(0.9)  WITHIN GROUP (ORDER BY salary) AS p90_disc
FROM employees
```

Output:

```text
p50_cont  p50_disc  p75_cont  p75_disc  p90_cont  p90_disc
--------  --------  --------  --------  --------  --------
120       120       150       150       160       200
```

Sessionization uses LAG and a running sum. Flag a new session when there is no previous event or the gap exceeds the threshold, then SUM the flags per user to number the sessions:

```sql
WITH gaps AS (
  SELECT user_id, ts,
         CASE WHEN LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) IS NULL
                OR ts - LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) > INTERVAL 30 MINUTE
              THEN 1 ELSE 0 END AS new_session
  FROM clicks
),
sess AS (
  SELECT user_id, ts, SUM(new_session) OVER (PARTITION BY user_id ORDER BY ts) AS session_id
  FROM gaps
)
SELECT user_id, session_id, MIN(ts) AS started, MAX(ts) AS ended,
       COUNT(*) AS events, date_diff('minute', MIN(ts), MAX(ts)) AS minutes
FROM sess
GROUP BY user_id, session_id
ORDER BY user_id, session_id
```

Output:

```text
user_id  session_id  started              ended                events  minutes
-------  ----------  -------------------  -------------------  ------  -------
1        1           2024-03-01 09:00:00  2024-03-01 09:25:00  3       25
1        2           2024-03-01 10:30:00  2024-03-01 10:35:00  2       5
1        3           2024-03-01 14:00:00  2024-03-01 14:00:00  1       0
2        1           2024-03-01 09:05:00  2024-03-01 09:05:00  1       0
2        2           2024-03-01 09:50:00  2024-03-01 09:55:00  2       5
```

User 1 has three sessions and user 2 has two. The threshold is a definition, a gap of exactly 30 minutes stays in the same session under a greater than rule, and sessions that cross the edge of a date range need a lookback.

Distinct counts and medians are not decomposable: the answer for a union of groups cannot be built from the group answers. Sketches fix this at the price of approximation. HyperLogLog gives mergeable distinct counts, quantile sketches such as t-digest give mergeable percentiles, and DuckDB exposes them as `approx_count_distinct` and `approx_quantile`. Store daily sketches and merge them for any window, and use exact counts for billing.

> **Remember:** aggregates can be rolled up exactly when they are decomposable: SUM, COUNT, MIN, MAX, and AVG from sum and count.

## X but not Y, and pivoting

Users who did X but not Y is a set difference over a user's whole history. Four forms agree on the sample (viewed but never purchased gives users 2 and 3): NOT EXISTS, a LEFT JOIN with IS NULL on the purchase side, EXCEPT of two user sets, and a single pass GROUP BY with conditional counts in HAVING:

```sql
SELECT user_id
FROM events
GROUP BY user_id
HAVING COUNT(*) FILTER (WHERE event_name = 'view') > 0
   AND COUNT(*) FILTER (WHERE event_name = 'purchase') = 0
ORDER BY user_id
```

Output:

```text
user_id
-------
2
3
```

Avoid `WHERE event_name <> 'purchase'`, which keeps the view rows of users who also purchased, and avoid NOT IN against a nullable column. The grouped form extends to "A and B but not C" by adding one more FILTER count. Users who did both X and Y use INTERSECT or `COUNT(DISTINCT event_name) = 2`, and X followed by Y within 24 hours uses a self join with an ordering and time condition, with DISTINCT on the user. On the sample, 3 users carted and purchased, and 2 did within 24 hours.

Pivoting turns row values into columns. Conditional aggregation is portable: group by the row key and write one SUM with a condition per output column, using a half open date range for months:

```sql
SELECT customer_id,
       SUM(CASE WHEN order_date >= '2024-01-01' AND order_date < '2024-02-01' THEN amount ELSE 0 END) AS jan,
       SUM(CASE WHEN order_date >= '2024-02-01' AND order_date < '2024-03-01' THEN amount ELSE 0 END) AS feb,
       SUM(amount) AS total
FROM orders
GROUP BY customer_id
ORDER BY customer_id NULLS LAST
```

Output:

```text
customer_id  jan  feb  total
-----------  ---  ---  -----
1            80   60   140
2            20   70   90
3            0    40   40
NULL         0    10   10
```

The columns are fixed in the query text, so a dynamic set of columns needs generated SQL or a long format result. DuckDB has native PIVOT and UNPIVOT, SQL Server, Oracle, Snowflake and BigQuery have PIVOT, and PostgreSQL has crosstab in an extension (from memory except DuckDB). A dynamic pivot cannot be a plain query in most engines because a plan needs a fixed set of output columns.

Unpivoting is a UNION ALL with one SELECT per column (key, a constant name, the value), or UNPIVOT, or a lateral join to a VALUES list that reads the table once. Filter NULL cells if needed, keep the types compatible across branches, and use UNION ALL, not UNION, so identical rows survive.

> **Say it like this:** "For users who did X but not Y I use NOT EXISTS, and a grouped HAVING when there are several conditions."

> **Interview trap:** a row level filter that cannot express a condition about a user's whole history.

## Performance and indexes

An index is a separate sorted structure, usually a B tree, mapping column values to row locations. It speeds up equality and range filters, join lookups and ORDER BY on the indexed columns, and it costs space and write speed, because every insert, update or delete maintains it. Index what queries filter, join and sort on, and confirm with the plan.

A predicate is sargable when an index can serve it. Wrapping the column in a function or arithmetic hides it from the index. Use The Index, Luke, which I read, states the rule: keep functions off the indexed column, move conversions to the constant side, and write continuous periods as range conditions with a half open upper bound. On a 50,000 row SQLite table with an index on the date column:

`WHERE date(order_date) = '2023-03-05'` gives the plan

```text
SCAN big_orders
```

and the range rewrite `order_date >= '2023-03-05' AND order_date < '2023-03-06'` gives

```text
SEARCH big_orders USING INDEX idx_date (order_date>? AND order_date<?)
```

A leading wildcard LIKE also scans. If you cannot change the query, an expression index on the exact expression helps.

A composite index on (a, b) is sorted by a then b, so it serves a, or a plus b, but not b alone. Put equality columns first and the range or sort column last. A covering index holds every column the query needs, so the table is never read, and an index in the right order removes a sort. Keyset pagination (`WHERE id > last ORDER BY id LIMIT n`) seeks, while a large OFFSET reads and discards rows.

EXPLAIN shows the chosen plan with estimates, EXPLAIN ANALYZE runs the query and adds actual rows and time. Read from the innermost step outward. Look for full scans of big tables, sorts, the join method and above all a large gap between estimated and actual rows, which points at stale statistics. A nested loop suits few outer rows with an indexed inner side, a hash join suits large unsorted equality joins and needs memory, a merge join suits presorted inputs. The optimizer chooses from estimated row counts, so wrong estimates (stale statistics, skew, correlated predicates) lead to bad plans. In a DuckDB plan on 200,000 rows, the filtered scan was estimated at about 100,000 rows and returned 10,000.

Columnar warehouses read only the needed columns and prune partitions by statistics and partition values instead of using B tree indexes. On Parquet files partitioned by month, DuckDB's plan read one column and one of 12 files. Avoid `SELECT *` there.

> **Interview trap:** DATE(col) or a month function on an indexed column, and treating plan cost as milliseconds.

## Pandas equivalents

Pandas maps to SQL as follows: `groupby().agg` is GROUP BY (size is COUNT(*), count is COUNT(col), nunique is COUNT DISTINCT), `merge(how=...)` covers the join types, `transform` is a window aggregate over a partition, `shift` is LAG and LEAD, `rank(method='first' | 'min' | 'dense')` is ROW_NUMBER, RANK and DENSE_RANK, `cumsum` is a running total, `rolling(n)` is a ROWS frame, `rolling('3D')` on a datetime index is a RANGE frame, and `pivot_table` and `melt` are conditional aggregation and UNPIVOT. HAVING is a boolean filter on the aggregated frame. Window functions need a prior sort, which plays the role of ORDER BY inside OVER.

The semantic differences come from missing values. I ran this on pandas 3.0.6:

```python
import pandas as pd, numpy as np
print("nan == nan:", np.nan == np.nan, "| pd.NA == pd.NA:", pd.NA == pd.NA)
s = pd.Series([1, np.nan, 3])
print("sum:", s.sum(), "mean:", s.mean(), "count:", s.count(), "len:", len(s))
print("all missing, float:", pd.Series([np.nan, np.nan]).sum(), "| all missing, Int64:", pd.Series([None, None], dtype="Int64").sum())
print("int with None ->", pd.Series([1, 2, None]).dtype, "| nullable ->", pd.Series([1, 2, None], dtype="Int64").dtype)
df = pd.DataFrame({"k": [np.nan, np.nan, 1.0], "v": [1, 2, 3]})
print("groupby default:", df.groupby("k")["v"].sum().to_dict())
print("groupby dropna=False:", df.groupby("k", dropna=False)["v"].sum().to_dict())
print("drop_duplicates keeps one NaN key:", len(df.drop_duplicates(subset="k")))
print("sort_values puts NaN:", df.sort_values("k")["k"].tolist())
print("boolean with NA:", pd.Series([True, False, None], dtype="boolean").tolist())
```

Output:

```text
nan == nan: False | pd.NA == pd.NA: <NA>
sum: 4.0 mean: 2.0 count: 2 len: 3
all missing, float: 0.0 | all missing, Int64: 0
int with None -> float64 | nullable -> Int64
groupby default: {1.0: 3}
groupby dropna=False: {1.0: 3, nan: 3}
drop_duplicates keeps one NaN key: 2
sort_values puts NaN: [1.0, nan, nan]
boolean with NA: [True, False, <NA>]
```

pandas drops NaN group keys by default (`dropna=False` keeps them), merge matches NaN keys with each other where SQL never does, the sum of an all missing column is 0 where SQL gives NULL, an integer column with a missing value becomes float unless you use the nullable `Int64` dtype, and NaN sorts last. Fan out exists in pandas too, so pass `validate='one_to_many'` to merge and make it an error. `pivot` raises on duplicate index and column pairs while `pivot_table` aggregates them.

`rolling('3D')` includes the current row and rows within the previous 3 days, excluding the left edge, so on the sample it gives 10, 15, 20, 35, 45, equal to the SQL RANGE frame, while `rolling(3, min_periods=1)` gives 10, 15, 20, 30, 40, equal to the ROWS frame. To count missing days as zero, `asfreq('D', fill_value=0)` first, the pandas version of a date spine.

Use pandas for exploration, plotting and models, and SQL for joins and aggregations at scale. DuckDB can query DataFrames directly with SQL, which suits mixing both.

> **Interview trap:** assuming groupby and merge treat missing keys the way SQL does.

## Analytics engineering

Normalisation stores each fact once: first normal form means atomic cells, second means non key columns depend on the whole key, third means they depend only on the key. Operational databases normalise to protect writes from update anomalies, and warehouses denormalise because they read far more than they write.

A star schema has a central fact table of events (one row per order line, with foreign keys and numeric measures) surrounded by dimension tables of descriptive attributes. A snowflake schema normalises the dimensions further and adds joins. The grain is what one row of a table represents. Declare it first, test that the grain columns are unique, and keep measures at their own grain: an order amount copied onto each order line sums to 420 when the orders with items really total 250.

Fact tables come in three types: transaction (one row per event), periodic snapshot (one row per entity per period, such as daily balances) and accumulating snapshot (one row per process instance, updated at milestones). Measures are additive (sales amount), semi additive (balances, which you must not sum across time) or non additive (ratios, rebuilt from numerator and denominator). Dimensions use surrogate keys, may be role playing (one date table joined as order date and ship date), are conformed when shared across facts, and include an unknown member (key minus 1) so facts with a missing value still join. The Kimball Group's list of techniques, which I fetched, names grain, the three fact table types, surrogate keys, degenerate, conformed and junk dimensions, and slowly changing dimension types 0 to 7.

Slowly changing dimensions: type 0 never changes, type 1 overwrites (history lost), type 2 closes the current row and adds a new version with a new surrogate key and validity dates, type 3 adds a previous value column. A point in time join to a type 2 dimension uses the business key and half open ranges, `valid_from <= date < valid_to`, with the current row's end set far in the future:

```sql
SELECT o.order_id, o.order_date, o.amount, d.city
FROM fact_orders o
JOIN dim_cust_scd2 d
  ON d.cust_id = o.cust_id
 AND o.order_date >= d.valid_from
 AND o.order_date <  d.valid_to
ORDER BY o.order_id
```

Output:

```text
order_id  order_date  amount  city
--------  ----------  ------  ------
1         2023-05-10  30      Pune
2         2023-07-01  45      Austin
3         2023-08-02  20      Austin
4         2023-08-02  25      Oslo
```

With inclusive BETWEEN and a shared boundary day, an order on the change date matches two versions and is duplicated:

```sql
SELECT o.order_id, o.order_date, d.cust_key, d.city
FROM fact_orders o
JOIN dim_cust_scd2 d
  ON d.cust_id = o.cust_id
 AND o.order_date BETWEEN d.valid_from AND d.valid_to
WHERE o.cust_id = 'C1'
ORDER BY o.order_id, d.cust_key
```

Output:

```text
order_id  order_date  cust_key  city
--------  ----------  --------  ------
1         2023-05-10  1         Pune
2         2023-07-01  1         Pune
2         2023-07-01  2         Austin
3         2023-08-02  2         Austin
```

Order 2 appears twice. Validate that versions do not overlap or leave gaps, and build `valid_to` from the next row's `valid_from` with LEAD.

dbt turns each model, a SELECT in a file, into a table or view, builds in the order given by `ref()` and `source()`, and supports view, table, incremental and ephemeral materializations. A dbt data test is a query returning the failing rows, zero rows is a pass, and the built in generic tests are unique, not_null, accepted_values and relationships (per the dbt documentation I read). Incremental models process rows newer than a watermark and merge by a unique key, and a strict watermark loses late arriving rows, which a lookback window fixes. Snapshots record changes to a mutable source as a type 2 dimension with `dbt_valid_from` and `dbt_valid_to`. I did not install dbt, so the model and test SQL in the cards was run as plain SQL in DuckDB.

When a metric changes after a refactor, compare totals and row counts, diff the two outputs by key with a full outer join and null safe equality, and look for fan out (inflated values), dropped keys (missing rows) and changed definitions:

```sql
SELECT COALESCE(o.customer_id, n.customer_id) AS customer_id, o.revenue AS old_revenue, n.revenue AS new_revenue,
       n.revenue - o.revenue AS diff
FROM rev_old o
FULL JOIN rev_new n ON o.customer_id IS NOT DISTINCT FROM n.customer_id
WHERE o.revenue IS DISTINCT FROM n.revenue
ORDER BY customer_id NULLS LAST
```

Output:

```text
customer_id  old_revenue  new_revenue  diff
-----------  -----------  -----------  ----
1            140          310          170
2            90           70           -20
NULL         10           NULL         NULL
```

> **Say it like this:** "I state the grain, test its uniqueness, join facts to dimensions on unique keys, and reconcile totals to the source."

> **Interview trap:** joining a type 2 dimension with inclusive ranges and double counting boundary rows.

## What interviewers probe

Expect each of these chains, in roughly this order.

- **Logical order:** order of clauses, then why an alias works in ORDER BY and not in WHERE, then where windows fit, then LIMIT without ORDER BY.
- **NULL:** `= NULL`, then three valued logic, then COUNT(*) versus COUNT(col), then NOT IN with a NULL, then NULLs in joins and GROUP BY.
- **Joins:** inner versus left, then the row count with duplicate keys, then ON versus WHERE for outer joins, then semi and anti joins, then self joins.
- **Fan out:** revenue doubled after a join, then how to detect it, then why SUM(DISTINCT) is wrong, then fixing at the grain.
- **Windows:** ROW_NUMBER versus RANK versus DENSE_RANK, then top N per group, then the second highest salary and ties, then LAG for growth, then running totals and the default frame.
- **Frames:** ROWS versus RANGE, then the LAST_VALUE trap, then a moving average with missing days.
- **Patterns:** deduplicate keeping the latest, consecutive day streaks (gaps and islands), retention and cohort tables, funnels with order and time windows, sessionization, median and percentiles.
- **Performance:** what an index is, then sargable predicates, then composite index order, then reading EXPLAIN ANALYZE, then estimated versus actual rows.
- **Pandas:** groupby and merge, then NaN versus NULL differences, then rolling windows.
- **Modelling:** star schema and grain, then slowly changing dimensions, then point in time joins, then dbt tests and incremental models, then debugging a changed metric.

## Sources

- [SQLite documentation: window functions](https://www.sqlite.org/windowfunctions.html)
- [PostgreSQL documentation: WITH queries (common table expressions)](https://www.postgresql.org/docs/current/queries-with.html)
- [Use The Index, Luke: obfuscated conditions on dates](https://use-the-index-luke.com/sql/where-clause/obfuscation/dates)
- [dbt documentation: data tests](https://docs.getdbt.com/docs/build/data-tests)
- [dbt documentation: snapshots](https://docs.getdbt.com/docs/build/snapshots)
- [Kimball Group: dimensional modeling techniques](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/)
- [The SQL patterns behind almost every interview question (techinterview.org)](https://www.techinterview.org/post/3233476262/sql-interview-question-patterns/)
- [SQL Interview Questions (2026): Window Functions, Optimization, and Advanced Queries (techinterview.org)](https://www.techinterview.org/post/3233460377/sql-interview-questions-2026-window-functions-optimization-and-advanced-queries/)
