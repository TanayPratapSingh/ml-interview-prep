# SQL and Analytics for Interviews, Plain Version

SQL interviews test a short list of ideas again and again: the order a query runs in, how NULL behaves, how joins multiply rows, window functions, and a dozen patterns built from them. This guide explains each idea in everyday terms with one small worked example. Every example uses the same tiny tables (nine employees, seven orders, a few events) so the numbers carry over from section to section. The deep version has the full queries and their real output.

## How SQL runs a query

A query is written starting with SELECT but runs starting with FROM. Picture an assembly line. First the tables are chosen and joined. Then WHERE throws away single rows, GROUP BY bundles the survivors, HAVING throws away whole bundles, SELECT builds the output columns, ORDER BY sorts, and LIMIT cuts the list.

Take nine employees. `WHERE salary >= 100` leaves eight. Grouping by department makes three bundles, and `HAVING COUNT(*) >= 2` keeps two of them. Each step only sees what the step before produced, which is why WHERE cannot see a COUNT but HAVING can.

The engine is free to reorder the real work for speed. The order is a rule about meaning, not about timing.

> **Interview trap:** a SELECT alias is not visible in WHERE under the standard rules, although DuckDB and SQLite allow it.

## NULL and three valued logic

NULL means "unknown", not zero and not an empty string. Any comparison with an unknown gives unknown, so `col = NULL` never matches anything. WHERE keeps a row only when the condition is true, and both false and unknown are thrown away.

Ian has no department. The filter `dept_id <> 10` silently drops him, because `NULL <> 10` is unknown. `COUNT(*)` counts all 9 employees while `COUNT(dept_id)` counts the 8 who have a department.

The classic disaster is `NOT IN`. If any order has a NULL customer, `customer_id NOT IN (SELECT customer_id FROM orders)` returns no rows at all, even though one customer has never ordered. Use `NOT EXISTS` instead.

> **Interview trap:** NOT IN plus a NULL in the subquery returns nothing, and it does so without an error.

## GROUP BY, HAVING and aggregates

GROUP BY turns many rows into one row per group. Think of sorting receipts into piles by customer and then writing one summary line per pile. Every column you select must be a pile label or a summary of the pile.

HAVING filters piles after they are built, WHERE filters receipts before. "Customers with at least two paid orders and over 100 spent" is WHERE for paid, then HAVING for the count and total.

An average of averages is wrong. Three customers average 46.67, 45 and 40 over three, two and one orders. Their plain mean is 43.89, but the true average of the six orders is 45. Carry sums and counts, then divide at the end.

An aggregate with no GROUP BY always returns one row, even over nothing: COUNT is 0 and SUM is NULL.

## Joins

A join pairs rows from two tables. An inner join keeps only the pairs that match. A left join also keeps every left row that found no partner, filling the right side with NULL. A full join keeps unmatched rows from both sides.

With 9 employees and 4 departments, an inner join returns 8 rows (Ian has no department, Legal has no staff), a left join 9, and a full join 10.

A filter on the right table belongs in ON if you want to keep the left rows, because the same filter in WHERE turns the left join back into an inner join. A semi join (EXISTS) asks whether a match exists, and an anti join (NOT EXISTS) asks whether none does. UNION removes duplicate rows, UNION ALL keeps them and is cheaper.

> **Interview trap:** NULL join keys never match each other.

## Join fan out

Fan out is a join that repeats rows. Each order has one amount, but an order with three items becomes three rows after joining to the items table, so its amount is counted three times.

The seven orders sum to 280. Join them to their eight item rows and the sum becomes 420, with no error and no warning. Orders without items also disappear in an inner join, which is a second silent change.

The fix is to aggregate the many side down to one row per order before joining, so every join is one to one. Then check the answer: the total must equal the plain orders total. `SUM(DISTINCT amount)` is not a fix, because it also removes different orders that happen to share an amount.

Joining two child tables at once multiplies them: 3 items times 2 shipments gives 6 rows per order.

## CASE, dates and strings

CASE is an if statement that returns a value. It checks conditions top to bottom and returns the first match, so put the narrow condition first. Inside an aggregate it gives conditional counting: `SUM(CASE WHEN status = 'refunded' THEN amount ELSE 0 END)`.

For dates, always filter with a half open range: `ts >= '2024-03-01' AND ts < '2024-03-02'`. BETWEEN includes both ends and treats a date as midnight, so it loses or double counts rows near the boundary.

Dividing an integer by an integer can drop the fraction. Six paid orders out of seven is 0 in SQLite and 0.857 in DuckDB. Multiply by 1.0 first.

For month reports, group by the truncated date, not by the month number, or January of two years merges.

## Subqueries and CTEs

A subquery is a query inside a query. A correlated subquery looks at the current outer row, such as "the average salary of this employee's department". The employees above their department average are Ada (200 against 155) and Eli (130 against 116.7).

A CTE, written WITH name AS (...), names a step so a long query reads top to bottom like a recipe. It is mostly a readability tool. In PostgreSQL 12 and later, a CTE used once is folded into the main query, and one used twice is computed once, unless you say otherwise.

EXISTS and IN return each row once, while a join returns one row per match. For the "not" version, prefer NOT EXISTS.

## Recursive CTEs

A recursive CTE repeats a step until nothing new appears. Think of asking "who reports to Ada?", then "who reports to those people?", and so on. You write a starting query and a repeating query joined by UNION ALL.

On the employee table, step 0 finds Ada, step 1 finds four direct reports, step 2 finds four more, and step 3 finds nobody, so it stops. Carry a depth number and a path string to show the structure.

It breaks when the data contains a loop, such as Ada reporting to her own report: the query never ends. Guard it with a depth limit, a visited path, or UNION instead of UNION ALL when the rows are otherwise identical.

Recursive CTEs also build date series, which fill gaps in reports.

## Window functions

A window function calculates across related rows but keeps every row. GROUP BY gives one line per department. `AVG(salary) OVER (PARTITION BY dept_id)` keeps all nine employees and writes the department average beside each.

Ranking functions differ on ties. Two people tie at 150. ROW_NUMBER numbers them 2 and 3, RANK gives both 2 and skips to 4, and DENSE_RANK gives both 2 and continues with 3.

LAG looks at the previous row, such as last month's revenue. Windows run after WHERE and GROUP BY, so you cannot filter on them directly. Put the window in a subquery or CTE and filter outside, or use QUALIFY where it exists.

> **Interview trap:** ROW_NUMBER with tied values numbers them in an arbitrary order, so add a unique tiebreaker.

## Window frames

A frame says how far from the current row the calculation looks. `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` means this row and the two before it, which is a 3 row moving average.

When you add ORDER BY and no frame, the default is a running window that includes ties. If two rows share a date, they both show the same running total. Add a unique tiebreaker and say ROWS to get a strict row by row total.

LAST_VALUE usually just returns the current row, because the default frame stops there. Extend the frame to UNBOUNDED FOLLOWING, or use FIRST_VALUE on the reverse order.

ROWS counts rows, so a missing day shifts the window. A RANGE frame measured in days stays true to the calendar.

## Top N, duplicates and Nth value

"Top 2 earners per department" is a ranking problem. Number the employees inside each department with ROW_NUMBER, then keep numbers 1 and 2. GROUP BY cannot do it because it destroys the rows you want to keep.

The second highest salary is 150 here (Ada earns 200, Ben and Cleo tie at 150). `ORDER BY salary DESC LIMIT 1 OFFSET 1` also gives 150, but for the third highest it gives 150 again, because the tie takes two places. Use DISTINCT or DENSE_RANK to count distinct values.

To keep the latest row per email, number the rows per email by date descending and keep number 1. SELECT DISTINCT does not help when rows differ in other columns. Preview the rows to remove before deleting anything.

## Gaps, islands and streaks

An island is a run of consecutive values, like days in a row with a login. The trick: number the days 1, 2, 3, and subtract that number from each date. Inside a run, the date and the number both go up by one, so the difference stays the same. After a gap the difference jumps.

User 2 logged in on January 1, then on January 3 to 7. The difference is one value for the five consecutive days and another for January 1, so the longest streak is 5 days. Remove duplicate days first, or a repeated login breaks the count.

Use LAG instead when you want to tolerate a one day gap, or to collapse a status log into periods. For a current streak, define whether yesterday still counts.

## Retention, cohorts and funnels

Retention asks what share of a group of new users comes back. A cohort is the group, such as everyone who signed up on January 1. Day 1 retention is the share active exactly one day after signup, and the denominator is the whole cohort.

Of three users who signed up on January 1, one was active on January 2, so day 1 retention is 33.3 percent. A cohort too recent to have reached day 7 shows 0 percent, which is missing data, not churn.

A funnel counts users at each step. Counting distinct users per event name gave 6 viewers, 4 carters and 4 purchasers, which hides that order and skipped steps matter. Requiring each step to follow the last gives 6, 3 and 2.

## Growth, percentiles and sessions

Month over month growth is this month minus last month, divided by last month. Revenue of 100 then 120 is 20 percent growth, then 90 is minus 25 percent. LAG fetches last month. Divide by NULLIF(last month, 0) so a zero month gives NULL, not an error.

The median is the middle value. For the nine salaries it is 120, while the mean is pulled up by the 200. A 90th percentile can be interpolated (160 here) or an actual value from the data (200).

A session is a burst of activity separated by quiet gaps. Flag a new session when the gap since the previous click exceeds 30 minutes, then count the flags to number the sessions.

Distinct counts and medians cannot be added across groups.

## X but not Y, and pivoting

"Users who viewed but never purchased" is a set difference. Everyone who viewed is users 1 to 6, purchasers are 1, 4, 5 and 6, so the answer is users 2 and 3. Write it with NOT EXISTS, a left join with IS NULL, or EXCEPT. Do not filter rows with `event_name <> 'purchase'`, because that keeps the view rows of people who also purchased.

Pivoting turns row values into columns. Group by the row label and add one conditional sum per column: `SUM(CASE WHEN month = 'Jan' THEN amount ELSE 0 END)`. The columns are fixed in the query text. Unpivoting is the reverse, usually a UNION ALL of one SELECT per column.

## Performance and indexes

An index is a sorted lookup structure, like the index at the back of a book. Without one, the database reads every row. With one on the filter column, it jumps to the matching entries. The price is slower writes and extra space.

An index only helps if the query leaves the column bare. `WHERE date(order_date) = '2023-03-05'` hides the column inside a function and forces a full scan. The rewrite is a range: `order_date >= '2023-03-05' AND order_date < '2023-03-06'`.

A composite index on (customer, date) serves a filter on customer, or customer plus date, but not date alone. Read plans with EXPLAIN, and compare estimated and actual rows. Columnar warehouses replace indexes with partition pruning and column pruning.

## Pandas equivalents

Pandas does the same jobs with different names. `groupby` is GROUP BY, `merge` is JOIN, `transform` is a window aggregate, `shift` is LAG, `rank` covers the ranking functions, and `rolling` is a frame.

The surprises are about missing values. Pandas drops rows with a NaN group key unless you pass `dropna=False`, while SQL keeps a NULL group. A merge matches NaN keys to each other, which SQL never does. The sum of an all missing column is 0 in pandas and NULL in SQL.

`rolling('3D')` on a date index behaves like a RANGE frame measured in days. Pass `validate='one_to_many'` to merge so fan out becomes an error.

## Analytics engineering

Analytics engineering is the craft of shaping raw data into trustworthy tables. The usual shape is a star schema: one fact table of events, such as one row per order line, surrounded by dimension tables for product, customer and date.

Start every table with its grain, meaning what one row stands for, and test that the grain columns are unique. Measures only sum safely at their own grain.

Attributes that change, like a customer's city, are handled by slowly changing dimensions. Type 1 overwrites and loses history. Type 2 closes the old row with an end date and adds a new row, so old orders keep the old city.

dbt writes each model as a SELECT, builds them in dependency order, and tests them. A test is a query that returns the bad rows, and zero rows means pass.
