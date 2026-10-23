# SQL and Analytics for Interviews, Plain Version

SQL (Structured Query Language, the language for asking questions of data stored in tables) interviews test a short list of ideas again and again. The first is the order a query runs in. The next is how NULL (a marker for a missing value) behaves. Then come how joins (ways to combine tables) multiply rows, and window functions (calculations across related rows). Last are a dozen patterns built from these. This guide explains each idea in everyday terms with one small worked example. Every example uses the same tiny tables (nine employees, seven orders, a few events), so the numbers carry over from section to section. The deep version has the full queries and their real output.

## How SQL runs a query

A query is a written question to the database. You write it starting with SELECT (which lists the columns you want), but it runs starting with FROM (which names the tables). Picture an assembly line. First the tables are chosen and joined. Then WHERE throws away single rows. GROUP BY bundles the survivors. HAVING throws away whole bundles. SELECT builds the output columns, ORDER BY sorts, and LIMIT cuts the list.

Take nine employees. `WHERE salary >= 100` leaves eight. Grouping by department makes three bundles, and `HAVING COUNT(*) >= 2` keeps two of them. COUNT(*) counts the rows in each bundle. Each step only sees what the step before produced, which is why WHERE cannot see a COUNT but HAVING can.

The engine (the database program) is free to reorder the real work for speed. The order is a rule about meaning, not about timing.

> **Interview trap:** a SELECT alias (a nickname you give an output column) is not visible in WHERE under the standard rules, although DuckDB and SQLite allow it. DuckDB and SQLite are two small database programs.

## NULL and three valued logic

NULL means "unknown", not zero and not an empty string. Any comparison with an unknown gives unknown, so `col = NULL` never matches anything. A condition can therefore be true, false or unknown, which is called three valued logic. WHERE keeps a row only when the condition is true, so both false and unknown are thrown away.

Ian has no department. The filter `dept_id <> 10` silently drops him, because `NULL <> 10` is unknown. The symbol `<>` means "not equal to". `COUNT(*)` counts all 9 employees while `COUNT(dept_id)` counts the 8 who have a department.

The classic disaster is `NOT IN`. If any order has a NULL customer, `customer_id NOT IN (SELECT customer_id FROM orders)` returns no rows at all, even though one customer has never ordered. The part in brackets is a subquery, a query written inside another query. Use `NOT EXISTS` instead. It asks whether no matching row exists.

> **Interview trap:** NOT IN plus a NULL in the subquery returns nothing, and it does so without an error.

## GROUP BY, HAVING and aggregates

GROUP BY turns many rows into one row per group. Think of sorting receipts into piles by customer and then writing one summary line per pile. Every column you select must be a pile label or a summary of the pile. A summary, such as a count or a sum, is called an aggregate.

HAVING filters piles after they are built, WHERE filters receipts before. "Customers with at least two paid orders and over 100 spent" is WHERE for paid, then HAVING for the count and total.

An average of averages is wrong. Three customers average 46.67, 45 and 40 over three, two and one orders. Their plain mean (the simple average of those three numbers) is 43.89, but the true average of the six orders is 45. Carry sums and counts, then divide at the end.

An aggregate with no GROUP BY always returns one row, even over nothing: COUNT is 0 and SUM is NULL.

## Joins

A join pairs rows from two tables. The left table is the one named first. An inner join keeps only the pairs that match. A left join also keeps every left row that found no partner, filling the right side with NULL. A full join keeps unmatched rows from both sides.

With 9 employees and 4 departments, an inner join returns 8 rows. Ian has no department, and Legal has no staff. A left join returns 9 rows, and a full join returns 10.

A filter on the right table belongs in ON (the clause that says how rows pair up) if you want to keep the left rows. The same filter in WHERE turns the left join back into an inner join. A semi join (written with EXISTS) asks whether a match exists. An anti join (written with NOT EXISTS) asks whether none does. UNION stacks two result lists and removes duplicate rows. UNION ALL keeps them and is cheaper.

> **Interview trap:** NULL join keys never match each other.

## Join fan out

Fan out is a join that repeats rows. Each order has one amount. But an order with three items becomes three rows after joining to the items table. So its amount is counted three times.

The seven orders sum to 280. Join them to their eight item rows and the sum becomes 420, with no error and no warning. Orders without items also disappear in an inner join, which is a second silent change.

The fix is to aggregate the many side down to one row per order before joining, so every join is one to one. The many side is the table with several rows per order. Then check the answer: the total must equal the plain orders total. `SUM(DISTINCT amount)` is not a fix. DISTINCT keeps one copy of each value, so it also removes different orders that happen to share an amount.

Joining two child tables at once multiplies them: 3 items times 2 shipments gives 6 rows per order.

## CASE, dates and strings

CASE is an if statement that returns a value. It checks conditions top to bottom and returns the first match, so put the narrow condition first. Inside an aggregate it gives conditional counting: `SUM(CASE WHEN status = 'refunded' THEN amount ELSE 0 END)`. That adds up the amounts of refunded orders only.

For dates, always filter with a half open range: `ts >= '2024-03-01' AND ts < '2024-03-02'`. A half open range includes its start but not its end. Here ts is a timestamp, a column holding a date and a time. BETWEEN includes both ends and treats a date as midnight, so it loses or double counts rows near the boundary.

Dividing an integer by an integer can drop the fraction. An integer is a whole number. Six paid orders out of seven is 0 in SQLite and 0.857 in DuckDB. Multiply by 1.0 first.

For month reports, group by the truncated date, not by the month number, or January of two years merges. Truncating a date cuts it back to the start of its month.

## Subqueries and CTEs

A subquery is a query inside a query. A correlated subquery looks at the current outer row, such as "the average salary of this employee's department". The employees above their department average are Ada (200 against 155) and Eli (130 against 116.7).

A CTE (common table expression), written WITH name AS (...), names a step so a long query reads top to bottom like a recipe. It is mostly a readability tool. PostgreSQL is a widely used database program. In PostgreSQL 12 and later, a CTE used once is folded into the main query. That means the database treats it as part of that query. One used twice is computed once, unless you say otherwise.

EXISTS and IN return each row once, while a join returns one row per match. For the "not" version, prefer NOT EXISTS.

## Recursive CTEs

A recursive CTE repeats a step until nothing new appears. Think of asking "who reports to Ada?", then "who reports to those people?", and so on. You write a starting query and a repeating query joined by UNION ALL.

On the employee table, step 0 finds Ada. Step 1 finds four direct reports, and step 2 finds four more. Step 3 finds nobody, so it stops. Carry a depth number and a path string to show the structure. A path string is a text value that lists the chain of people from the top.

It breaks when the data contains a loop, such as Ada reporting to her own report: the query never ends. Guard it with a depth limit, a visited path, or UNION instead of UNION ALL when the rows are otherwise identical. A visited path records the people already reached, so the query can stop when it meets one again.

Recursive CTEs also build date series, which fill gaps in reports. A date series is one row per day in a range.

## Window functions

A window function calculates across related rows but keeps every row. GROUP BY gives one line per department. `AVG(salary) OVER (PARTITION BY dept_id)` keeps all nine employees and writes the department average beside each. PARTITION BY splits the rows into groups, and OVER says to calculate across each group's rows.

Ranking functions differ on ties. Two people tie at 150. ROW_NUMBER numbers them 2 and 3, RANK gives both 2 and skips to 4, and DENSE_RANK gives both 2 and continues with 3.

LAG looks at the previous row, such as last month's revenue. Windows run after WHERE and GROUP BY, so you cannot filter on them directly. Put the window in a subquery or CTE and filter outside, or use QUALIFY where it exists. QUALIFY is a clause that some databases offer for filtering on window results.

> **Interview trap:** ROW_NUMBER with tied values numbers them in an arbitrary order, so add a unique tiebreaker. A tiebreaker is an extra sort column that makes every row's position certain.

## Window frames

A frame says how far from the current row the calculation looks. `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` means this row and the two before it. That makes a 3 row moving average, which is an average over a sliding set of rows.

When you add ORDER BY and no frame, the default is a running window that includes ties. A running total adds up everything so far. If two rows share a date, they both show the same running total. Add a unique tiebreaker and say ROWS to get a strict row by row total.

LAST_VALUE returns the value from the last row of the frame, and FIRST_VALUE returns it from the first row. LAST_VALUE usually just returns the current row, because the default frame stops there. Extend the frame to UNBOUNDED FOLLOWING, which means all the way to the end of the group. Or use FIRST_VALUE on the reverse order.

ROWS counts rows, so a missing day shifts the window. A RANGE frame measured in days stays true to the calendar.

## Top N, duplicates and Nth value

"Top 2 earners per department" is a ranking problem. Number the employees inside each department with ROW_NUMBER, then keep numbers 1 and 2. GROUP BY cannot do it because it destroys the rows you want to keep.

The second highest salary is 150 here (Ada earns 200, Ben and Cleo tie at 150). `ORDER BY salary DESC LIMIT 1 OFFSET 1` also gives 150. DESC sorts from high to low, and OFFSET 1 skips the first row. For the third highest, the same pattern gives 150 again, because the tie takes two places. Use DISTINCT or DENSE_RANK to count distinct values.

To keep the latest row per email, number the rows per email by date descending and keep number 1. SELECT DISTINCT does not help when rows differ in other columns. Preview the rows to remove before deleting anything.

## Gaps, islands and streaks

An island is a run of consecutive values, like days in a row with a login. A gap is a break between runs, and a streak is a run of days in a row. The trick: number the days 1, 2, 3, and subtract that number from each date. Inside a run, the date and the number both go up by one, so the difference stays the same. After a gap the difference jumps.

User 2 logged in on January 1, then on January 3 to 7. The difference is one value for the five consecutive days and another for January 1, so the longest streak is 5 days. Remove duplicate days first, or a repeated login breaks the count.

Use LAG instead when you want to tolerate a one day gap, or to collapse a status log into periods. For a current streak, define whether yesterday still counts.

## Retention, cohorts and funnels

Retention asks what share of a group of new users comes back. A cohort is the group, such as everyone who signed up on January 1. Day 1 retention is the share active exactly one day after signup. The denominator, the number you divide by, is the whole cohort.

Of three users who signed up on January 1, one was active on January 2, so day 1 retention is 33.3 percent. A cohort too recent to have reached day 7 shows 0 percent. That is missing data, not churn. Churn means users who left.

A funnel counts users at each step of a path, such as view, add to cart and purchase. Counting distinct users per event name gave 6 viewers, 4 carters and 4 purchasers. A carter is a user who added to the cart. That count hides that order and skipped steps matter. Requiring each step to follow the last gives 6, 3 and 2.

## Growth, percentiles and sessions

Month over month growth is this month minus last month, divided by last month. Revenue of 100 then 120 is 20 percent growth, then 90 is minus 25 percent. LAG fetches last month. Divide by NULLIF(last month, 0) so a zero month gives NULL, not an error. NULLIF returns NULL when its first value equals the second.

The median is the middle value. For the nine salaries it is 120, while the mean (the plain average) is pulled up by the 200. A 90th percentile is the value that 90 percent of the data falls at or below. It can be interpolated (160 here), which means estimated between two neighbouring values. Or it can be an actual value from the data (200).

A session is a burst of activity separated by quiet gaps. Flag a new session when the gap since the previous click exceeds 30 minutes, then count the flags to number the sessions.

Distinct counts and medians cannot be added across groups. For example, a user who appears in two groups is counted twice if you add the group counts.

## X but not Y, and pivoting

"Users who viewed but never purchased" is a set difference, which means the items in one group that are not in another. Everyone who viewed is users 1 to 6, purchasers are 1, 4, 5 and 6, so the answer is users 2 and 3. Write it with NOT EXISTS, a left join with IS NULL, or EXCEPT. EXCEPT returns the rows of the first result that are missing from the second. Do not filter rows with `event_name <> 'purchase'`, because that keeps the view rows of people who also purchased.

Pivoting turns row values into columns. Group by the row label and add one conditional sum per column: `SUM(CASE WHEN month = 'Jan' THEN amount ELSE 0 END)`. The columns are fixed in the query text. Unpivoting is the reverse, usually a UNION ALL of one SELECT per column.

## Performance and indexes

An index is a sorted lookup structure, like the index at the back of a book. Without one, the database reads every row. With one on the filter column, it jumps to the matching entries. The price is slower writes (adding or changing data) and extra space.

An index only helps if the query leaves the column bare. `WHERE date(order_date) = '2023-03-05'` hides the column inside a function and forces a full scan, which reads every row. The rewrite is a range: `order_date >= '2023-03-05' AND order_date < '2023-03-06'`.

A composite index on (customer, date) covers several columns in order. It serves a filter on customer, or customer plus date, but not date alone. Read plans with EXPLAIN, which shows the steps the database will take. Compare estimated and actual rows. Columnar warehouses store data by column. They replace indexes with partition pruning and column pruning. Partition pruning skips whole chunks of data that cannot match, and column pruning reads only the columns the query needs.

## Pandas equivalents

Pandas, a Python library for working with tables, does the same jobs with different names. `groupby` is GROUP BY, `merge` is JOIN, `transform` is a window aggregate, `shift` is LAG, `rank` covers the ranking functions, and `rolling` is a frame.

The surprises are about missing values. NaN (not a number) is how pandas marks a missing value. Pandas drops rows with a NaN group key unless you pass `dropna=False`, while SQL keeps a NULL group. A merge matches NaN keys to each other, which SQL never does. The sum of an all missing column is 0 in pandas and NULL in SQL.

`rolling('3D')` on a date index behaves like a RANGE frame measured in days. Here 3D means three days, and a date index uses the date column as the row labels. Pass `validate='one_to_many'` to merge so fan out becomes an error.

## Analytics engineering

Analytics engineering is the craft of shaping raw data into trustworthy tables. The usual shape is a star schema: one fact table of events, such as one row per order line, surrounded by dimension tables for product, customer and date. A fact table records the events you measure. A dimension table describes the things involved.

Start every table with its grain, meaning what one row stands for, and test that the grain columns are unique. Measures only sum safely at their own grain. A measure is a number you add up, such as an amount.

Attributes that change, like a customer's city, are handled by slowly changing dimensions. Type 1 overwrites and loses history. Type 2 closes the old row with an end date and adds a new row, so old orders keep the old city.

dbt (short for data build tool) writes each model as a SELECT. A model is one table built from a query. dbt builds the models in dependency order, which means each one is built after the tables it reads from. It also tests them. A test is a query that returns the bad rows, and zero rows means pass.
