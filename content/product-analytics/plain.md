# Product and business analytics, explained plainly

This is the easy version of the product analytics module. It covers how analyst interviews work and the metrics every product team tracks. It also covers how to find out why a number moved, and how to say the answer to an executive. Each section matches a section of the deep version, so you can switch whenever you want the formulas and edge cases. SQL (Structured Query Language) is a language for asking questions of data stored in tables. A/B testing is an experiment that splits users at random into groups A and B. Both have their own modules, so they get only a mention here. Every number below was checked by running code. The companies and figures in the worked examples are made up for practice, not taken from any real business.

## How the interview rounds work

An analytics interview is a set of short exams on different skills. You usually get a screen with SQL and basic statistics. You also get a product sense question ("how would you measure this feature?"). Another is a root cause question ("this number fell, why?"). Then come an estimation question, and a case where you are handed data and asked for a recommendation. Business intelligence roles (jobs that build reports and dashboards) add a tool round in Excel, Tableau or Power BI. Those are three programs for analysing data and drawing charts, and BI is short for business intelligence. Think of it as a driving test with separate parts: parking, motorway, night driving. Passing one part does not pass the others.

The roles differ in weight. A product analyst works next to product managers on metrics (numbers that track the product) and funnels (series of steps that users move through). A business intelligence analyst builds trusted reports and dashboards (screens of charts and key numbers). A product data scientist adds deeper experiments and modelling, which means building mathematical models of behaviour. Titles vary, so ask the recruiter what each round covers.

> **Say it like this:** "Before I answer, let me make sure I understand the metric and the goal. I will look at three things, in this order, and then give a recommendation."

Where it breaks: most of these questions are open ended, which means they have no single right answer. Interviewers score your structure, your judgement and your clarity more than the final answer.

## Engagement metrics

DAU, WAU and MAU mean daily, weekly and monthly active users. They count the distinct people (each person once) who did something in a day, a week and a month. The catch is the word "something". If opening the app counts, a messaging app might have 10 million daily users. If sending a message counts, it might have 6 million. So always say what counts as active, which window and which time zone.

Stickiness is average daily users divided by monthly users. With 120,000 daily and 400,000 monthly users it is 30 percent. That means the average monthly user showed up about 9 days out of 30. Activation is different: it is the share of new signups who reach a first real benefit. If 380 of 1,000 signups share a first project within 7 days, activation is 38 percent.

Where it breaks: if 100,000 casual users stop coming, monthly users fall and the stickiness ratio rises, even though you lost a fifth of your audience.

## Retention, churn and resurrection

Imagine a bucket with a hole. Acquisition (getting new users) pours users in, and churn (users leaving) is the leak. Retention asks what share of one signup group is still around later. Be careful with the definition. Take a small example. "Active on exactly day 7" gave 40 percent. "Active on day 7 or later" gave 70 percent. "Active in days 7 to 13" gave 60 percent.

A retention curve, a line showing the share still active as time passes, tells a story. A curve that flattens means a core group found lasting value. A curve that keeps sliding to zero means a leaky bucket. Churn compounds, which means each month's loss comes out of a base that is already smaller. So 5 percent a month sounds like 60 percent a year. But the base keeps shrinking as people leave, so each month's loss takes fewer people, and annual churn is 46 percent.

Growth accounting splits the change in active users into new, retained, resurrected and churned. Resurrected users are people who left and then came back. Start at 100,000, add 12,000 new and 3,000 returning, lose 9,000, and you end at 106,000.

Where it breaks: recent signups have not had time to come back, so counting them as lost drags retention down.

## Revenue and unit economics

Average revenue per user (ARPU) is revenue divided by all users. Divide by paying users only and you get ARPPU (average revenue per paying user). Amounts are in USD, which means United States dollars. With USD 500,000 revenue, 400,000 users and 20,000 payers, ARPU is USD 1.25 and ARPPU is USD 25.

In a marketplace, GMV (gross merchandise value) is the total value sold, and revenue is only the slice the marketplace keeps. The share of each sale it keeps is the take rate. A 20 percent take rate on USD 50 million GMV is USD 10 million. Driver incentives of USD 3 million cut it to USD 7 million.

Lifetime value is the margin a customer brings over their life. Margin is revenue minus the cost of serving the customer. A customer paying USD 30 with USD 12 of variable cost (cost that grows with each customer) leaves USD 18 a month. At 4 percent monthly churn the life is 25 months, so lifetime value is USD 450. If acquiring them costs USD 150, payback takes 8.3 months on the simple formula, and about 10 once churn is counted. Payback is the time to earn that cost back.

> **Interview trap:** Computing lifetime value from revenue instead of margin. That gives USD 750 and overstates value by two thirds.

## North star and metric trees

A north star metric is the one number that best captures the value customers get and that leads revenue. Leads means it moves before revenue does. For a messaging app it might be messages sent between friends each week. Revenue itself is a poor choice because it lags (it moves after the value is delivered) and can be raised in ways that hurt customers.

A metric tree breaks the top number into pieces you can act on. Weekly sales (GMV) for a delivery app is active customers times orders per customer times average order value: 2 million times 1.6 times USD 28 is USD 89.6 million. A 5 percent rise in order frequency, which means orders per customer, lifts the total by 5 percent, about USD 4.48 million a week. Each piece gets an owner, and when the top number falls, you check which branch moved.

Where it breaks: a metric nobody can move is decoration. A cumulative metric, a running total such as total signups, only goes up, so it can never warn you.

## Good metrics and counter metrics

A good metric is measurable, easy to explain, tied to value, sensitive enough to move when you act, and hard to game. To game a metric means to push the number up without creating real value. Average session time looks fine and fails: it rises when the app is slow or confusing.

So pair every target with a counter metric that catches the harm. If the target is notification opens, watch unsubscribes. If it is delivery speed, watch order accuracy. When a measure becomes a target, people optimise the measure. Reward call handle time (how long an agent spends on each call) and calls get shorter while problems stay unsolved.

Ratios hide a second trap. An impression is one time an ad is shown. Suppose one user clicked 1 of 2 ads, and another clicked 10 of 1,000. Total clicks over total impressions is 1.1 percent. But the average of the two user rates is 25.5 percent. Both are valid, they answer different questions, so say which one you mean.

> **Remember:** Name the numerator, the denominator and the unit for every ratio. The numerator is the top number in a division, and the denominator is the bottom number.

## Product sense case framework

A product sense question gives you a feature and asks how you would know it worked. Use seven steps:

1. Clarify the goal and users.
2. Map what each side gains.
3. Choose one success metric.
4. Break it into input metrics.
5. Add counter metrics.
6. Say how you would measure it.
7. Finish with a decision rule.

Input metrics are the smaller numbers that add up to the success metric. A decision rule says in advance what result would make you ship.

Example: a photo app launches a short video feed. Measure total time in the whole app, not time in the new feed. Control users (who do not get the feed) spend 50 minutes a day. Treatment users (who do get it) spend 52, with 9 minutes in the feed. Only 2 minutes are new, and 7 of the 9 were taken from elsewhere.

For shipping decisions, put both effects on one scale. An upsell banner, which pushes users to buy more, lifts revenue per user 12.5 percent and raises churn 12.5 percent. Value is revenue divided by churn, so it breaks even, and the confidence interval on churn decides. A confidence interval is a range of values likely to contain the true effect.

## Diagnosing a metric drop

Treat a drop like a detective case. First check that the data is right, because logging bugs look exactly like real drops. A logging bug is a fault in how events are recorded. Then ask whether the move is expected, and look at its shape and timing. Separate internal causes (releases, experiments, outages) from external ones (holidays, competitors). A release is a new version of the product, and an outage is a time when the service is down. Finally, split the metric by platform, country, new versus existing users and funnel step until you find where it is concentrated.

Case one: daily users fell 10 percent. Server side activity (what the company's servers recorded) was down only 0.3 percent, and Android accounted for 96 percent of the drop. A release had renamed the event the metric counts. It was a measurement fault, not users leaving.

Case two: weekly orders fell 12 percent. Sessions and early funnel steps were flat, which means they did not change. Checkout to paid, the share who finish paying, fell from 80 to 70.4 percent. The fall was entirely on card payments, which broke when a new payment flow shipped.

> **Interview trap:** Naming a cause before verifying the data.

## Estimation and Fermi questions

An estimation question asks for a rough number reasoned from scratch. Such questions are also called Fermi questions. Break the problem into factors you can guess, say each assumption out loud, multiply, and sanity check, which means asking whether the answer is plausible.

Example: daily ride hailing trips (rides ordered through an app) in a city of 2 million. Assume 80 percent are adults, which is 1.6 million. Assume 4 percent ride on a given day, which is 64,000 riders. Assume 2.2 trips each. That gives about 140,000 trips. Check it from the other side: at 12 trips per driver per day you need about 11,700 drivers, so ask whether that sounds plausible for a city that size.

Errors multiply in a product. Four factors that are each typically off by a factor of 2 leave the answer typically off by a factor of about 4, not 16. Off by a factor of 2 means twice too high or half too low. The total error stays small because over and under estimates partly cancel. So improve the shakiest factor first, and give a range rather than one number.

## Funnels and attribution

A funnel counts how many people reach each step toward a goal. Take 100,000 visitors, 8,000 signups, 4,400 activated, 1,100 trials and 330 paid. The step rates are 8, 55, 25 and 30 percent. Overall conversion, the share of visitors who end up paying, is 0.33 percent.

Overall conversion is the product of the steps. So a 10 percent relative lift in any step gives 363 paid users instead of 330. A relative lift of 10 percent means that step's rate is multiplied by 1.1. The arithmetic does not pick the step. Effort and headroom (how much room a step has to improve) do.

Attribution splits credit among the touches before a purchase. A touch is one marketing contact, such as an ad view or an email. Take one USD 120 sale with display, social, email and paid search touches. Display means banner ads, and paid search means ads on search results. First touch gives display everything, and last touch gives paid search everything. Linear splits the credit equally, which gives USD 30 each. The channel ranking flips with the rule, so credit is not cause. To measure cause, hold some users out of the ads.

## Segmentation and cohorts

A cohort is a group that started together, such as everyone who signed up in January. Comparing cohorts at the same age avoids mixing old and new users. Segments slice users by behaviour, value or source.

Segments can fool you. Design B converts better than A on desktop (21 versus 20 percent) and on mobile (6 versus 5 percent). Converting means completing the goal. Yet overall B shows 9.0 percent against A's 18.5, because B's traffic is mostly mobile. This is Simpson's paradox: a trend in each group reverses when the groups are combined. Compare within segments.

The same idea explains a falling average with no real problem. Average order value, the average amount per order, dropped from USD 29.0 to 27.8. Both markets improved. But a lower value market grew from 10 to 32 percent of orders.

Where it breaks: slice into 20 segments and one will look special by chance about 64 percent of the time. Each slice has a 5 percent chance of looking special by luck alone.

## Forecasting for analysts

A forecast is a guess with a track record. Start with simple baselines, such as repeating yesterday or repeating the same weekday last week. Make any fancier method beat them in a backtest, which hides recent weeks and predicts them.

Take a simulated series (a list of numbers over time) with a weekday pattern. Yesterday's value was off by 60.8 on average. A seven day average was off by 51.7. The same weekday last week was off by only 14.5. Weekly rhythm was most of the error.

Growth maths: going from 100,000 to 180,000 users in six months is 10.29 percent a month compounded, and doubling takes about seven months. Compounded means each month's growth builds on the last month's total. Do not extrapolate that for years, which means do not extend the trend that far, because markets saturate. A market saturates when it runs out of new people to reach. Always give a range and say what you assumed.

## Dashboard and KPI design

A good dashboard answers one question for one audience. Put a few key numbers at the top, each with a target or last period for context, then trends, then breakdowns. Mix leading indicators, like activation rate, with lagging ones, like churn. Leading indicators move before the outcome, and lagging ones move after. The mix lets you see trouble before the revenue does.

A KPI (key performance indicator) is a metric with a target, an owner and a review rhythm. Write its definition once. Chart choices matter: lines for time, sorted bars for categories, and bars that start at zero. Avoid dual axes (two scales on one chart), pie charts with many slices and rainbow colours.

An alert is an automatic warning. It should compare each day with normal variation, adjusted for the weekday, and require the problem to last. Too many alerts and people ignore them all.

> **Say it like this:** "Who is this dashboard for, and what decision does it support?"

## Data quality and trust

Numbers lose trust the first time they are wrong. Automated checks catch most problems. Is the table fresh (recently updated)? Is the row count normal? Are keys unique, meaning no value repeats? Do totals match finance?

A classic bug comes from a join, which combines two tables by matching rows. Join orders to order items and each order repeats once per item. Three orders worth 100, 200 and 300 sum to 600. After joining to two items each, the sum is 1,200. Know the grain of every table, meaning what one row stands for, and compare totals before and after each join.

When two dashboards disagree, such as 1.02 million versus 0.96 million daily users, do not pick a winner. Write down both definitions and walk the gap. Bots (automated programs that act like users) explain 25,000. A day boundary, where one dashboard starts a day at a different time, explains 20,000. Background pings (signals an app sends without the user doing anything) explain 15,000. Together they explain the whole 60,000. Then define the metric once.

## Communicating to executives

Executives decide, so give them the decision first. Lead with the headline and its size. Then give two or three reasons, then what it means in business terms, then your recommendation, then your confidence. Method goes in an appendix, a section at the end for details.

For example: "Checkout success fell 10 points after Tuesday's payment change, costing about 100,000 orders a week. I recommend rolling it back today. I am confident because the timing and error codes agree." Rolling back means returning to the old version. Ten points means ten percentage points.

Express uncertainty as a range and a plain sentence, such as "most likely a 0.6 point lift, plausibly between 0.2 and 1.1". Do not hide it. A stakeholder is a person who cares about the result. When one pushes back, restate their concern. Find whether they dispute the data, the method or the conclusion. Then offer a quick check that could settle it.

## Excel for analysts

A pivot table summarises a table by dragging fields into rows, columns and values. A field is a column of the source table. Watch the averages. Take region averages of 20 and 50 from 2 and 8 orders. The grand total average is 44, not 35, because it weights by order count.

A lookup finds a value in a table by using a key. VLOOKUP needs the key in the first column and breaks when columns move, so pass FALSE for exact match. INDEX with MATCH is more flexible. INDEX returns a cell by its position, and MATCH finds the position of a value. XLOOKUP is the modern choice: exact by default, looks left (it can return values from columns to the left of the key), and has a not found value.

SUMIFS adds values that meet several conditions. Dollar signs lock references (cell addresses such as B2) when you copy a formula across a grid. Number errors have meanings: #N/A is a failed lookup and #DIV/0! is division by zero. Wrapping everything in IFERROR, which swaps any error for a value you choose, hides real problems.

## Tableau and Power BI

Tableau and Power BI are business intelligence tools for building charts and reports. In Tableau, dimensions slice and measures add up. A dimension is a column you group by, such as region. A measure is a number you add up, such as sales.

A level of detail expression computes at a level you choose. FIXED uses only the dimensions you name, INCLUDE adds dimensions to the view, and EXCLUDE removes them. With two regions, a plain average of order rows gave 60 and 80. Averaging customer totals gave 90 and 120.

In Power BI, a calculated column is computed per row at refresh and stored, so you can slice by it. Refresh means reloading the data. A measure is computed on demand from whatever filters apply. Ratios must be measures. Two sales rows with margins of 40 and 10 percent average to 25 percent, but the true margin from totals is 13 percent. A margin is the share of sales kept as profit.

Model data as a star: one fact table in the middle, dimension tables around it, one to many relationships. A fact table records events, and a dimension table describes things such as products. One to many means one dimension row matches many fact rows.

> **Interview trap:** Writing a ratio as a calculated column and averaging it.
