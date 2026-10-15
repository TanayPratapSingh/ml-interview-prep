# Product and business analytics cheatsheet

One page for the ten minutes before the interview: the answer skeleton, the formulas, the numbers worth remembering and the lines to say. Worked numbers were checked by running code. Example companies and figures are illustrative.

## The answer skeleton

1. Restate and clarify: metric definition, size of change, timeline, decision at stake.
2. Say a structure in three or four buckets.
3. Work the buckets in priority order, with rough numbers.
4. Close with the answer first, then evidence, recommendation, confidence, what would change your mind.
5. Check in at forks. If the interviewer will not clarify, state an assumption.

## Metric drop order

Clarify, verify the data (independent source), expected (same weekday, last year), shape and timing, internal versus external, segment (platform, version, country, new versus existing, channel, funnel step), decompose (metric tree, growth accounting), rank hypotheses, quantify, recommend, monitor.

| Clue | Likely cause |
| --- | --- |
| Sharp step at a deploy time, one source only | Tracking or definition change |
| Concentrated in one version or platform | Release bug |
| Only one funnel step moved | Fault in that step |
| Gradual, overlapping segments | Competitor or novelty fading |
| Repeats each week or year | Seasonality |

## Formulas

| Metric | Formula |
| --- | --- |
| Stickiness | average DAU / MAU = mean active days / days in month (30% is 9 days of 30) |
| Activation | signups reaching the value event in T / signups |
| Retention | exact day N, unbounded (day N or later), bracket (a range) |
| Annual churn | 1 - (1 - monthly churn)^12 (5% monthly is 46%, not 60%) |
| Lifetime | 1 / churn (fails with mixed segments) |
| MAU change | new + resurrected - churned |
| Quick ratio | (new + resurrected) / churned |
| ARPU | paying rate x ARPPU |
| Take rate | revenue / GMV |
| CAC | blended: spend / all new, paid: spend / paid new |
| LTV | margin per month / churn (discounted: m(1+d)/(c+d)) |
| Payback | CAC / monthly margin (adjust for churn) |
| NDR, GDR | (start + expansion - contraction - churn) / start, without expansion |
| CMGR | (last / first)^(1 / intervals) - 1 |
| Share of drop | segment change / total change |
| Impact | users x effect x value |

## Numbers worth remembering

- 120,000 DAU, 400,000 MAU: 30%. DAU 100,000 and MAU 500,000 is 20%, and 25% after 100,000 casual users leave.
- Margin USD 18 (revenue 30, cost 12), churn 4%: LTV USD 450, CAC 150 gives 3.0. Payback 8.3 months, 9.9 with churn. Discounted at 1% a month: USD 363.60.
- Mixed churn: 60% at 2% and 40% at 10% gives 19.2 months implied, 34 true.
- Funnel 100,000, 8,000, 4,400, 1,100, 330: 0.33%. Any step up 10% gives 363.
- Click ratio of sums 1.1% versus mean of ratios 25.5%.
- Twenty segments at 5%: 64% chance of a false positive.
- Four Fermi factors each within 2x: answer within about 4x.
- Same weekday baseline beat naive 14.5 to 60.8 on a simulated weekly series.
- Weekly mean 100,000, sd 2,000: three sigma limit 94,000.

## Good metric test

Measurable, understandable, actionable, sensitive, low noise, aligned with value, hard to game, stable definition. Ask what you would do if it dropped, and whether it can drop at all. For ratios, say the numerator, the denominator and the unit: a ratio of sums is impression weighted, a mean of ratios is user weighted, and experiments need the delta method because users are the independent units.

## North star and counter metrics

North star: customer value, leads revenue, measurable, actionable, hard to game. Not revenue, not registered users. Use input metrics teams can move, validated by experiments.

| Metric | Counter metric |
| --- | --- |
| Notification opens | Unsubscribes, mutes |
| Time spent | Satisfaction, return rate |
| Delivery speed | Accuracy, courier incidents |
| Tickets closed | Reopens, CSAT |
| Conversion | Refunds, day 30 retention |
| Revenue per user | Churn, satisfaction |

## Product sense in seven steps

Clarify goal and users, value map, one success metric (incremental total impact), inputs, counter metrics, measurement plan (holdout, long term holdout), decision rule. Watch cannibalisation (9 minutes in the new feed, only 2 net), novelty and interference. For ship decisions, put effects on one scale: value is revenue over churn, so a 12.5% revenue gain and 12.5% more churn break even.

## Funnels, attribution, segments

- Funnel: state unit, window, order. Overall equals the product of steps. Choose by headroom and effort.
- Attribution: first, last, linear, time decay, position based, data driven. Credit is not cause. Measure incrementality with holdouts (attributed 22,000, incremental 2,000).
- Cohorts: compare at the same age. Diagonal dips are calendar events.
- Simpson: compare within segments or at a common mix. Rate effect plus mix effect explains an average that falls while every segment rises.

## Estimation

Clarify, decompose, assume with reasons, compute, sanity check from another side, name the weakest assumption, give a range. Example: 2,000,000 people x 0.8 adults x 4% riding x 2.2 trips is about 140,000 trips a day.

## Forecasting

Baselines first (naive, seasonal naive), then a rolling backtest. Use WAPE when actuals can be zero. Quote ranges and scenarios. Active users from cohorts equal new users per month times the area under the retention curve, and a plateau means no steady state.

## Dashboards and data quality

One question per dashboard, summary to detail, context on every number, definitions and freshness visible. Lines for time, sorted bars for categories, bars from zero, no dual axes. KPI: definition, target, owner, cadence, counter metric. Alerts need weekday aware thresholds and persistence.

Data checks: freshness, volume, nulls, unique keys, integrity, reconciliation. Join fan out turns 600 into 1,200. Two dashboards disagree: compare definitions and walk the gap, then define once.

## Tools in one line each

- Pivot: grand total average weights by rows, calculated fields use sums.
- VLOOKUP: leftmost key, use FALSE. XLOOKUP: exact default, looks left, if not found.
- SUMIFS: sum range first. Lock references with dollar signs.
- Tableau LOD: FIXED named dimensions only, ignores view filters except context. INCLUDE adds, EXCLUDE removes. Table calculations run last.
- Power BI: column at refresh, stored. Measure at query time, filter context. Ratios are measures. Star schema, one to many, single direction.
- DAX year over year: CALCULATE with SAMEPERIODLASTYEAR, DIVIDE, marked date table.

## Say it like this

- "Before I explain it, let me check the data is right."
- "Stickiness of 30 percent means the average monthly user is active 9 days in 30."
- "I would use contribution margin for lifetime value, not revenue."
- "Attribution gives credit, not cause. I would test incrementality."
- "My recommendation is X, I am fairly confident because Y, and I would change my mind if Z."
