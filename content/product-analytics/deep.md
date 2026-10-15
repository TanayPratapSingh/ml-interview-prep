# Product and business analytics, in depth

This is the deep version of the product analytics module: definitions, formulas, worked numbers, failure modes and the follow up chains interviewers use. It covers how product analytics, product data science and business intelligence rounds work, the metric toolkit (engagement, retention, revenue, north star, counter metrics), the product sense and metric drop cases, estimation, funnels and attribution, segmentation, forecasting, dashboards, data quality, executive communication, and Excel, Tableau and Power BI. SQL and experiment design have their own modules. All worked numbers were produced by running code. The companies and figures in the examples are illustrative, not real.

## How the interview rounds work

The product data science interview book (chapter 6) lists four question types: measure something for the first time, find the root cause of a recent metric change, estimate a counterfactual impact, and weigh a trade off. Around them sit SQL, experiments and statistics, estimation, BI tool questions and behavioural rounds.

| Round | Example prompt | Tests |
| --- | --- | --- |
| SQL | Retention by signup cohort | Joins, windows, grain |
| Experiments | Is a 0.3 point lift real? | Power, variance, pitfalls |
| Product sense | How would you measure Stories? | Metric design, user empathy |
| Root cause | DAU fell 10 percent | Structured diagnosis |
| Estimation | Rides per day in a city | Assumptions, arithmetic |
| BI tools | LOD expression, DAX measure | Tool fluency |
| Behavioural | A time data changed a decision | Influence, communication |

Roles differ in weight, not in kind. A product analyst leans on metrics, funnels and product sense. A BI analyst leans on SQL, Tableau or Power BI, Excel, data modelling and stakeholder reporting. A product data scientist adds experiment depth, causal inference and some modelling. Titles vary by company, so ask for the round list.

**The five moves for any open question.** Restate and clarify (metric definition, size of change, timeline, decision at stake). State a structure out loud in three or four buckets. Work the buckets in priority order with rough numbers. Wrap up answer first, then evidence, recommendation, confidence and what would change your mind. Check in at each fork. If the interviewer will not answer a clarifying question, state an assumption and continue.

**What gets scored.** Structure, business and product judgement, quantitative reasoning, communication, and collaboration under ambiguity. This rubric is a synthesis of how such rounds are commonly graded, not an official one. Typical failures: naming a cause before verifying the data, an unordered list of a dozen hypotheses, metrics with no definition, no counter metric, ignoring magnitude, and ending with no recommendation. The book advises engaging the interviewer even when they are unhelpful.

## Engagement metrics

**Active users.** $\text{DAU}_t$ is the number of distinct users with at least one qualifying event on day $t$. WAU and MAU use windows of 7 and 28 (or 30) days. "Active" is a definition: which event (a push delivery should never count), which identity (user, device, cookie), which time zone, which exclusions (bots, staff, test accounts). A rolling 28 day window holds exactly four of each weekday. Version the definition, since changing it breaks the trend.

**Stickiness.** Average DAU divided by MAU equals the average fraction of days a monthly user is active: with $d_i$ active days for user $i$ among $N$ monthly users in a $D$ day month, $\text{avg DAU} = \sum d_i / D$ and $\text{MAU} = N$, so the ratio is $\bar d / D$ (confirmed on 2,000 simulated users, both sides 0.3194). With DAU 120,000, WAU 250,000 and MAU 400,000: DAU/MAU is 30% (9 days of 30), DAU/WAU is 48% (3.4 days of 7), WAU/MAU is 62.5%. L7 and L28 are per user counts of active days and give a distribution.

**When it misleads.** The ratio assumes daily use is the goal, and it moves with the denominator. DAU 100,000 and MAU 500,000 is 20%. If 100,000 casual users vanish, MAU is 400,000 and the ratio rises to 25% while reach fell 20%. Track reach and frequency separately and plot the L28 histogram.

**Activation.** The share of a signup cohort reaching a value moment within a window: 380 of 1,000 within 7 days is 38%. Choose the event by comparing day 30 retention for users who did it early against those who did not, prefer events that are early, reachable and nudgeable, and set the window from the time to first action distribution. Validate causally. Example: 20% of users add 3 or more friends and retain at 52% (others 9.5%, a 42.5 point gap). A randomised nudge lifts retention 2 points and the share reaching 3 friends 11 points, so the effect per induced user is 2/11, about 18.2 points (a Wald or instrumental variable estimate, valid if the nudge acts only through the friend count, and local to compliers).

```python
obs_share, obs_hit, base = 0.20, 0.52, 0.18
others = (base - obs_share * obs_hit) / (1 - obs_share)
print("D30 of users who did not hit 3 friends:", round(others, 3))
print("observational gap:", round(obs_hit - others, 3))
itt = 0.20 - 0.18          # D30 lift from randomised nudge
first_stage = 0.31 - 0.20  # extra users reaching 3 friends because of the nudge
print("effect per induced user:", round(itt / first_stage, 3))
```

**Frequency, intensity, depth.** Days active per month, actions per active day, features used. Report median and 90th percentile because usage is skewed. Keep composite scores split into components. In the forum example in the book, a 3 standard deviation drop in a weighted engagement score was traced to likes and replies, while posts stayed flat.

## Retention, churn and resurrection

**Retention types.** For a signup cohort, N day retention counts users active on exactly day N, unbounded retention counts day N or later, bracket retention counts a range.

```python
import pandas as pd

activity = {1: [0, 1, 2, 7, 8, 14], 2: [0, 1, 3], 3: [0, 7], 4: [0], 5: [0, 2, 6, 9, 12],
            6: [0, 1, 7, 8, 9], 7: [0, 8, 10], 8: [0, 1], 9: [0, 3, 7], 10: [0, 14]}
events = pd.DataFrame([(u, d) for u, days in activity.items() for d in days],
                      columns=["user", "day"])   # day = days since signup
n = events["user"].nunique()
print("D1 exact     ", events.query("day == 1")["user"].nunique() / n)
print("D7 exact     ", events.query("day == 7")["user"].nunique() / n)
print("D7 unbounded ", events.query("day >= 7")["user"].nunique() / n)
print("days 7 to 13 ", events.query("7 <= day <= 13")["user"].nunique() / n)
```

Output: 0.4, 0.4, 0.7, 0.6. Same data, three "day 7" numbers, so name the type. Match it to natural frequency.

**Curve shapes.** A plateau (100, 40, 30, 26, 24, 23, 23) means a durable core. Decay to zero (100, 35, 20, 12, 7, 4, 2) is a leaky bucket. A smile means users return. The early drop is an activation problem and the plateau height is a core product problem. Overlay cohorts and segments, since a blend averages different curves. A good retention rate depends on cadence, shape and trend, and I quote no benchmark because I did not verify one.

**Churn.** Unit churn is lost customers over customers at the prior period end. Gross dollar churn is revenue lost to cancellations and downgrades over starting revenue. Net dollar churn subtracts expansion and can be negative (definitions follow the a16z article). With constant monthly churn $c$, annual churn is $1 - (1-c)^{12}$: 5% monthly gives 45.96%, not 60%. Expected lifetime is $1/c$ (geometric distribution), so 20 months. It fails when customers differ: 60% churning at 2% and 40% at 10% gives blended 5.2% and an implied 19.2 months, but the true mean is $0.6 \times 50 + 0.4 \times 10 = 34$ months.

**Growth accounting.** New (first ever active), retained (active last and this period), resurrected (active now, not last period, active before), churned (active last, not now). $\text{MAU}_t - \text{MAU}_{t-1} = \text{new} + \text{resurrected} - \text{churned}$. Start 100,000, new 12,000, resurrected 3,000, churned 9,000: end 106,000, growth 6%, quick ratio 15,000/9,000 = 1.67.

**Non contractual churn.** Without a cancel event, define churn as inactivity beyond a threshold taken from a high percentile of inter visit gaps (for example 19 days at the 95th percentile, so 28 days), check sensitivity at half and double, and treat returners as resurrected. Infrequent buyers need purchase based definitions or probabilistic models such as BG/NBD.

**Right censoring.** With 100 signups a day for 60 days and true D30 retention of 25%, dividing retained users by all 6,000 signups gives 12.9% because only 3,100 users have reached day 30. Use mature cohorts. For permanent churn, Kaplan Meier: $\hat S(t) = \prod (1 - d_i/n_i)$.

**Extrapolating from six months.** Churn falls as heterogeneous customers sort themselves. The shifted beta geometric model has $S(t) = B(\alpha, \beta + t)/B(\alpha, \beta)$. Fitted to survival 1.00, 0.70, 0.56, 0.48, 0.43, 0.39, it gives 25.7% at month 12 against 19.7% from constant churn at the last month's 9.3%, and 8.0 expected paid months in 24. Backtest on older cohorts.

## Revenue and unit economics

**ARPU, ARPPU.** ARPU = revenue / users, ARPPU = revenue / payers, ARPU = paying rate x ARPPU. Revenue USD 500,000, MAU 400,000, payers 20,000: ARPU USD 1.25, ARPPU USD 25, paying rate 5%. Growth rates multiply: +8% rate and +2% ARPPU gives +10.2% ARPU.

**GMV, revenue, take rate.** GMV is total sales through the marketplace, revenue is what it keeps, take rate is revenue / GMV. GMV USD 50 million at 20% gives USD 10 million gross revenue, USD 3 million of incentives leaves USD 7 million net, effective take rate 14%. Principal versus agent accounting changes reported revenue for the same economics.

**Margins.** Gross margin subtracts direct delivery cost. Contribution margin subtracts all variable cost. A software customer at USD 100 a month with USD 20 hosting and support has an 80% gross margin, and after USD 20 of other variable cost, 60% contribution margin. Build LTV and payback on contribution margin.

**CAC.** Fully loaded acquisition cost per new customer. Blended divides by all new customers, paid by paid channel customers only. USD 200,000 and 2,000 new customers gives blended USD 100, and with 800 from paid, paid CAC is USD 250. Marginal CAC exceeds average CAC as spend scales.

**LTV and payback.** With margin $m$ per month and churn $c$, LTV $= m/c$. USD 30 revenue, USD 12 variable cost, 4% churn: $m = 18$, lifetime 25 months, LTV USD 450. CAC USD 150 gives LTV to CAC of 3 (a rule of thumb, not a law). Revenue based LTV is USD 750, 67% too high. Simple payback is $150/18 = 8.33$ months. Churn adjusted payback solves $m(1-(1-c)^n)/c = \text{CAC}$, giving $n = 9.93$ months. Discounting at monthly rate $d$: $\text{LTV} = m(1+d)/(c+d)$, so USD 363.60 at 1% monthly.

```python
def ltv(monthly_revenue, monthly_variable_cost, monthly_churn, monthly_discount=0.0):
    margin = monthly_revenue - monthly_variable_cost
    return margin * (1 + monthly_discount) / (monthly_churn + monthly_discount)

print(round(ltv(30, 12, 0.04), 1))          # 450.0
print(round(ltv(30, 12, 0.04, 0.01), 1))    # 363.6
```

**Retention of revenue.** NDR = (start + expansion - contraction - churn) / start over existing customers only, GDR leaves expansion out. Start USD 1,000,000, expansion 80,000, contraction 20,000, churn 50,000: NDR 101%, GDR 93%.

**ARR, bookings, billings, revenue.** A 3 year USD 360,000 contract billed annually in advance: bookings 360,000, ARR 120,000, first quarter revenue 30,000, billings 120,000 (revenue plus 90,000 added deferred revenue). Do not annualise one month of bookings.

**Order level economics.** AOV USD 30, take rate 20%: revenue 6.00, payment cost 2.9% x 30 + 0.30 = 1.17, support 0.40, promotions 1.00, contribution 3.43. CAC USD 25 pays back in 7.3 orders. Courier cost is the missing line in a real delivery P and L.

**Traps.** Revenue instead of margin, blended CAC only, one churn number, simple payback, extrapolating young cohorts, average instead of marginal CAC. Use realised cohort curves.

## North star and metric trees

**North star metric.** One metric that expresses customer value and predicts business success. Amplitude's playbook introduces it as the link between three languages teams speak: customer, product and business. Criteria: customer value, leads revenue, measurable, actionable, understandable, hard to game. Typical shapes: messages sent between connected users (messaging), successful transactions (marketplace), hours watched (streaming), weekly active teams doing the core job (collaboration software). Not a north star: total registered users, revenue, page views.

**Choosing for music streaming.** Compare monthly active listeners (reach, not value), total hours (volume, inflatable by autoplay), weekly listening hours per active listener (value and frequency), paid subscribers (lagging), revenue (lagging). Choose weekly listening hours per active listener with counter metrics skip rate, satisfaction, discovery, subscription churn, and inputs such as new listener activation and recommendation quality.

**Metric tree.** Weekly GMV = active customers x orders per active customer x AOV = 2,000,000 x 1.6 x USD 28 = USD 89.6 million. A 5% lift in frequency or AOV adds USD 4.48 million a week. Active customers = retained + new + resurrected. Use the tree to size ideas and to find the branch that moved.

```python
weekly_active_customers = 2_000_000
orders_per_active = 1.6
aov = 28
print(weekly_active_customers * orders_per_active * aov)           # 89,600,000
print(weekly_active_customers * orders_per_active * 1.05 * aov)    # 5 percent more frequency
```

**Input and output metrics.** Outputs are slow results. Inputs are leading levers a team controls and moves in a quarter. Validate inputs with past experiments. **Mistakes.** Revenue as north star, cumulative metrics, several north stars, no owner, gameable metrics, changing it each quarter. Change it only when strategy shifts or it decouples from value, and run old and new side by side.

## Good metrics and counter metrics

**Properties.** Measurable, understandable, actionable, sensitive, low noise, aligned with value, hard to game, stable definition, attributable. Fast metrics tend to align poorly with long term value, so pair a fast proxy with a slow outcome. **Vanity versus actionable.** Test: what would you do if it dropped, and can it drop. Convert totals into rates and cohort views.

**Counter metrics and guardrails.** Notification opens pair with unsubscribe and mute rates, time spent with satisfaction and return rate, delivery speed with accuracy and courier incidents, tickets closed with reopen rate and CSAT, conversion with refunds and day 30 retention, revenue per user with churn. Set thresholds in advance and keep one long horizon metric.

**Goodhart's law.** Once a measure is a target it stops tracking the goal: gaming, selection, tunnel vision, definition drift. Defend with counter metrics, outcome targets, audits.

**HEART.** Happiness, Engagement, Adoption, Retention, Task success, plus the Goals, Signals, Metrics process from the Google research paper. For search: satisfaction survey, queries per user per week, share of new users searching in week 1, repeat search after 4 weeks, share of queries ending in a satisfied click.

**Ratio of sums versus mean of ratios.** One user clicked 1 of 2 impressions, another 10 of 1,000. Ratio of sums is 11/1,002 = 1.1%. Mean of ratios is 25.5%. In experiments the user is the independent unit. The delta method gives $\operatorname{Var}(\hat R) \approx \frac{1}{n\mu_y^2}[\sigma_x^2 + R^2\sigma_y^2 - 2R\sigma_{xy}]$. In a simulation of 4,000 users the naive standard error was 0.00074, the delta method 0.00117, and the empirical spread 0.00119.

```python
import numpy as np

rng = np.random.default_rng(11)
n_users, true_mean_ctr = 4000, 0.05

def one_experiment():
    imps = rng.poisson(rng.gamma(2.0, 10.0, n_users)) + 1            # impressions per user, heavy tailed
    user_ctr = rng.beta(1.0, 19.0, n_users)                          # each user has their own click rate
    clicks = rng.binomial(imps, user_ctr)
    return clicks, imps

clicks, imps = one_experiment()
ratio = clicks.sum() / imps.sum()
naive_se = np.sqrt(ratio * (1 - ratio) / imps.sum())                 # treats every impression as independent
n = n_users
mx, my = clicks.mean(), imps.mean()
vx, vy = clicks.var(ddof=1), imps.var(ddof=1)
cxy = np.cov(clicks, imps, ddof=1)[0, 1]
delta_se = np.sqrt((vx + ratio**2 * vy - 2 * ratio * cxy) / (n * my**2))

def ratio_once():
    c, i = one_experiment()
    return c.sum() / i.sum()

replicates = [ratio_once() for _ in range(2000)]
print("ratio", round(ratio, 4))
print("naive SE   ", round(naive_se, 5))
print("delta SE   ", round(delta_se, 5))
print("empirical SD", round(float(np.std(replicates)), 5))
```

**Surrogates.** A short term proxy can mislead through novelty and mechanism mismatch. Check across past experiments whether effects on the proxy predicted effects on the long term outcome, and keep long term holdouts.

## Product sense case framework

**Measuring a feature.** Seven steps: clarify, value map, one success metric (prefer incremental total impact), input metrics (HEART as a checklist), counter metrics, measurement plan, decision rule. **Improving a product.** Goal, segment, evidence of need from data, options, prioritise by impact, confidence and effort, measure, recommend.

**Case 1: short video feed in a photo app.** Headline: incremental total time per user and day 28 retention. Inputs: adoption, videos per session, completion, creator supply. Counters: time on other surfaces, reports per view, ad revenue per user. Numbers: control 50 minutes, treatment 52, of which 9 in the feed, so net +2 and 7 of 9 (78%) cannibalised. Randomised holdout, four weeks or more, long term holdout, rule: ship if incremental time at least +1 minute, retention not negative, creator supply healthy.

**Case 2: delivery marketplace health.** North star: weekly orders delivered on time and correct. Demand (actives, frequency, retention, AOV), supply (courier utilisation and earnings, restaurant acceptance and prep accuracy), match (fill rate, on time rate, accuracy, cancellations), business (take rate, contribution per order). Delivery times 22, 24, 25, 26, 27, 28, 30, 34, 45, 70 minutes have mean 33.1, median 27.5, p90 47.5: report percentiles.

**Case 3: upsell banner.** Paid conversion 4.0% to 4.5% at USD 10 per payer: revenue per active user USD 0.40 to 0.45 (+12.5%). Monthly churn 8% to 9% (+12.5%). Value per user is revenue over churn, 5.00 in both arms. At 8.5% churn it is 5.29 (+5.9%), at 9.5% it is 4.74 (-5.3%). The interval on churn decides. Test a capped or targeted variant with a long term holdout.

**Second order effects.** Cannibalisation (measure total usage), novelty and primacy (days since exposure plots, long term holdout), interference (violates stable unit treatment values, use cluster, geo or switchback designs). Toy market: 100 rides per hour, 2,000 users, treated request rate 7% against 5%. A user level test shows treated users completing 40% more rides, while a full launch leaves completed rides at 5% per user, a true effect of zero.

```python
users, capacity = 2000, 100          # 2000 users, 100 rides available per hour
base_rate, treated_rate = 0.05, 0.07  # request rate per user, the treatment lifts it for those who get it

# user level A/B test: half treated, half control, sharing one supply pool
req_t, req_c = 1000 * treated_rate, 1000 * base_rate
fill = min(1.0, capacity / (req_t + req_c))
done_t, done_c = req_t * fill, req_c * fill
print("A/B completed per user:", round(done_t / 1000, 4), round(done_c / 1000, 4),
      "relative lift", round((done_t / done_c) - 1, 3))

# launch to everyone
req_all = users * treated_rate
done_all = min(capacity, req_all)
print("full launch completed per user:", done_all / users, "vs baseline", min(capacity, users * base_rate) / users)
```

## Diagnosing a metric drop

**Framework.** Clarify. Verify the data. Check whether it is expected (same weekday, same period last year). Look at shape and timing. Separate internal causes (releases, experiments, ranking, pricing, spend, outages, tracking) from external (holidays, competitors, platform changes, news, weather, regulation). Segment by platform, version, country, new versus existing, channel, device, plan and funnel step. Decompose with the metric tree or growth accounting. Rank hypotheses by likelihood and cost to check. Quantify, recommend, monitor. A candidate guide frames it as understand the change, build a funnel, slice, then propose fixes.

**Data quality first.** Check freshness, renamed or dropped events, tracking coverage by version, time zones, filters and joins, identity stitching, bots, backfills. The best test is an independent source such as server side counts or finance. Flat revenue next to a sharp 10% DAU fall suggests measurement.

**Segmentation.** Share of drop for segment $i$ is $(x_i' - x_i)/\Delta$. Compare with its share of the base. Check for mix shift and avoid over slicing.

**Case 1: DAU fell 10 percent (900,000 against 1,000,000 on the same weekday).** Client DAU counts `app_open`. Server side authenticated DAU is 997,000, down 0.3%. Platforms: iOS -2,000, Android -96,000 (96% of the drop, 24% of Android), Web -2,000. Release 8.2 reached a staged 24% of Android users and renamed `app_open` to `session_start`. Server side Android DAU is 398,000 against 304,000 on the client, a gap of 94,000. Fix the definition, backfill and restate, add a test for events used by headline metrics.

**Case 2: weekly orders fell 12 percent.** Orders = sessions x browse to cart x cart to checkout x checkout to paid. Sessions 5,000,000 flat, 30% and 70% flat, checkout to paid 80% to 70.4%, so orders 840,000 to 739,200 (-12.0%). Cards are 40% of checkouts both weeks and success fell from 80% to 56%, which is $0.24 \times 0.40 = 9.6$ points overall. The cause is `authentication_timeout` after a new card flow shipped at 14:00 Tuesday. Impact 100,800 orders a week, about USD 2.82 million GMV at USD 28 AOV. Roll back for cards, alert on payment success by method.

**Shapes.** A sharp step suggests deploys or tracking, a ramp suggests rollouts or competitors, a spike suggests outages or glitches, repetition suggests seasonality. **Noise.** Weekly mean 100,000 with standard deviation 2,000 gives a three sigma lower limit of 94,000, and 91,500 is 4.25 sigma below. Check persistence and related metrics, and tier alerts since many monitored metrics produce false alarms. **Good news.** Investigate rises as carefully: duplicated events, definition changes, bots, low quality incentive traffic.

**Automated search.** For each dimension compute explanatory power per value and surprise (Jensen Shannon divergence of expected versus actual share). Platform gives 2%, 96%, 2% with surprise 0.0030 bits, country gives 60%, 20%, 20% with surprise 0, since every country fell 10%. Recurse into the top value. It finds where, not why.

## Estimation and Fermi questions

**Method.** Clarify scope and unit, decompose into factors, assume values with reasons, compute with round numbers, sanity check a second way, name the most sensitive assumption, give a range.

**Example.** City of 2,000,000, 80% adults (1.6 million), 4% ride on a given day (64,000), 2.2 trips each: 140,800, about 140,000 trips a day. At 12 trips per driver per day that implies about 11,700 drivers. The 4% factor is the weakest: at 1% to 8% the answer ranges from about 35,000 to 280,000.

**Top down and bottom up.** Top down narrows a total with shares. Bottom up builds from a unit. Do both. TAM is the demand if everyone bought, SAM the reachable part, SOM what you can win. Example: 50,000 companies x 10% adoption x USD 12,000 = USD 60 million, against 100 reps closing 25 deals at USD 12,000 for USD 30 million of new sales a year.

**Impact of a lift.** Users affected x effect x value. 10,000,000 visitors x 0.005 x USD 40 = USD 2 million revenue a month, USD 0.5 million contribution at 25%. Adjust for incrementality, population, duration, cost and give a range from the interval (0.2, 0.5, 0.8 points give USD 0.8, 2.0, 3.2 million).

**Error propagation.** Log errors add: $\text{sd}(\log Y) = \sqrt{\sum \sigma_j^2}$. Four factors each within 2x at one sigma give a product within about 4x (simulated 3.99), and 16x at two sigma. Use geometric means for ranges ($\sqrt{50{,}000 \times 200{,}000} = 100{,}000$). Refine the widest factor first.

## Funnels and attribution

**Funnel.** Ordered steps, step conversion, overall conversion as the product. State unit (user or session), window, order (strict or loose), entry cohort and step definition.

```python
steps = [("visit", 100000), ("signup", 8000), ("activated", 4400), ("trial", 1100), ("paid", 330)]
for (a, na), (b, nb) in zip(steps, steps[1:]):
    print(f"{a} to {b}: {nb / na:.1%}")
print("overall:", f"{steps[-1][1] / steps[0][1]:.2%}")
```

Step rates 8%, 55%, 25%, 30%, overall 0.33%. A 10% relative lift in any one step gives 363 paid, so choose by headroom (best performing segment as target), effort and downstream quality. Improving all four steps by 10% lifts overall by $1.1^4 - 1 = 46\%$. Funnel pitfalls: mixed units, short windows, order enforcement, re entry, bots, step redefinition, channel mix shift, ignoring time to convert.

**Attribution models.** First touch, last touch, linear, time decay, position based (40, 20, 40), data driven. Rule based models allocate credit and say nothing about cause. For a USD 120 sale with touches 10, 6, 3 and 1 days before: first touch gives display 120, last touch gives paid search 120, linear 30 each, position based 48, 12, 12, 48, time decay with a 7 day half life 17.3, 25.8, 34.7, 42.3.

```python
import numpy as np

touches = ["display", "social", "email", "paid search"]
age_days = np.array([10, 6, 3, 1])      # days before the purchase
revenue = 120
w = 2.0 ** (-age_days / 7)              # half life of 7 days
print({t: round(float(c), 1) for t, c in zip(touches, w / w.sum() * revenue)})
```

**Shapley and Markov.** Shapley: with $v(A) = 10$, $v(B) = 20$, $v(AB) = 50$, credit is A 20 and B 30. Markov chain removal effect: in the worked example conversion probability is 0.2837, removing A leaves 0.09 (effect 68.3%), removing B leaves 0.14 (50.6%), credit shares 57.4% and 42.6%. Both are observational and inherit targeting bias.

**Incrementality.** 1,000,000 exposed users at 2.2% against a holdout at 2.0%: 22,000 attributed, 2,000 incremental (9.1%). With USD 60 margin and USD 50,000 spend, last touch ROAS is 26.4 and incremental ROAS 2.4. Use user holdouts, geo experiments, matched markets or marketing mix models.

## Segmentation and cohorts

**Segmentation.** Demographic, behavioural, lifecycle, value, acquisition. RFM scores recency, frequency and monetary value. A useful segmentation is actionable, measurable, stable, distinct and sizeable. Start with rules, not clustering.

**Cohorts.** Group by a shared start event, compare at the same age. In a retention table rows are cohorts, columns are age, diagonals are calendar months. Example: Jan cohort 100, 40, 32, 22; Feb 100, 40, 26; Mar 100, 34; Apr 100. The April cells (22, 26, 34) each sit 6 points below trend, a calendar event, not cohort quality. A growing business can show falling blended retention while every cohort improves.

**Simpson's paradox.** Desktop: A 180 of 900 (20%), B 42 of 200 (21%). Mobile: A 5 of 100 (5%), B 48 of 800 (6%). Overall A is 18.5% and B is 9.0%. At a common 50/50 mix A is 12.5% and B 13.5%. The segment view is right when the segment is a pre treatment variable.

**Rate and mix.** $\Delta A = \sum w_i^0 (x_i^1 - x_i^0) + \sum (w_i^1 - w_i^0) x_i^1$. AOV falls from 29.0 to 27.8: rate effect +1.0, mix effect -2.2.

```python
import pandas as pd

seg = pd.DataFrame({"segment": ["mature", "emerging"],
                    "orders0": [900, 100], "aov0": [30, 20],
                    "orders1": [850, 400], "aov1": [31, 21]})
w0 = seg.orders0 / seg.orders0.sum()
w1 = seg.orders1 / seg.orders1.sum()
total0 = (w0 * seg.aov0).sum()
total1 = (w1 * seg.aov1).sum()
rate = (w0 * (seg.aov1 - seg.aov0)).sum()      # each segment's own change, at old weights
mix = ((w1 - w0) * seg.aov1).sum()             # change in weights, at new values
print(total0, total1, round(rate, 2), round(mix, 2), round(rate + mix, 2))
```

**Slicing pitfalls.** Twenty segments at the 5% level give $1 - 0.95^{20} = 64\%$ chance of a false positive. Do not define segments from outcomes, condition on post treatment variables or trust tiny segments. **A segment that retains better** (52% against 18% overall) is a lead, not a lever: check selection, size the prize, run a nudge test, and plan with the experimental effect.

## Forecasting for analysts

**Workflow.** Decision and horizon, plot, baselines (naive, seasonal naive, moving average), rolling origin backtest, structure only when it helps, report a range. No random splits on time series.

```python
import numpy as np

rng = np.random.default_rng(7)
t = np.arange(140)
weekday_effect = np.tile([0, 50, 60, 60, 50, 20, -140], 20)
y = 1000 + t + weekday_effect + rng.normal(0, 15, 140)   # synthetic daily metric

test_idx = range(112, 140)                                 # last 4 weeks, one step ahead
actual = y[112:]
naive = np.array([y[i - 1] for i in test_idx])
mean7 = np.array([y[i - 7:i].mean() for i in test_idx])
seasonal = np.array([y[i - 7] for i in test_idx])
for name, f in [("naive", naive), ("7 day mean", mean7), ("seasonal naive", seasonal)]:
    print(name, round(np.abs(actual - f).mean(), 1))
```

On this synthetic series the MAE was 60.8 for naive, 51.7 for a seven day mean and 14.5 for seasonal naive. Across 300 seeds the averages were 62.1, 51.3 and 17.8, with the same ordering every time.

**Growth.** $\text{CMGR} = (\text{last}/\text{first})^{1/n} - 1$ with $n$ intervals. 100,000 to 180,000 in 6 months is 10.29% (the arithmetic mean of monthly rates in the example is 10.49%). Doubling time is $\ln 2 / \ln(1+g) = 7.08$ months.

**Cohort forecast.** $A_t = \sum_a N_{t-a} r(a)$. With 10,000 new users a month and retention 1.00, 0.40, 0.30, 0.25, 0.22 then 0.20, actives are 23,700 at month 6 and 35,700 at month 12. A plateau means no steady state.

**Accuracy.** MAE, RMSE, MAPE (undefined at zero actuals), WAPE (actuals 100, 0, 50 with forecasts 90, 10, 55 give 25/150 = 16.7%), bias. Use backtest error quantiles for intervals, scenarios and stated assumptions.

**Hierarchies.** Total 105 with regions 62 and 38 is incoherent. Bottom up gives 100, top down at 60/40 gives 63 and 42, least squares reconciliation gives 103.33, 63.67 and 39.67. Choose by backtest.

## Dashboard and KPI design

**Principles.** One question per dashboard and audience, summary to detail layout, context on every number, few charts, visible definitions and freshness, annotated events, performance, usage tracking.

**Leading and lagging.** Revenue, churn and satisfaction lag. Activation, qualified pipeline, late delivery share and support wait time lead. Check that a leading metric moves ahead of the outcome and predicts it.

**KPI spec.** Name, goal, definition (numerator, denominator, window, exclusions), source and refresh, segments, target, action threshold, owner, cadence, counter metric. Three to six per team.

**Charts.** Lines for time, sorted horizontal bars for categories, histograms or box plots for distributions, scatter for relationships, bars with step rates for funnels, heatmaps for cohorts. Mistakes: truncated bar axes, dual axes, many slice pies, 3D, rainbow palettes, spaghetti lines, cumulative charts for flows, averages only, no target. Put the takeaway in the title.

**Alerts.** Static, statistical (same weekday history), forecast based, rate of change. Require persistence, assign owners and runbooks, tier severity, separate data quality alerts from business alerts, track precision.

**Executive dashboard for a subscription app.** Five KPIs (weekly actives, new paid subscribers, net revenue retention, monthly churn, CAC payback), leading indicators, annotated trends, cohort heatmap and funnel, segment table with drill down, definitions and freshness on every tile.

## Data quality and trust

**Dimensions and checks.** Completeness, validity, uniqueness, consistency, timeliness, accuracy, integrity. Automate freshness, volume anomaly, distribution drift, schema change, row rules (dbt style tests) and reconciliation with finance. Test at ingestion, after transformation and at the metric layer, and block on critical failures.

**Tracking pitfalls.** Duplicate events, late arrivals, renamed or dropped events, time zones, bots, client versus server logging, consent and ad blockers, identity stitching, schema drift, sampling. Log critical events server side, keep a tracking plan, monitor volume per event per version.

**Two dashboards disagree.** Product 1,020,000, finance 960,000. Write both definitions, align one difference at a time: bots and internal traffic -25,000, local versus UTC day -20,000, background pings -15,000, reaching 960,000. Name a standard definition and put it in a shared layer with a reconciliation alert.

**Join fan out.** A one to many join repeats the one side. Orders 100, 200, 300 with two items each sum to 1,200 after the join against a true 600.

```python
import pandas as pd

orders = pd.DataFrame({"order_id": [1, 2, 3], "revenue": [100, 200, 300]})
items = pd.DataFrame({"order_id": [1, 1, 2, 2, 3, 3], "sku": list("abcdef")})
joined = orders.merge(items, on="order_id")
print(orders["revenue"].sum(), joined["revenue"].sum())          # 600 1200
print(joined.drop_duplicates("order_id")["revenue"].sum())       # 600
```

Know each table's grain, aggregate before joining, assert key uniqueness, compare totals before and after.

**Metric layer.** One definition in code, owner, dictionary, versions, tests, lineage, freshness, change management that restates history. Trust is built by consistency and transparent correction.

## Communicating to executives

**Structure.** Headline with size, why (two or three facts), so what in business units, recommendation and ask, confidence and risk, next steps, appendix for method. This follows the pyramid principle of conclusion first.

**Example.** "Checkout success fell from 80% to 70.4% after Tuesday's card payment change, costing about 100,800 orders last week, roughly USD 2.8 million of GMV. I recommend rolling back the card flow today. Confidence is high because the timing, the single payment method and the error code agree."

**Uncertainty.** Give a range and a plain sentence ("most likely 0.6 points, plausibly 0.2 to 1.1"). Separate known, estimated and assumed. Use a fixed vocabulary of high, medium and low confidence and say what would change your mind. Tie uncertainty to the decision.

**Pushback.** Restate, locate the disagreement (data, method, interpretation or recommendation), show evidence for the contested point, offer a fast test, agree what changes minds, admit unknowns, prewire the key people and anticipate objections.

**Rewrite.** "The regression coefficient was positive and significant" becomes "Adopters retain 8 points better at day 30, but motivated users adopt more, so I recommend a two week nudge test before building more. Confidence medium."

## Excel for analysts

**Pivot tables.** Rows, columns, values, filters, show values as, date grouping, slicers, calculated fields, refresh. The grand total average weights by rows: with region averages 20 (2 orders) and 50 (8 orders) it is 44, while the average of averages is 35. Calculated fields work on sums, so profit over sales across rows (100, 40) and (900, 90) is 13% against 25% for the average of row margins. Use an Excel Table as the source.

**Lookups.** VLOOKUP needs the key leftmost, takes a column number and defaults to approximate match. INDEX and MATCH looks anywhere. XLOOKUP is exact by default, has an if not found argument, looks left and needs Microsoft 365 or Excel 2021 or later. For tiers (0, 1,000, 5,000 giving 0%, 5%, 8%), 3,200 returns 5%.

**Conditional aggregation.** `SUMIFS(sum_range, criteria_range, criteria, ...)` with the sum range first, `COUNTIFS`, `AVERAGEIFS`, and mixed references such as `$A2` and `B$1` to copy across a grid. For the six row sample, East and January sum to 150.

**Dynamic arrays and Power Query.** FILTER, UNIQUE, SORT and LET spill results. Power Query records repeatable steps (unpivot, merge, append). The Data Model gives distinct counts.

**Errors and hygiene.** #N/A, #REF!, #VALUE!, #DIV/0!, #NAME?, #SPILL!. Avoid blanket IFERROR, numbers as text, merged cells, volatile functions, hard coded inputs, and always reconcile totals.

**Retention table.** Cohort month `=EOMONTH(B2, -1) + 1`, age `=(YEAR(C2)-YEAR(B2))*12 + MONTH(C2)-MONTH(B2)`, dedupe user and month, count by cohort and age, divide by the age 0 count.

```python
import pandas as pd

activity = pd.DataFrame({
    "user": ["u1", "u1", "u1", "u2", "u2", "u3", "u4", "u4", "u5", "u6", "u6"],
    "signup": ["2026-01-10"] * 3 + ["2026-01-20"] * 2 + ["2026-01-25"]
              + ["2026-02-03"] * 2 + ["2026-02-15"] + ["2026-02-20"] * 2,
    "active": ["2026-01-12", "2026-02-02", "2026-03-05", "2026-01-21", "2026-02-10", "2026-01-26",
               "2026-02-04", "2026-03-01", "2026-02-16", "2026-02-21", "2026-03-09"],
})
for col in ("signup", "active"):
    activity[col] = pd.to_datetime(activity[col])
activity["cohort"] = activity["signup"].dt.to_period("M")
activity["age"] = ((activity["active"].dt.year - activity["signup"].dt.year) * 12
                   + activity["active"].dt.month - activity["signup"].dt.month)
unique = activity.drop_duplicates(["user", "cohort", "age"])          # one row per user per month
table = unique.pivot_table(index="cohort", columns="age", values="user", aggfunc="nunique")
print(table)
print(table.div(table[0], axis=0).round(3))
```

The Excel formulas are written from known syntax. I checked their logic and results in pandas, not in Excel.

## Tableau and Power BI

**Tableau basics.** Dimensions and measures, discrete (blue) and continuous (green), row level versus aggregate calculations, table calculations, LOD expressions, live versus extract, joins versus relationships versus blends, parameters, sets.

**LOD expressions.** `{ FIXED | INCLUDE | EXCLUDE dims : aggregate }`. FIXED uses only the named dimensions and ignores view filters except context, data source and extract filters. INCLUDE adds dimensions to the view, EXCLUDE removes them. The inner expression must be an aggregate, ATTR and table calculations are not allowed in it (per Tableau's help page).

```python
import pandas as pd

orders = pd.DataFrame({"region": ["East", "East", "East", "West", "West", "West"],
                       "customer": ["C1", "C1", "C2", "C3", "C4", "C4"],
                       "sales": [100, 50, 30, 200, 20, 20]})
row_avg = orders.groupby("region")["sales"].mean()                    # AVG([Sales]) per region
per_customer = orders.groupby(["region", "customer"])["sales"].sum()
include_avg = per_customer.groupby("region").mean()                   # AVG({ INCLUDE [Customer] : SUM([Sales]) })
east = orders.loc[orders["region"] == "East", "sales"].sum()
east_share_fixed = east / orders["sales"].sum()                       # SUM([Sales]) / { FIXED : SUM([Sales]) }
print(row_avg.to_dict(), include_avg.to_dict(), round(east_share_fixed, 3))
```

Plain average of order rows gives 60 and 80, the average of customer totals gives 90 and 120, and East's share of all sales is 42.9%.

**Order of operations.** Extract, data source, context filters, then sets and Top N, FIXED, dimension filters, INCLUDE and EXCLUDE, measure filters, table calculations. Only the FIXED placement is from the fetched help page, so confirm the rest against Tableau's order of operations page. Promote a filter to context to make FIXED or Top N respect it.

**LOD versus table calculation.** Table calculations run last on the marks in the view (running totals, ranks, differences). LOD runs at a named level in the query (first purchase date, customer level aggregates, cohorts, percent of a fixed total).

**Power BI measure versus calculated column.** A column is computed per row at refresh, stored and usable on axes and slicers. A measure is computed at query time in filter context and recalculates as users click. Averaging a margin percentage column of 40% and 10% gives 25%, while the measure from totals gives 13%.

```python
import pandas as pd

sales = pd.DataFrame({"amount": [100, 900], "cost": [60, 810]})
sales["margin_pct_row"] = (sales["amount"] - sales["cost"]) / sales["amount"]     # calculated column style
print(sales["margin_pct_row"].mean())                                               # 0.25, average of row percentages
print((sales["amount"].sum() - sales["cost"].sum()) / sales["amount"].sum())        # 0.13, measure style, ratio of sums
```

**DAX.** Filter context, row context, `CALCULATE`, context transition, iterators such as `SUMX` and `AVERAGEX`, `DIVIDE`, variables. Year over year: `Sales LY = CALCULATE([Total Sales], SAMEPERIODLASTYEAR('Date'[Date]))` and `YoY % = DIVIDE([Total Sales] - [Sales LY], [Sales LY])`, which needs a marked date table. 1,200 to 1,380 is 15%. The DAX is written from standard syntax and not run.

**Modelling.** Star schema, one to many relationships with single direction filters, an inactive relationship plus `USERELATIONSHIP` for role playing dates, Import versus DirectQuery versus composite, Power Query for shaping and DAX for calculation, remove unused columns, row level security.

## What interviewers probe

- **Metrics:** you define DAU, so what counts as active? Why did stickiness rise? Which retention definition? Is LTV on revenue or margin, and is churn constant?
- **North star:** why not revenue? What are the inputs and counter metrics? How do you know the input moves the output?
- **Product sense:** how do you handle cannibalisation, novelty and interference? What decision rule would make you ship?
- **Metric drop:** is the data right, what independent source confirms it, where is the change concentrated, how would you rule out a mix shift, what is the revenue impact?
- **Estimation:** what is your weakest assumption, how would you check it, and what range would you quote?
- **Funnels and attribution:** which step first and why, what does last touch miss, how would you measure incrementality?
- **Segmentation:** could Simpson's paradox be at work, how many cuts did you try, is the segment cause or selection?
- **Forecasting:** what is the baseline, how did you backtest, what range do you quote?
- **Dashboards:** who is it for, what decision, what would you remove, how do you know it is used?
- **Data quality:** how do you catch silent failures, how do you reconcile two numbers, what is the grain?
- **Communication:** give me the answer in 30 seconds, what is your confidence, what would change your mind?
- **Tools:** VLOOKUP versus XLOOKUP, FIXED versus INCLUDE, measure versus calculated column, import versus DirectQuery.

## Sources

- [Product Data Science Interview: The Bar to Get Hired, chapter 6 Metrics and problem solving](https://bookdown.org/madinterviewllc/Product-Data-Science-Interview-The-Bar-to-Get-Hired/metrics.html): Four metric question types, a response framework, guardrail metrics and a worked engagement drop example with a 3 standard deviation move.
- [a16z: 16 startup metrics](https://a16z.com/16-startup-metrics/): Definitions of CAC (blended and paid), LTV on contribution margin, churn variants, cohort retention, GMV versus revenue, ARR, billings and CMGR.
- [Mandy Liu on Substack: how would you investigate a 20 percent drop](https://mandyliu.substack.com/p/how-would-you-investigate-a-20-drop): Interview framing for a metric drop: understand the change, build a funnel, slice the data, recommend. Only the free part of the post loaded.
- [Google Research: Measuring the user experience on a large scale (HEART)](https://research.google/pubs/measuring-the-user-experience-on-a-large-scale-user-centered-metrics-for-web-applications/): Publication page for the HEART framework and the process that maps product goals to metrics. Only the abstract loaded here.
- [Tableau Help: Create level of detail expressions](https://help.tableau.com/current/pro/desktop/en-us/calculations_calculatedfields_lod.htm): Official syntax, FIXED, INCLUDE and EXCLUDE behaviour, filter interaction, limitations and example formulas.
- [Microsoft Learn: Measures in Power BI Desktop](https://learn.microsoft.com/en-us/power-bi/transform-model/desktop-measures): Official description of measures, implicit measures, DAX as an Excel like formula language, and measure organisation.
- [Amplitude: North Star Playbook](https://amplitude.com/en-us/books/north-star/about-north-star-framework): Introduces the north star framework as a link between the language of the customer, the product and the business. Fetched in earlier research notes, introduction only.
- [Amplitude: Get specific, defining your North Star](https://amplitude.com/en-us/books/north-star/defining-your-North-Star): Playbook chapter on defining a north star metric. Page loaded in earlier research notes but only the shared playbook introduction was captured.
- [Amplitude: The North Star checklist and statement exercise](https://amplitude.com/en-us/books/north-star/the-north-star-checklist): Playbook chapter on checking a north star metric. Page loaded in earlier research notes but only the shared playbook introduction was captured.
