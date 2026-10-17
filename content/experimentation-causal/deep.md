# Experimentation and causal inference: deep guide

This is the deep version of the experimentation and causal inference module, for the interviewer who asks follow up after follow up. It covers A/B testing end to end (design, metrics, sizing, duration, peeking, trust checks, novelty, interference, variance reduction, ratio metrics, heterogeneity, ranking and bandits), then causal inference without experiments (potential outcomes, difference in differences, synthetic control, instruments, discontinuities, propensity scores, uplift), then the decision questions that combine them. Formulas were derived and every number was checked by running code; simulations are synthetic and labelled as such. Numbers that come from a paper name the paper.

## Experiment design basics

**Potential outcomes.** Each user $i$ has $Y_i(1)$ and $Y_i(0)$ and you observe one. The average treatment effect is $\tau = E[Y(1) - Y(0)]$. Randomisation makes assignment $T$ independent of both potential outcomes, so $E[Y \mid T=1] - E[Y \mid T=0] = \tau$ whatever the confounders, measured or not. In a finite sample the groups differ by chance; the standard error $\sqrt{s_1^2/n_1 + s_0^2/n_0}$ measures how much.

**What the claim rests on.** Assignment is truly random, units do not affect each other (no interference), both arms are measured the same way, and the analysis keeps users in the arm they were assigned. Control and treatment run concurrently so seasonality and other launches hit both arms.

**Hypothesis template.** Changing X for population P will move primary metric M in direction d by at least D, because of mechanism R. Add the decision rule: ship if the lower end of the confidence interval clears the practical threshold and no guardrail regresses. Write it before launch; choosing the metric afterwards multiplies the comparisons you could have made, so the nominal 5 percent no longer describes the false positive rate.

**Design checklist.** (1) Decision and hypothesis. (2) Primary, secondary and guardrail metrics. (3) Unit, population, trigger. (4) Minimum detectable effect, $\alpha = 0.05$, power 0.8, sample size. (5) Duration rounded up to full weeks. (6) Allocation, ramp, layers. (7) Pre launch checks: logging, A/A, sample ratio monitoring. (8) Run without stopping on a significant look. (9) Analyse: sample ratio first, then primary with an interval, guardrails, pre registered segments. (10) Decide and record; keep a holdout for long term follow up.

> **Remember:** The strongest answers say where the recipe bends: interference, tiny traffic, long term outcomes.

## Metrics for experiments

**Roles.** One primary metric (or one pre defined composite) decides the launch and spends the significance level. Secondary metrics explain the mechanism and are exploratory or false discovery corrected. Guardrails come in two families: harm to users or the business (retention, refunds, unsubscribes, revenue per user), and system health (latency, error rate, crash rate, data quality such as sample ratio).

**Good metrics.** Measurable at the randomisation unit, attributable, sensitive, directional, hard to game. Kohavi et al. (2014) argue for overall organisational metrics such as sessions per user over feature metrics, because feature clicks are easy to raise and often cannibalise elsewhere, and report that most Bing experiments fail and the successful ones improve key metrics by 0.1 to 1.0 percent once diluted.

**Dilution.** If a segment holds share $w$ of the metric and the treatment lifts it by $\ell$ inside the segment, the overall lift is about $w\ell$: 10 percent on a 1 percent segment is 0.1 percent; on a segment that is 1 percent of users but 3 percent of revenue it is 0.3 percent. Sample size scales with $1/\delta^2$, so a diluted effect needs far more users, which is the case for triggered analysis.

**Guardrails as non inferiority tests.** With tolerated loss $\Delta$: $H_0: \tau \le -\Delta$ against $H_1: \tau > -\Delta$; reject if the lower confidence bound exceeds $-\Delta$. Sample size at true difference 0, one sided $\alpha = 0.05$, power 0.8: $n = (z_{1-\alpha} + z_{1-\beta})^2\, 2p(1-p)/\Delta^2$. For $p = 0.10$ and $\Delta = 0.005$ this is 44,514 per arm, three times the 14,750 for a 1 point lift on the primary. A non significant "difference" test from a small experiment is not evidence of safety.

## Randomisation and the unit

**Choosing the unit.** Consistency (a user should not flip arms; cookies churn and differ across devices), independence (coarser units when units interact), analysis alignment, and power through the design effect $1 + (m-1)\rho$. Page view and session units suit stateless changes; user id suits most features; accounts for B2B; clusters, markets or time slices for interference.

**Analysing the wrong unit.** With $m$ sessions per user and intra user correlation $\rho$, treating sessions as independent understates the standard error by $\sqrt{1 + (m-1)\rho}$. Simulation (A/A, 400 users, about 11 sessions each, user SD 1, noise SD 1, $\rho = 0.5$): the inflation is $\sqrt{6} = 2.45$, so $P(|Z| > 1.96/2.45) \approx 0.42$; observed false positive rate 42.3 percent at session level against 4.9 percent at user level. Fix: aggregate to the user, use clustered standard errors, or the delta method.

**Hash assignment.** `bucket = int(md5(salt + ":" + user_id), 16) % 1000`, buckets map to variants. Fabijan et al. (2019) list three requirements: equal probability, consistency, independence across experiments. In a 200,000 id simulation the split was 50.05 percent with sample ratio p of 0.63, correlation of assignments under different salts was $-0.0017$ and under the same salt 1.0. With 365 seeds, 23 experiments give a 50.7 percent chance two share a seed. A bucket bug giving control one bucket fewer of 1,000 (50.05 versus 49.95 in shares) reaches $p < 0.001$ only near 11 million users.

**Intent to treat and triggering.** Analyse everyone assigned: $E[Y \mid Z=1] - E[Y \mid Z=0]$. Dropping post assignment behaviour (did not open the feature, left early) breaks randomisation. Under one sided non compliance, ITT equals compliance times the effect on compliers. A triggered analysis keeps users who could have seen a difference, identified symmetrically with counterfactual logging in control. With $\sigma = 20$, $\delta = 1$ and trigger rate 0.10 the full population needs 627,900 per arm and the triggered analysis 6,280 triggered users (62,800 of traffic), $1/f$ times fewer. A sample ratio mismatch in the triggered scorecard only points to a wrong trigger or filter.

**Overlapping experiments.** Independent salts make experiments orthogonal; each estimates its effect averaged over the other experiments' arms. Interacting changes belong in one layer with mutually exclusive buckets, at the cost of traffic. Interactions can also cause SRM, for example a crash only when both treatments are on.

## Sample size and power

**Derivation for two proportions.** With control $p_1$, treatment $p_2 = p_1 + \delta$, $n$ per arm, reject when $|Z| > z_{1-\alpha/2}$, $\mathrm{SE}_0 = \sqrt{2\bar p\bar q/n}$ under the null and $\mathrm{SE}_1 = \sqrt{(p_1q_1 + p_2q_2)/n}$ under the alternative. Power $1-\beta$ requires $\delta = z_{1-\alpha/2}\mathrm{SE}_0 + z_{1-\beta}\mathrm{SE}_1$, so

$$
n = \frac{\left(z_{1-\alpha/2}\sqrt{2\bar p\bar q} + z_{1-\beta}\sqrt{p_1q_1 + p_2q_2}\right)^2}{\delta^2} \approx \frac{(1.96 + 0.8416)^2\,(p_1q_1 + p_2q_2)}{\delta^2}, \quad (1.96 + 0.8416)^2 = 7.849 .
$$

For 10 percent versus 11 percent: 14,748 (simplified), 14,751 (pooled null variance), 14,744 (statsmodels, Cohen's $h$). A simulation of 20,000 experiments at 14,748 per arm had 79.9 percent power and 5.2 percent size.

```python
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def n_per_arm(p1, p2, alpha=0.05, power=0.80):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return (za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2


print(round(n_per_arm(0.10, 0.11)))
h = proportion_effectsize(0.11, 0.10)  # Cohen's h, arcsine scale
print(round(NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.80, ratio=1.0)))
```

**Continuous metrics.** $n = 2(z_{1-\alpha/2} + z_{1-\beta})^2\sigma^2/\delta^2 \approx 16\sigma^2/\delta^2$ (Lehr's rule, also in Evan Miller's write up). With $\sigma = 20$ and $\delta = 0.5$: 25,116 exact, 25,600 by Lehr, 25,117 from `TTestIndPower`.

**MDE.** $\mathrm{MDE} = (z_{1-\alpha/2} + z_{1-\beta})\mathrm{SE} \approx 2.8\,\mathrm{SE}$; at 14,748 per arm on a 10 percent baseline it is 1.00 points. A 1 point absolute lift needs 3,825 per arm at a 2 percent baseline, 14,751 at 10 percent and 39,239 at 50 percent; a 10 percent relative lift needs 80,681, 14,751 and 1,565.

**What moves $n$.** Halving a 10 percent relative MDE to 5 percent: 14,751 to 57,763 (3.9 times). Power 80 to 90 percent: 1.34 times. $\alpha$ 5 to 1 percent: 1.49 times. One sided 5 percent: 0.79 times, which equals the bar of a two sided 10 percent test. Allocation: total users relative to 50/50 is $1/(4p(1-p))$, so 20/80 costs 1.56 times, 10/90 costs 2.78 times, 5/95 costs 5.26 times.

**Observed power.** It is monotone in $|z|$: $p = 0.05$ gives 50.0 percent, $p = 0.2$ gives 24.9 percent, $p = 0.5$ gives 10.4 percent. Report the interval against the smallest worthwhile effect.

**Winner's curse.** Conditioning on $|\hat\tau| > 1.96\,\mathrm{SE}$ truncates the sampling distribution. Simulation: true effect over SE of 1.0 gives power 17.0 percent and significant estimates averaging 2.49 times the truth; 2.0 gives 51.7 percent and 1.39; 2.8 gives 80.1 percent and 1.12; 4.0 gives 97.9 percent and 1.01.

## Duration and weekly cycles

Days needed equal total sample over eligible users entering per day: $29{,}496 / 2{,}000 = 14.75$, so 15 days, rounded up to three weeks. Add the outcome window for retention style metrics. If users return, daily new unique users fall, so use the cumulative unique trajectory.

**Why full weeks.** With baseline 1.0 on weekdays and 1.3 on weekends and relative effects of 2 percent and 10 percent, the weekly effect is $(5 \times 0.02 + 2.6 \times 0.10)/7.6 = 4.74$ percent. Five weekdays show 2.00 percent, ten days from Monday 3.96 percent, fourteen days 4.74 percent. The comparison is internally valid for the days observed but unrepresentative of the steady state.

**Early sample size.** Duration is the maximum of the time for the sample size and the calendar time for weekly cycles and novelty. Legitimate early stops are pre registered sequential rules and pre agreed harm stops on guardrails. Ramps are for safety; analyse only the stable allocation period because changing the split mixes periods with different base rates (Simpson's paradox).

## Peeking and sequential testing

**Numbers.** A/A test, stop at the first $|z| > 1.96$, equally spaced looks: 1 look 5.0 percent, 2 looks 8.3, 5 looks 14.2, 10 looks 19.2, 20 looks 24.8, 100 looks 37.3. Evan Miller's logo example (50 percent baseline, stopping at first significance) reports 26.1 percent. Stopping also inflates the estimate.

**Maths.** $Z_k = S_k/\sqrt{2\sigma^2 k n}$ is a rescaled random walk with $\mathrm{Corr}(Z_s, Z_t) = \sqrt{s/t}$; two looks at half and full sample correlate 0.71, giving 8.3 not 10 percent. By the law of the iterated logarithm, $\limsup S_t/\sqrt{2\,\mathrm{Var}\,t\log\log t} = 1$, so $|Z_t|$ crosses any fixed bound infinitely often; checking after every observation from 30 to 5,000 rejected 50 percent of null experiments.

**Group sequential designs.** Five equally spaced looks at overall $\alpha = 0.05$: Pocock critical value 2.41 at every look (nominal p 0.0158); O'Brien-Fleming 4.56, 3.22, 2.63, 2.28, 2.04. Maximum sample size for 80 percent power: 1.22 times fixed for Pocock, 1.03 times for O'Brien-Fleming. Miller's table of adjusted per look thresholds (2.9, 2.2, 1.8, 1.4, 1.0 percent for 1, 2, 3, 5, 10 peeks) matches a Pocock boundary for 2, 3, 4, 6, 11 looks (simulated 2.96, 2.20, 1.83, 1.42, 1.01 percent). Lan and DeMets spending: the O'Brien-Fleming type $2 - 2\Phi(z_{\alpha/2}/\sqrt t)$ spends cumulatively 0.00001, 0.0019, 0.0114, 0.0284, 0.05 at $t = 0.2, \ldots, 1$; the Pocock type $\alpha\ln(1 + (e-1)t)$ spends 0.0148, 0.0262, 0.0354, 0.0432, 0.05.

**Always valid inference.** For paired differences with variance $V$ and a $N(0, \tau^2)$ mixing prior, $\Lambda_n = \sqrt{V/(V + n\tau^2)}\exp\!\big(n^2\tau^2\bar d_n^2/(2V(V + n\tau^2))\big)$ is a martingale under $H_0$; by Ville's inequality $P(\sup_n \Lambda_n \ge 1/\alpha) \le \alpha$. Simulation (effect 0.1 SD, $\tau^2 = 0.01$): type I error 3.5 percent over 5,000 looks (naive 50.1 percent), power 55.7 percent at the fixed horizon sample of 1,570 per arm and 88.5 percent at twice that.

**Bayesian monitoring.** Beta(1, 1) priors, equal 10 percent arms, 500 users per look: stopping when $P(\text{B better}) > 0.95$ or $< 0.05$ falsely stopped 10.7 percent with one look at 10,000 per arm and 42.8 percent over 20 looks. The posterior is valid if the prior is right; a flat prior is not how experiment effects behave.

> **Say it like this:** "Looking is fine, acting on a significant look is the problem. I would hide significance until the planned end, or use a planned group sequential design."

## Trust checks and sample ratio mismatch

**Test.** Chi square goodness of fit against the planned allocation, $\chi^2 = \sum (O - E)^2/E$ with $k-1$ degrees of freedom; for two arms planned 50/50, $(n_1 - n_2)^2/(n_1 + n_2)$.

```python
from scipy.stats import chisquare

stat, p = chisquare([821588, 815482])  # expected counts default to an equal split
print(round(stat, 2), f"{p:.1e}")

# Planned 90/10 split: pass the expected counts explicitly.
observed = [89600, 10400]
total = sum(observed)
stat, p = chisquare(observed, f_exp=[0.9 * total, 0.1 * total])
print(round(stat, 2), f"{p:.1e}")
```

The first example is the Fabijan et al. (2019) case: 821,588 versus 815,482 gives $\chi^2 = 22.77$, $p = 1.8 \times 10^{-6}$, about 1 in 550,000. Other cases: 50,000 versus 50,800 gives 0.012; 10,000 versus 10,500 gives 0.0005.

**Why it invalidates.** Differential loss that depends on behaviour is selection bias and can flip the sign. In their carousel example the most engaged treatment users were removed by a bot filter, so a significant decrease reversed once corrected. About 6 percent of Microsoft experiments in their study had an SRM, and they cite about 10 percent of LinkedIn triggered analyses. SRMs can have positive causes (a faster treatment logs more users). Do not reweight to the planned ratio.

**Taxonomy.** Assignment (bucket bugs, unstable ids, correlated seeds), execution (late variant start, a config refresh that logged the wrong variant for about 30 percent of Skype sessions), log processing (bot filters reacting to treatment, bad joins), analysis (wrong trigger or filter, as in a Teams example), interference (URLs forcing a variant, telemetry tampering, treatments that crash together).

**Diagnosis (ten rules of thumb).** Compare scorecards (triggered versus full), user segments, time segments (day 1 only suggests caching or a late start), frequency across experiments, A/A tests, severity, performance and engagement metrics (who is being lost), downstream stages and across pipelines.

**Threshold and sensitivity.** In 10,000 healthy 50/50 experiments, $p < 0.05$ flagged 5.1 percent and $p < 0.001$ flagged 0.11 percent, so alerts use a strict level (commonly 0.001 or 0.01; no primary source fixes the number). Detectable split at 80 percent power and $p < 0.001$: 51.46/48.54 at 20,000 users, 50.46/49.54 at 200,000, 50.15/49.85 at 2 million. A 50.05/49.95 bug gives $p = 0.32$ at 1 million users, 0.0016 at 10 million.

**A/A tests.** Healthy: no SRM, balanced pre period metrics, uniform p values with 5 percent below 0.05. Failures: too many small p values (variance understated), too few (conservative), SRM, pre period imbalance. Simulated A/A false positive rates: session level analysis of user randomisation 42.3 percent versus 4.9 percent; naive view level CTR test 11.7 percent versus 5.4 percent for the delta method.

## Novelty, primacy and long term effects

**Detection.** Plot the daily effect against days since each user's first exposure, split new versus returning users, fit a decay curve. Kohavi et al. (2014) call novelty and primacy effects uncommon in practice but show them in recommenders (the People You May Know diversity bump followed by a finite pool) and in a new tab feature whose negative feedback fell from 20 percent of messages in week one to 2 percent by weeks three and four while the metric lift was sustained. They recommend two weeks.

**Worked curve** (synthetic): lift $= 0.02 + 0.06e^{-(d-1)/4}$ gives 8.0 percent on day 1, 3.3 on day 7, 2.2 on day 14, 2.0 on day 28. Averages: 5.2 percent over days 1 to 7, 3.9 over 14, 3.0 over 28, 2.0 for the last week. Report the steady state, fit $a + be^{-d/\tau}$ for the asymptote, forecast with the steady state, and never extrapolate a still falling curve.

**Long term gaps.** Learning and habit, fatigue, ecosystem response (sellers, creators), retention and lifetime value, cookie churn, overlapping tests. Remedies: long runs, cohort analysis, holdouts, validated surrogates (the treatment effect on the surrogate must predict the effect on the outcome across past experiments).

**Holdouts.** MDE at 80 percent power on a 10 percent baseline with 1 million users: 1 percent holdout 0.84 points, 2 percent 0.60, 5 percent 0.39, 10 percent 0.28. Costs: a worse experience for held out users, a second code path, interference.

## Interference and network effects

**SUTVA.** $Y_i(T_1, \ldots, T_N) = Y_i(T_i)$ fails when treatments spill over. The launch estimand is the global effect, all treated versus none. Positive spillovers (networks, messaging) bias a user level test toward zero; competition for shared supply (marketplaces, ad budgets) biases away from zero.

**Network simulation.** 400 communities of 50, $y = \tau T + \gamma\cdot(\text{fraction of others treated}) + \text{noise}$, $\tau = \gamma = 1$. User level randomisation: mean 1.00 (SD 0.042), because spillover cancels. Cluster randomisation: 2.00 (SD 0.100), the global effect.

**Cluster design effect.** $\mathrm{DEFF} = 1 + (m-1)\rho$; with $m = 50$, $\rho = 0.1$ it is 5.9 and the empirical variance ratio was 5.92. Analyse cluster means or use clustered standard errors; leakage across boundaries biases toward zero.

**Marketplace simulation.** 100,000 buyers, demand 0.30 versus 0.36, supply for 33,000. User level A/B: purchase rate 36.11 versus 29.65 percent, +21.8 percent. Global rollout: 30.13 to 33.00 percent, +9.5 percent. Demand rose 20 percent but supply caps purchases. Better designs: market randomisation, switchback, partitioned supply or budgets.

**Switchback.** A system moving toward the treated state with a 3 hour time constant, true effect 1.0: 1 hour slices estimate 0.33, 2 hours 0.44, 6 hours 0.70, 6 hours with 3 hour washout 0.87, 24 hours 0.91, 24 hours with 3 hour washout 0.97, with 6 hour washout 0.99 (SD 0.18 versus 0.16). Slices are the independent units; block by day of week and hour.

**Few units.** With $J$ cities and $J/2$ treated there are $\binom{J}{J/2}$ assignments; 6 cities give 20 so the minimum two sided permutation p is 0.10; 8 give 70 (0.029); 10 give 252 (0.008). Match or pair on pre period trends, use DiD or synthetic control, and use randomisation inference.

## Variance reduction and CUPED

**Link to power.** $n \propto \sigma^2/\delta^2$. Deng, Xu, Kohavi and Walker (2013) report variance reductions of 45, 52 and 49 percent in three Bing experiments with one week of data on each side, equal to the power of about twice the traffic.

**Derivation.** $\mathrm{Var}(Y - \theta X) = \mathrm{Var}(Y) + \theta^2\mathrm{Var}(X) - 2\theta\mathrm{Cov}(Y, X)$; setting the derivative to zero gives $\theta^\ast = \mathrm{Cov}(Y, X)/\mathrm{Var}(X)$, the OLS slope, and the minimum is $\mathrm{Var}(Y)(1 - \rho^2)$. The estimator $\Delta_{cv} = \Delta - \theta(\bar X_t - \bar X_c)$ is unbiased because $E[\bar X_t - \bar X_c] = 0$ for a pre treatment $X$. Several covariates reduce variance by $1 - R^2$; the optimal nonlinear adjustment is $E[Y \mid X]$.

```python
import numpy as np

rng = np.random.default_rng(1)
n = 20000
x = rng.normal(20, 10, n)                      # pre experiment metric
t = rng.permutation(np.repeat([0, 1], n // 2))  # random assignment
y = 20 + 0.7 * (x - 20) + rng.normal(0, 7.14, n) + 0.5 * t  # in experiment metric, true effect 0.5

theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
y_adj = y - theta * (x - x.mean())


def effect(v):
    d = v[t == 1].mean() - v[t == 0].mean()
    se = np.sqrt(v[t == 1].var(ddof=1) / (t == 1).sum() + v[t == 0].var(ddof=1) / (t == 0).sum())
    return round(float(d), 3), round(float(se), 3)


print("raw", effect(y), "cuped", effect(y_adj), "theta", round(theta, 3))
print("variance ratio", round((effect(y_adj)[1] / effect(y)[1]) ** 2, 3), "1 - rho^2", round(1 - np.corrcoef(x, y)[0, 1] ** 2, 3))
```

Output: raw 0.441 (SE 0.141), CUPED 0.519 (SE 0.101), variance ratio 0.513 versus $1 - \rho^2 = 0.515$. Monte Carlo variance ratios: $\rho$ 0.3 gives 0.918, 0.5 gives 0.743, 0.7 gives 0.492, 0.9 gives 0.186, with means at the true 0.5.

**Regression equivalence.** OLS of $Y$ on $T$ and the centred covariate gives the same effect (0.5201 versus 0.5201, SE 0.1014). A treatment by covariate interaction is a known refinement. Finite sample bias from estimating $\theta$ is of order $1/n$.

**Covariate safety.** Pre treatment only. In 300 experiments (true effect 0.5): raw 0.503 (SD 0.101), pre covariate 0.501 (SD 0.069), in experiment metric driven by the outcome $-0.451$ (SD 0.021). Deng et al. show a Bing case where a metric highly correlated with queries per user flipped the sign. One to two weeks of pre period works well; too short matches users poorly, too long lowers the correlation.

**Missing pre period.** Indicator and constant covariate. With 30 percent new users and $\rho = 0.7$ the variance ratio was 0.669 and the estimate unbiased.

**Stratification.** $\mathrm{Var}(Y) = \sum w_k\sigma_k^2 + \sum w_k(\mu_k - \mu)^2$; stratifying removes the between part. Light, medium, heavy strata (50, 30, 20 percent; means 5, 20, 80; SDs 4, 10, 30): standard error 0.458 raw versus 0.213 stratified, variance ratio 0.216, between share of variance 78.8 percent.

**Skewed metrics.** Kohavi et al. (2014) suggest at least $355\,s^2$ users per variant when $|s| > 1$ (skewness). Their Bing table: revenue per user skewness 17.9 needs about 114,000; sessions per user 3.6 needs 4,700; time to success 2.1 needs 1,550. Capping revenue at USD 10 per user per week lowered skewness from 18 to 5.3 and detected a 30 percent smaller change. In a synthetic revenue metric (5 percent buyers, lognormal spend) skewness fell from 30.9 to 13.3 by capping at the 99th percentile of buyer spend, variance fell 44 percent. Capping changes the estimand; choose the cap before the test and keep an uncapped guardrail.

## Ratio metrics and the delta method

**Estimands.** Ratio of sums $\sum Y/\sum X$ weights heavy users; mean of per user ratios weights users equally (0.0997 versus 0.0985 in a simulation; 0.18 versus 0.1375 on the toy table).

**Delta method.** If $\sqrt n(\bar X - \mu) \to N(0, \sigma^2)$ then $\sqrt n(g(\bar X) - g(\mu)) \to N(0, g'(\mu)^2\sigma^2)$; in several variables $\mathrm{Var}(g(\bar U)) \approx \nabla g^\top\Sigma\nabla g/n$. Deng, Knoblich and Lu present it for metric analytics with closed form, distributed computation as an alternative to the bootstrap.

**Ratio variance.** With gradient $(1/\mu_x, -R/\mu_x)$:

$$
\mathrm{Var}(\hat R) \approx \frac{1}{n\mu_x^2}\left[\sigma_y^2 + R^2\sigma_x^2 - 2R\sigma_{xy}\right] = \frac{\mathrm{Var}(Y - RX)}{n\mu_x^2} .
$$

Toy table views (10, 20, 5, 15), clicks (1, 5, 0, 3): $R = 0.18$, variance 0.001867, SE 0.0432.

```python
import numpy as np


def ratio_and_se(y, x):
    # Ratio of sums, sum(y) / sum(x), with the delta method standard error (one row per user).
    n = len(y)
    r = y.sum() / x.sum()
    resid = y - r * x                       # linearised value for each user
    return round(float(r), 4), round(float(resid.std(ddof=1) / np.sqrt(n) / x.mean()), 4)


views = np.array([10, 20, 5, 15.0])
clicks = np.array([1, 5, 0, 3.0])
print(ratio_and_se(clicks, views))          # (0.18, 0.0432)

# Same answer from the textbook formula
n, mx, my = len(views), views.mean(), clicks.mean()
r = my / mx
cov = np.cov(clicks, views, ddof=1)
var = (cov[0, 0] + r**2 * cov[1, 1] - 2 * r * cov[0, 1]) / (n * mx**2)
print(round(float(np.sqrt(var)), 4))
```

**Simulation (20,000 users).** Delta method SE 0.000891, linearised 0.000891, user bootstrap 0.000875, naive view level 0.000703 (79 percent of correct). A/A false positives (4,000 users per arm, 2,000 runs): naive 11.7 percent, delta 5.4 percent.

**Linearisation.** Build $L_i = Y_i - RX_i$ and run a t-test; use the pooled $R$ under the null; CUPED composes on $L$.

**Relative lift.** $\mathrm{Var}(\bar Y_t/\bar Y_c - 1) \approx \mathrm{Var}(\bar Y_t)/\mu_c^2 + \mu_t^2\mathrm{Var}(\bar Y_c)/\mu_c^4$. With $\mu_c = 10$, $\mu_t = 10.5$, SD 8, $n = 5{,}000$: SE 0.0164 (simulated SD 0.0164), interval 5 percent $\pm 3.2$ points. Fieller or a user bootstrap is safer when the control mean is imprecise.

## Heterogeneous effects and multiple testing

**CATE.** $\tau(x) = E[Y(1) - Y(0) \mid X = x]$. A segment with 10 percent of users has a standard error 3.2 times larger. Subgroup fishing: 20 subgroups under the null give at least one significant in 65 percent of experiments (theory 64.2 percent), 1.05 on average. Use pre specified subgroups, one interaction test, multiplicity control and a confirmation test; "significant here, not there" is not evidence of a difference.

**Machine learning CATE.** S learner (treatment as a feature; can shrink effects to zero), T learner (two models; noisy), X, R and DR learners, causal forests with honest splitting. Evaluate on held out randomised data by calibration per decile and uplift curves; use only pre treatment features.

**Multiple metrics.** At least one false positive among $m$ independent null metrics is $1 - 0.95^m$: 22.6 percent for 5, 40.1 for 10, 53.7 for 15. Bonferroni levels 0.01, 0.005, 0.0033 raise $n$ by 1.49, 1.70 and 1.82 times. Policy: one primary at full $\alpha$, guardrails with margins, secondary with Benjamini-Hochberg, gatekeeping order. Kohavi et al. note that thousands of experiments at 0.05 produce hundreds of false positives per metric and treat borderline results as tentative.

**A/B/n allocation.** Minimising $k^2/(N - n_0) + k/n_0$ gives $n_0/n_t = \sqrt k$: optimal control share 41.4 percent for 2 variants, 33.3 for 4, 26.1 for 8, 20.5 for 15 (equal splits give 33.3, 20.0, 11.1, 6.2). Use Dunnett or Holm; screen many variants, confirm the best few.

**Winner's curse across variants.** 10 variants with zero true effect and SE 0.5 points: best estimate averages 0.77. With true effects $N(0, 0.25^2)$: the winner's estimate averages 0.86 against a true 0.17; empirical Bayes shrinkage $\tau^2/(\tau^2 + \sigma^2) = 0.2$ gives $0.86 \times 0.2 = 0.17$.

## Ranking, bandits and when not to test

**Interleaving.** Team draft: rankers alternate picks (coin flip for who picks first each round), clicks credit the contributing ranker, and a sign test or t-test on per query scores decides. Within query comparison removes between user variance and balances position bias. Toy simulation (relevance Beta(1, 6), position bias $1/(1 + 0.6(k-1))$, ranker noise 0.14 versus 0.10): interleaving power 90 percent at 2,000 queries, 58 percent at 1,000; A/B on clicks 30.5 percent at 2,000 and 89 percent at 8,000. Limits: ranking only, preference not magnitude, needs custom serving, calibrate against A/B.

**Bandits.** 30,000 users, 10 versus 11 percent: A/B 3,151 conversions, power 79.8 percent; Thompson sampling 3,267 conversions (oracle 3,300), 87 percent of traffic to B, naive z test power 52.7 percent, and 14.3 percent false positives when arms are equal. Thompson sampling draws $\theta_a \sim \mathrm{Beta}(1 + s_a, 1 + f_a)$ and plays the maximum. Adaptive allocation makes arm samples dependent, biases means for unlucky arms, and breaks the standard error. Remedies: inverse propensity weighted estimates with valid intervals, an exploration floor, batch designs, a final fixed test.

**Off policy evaluation.** $\hat V_{IPS}(\pi) = \frac1n\sum_i \mathbb 1[\pi(x_i) = a_i]\,r_i/\mu(a_i \mid x_i)$, unbiased with overlap and correct propensities. Simulation: true 0.2198, IPS 0.2205, naive average over matching rows 0.1976. Variance remedies: clipping, self normalisation, doubly robust estimators.

**When not to test.** Too little traffic for any plausible effect, outcome too slow, interference with no feasible design, ethics, mandatory or obvious changes, one off events, cost above value, no clear hypothesis. Alternatives: quasi experiments, observational analysis, qualitative research, staged rollout with a holdout.

## Potential outcomes and confounding

**Selection bias decomposition.** $E[Y \mid T=1] - E[Y \mid T=0] = E[Y(1) - Y(0) \mid T=1] + \big(E[Y(0) \mid T=1] - E[Y(0) \mid T=0]\big)$. Simulation (200,000 users, constant effect 1, motivated users adopt): naive 3.12 = ATT 1.00 + selection bias 2.12; regression on motivation 0.997; randomised 1.015.

**Confounders, mediators, colliders.** Confounder: naive 3.26, adjusting 1.006 (truth 1). Mediator with direct 1.0 and indirect $0.8 \times 1.5 = 1.2$: total 2.20 unadjusted, 0.995 adjusting. Collider with zero effect: $-0.005$ unadjusted, $-0.500$ adjusting for it, $-0.352$ when selecting on $C > 1$. The backdoor criterion: block all paths into $T$ without conditioning on descendants of $T$.

**Estimands.** ATE, ITT, ATT, LATE, local effect at a cutoff, CATE; DiD and synthetic control give ATT, IV gives LATE, RD gives a local effect.

**Assumptions.** Consistency and no interference, ignorability $(Y(0), Y(1)) \perp T \mid X$ (untestable), overlap $0 < e(x) < 1$ (checkable), correct measurement and model. Design specific: parallel trends, convex donor weights, instrument conditions, continuity.

## Difference in differences and synthetic control

**DiD.** $\hat\tau = (\bar Y_{T,post} - \bar Y_{T,pre}) - (\bar Y_{C,post} - \bar Y_{C,pre})$, the ATT under parallel trends. Snow's cholera study (Cunningham, Table 9.2): Lambeth 62 to 14 deaths per 10,000 households ($-48$), Southwark and Vauxhall 565 to 614 ($+49$), DiD $-97$. Card and Krueger (Table 9.12): Pennsylvania $-2.16$, New Jersey $+0.59$, DiD about $+2.75$ (the chapter reports 2.76, standard error about 1.36). Simulation: pre post change of the treated group 5.0 versus a true 2.0 because of trend; DiD 1.99 (SE 0.059, clustered by unit).

**Parallel trends.** Event study coefficients in a parallel panel: pre $-0.16, -0.06, -0.03, -0.09, -0.10$, post $1.98, 1.89, 1.96, 1.95$. With a 0.3 per period divergence: pre $-1.72$ rising to $-0.28$, post 2.28 to 3.23, DiD 3.82 (bias $0.3 \times 6 = 1.8$). Pre trends are suggestive, not a formal test. Failures: non parallel trends, anticipation, composition change, spillovers, concurrent shocks, selection on pre outcomes (Ashenfelter dip), scale dependence.

**Staggered adoption.** TWFE is a weighted average of 2 by 2 comparisons including already treated units as controls (Goodman-Bacon, 2021). Simulation (cohorts treated at 4 and 8, never treated, effect $1 + $ periods since adoption): true 3.83, TWFE 2.50, clean cohort by period DiD 3.86. Use Callaway and Sant'Anna (2021) or Sun and Abraham (2021).

**Synthetic control.** Weights $w_j \ge 0$, $\sum w_j = 1$ minimising the pre period squared error. Simulation (39 donors, 40 pre periods, effect 5.0): pre RMSPE 0.26, 6 donors above 0.02, estimate 4.80, a simple donor average gave 6.75. Placebo inference: treated post to pre RMSPE ratio 18.5, maximum placebo 2.0, $p = 1/40 = 0.025$, the minimum $1/(J + 1)$.

**Choosing.** DiD for many treated and control units with plausible parallel trends; synthetic control for one or few treated units with long pre periods; run both when both are plausible. Bayesian structural time series approaches such as Google's CausalImpact project a treated series forward from controls.

> **Say it like this:** "A before and after comparison is confounded by trend. I would use matched control cities, check pre trends and placebos, and use permutation inference because there is one treated city."

## Instrumental variables and regression discontinuity

**IV.** Conditions: relevance, independence, exclusion, monotonicity. With $Y = \alpha + \tau T + U$ and $\mathrm{Cov}(Z, U) = 0$: $\tau = \mathrm{Cov}(Y, Z)/\mathrm{Cov}(T, Z)$, the Wald estimator, equal to 2SLS with one instrument. Simulation (200,000 users, true 2.0): OLS 5.30, Wald 2.08, 2SLS 2.08, first stage 0.213 (F about 10,026), reduced form 0.444. Manual second stage standard errors are wrong.

**LATE and weak instruments.** The Wald ratio is the effect for compliers. Under one sided non compliance ITT = compliance $\times$ LATE: ITT 1.206, compliance 0.399, ratio 3.02 (truth 3.0). Weak instruments: first stage 0.0045 with F of 5.0 gave 7.17 for a truth of 2.0; with 2,000 users a strong instrument gave a 5th to 95th percentile of 0.8 to 3.0 and a weak one $-18.5$ to 27.8. Rule of thumb: first stage F above 10; use Anderson-Rubin intervals otherwise.

**Sharp RD.** $\tau_{RD} = \lim_{x\downarrow c}E[Y \mid x] - \lim_{x\uparrow c}E[Y \mid x]$ under continuity. Local linear fit $Y = \alpha + \tau D + \beta(x - c) + \gamma D(x - c)$ within bandwidth $h$. Simulation (true jump 2.0): $h = 1.0$ gives 1.993 (SE 0.028), 0.5 gives 1.942 (0.040), 0.25 gives 1.898 (0.057), 0.1 gives 2.021 (0.092); naive difference of means 3.508.

**RD checks.** Density test (counts in $\pm 0.05$: 488 and 508 clean, 308 and 688 manipulated), covariate continuity, placebo cutoffs, donut holes, sensitivity to bandwidth and order, no other treatment at the cutoff.

**Fuzzy RD.** Outcome jump over treatment jump, IV with the cutoff indicator: 1.90 over 0.609 gives 3.12 (truth 3.0), the LATE for compliers at the cutoff.

## Propensity scores and uplift

**Propensity score.** $e(x) = P(T=1 \mid X=x)$; balancing property $T \perp X \mid e(X)$ (Rosenbaum and Rubin, 1983). Simulation (50,000 users, true 2.0): naive 3.729; SMD before 0.760, 0.483, $-0.411$ and after IPW $-0.004$, 0.002, 0.002.

**Matching.** 1:1 nearest neighbour with replacement gave ATT 2.009, SMD after matching near 0.001 to 0.004. Calipers, replacement and neighbour count trade bias and variance; a plain bootstrap can fail for nearest neighbour matching.

**IPW.** $\hat\tau = \frac1n\sum[T Y/\hat e - (1-T)Y/(1-\hat e)]$ or the normalised version: 2.001 here. Maximum weight 42, effective sample $(\sum w)^2/\sum w^2 = 32{,}887$ of 50,000. Poor overlap (propensity from $10^{-7}$ to 1): maximum weight 797, effective sample 2,236; over 300 runs of 5,000 users the mean was 2.217 and the SD 0.589 versus 1.998 and 0.066 with good overlap. Trimming to $[0.05, 0.95]$ kept 54 percent and gave 2.02 for the overlap population.

**Doubly robust estimator.** $\hat\tau_{AIPW} = \frac1n\sum[\hat\mu_1 - \hat\mu_0 + T(Y - \hat\mu_1)/\hat e - (1 - T)(Y - \hat\mu_0)/(1 - \hat e)]$. Both models right 2.003, wrong outcome and right propensity 2.000, right outcome and wrong propensity 2.004. Cross fit with machine learning nuisance models.

**Unmeasured confounding.** The E value $RR + \sqrt{RR(RR - 1)}$ is 2.37 for $RR = 1.5$, 3.41 for 2, 5.45 for 3. Other tools: Rosenbaum bounds, negative controls.

**Uplift.** $\tau(x) = P(Y=1 \mid T=1, x) - P(Y=1 \mid T=0, x)$; persuadables, sure things, lost causes, sleeping dogs. Simulation (200,000 users, randomised offer): observed uplift in the top 20, 40 and 100 percent is 0.9, 0.9 and 1.6 percent for a response model and 4.9, 3.9 and 1.6 percent for a T learner; the response model's top 20 percent is 62 percent sure things, the T learner's is 39 percent persuadables. Evaluate with uplift by decile and Qini curves on held out randomised data; treat when predicted uplift times value exceeds cost.

## Decision questions

**Significant but tiny.** 10 million users per arm, 10.00 versus 10.05 percent: $z = 3.72$, $p = 0.0002$, interval 0.024 to 0.076 points; at 0.02 points $p = 0.14$. At 50 million users and USD 20 per conversion the interval is worth USD 237,000 to USD 763,000 a year. Compare with cost, not with 0.05.

**Primary wins, guardrail loses.** Verify the loss, compare with the margin, convert to a common currency, and choose fix, segment, ramp or reject. Kohavi et al. report 0.6 percent revenue per 100 ms at Bing, so 80 ms is about 0.5 percent there; measure your own rate.

**Conflicting metrics.** Use the hierarchy, validate both moves, check mechanism (cannibalisation, short versus long term), convert to value; if the fallen metric matters more, do not ship.

**Flat result.** Trust checks, read the interval, check dilution, sensitivity, strength of the change, pre specified segments; narrow around zero means stop, wide means iterate.

**Too good to be true.** $P(\text{true} \mid \text{sig}) = (1-\beta)\pi/((1-\beta)\pi + \alpha(1-\pi))$: 88.9 percent at $\pi = 1/3$, 73.8 at 0.15, 64.0 at 0.10, 3.1 at $1/500$ (Kohavi et al.). Twyman's law: effects with SD 0.25 percent make +2.0 percent an eight sigma event, $p \approx 1.2 \times 10^{-15}$. Check SRM and instrumentation, replicate, combine with Fisher's method (two p values of 0.04 give 0.012).

**Startup traffic.** At 2,000 users a week, a 20 percent relative lift on 10 percent needs about 3,840 per arm (3.8 weeks); 10 percent needs 14,751 (14.7 weeks). Levers: bolder changes, sensitive metrics, CUPED (25 percent less at $\rho = 0.5$), $\alpha = 0.10$ (about 3,020 per arm), sequential tests, priors.

**Explaining to stakeholders.** Recommendation, effect in business terms, range, value (200,000 visitors times 0.009 is 1,800 orders, range 800 to 2,800), plain confidence statement, guardrails, caveats.

**Finance sees nothing.** Dilution (3 percent on 10 percent exposure is 0.3 percent), same version, novelty, winner's curse, interference (21.8 versus 9.5 percent), metric definitions, concurrent launches; settle with a holdout.

## What interviewers probe

- "Derive the sample size formula." Then: why $n \propto 1/\delta^2$; pooled versus unpooled variance; what changes with unequal allocation or one sided tests.
- "Why does peeking inflate false positives?" Then: show the numbers; group sequential boundaries; always valid tests; why Bayesian stopping rules still need checking.
- "You see an SRM." Then: chi square check; why it invalidates; taxonomy by pipeline stage; how you localise it.
- "Explain CUPED." Then: derive $\theta^\ast$ and $1 - \rho^2$; why unbiased; which covariates are safe; what to do for new users.
- "CTR is a ratio metric." Then: delta method derivation; linearisation; why the naive test is wrong.
- "How do you handle network effects or a marketplace?" Then: direction of bias; cluster design effect; switchback slice length and washout; few units and randomisation inference.
- "Novelty effect?" Then: detect by days since exposure; which number to report; holdouts and surrogates.
- "Multiple metrics or variants?" Then: metric roles; Bonferroni cost; control allocation $\sqrt k$; winner's curse.
- "Bandit or A/B?" Then: regret versus inference; why the naive test breaks; off policy evaluation.
- "No experiment possible." Then: DiD and parallel trends checks; synthetic control inference; IV conditions and weak instruments; RD manipulation checks; propensity scores, overlap, doubly robust estimators, sensitivity analysis; uplift evaluation.
- "Ship or not?" Then: practical significance, guardrail trade offs, conflicting metrics, Twyman's law, communication.

## Sources

- [Deng, Xu, Kohavi, Walker: CUPED, improving the sensitivity of online controlled experiments with pre experiment data (WSDM 2013)](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Evan Miller: How Not To Run an A/B Test](https://www.evanmiller.org/how-not-to-run-an-ab-test.html)
- [Fabijan et al.: Diagnosing Sample Ratio Mismatch in Online Controlled Experiments (KDD 2019)](https://exp-platform.com/Documents/2019_KDDFabijanGupchupFuptaOmhoverVermeerDmitriev.pdf)
- [Deng, Knoblich, Lu: Applying the Delta method in metric analytics (arXiv 1803.06336)](https://arxiv.org/abs/1803.06336)
- [Kohavi, Deng, Longbotham, Xu: Seven Rules of Thumb for Web Site Experimenters (KDD 2014)](https://exp-platform.com/Documents/2014%20experimentersRulesOfThumb.pdf)
- [Cunningham: Causal Inference, The Mixtape, chapter 9 Difference in differences](https://mixtape.scunning.com/09-difference_in_differences)
- [Amplitude: 22 A/B testing interview questions and how to answer them](https://www.amplitude.com/blog/a-b-testing-interview-questions.md)
