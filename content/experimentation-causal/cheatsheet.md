# Experimentation and causal inference cheatsheet

Ten minute review before a product data science screen: formulas, decision tables, numbers worth memorising, and one line answers. Every number was produced by a formula or simulation that was run (synthetic data where stated); paper figures name the paper.

## Design and metrics

- Order of work: decision, hypothesis, one primary metric, guardrails, unit, sample size, duration in full weeks, checks, one analysis, decision rule fixed beforehand.
- Primary decides; secondary explains; guardrails block. Guardrail test is non inferiority: reject $\tau \le -\Delta$. Guardrail sample size $= (z_{1-\alpha} + z_{1-\beta})^2\,2p(1-p)/\Delta^2$; 44,514 per arm for $\Delta = 0.5$ points at 10 percent.
- Dilution: overall lift is about segment share times segment lift (10 percent on 1 percent of traffic is 0.1 percent). Triggered analysis cuts traffic by about the trigger rate; control needs counterfactual logging.
- Unit: user by default. Randomise users and analyse sessions: 42 percent false positives at nominal 5 percent. Hash user id with a per experiment salt.
- Analyse everyone assigned (intent to treat). ITT = compliance times effect on compliers.

## Sample size, power, duration

| Quantity | Formula or value |
| --- | --- |
| Two proportions | $n = (z_{1-\alpha/2} + z_{1-\beta})^2\,(p_1q_1 + p_2q_2)/\delta^2$, constant 7.849 |
| Continuous | $n = 2(z_{1-\alpha/2}+z_{1-\beta})^2\sigma^2/\delta^2 \approx 16\sigma^2/\delta^2$ |
| MDE | about $2.8\,\mathrm{SE}$ |
| 10 to 11 percent | 14,748 per arm (statsmodels 14,744) |
| 10 to 12 percent | about 3,840 per arm |
| 10 to 10.5 percent | 57,763 per arm |
| Power 80 to 90 percent | 1.34 times the users |
| $\alpha$ 5 to 1 percent | 1.49 times |
| One sided at 5 percent | 0.79 times (same bar as two sided 10 percent) |
| 90/10 split vs 50/50 | 2.78 times total users |
| Skewed metric floor | $355\,s^2$ per arm for $|s| > 1$ (Kohavi et al., 2014) |

- Halving the effect quadruples $n$. Variance enters linearly; effect enters squared.
- Duration $=$ total $n$ over eligible users per day, rounded up to whole weeks, at least one to two weeks. 29,496 users at 2,000 a day is 14.75 days, so 3 weeks.
- Reached $n$ on day 3? Keep running: partial week, novelty, optional stopping.
- Weekly cycle example: weekday lift 2 percent, weekend 10 percent, true weekly 4.7 percent, five weekdays show 2.0 percent.
- Observed power is the p value in disguise (p of 0.05 gives 50 percent). Report the interval instead.
- Winner's curse: at 17 percent power significant estimates average 2.5 times the truth; at 80 percent power 1.12 times.

## Peeking, SRM, A/A

| Looks in an A/A test | False positive rate |
| --- | --- |
| 1 | 5.0 percent |
| 2 | 8.3 percent |
| 5 | 14.2 percent |
| 10 | 19.2 percent |
| 20 | 24.8 percent |
| 100 | 37.3 percent |

- Five look boundaries: Pocock 2.41 each; O'Brien-Fleming 4.56, 3.22, 2.63, 2.28, 2.04. Max sample cost 1.22 times versus 1.03 times.
- Always valid (mSPRT): reject when the mixture likelihood ratio reaches $1/\alpha$; valid at any look, costs power. Bayesian stopping at $P > 0.95$ still falsely stopped 42.8 percent over 20 looks.
- SRM: $\chi^2 = \sum (O-E)^2/E$; 821,588 versus 815,482 gives $p = 1.8 \times 10^{-6}$. Use a strict alert level (0.001 is common). SRM means selection bias: do not reweight, find the stage (assignment, execution, log processing, analysis, interference). About 6 percent of Microsoft experiments had one (Fabijan et al., 2019).
- A/A: p values uniform, 5 percent below 0.05; too many means understated variance.

## Novelty, long term, interference

- Plot effect by days since first exposure. Report the steady state (8 percent on day 1 settling at 2 percent: first week average 5.2 percent). Use holdouts for long term effects; a 2 percent holdout of 1 million users detects 0.6 points at 10 percent.
- Interference direction: positive spillover biases a user level test toward zero (1.0 versus global 2.0 in simulation); competition for supply biases away (toy marketplace +21.8 percent A/B versus +9.5 percent rollout).
- Cluster design effect $1 + (m-1)\rho$ (5.9 for $m = 50$, $\rho = 0.1$). Switchback: slice longer than adjustment time, drop a washout, slices are the units. Six cities with three treated cannot give $p < 0.10$.

## Variance reduction and ratio metrics

- CUPED: $\tilde Y = Y - \theta(X - \bar X)$, $\theta = \mathrm{Cov}(Y,X)/\mathrm{Var}(X)$, variance times $1 - \rho^2$ (0.51 at $\rho = 0.7$). Unbiased because $X$ is pre treatment. Post treatment covariate gave $-0.45$ for a true $+0.5$. Pre period of one to two weeks; Bing variance cuts 45, 52, 49 percent (Deng et al., 2013).
- Stratification removes between group variance. Cap skewed metrics before the test; keep an uncapped guardrail.
- Ratio of sums: $\mathrm{Var}(\hat R) \approx \mathrm{Var}(Y - RX)/(n\mu_x^2)$. Toy table (R = 0.18) gives SE 0.0432. Naive view level test: 11.7 percent false positives versus 5.4 percent.
- Relative lift: $\mathrm{Var} \approx \mathrm{Var}(\bar Y_t)/\mu_c^2 + \mu_t^2\mathrm{Var}(\bar Y_c)/\mu_c^4$.

## Many metrics, variants, bandits

- $P(\text{any false positive}) = 1 - 0.95^m$: 22.6 percent for 5, 40.1 for 10 metrics. Bonferroni at 10 metrics raises $n$ by 1.70 times.
- A/B/n: control gets $\sqrt k$ times each variant's traffic. Best of 10 null variants shows +0.77 points; shrink by $\tau^2/(\tau^2 + \sigma^2)$.
- Subgroups: 20 slices give at least one significant in 65 percent of null experiments; use one interaction test and confirm.
- Interleaving: paired, more sensitive (toy: 90 percent power at 2,000 queries versus 31 percent for A/B). Bandit: more reward, naive test breaks (14.3 percent false positives).
- Off policy: IPS weights matching rows by $1/\mu(a \mid x)$.

## Causal inference without experiments

| Method | Estimates | Key assumption | One check |
| --- | --- | --- | --- |
| Randomised test | ATE or ITT | Random, no interference | Sample ratio, A/A |
| Difference in differences | ATT | Parallel trends | Event study, placebos |
| Synthetic control | Effect on treated unit | Donor mix reproduces counterfactual | Pre fit, placebo $p \ge 1/(J+1)$ |
| Instrumental variables | LATE (compliers) | Relevance, exclusion, monotonicity | First stage F above 10 |
| Regression discontinuity | Local effect at cutoff | Continuity, no manipulation | Density test, bandwidths |
| Propensity score methods | ATE or ATT | Ignorability, overlap | Balance (SMD below 0.1), weights |

- Naive adopter comparison = ATT + selection bias (3.12 = 1.00 + 2.12). Adjust for confounders, never mediators or colliders.
- DiD $= (\bar Y_{T,post} - \bar Y_{T,pre}) - (\bar Y_{C,post} - \bar Y_{C,pre})$; Snow cholera $-48 - 49 = -97$. Trend gap 0.3 per period turned 2.0 into 3.82. Staggered adoption: avoid plain TWFE.
- IV: Wald $= $ reduced form over first stage (OLS 5.30, IV 2.08, truth 2.0). Weak instruments explode the spread.
- RD: local linear within a bandwidth; naive difference 3.5 versus 2.0 truth. Fuzzy RD = jump in outcome over jump in treatment.
- IPW weights $1/e$ and $1/(1-e)$; effective sample $(\sum w)^2/\sum w^2$. A doubly robust estimator is right if either model is right. E value for $RR = 2$ is 3.41.
- Uplift: top 20 percent by uplift 4.9 percent versus 0.9 percent by response model. Needs randomised data; evaluate by decile or Qini.

## If asked, say

- Peeking: "Looking is fine, acting on a significant look is not. Fixed horizon or a planned sequential design."
- SRM: "Selection bias; I find the pipeline stage before reading any metric."
- Tiny significant effect: "Compare the interval with the smallest worthwhile effect and the cost."
- Guardrail loss: "Verify, compare with the pre agreed margin, convert to value, then fix, segment, ramp or reject."
- Too good to be true: "Twyman's law. Prior 1 in 3 gives 89 percent real; prior 1 in 500 gives 3.1 percent."
- One city launch: "Matched controls, DiD or synthetic control, pre trends, placebo, permutation inference."
- Few users: "Bold changes, sensitive metric, CUPED, accept a stated MDE."
