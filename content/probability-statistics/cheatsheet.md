# Probability and statistics cheatsheet

Ten minute review before a screen: the formulas, decision tables, numbers worth memorising, and the one line answers to the questions that come up most. All numbers were checked by simulation in the deep guide.

## Rules and Bayes

- Complement: $P(\text{at least one}) = 1 - P(\text{none})$. Two dice, at least one 4: $11/36$.
- Inclusion exclusion: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$. Divisible by 3 or 5 up to 300: 140.
- Conditional: $P(A \mid B) = P(A \cap B)/P(B)$. Independent: $P(A \cap B) = P(A)P(B)$. Disjoint events with positive probability are dependent.
- Total probability: $P(B) = \sum_i P(B \mid A_i)P(A_i)$.
- Bayes: posterior odds = prior odds × likelihood ratio. $\text{LR}^+ = \text{sensitivity}/\text{FPR}$.
- Medical test, prevalence 1 in 10,000, 1 percent FPR and FNR: $P(D \mid +) = 0.98$ percent; two independent positives: 49.5 percent.
- Two children, at least one boy: $1/3$. Tuesday boy: $13/27$. Three friends, truth $2/3$, prior 0.5: $8/9$ (independent) or $2/3$ (copying).

## Counting

| | Order matters | Order ignored |
| --- | --- | --- |
| No repeats | $n!/(n-k)!$ | $\binom{n}{k}$ |
| Repeats | $n^k$ | $\binom{n+k-1}{k}$ |

Same suit, two cards: $4/17$. Two queens: $1/221$. 12 people into 3 unnamed teams of 4: $5775$. Last ball red (3 red, 5 blue): $3/8$ by symmetry.

## Expectation and variance

- Linearity always: $E[\sum X_i] = \sum E[X_i]$. Fixed points of a permutation: 1. Empty slots, $n$ keys in $m$ slots: $m(1 - 1/m)^n$.
- Tower rule: $E[X] = E[E[X \mid Y]]$. Rolls until a 6: $m = 1 + \frac56 m = 6$.
- Tail sum: $E[X] = \sum_{k \ge 1}P(X \ge k)$. Max of two dice: $161/36 = 4.47$.
- Jensen: convex $g$ gives $E[g(X)] \ge g(E[X])$. $E[X^2] - E[X]^2 = \operatorname{Var}(X)$.
- $\operatorname{Var}(aX + b) = a^2\operatorname{Var}(X)$; $\operatorname{Var}(X \pm Y) = \operatorname{Var}X + \operatorname{Var}Y \pm 2\operatorname{Cov}$.
- Total variance: $\operatorname{Var}(X) = E[\operatorname{Var}(X \mid Y)] + \operatorname{Var}(E[X \mid Y])$. Random sum: $E[N]\operatorname{Var}X + \operatorname{Var}N \cdot E[X]^2$.
- Correlation in $[-1, 1]$ (Cauchy Schwarz); $\rho = 0.3$ means $R^2 = 0.09$. Zero correlation does not imply independence: $X \sim U(-1, 1)$, $Y = X^2$.
- Markov $P(X \ge a) \le E[X]/a$; Chebyshev $P(|X - \mu| \ge k\sigma) \le 1/k^2$ (25 percent at $k = 2$, normal has 4.6).

## Distributions

| Name | Mean | Variance | Remember |
| --- | --- | --- | --- |
| Bernoulli($p$) | $p$ | $p(1-p)$ | Max variance at 0.5 |
| Binomial($n, p$) | $np$ | $np(1-p)$ | Exactly one of $N$: $NP(1-P)^{N-1}$ |
| Geometric($p$) | $1/p$ | $(1-p)/p^2$ | Memoryless; scipy `geom` counts trials |
| Neg. binomial($r, p$) | $r/p$ | $r(1-p)/p^2$ | Overdispersed counts (gamma mixed Poisson) |
| Poisson($\lambda$) | $\lambda$ | $\lambda$ | Binomial limit; check variance against mean |
| Uniform($a, b$) | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | $k$th order stat of $n$: mean $\frac{k}{n+1}$ |
| Normal | $\mu$ | $\sigma^2$ | 68 / 95 / 99.7; 1.645, 1.96, 2.576 |
| Exponential($\lambda$) | $1/\lambda$ | $1/\lambda^2$ | Memoryless; min of rates $\lambda_1 + \lambda_2$; numpy takes the mean |
| Gamma($\alpha, \lambda$) | $\alpha/\lambda$ | $\alpha/\lambda^2$ | Sum of $\alpha$ exponentials |
| Beta($\alpha, \beta$) | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | Prior for a probability |

Poisson crash question: 5 per year gives $P(\ge 1 \text{ next month}) = 1 - e^{-5/12} = 0.34$. Inspection paradox: Poisson buses every 10 minutes means a 10 minute average wait.

Exponential family $b(y)\exp(\eta T(y) - a(\eta))$: Bernoulli's $\eta$ is the log odds, so the sigmoid falls out ([CS229 Lecture 4](https://www.youtube.com/watch?v=8gVi4Rk21Eg)).

## LLN, CLT, standard errors

- CLT: $\bar X \approx N(\mu, \sigma^2/n)$ if the variance is finite. It is about the mean, not the data.
- Breaks: Cauchy (no mean), Pareto tail index below 2 (no variance), dependence, one dominant term. Skewness of $\bar X$ is $\gamma/\sqrt n$; need $n \approx 100\gamma^2$ for skew below 0.1.
- $\operatorname{SE}(\bar X) = s/\sqrt n$; proportion $\sqrt{p(1-p)/n}$; difference $\sqrt{s_1^2/n_1 + s_2^2/n_2}$.
- Margin from 3 to 0.3: 100 times the sample. CTR within $\delta$: $n \ge 1.96^2 p(1-p)/\delta^2$ (about 7,300 at $p = 0.05$, $\delta = 0.005$).
- $s^2$ divides by $n - 1$ (unbiased); $s$ is still biased low; numpy `var` defaults to `ddof=0`, pandas to 1.
- Clustered data: design effect $1 + (m-1)\rho$. Ratio metrics: delta method or user level bootstrap.

## Intervals and tests

- 95 percent CI: the procedure covers the truth 95 percent of the time. Not a probability for this interval; not where data fall; a replication's mean lands inside about 83 percent of the time.
- Puppies: $1 \pm 1.96 \times 0.012 = (0.976, 1.024)$.
- Wald proportion interval fails near 0 or 1; use Wilson. Zero events in $n$: upper bound about $3/n$.
- Overlapping intervals can still be a significant difference. Interval excludes 0 iff $p < 0.05$ (duality).
- p value: $P(\text{data at least this extreme} \mid H_0)$. Uniform under the null. Not $P(H_0 \mid \text{data})$.
- Power: $n \approx 16\sigma^2/\delta^2$ per group at 5 percent and 80 percent. 10 versus 11 percent conversion: about 14,750 per arm.
- 10 interim looks at an A/A test: about 19 percent false positives.
- Share of significant results that are false: $\frac{\alpha\pi_0}{\alpha\pi_0 + \text{power}(1 - \pi_0)}$; 36 percent at $\pi_0 = 0.9$, power 0.8.
- 8 heads in 10: exact binomial $p = 0.109$.

| Situation | Test |
| --- | --- |
| Two independent means | Welch t (`equal_var=False`); Student's false positive rate hit 25 percent when the small group was noisier |
| Same units twice | Paired t on differences |
| 3 or more means | ANOVA, then Tukey or Holm |
| 2 by 2 table, big counts | Two proportion z or chi square (scipy applies Yates on 2 by 2) |
| Small expected counts | Fisher exact |
| Ordinal or skewed, "tends to be larger" | Mann Whitney (tests $P(X > Y)$, not medians) |
| Any statistic, randomised | Permutation test, $p = (\#\text{extreme} + 1)/(B + 1)$ |
| Paired binary (two models, same examples) | McNemar |

## Multiple comparisons

- 20 null tests: $1 - 0.95^{20} = 0.64$ chance of a false positive.
- Bonferroni $\alpha/m$; Holm: sorted $p_{(k)} \le \alpha/(m - k + 1)$, stop at first failure; BH: largest $k$ with $p_{(k)} \le k\alpha/m$, reject all below.
- FWER for confirmatory decisions, FDR for screening. BH valid under independence or positive dependence; BY divides by $\sum 1/i$.
- statsmodels `multipletests` defaults to `method='hs'` (Holm Šidák). Always pass the method.

## Estimation

- MLE Bernoulli $\hat p = k/n$; Gaussian $\hat\mu = \bar x$, $\hat\sigma^2 = \frac1n\sum(x - \bar x)^2$ (biased by $\frac{n-1}{n}$); exponential $\hat\lambda = 1/\bar x$.
- Least squares = Gaussian noise MLE. Cross entropy = NLL. Uniform over $K$ classes: loss $\log K$.
- MLE asymptotics: $\operatorname{Var} \approx 1/(nI(\theta))$; Bernoulli $I = 1/(p(1-p))$.
- MAP: Gaussian prior = L2, Laplace prior = L1, Beta(2, 2) = add one smoothing.
- Conjugates: Beta binomial (add successes and failures), Gamma Poisson ($a + \sum x$, $b + n$), Dirichlet multinomial, normal normal (precisions add).
- Beta(1, 1) + 8 of 12 heads = Beta(9, 5): mean 0.643, 95 percent credible (0.386, 0.861).
- Rule of succession $\frac{k+1}{n+2}$. One double headed coin in 1,000, 10 heads: 0.506.
- GDA implies a logistic posterior. EM: E step responsibilities, M step weighted MLE, never decreases the likelihood, local optima.
- Bootstrap: resample the independent unit; percentile, basic, BCa (scipy default); fails for the max, dependence, tiny $n$.

## Causal traps

- Correlation: causation, reverse, confounder, selection or collider, chance.
- Simpson (kidney stones): A wins both strata, loses pooled; stratify on confounders, never on post treatment variables.
- Regression to the mean: $E[Z_Y \mid Z_X = z] = \rho z$; needs a control group selected the same way.
- Small units fill both extremes (kidney failure in small cities); shrink with empirical Bayes.
- Survivorship: Wald's bombers. Collider: hired for coding or communication.

## Puzzles

| Puzzle | Answer | Method |
| --- | --- | --- |
| Flips until HH / HT | 6 / 4 | Two state first step equations |
| Any pattern, fair coin | Sum of $2^k$ over self overlaps | HTH 10, HHH 14 |
| Monty Hall | Switch wins 2/3 | Host knows; random host gives 1/2 |
| Birthday | 23 for a pair; 253 to match you | Complement, 253 pairs |
| Coupon collector | $nH_n$; die 14.7; 50 coupons 225 | Geometric stages |
| Gambler's ruin | $k/N$, duration $k(N - k)$ | Martingales |
| Random walk | $E|S_n| \approx \sqrt{2n/\pi}$ | Recurrent in 1D |
| Stick in three | 1/4; break longer piece $2\ln 2 - 1$ | Area in unit square |
| Blue before first red | $b/(r+1)$ | Gaps symmetry |
| Fair coin from biased | HT vs TH; $1/(p(1-p))$ flips | Rejection |
| German tank | $m + m/k - 1$ | Unbiased max correction |

## If asked X, say Y

- "What's a p value?" The chance of data this extreme if the null were true; a statement about the data, not the hypothesis.
- "Data aren't normal, so t-test is invalid?" The CLT covers the mean; heavy skew slows it; rank tests change the question.
- "Is this lift real?" Effect size, interval, power, multiplicity, and whether the unit of analysis matches randomisation.
- "Why switch in Monty Hall?" The host's forced choice carries information.
