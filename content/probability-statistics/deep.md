# Probability and statistics

This is the technical version of the probability and statistics module: the axioms and rules, conditional probability and Bayes, counting, random variables, expectation and variance with the laws of total expectation and variance, the distributions that interviewers expect you to recognise, the law of large numbers and the central limit theorem with their failure cases, sampling distributions, confidence intervals, hypothesis tests and the test menu, multiple comparisons, maximum likelihood and MAP with derivations, conjugate Bayesian updating, the bootstrap, the causal traps (Simpson's paradox, regression to the mean, selection and survivorship bias), and the classic quant and data science puzzles. Every number below was checked by running code in a Python environment with numpy 2.5, scipy 1.18 and statsmodels 0.15, and every puzzle has a simulation next to its derivation.

## Probability rules

A **probability** is a function $P$ that assigns a number to each event (a set of outcomes in the sample space $\Omega$) and obeys three axioms: $P(A) \ge 0$, $P(\Omega) = 1$, and for disjoint events $P(A_1 \cup A_2 \cup \dots) = \sum_i P(A_i)$. Everything else is derived from these. The CS229 probability review states them in this form and builds the rest of its notes on them.

The derived rules you use every day:

| Rule | Formula | When you reach for it |
| --- | --- | --- |
| Complement | $P(A^c) = 1 - P(A)$ | "At least one" questions: compute "none" and subtract |
| Inclusion exclusion | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | Overlapping events, FizzBuzz style counts |
| Monotonicity | $A \subseteq B \Rightarrow P(A) \le P(B)$ | Sanity checks and the conjunction fallacy |
| Union bound | $P(\bigcup_i A_i) \le \sum_i P(A_i)$ | Bonferroni correction, worst case error budgets |
| Product rule | $P(A \cap B) = P(A \mid B)P(B)$ | Sequential draws, chain rule of probability |

Inclusion exclusion generalises to $n$ events by alternating sums over intersections of size 1, 2, 3 and so on. A common screen question asks how many integers from 1 to 300 are divisible by 3 or 5: $100 + 60 - 20 = 140$, where 20 counts multiples of 15 that were counted twice.

The complement rule is the single most useful trick in screens. "Probability of at least one 4 in two dice" is $1 - (5/6)^2 = 11/36$. "Probability of seeing at least one shooting star in an hour when the chance in any 15 minute window is 0.2" is $1 - 0.8^4 = 0.5904$ if the windows are independent.

> **Interview trap:** The union bound is an inequality, not an identity. Bonferroni is conservative precisely because it uses $P(\bigcup A_i) \le \sum P(A_i)$, which is loose when the events overlap (correlated tests).

## Conditional probability and Bayes

The conditional probability of $A$ given $B$, for $P(B) > 0$, is $P(A \mid B) = P(A \cap B) / P(B)$. Conditioning restricts the sample space to $B$ and renormalises. Two events are **independent** when $P(A \cap B) = P(A)P(B)$, equivalently $P(A \mid B) = P(A)$. A collection is mutually independent only if the product rule holds for every subset, which is stronger than pairwise independence: with two fair coins, the events "first is heads", "second is heads" and "the coins match" are pairwise independent but not mutually independent.

Independence is different from being disjoint. Disjoint events with positive probability are maximally dependent: if one happens the other cannot.

The **law of total probability** splits a probability over a partition $A_1, \dots, A_n$ of the sample space:

$$
P(B) = \sum_{i=1}^{n} P(B \mid A_i) P(A_i).
$$

**Bayes' theorem** follows by writing the joint probability two ways:

$$
P(A_j \mid B) = \frac{P(B \mid A_j) P(A_j)}{\sum_i P(B \mid A_i) P(A_i)}.
$$

In words: posterior is likelihood times prior, divided by the evidence. In odds form, which is easier to do in your head, posterior odds equal prior odds times the likelihood ratio: $\frac{P(H \mid E)}{P(H^c \mid E)} = \frac{P(H)}{P(H^c)} \cdot \frac{P(E \mid H)}{P(E \mid H^c)}$.

### The medical test and the base rate

Chip Huyen's interview book asks this version: a disease affects 1 in 10,000 people, the test has a 1 percent false positive rate and a 1 percent false negative rate. What is the chance that someone who tests positive is sick?

$$
P(D \mid +) = \frac{0.99 \times 0.0001}{0.99 \times 0.0001 + 0.01 \times 0.9999} \approx 0.0098.
$$

Under 1 percent. The natural frequency version is how you explain it: in a million people, 100 are sick and 99 of them test positive; 999,900 are healthy and 9,999 of them test positive. Of the 10,098 positives, 99 are sick. The test is accurate, but the healthy population is so large that its 1 percent false positives swamp the true positives. This is **base rate neglect**.

A second independent positive test multiplies the odds by the likelihood ratio $0.99 / 0.01 = 99$ again. Prior odds of $1:9999$ become $99:9999$ after one positive and $9801:9999$ after two, so $P(D \mid ++) \approx 0.495$. Independence of the two tests given disease status is the assumption that makes this work; two runs of the same assay on the same sample are not independent.

```python
import numpy as np

prev, sens, fpr = 1 / 10_000, 0.99, 0.01
exact = sens * prev / (sens * prev + fpr * (1 - prev))

rng = np.random.default_rng(0)
n = 10_000_000
sick = rng.random(n) < prev
positive = np.where(sick, rng.random(n) < sens, rng.random(n) < fpr)
print(f"exact {exact:.4f}  simulated {sick[positive].mean():.4f}")
```

This printed `exact 0.0098  simulated 0.0099`.

### Conditioning puzzles that test the definition

- **Two children, at least one is a boy.** The equally likely families are BB, BG, GB, GG. Conditioning on "at least one boy" leaves three, so $P(BB) = 1/3$. If instead you meet a specific child and he is a boy, the answer is $1/2$, because the information is about that child. The difference is how the information was generated.
- **Boy born on a Tuesday.** There are $14^2 - 13^2 = 27$ equally likely (sex, day) pairs with at least one Tuesday boy, of which $7^2 - 6^2 = 13$ are two boys. The answer $13/27$ moves toward $1/2$ because the extra detail nearly identifies one child. A simulation with a million families gave 0.480 against $13/27 = 0.481$.
- **Three unreliable friends.** Facebook has asked: you ask three friends whether it is raining in Seattle; each tells the truth with probability $2/3$ and all say yes. With a prior of 0.5 and independent friends, posterior odds are $1 \times 2^3 = 8$, so $P = 8/9 \approx 0.889$. If the friends always copy each other, the three answers are one piece of evidence and the posterior is $2/3$. The dependence assumption is the whole question.

> **Say it like this:** "Bayes is prior odds times the likelihood ratio. A 99 percent accurate test for a 1 in 10,000 disease gives a likelihood ratio of 99, which turns odds of 1 to 9,999 into about 1 to 100. So a positive means roughly a 1 percent chance of disease."

## Counting and combinatorics

When outcomes are equally likely, probability is counting: $P(A) = |A| / |\Omega|$. The four counting formulas come from two questions: does order matter, and is repetition allowed?

| | Order matters | Order does not matter |
| --- | --- | --- |
| Without repetition | $n!/(n-k)!$ (permutations) | $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ (combinations) |
| With repetition | $n^k$ | $\binom{n+k-1}{k}$ (stars and bars) |

Worked numbers from screens:

- Ways to rank 3 people as 1st, 2nd, 3rd: $3! = 6$.
- Probability two cards drawn from a deck share a suit: whatever the first card, 12 of the remaining 51 match, so $12/51 = 4/17$.
- Probability of two queens: $\frac{4}{52} \cdot \frac{3}{51} = 1/221$, or $\binom{4}{2}/\binom{52}{2}$.
- Exactly 2 tails in 5 flips: $\binom{5}{2}/2^5 = 10/32$.
- Exactly one 5 in seven die rolls: $7 \cdot \frac{1}{6} \cdot (5/6)^6 \approx 0.391$.
- Splitting 12 people into 3 unlabeled teams of 4: $\frac{12!}{(4!)^3 \, 3!} = 5775$. The $3!$ divides out the team labels.

Two identities carry most derivations: Pascal's rule $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$ (the first item is in or out), and Vandermonde's identity $\binom{m+n}{k} = \sum_j \binom{m}{j}\binom{n}{k-j}$, which is the hypergeometric distribution summing to one.

Use **symmetry** before counting. "Probability the last ball drawn from an urn of 3 red and 5 blue is red" is $3/8$, the same as the first ball, because every position is exchangeable. A simulation of 100,000 shuffles gave 0.374.

> **Interview trap:** Overcounting unlabeled groups. If the teams have no names, divide by the number of ways to permute the teams.

## Random variables and expectation

A **random variable** is a function from outcomes to numbers. It is described by its cumulative distribution function $F(x) = P(X \le x)$, which always exists, and by a probability mass function $p(x) = P(X = x)$ when discrete or a density $f(x) = F'(x)$ when continuous. A density is not a probability and can exceed 1: the uniform density on $[0, 0.5]$ equals 2 everywhere on its support. Only areas under $f$ are probabilities.

The **expectation** is the probability weighted average, $E[X] = \sum_x x\,p(x)$ or $\int x f(x)\,dx$, defined when $E|X| < \infty$. For a nonnegative integer variable there is the tail sum formula $E[X] = \sum_{k \ge 1} P(X \ge k)$, handy for waiting times.

**Linearity of expectation** says $E[aX + bY + c] = aE[X] + bE[Y] + c$ for any random variables, independent or not. This is why indicator variables solve so many puzzles. Expected number of fixed points of a random permutation of $n$ items: write $X = \sum_i I_i$ where $I_i$ indicates item $i$ lands in place $i$; each $E[I_i] = 1/n$, so $E[X] = 1$ for every $n$, even though the $I_i$ are dependent. The same argument gives the expected number of letters reaching the right envelope and the expected number of empty hash buckets: $m(1 - 1/m)^n$ for $n$ keys in $m$ buckets.

Products are not linear. $E[XY] = E[X]E[Y]$ needs uncorrelatedness, and $E[g(X)] \ne g(E[X])$ in general. **Jensen's inequality** pins the direction: for convex $g$, $E[g(X)] \ge g(E[X])$. With $g(x) = x^2$ this is $E[X^2] \ge E[X]^2$, which answers PayPal's question of whether squaring one draw beats multiplying two independent draws (it does, by exactly $\operatorname{Var}(X)$). Jensen with the concave $\log$ is the inequality behind the evidence lower bound in EM.

### Law of total expectation

The **law of total expectation** (tower rule) says $E[X] = E[E[X \mid Y]]$: average the conditional means, weighting by how likely each condition is. It is the formal version of "condition on the first step". Expected number of die rolls until a 6: with $\mu$ the answer, the first roll is a 6 with probability $1/6$ and otherwise you restart, so $\mu = 1 + \frac{5}{6}\mu$ and $\mu = 6$.

The same move gives Wald's identity for a random sum. If $N$ is the number of orders in a day and each order has mean value $E[X]$, independent of $N$, then $E[\sum_{i=1}^{N} X_i] = E[N]E[X]$.

### Markov and Chebyshev

Two inequalities bound tails from moments alone. **Markov**: for $X \ge 0$, $P(X \ge a) \le E[X]/a$. **Chebyshev**: $P(|X - \mu| \ge k\sigma) \le 1/k^2$. Chebyshev guarantees at most 25 percent beyond two standard deviations for any finite variance distribution; a normal puts 4.6 percent there. These are what you cite when asked for a guarantee without a distributional assumption, and Chebyshev gives a one line proof of the weak law of large numbers.

## Variance, covariance and correlation

The **variance** is $\operatorname{Var}(X) = E[(X - \mu)^2] = E[X^2] - \mu^2$. Shifts do not change it and scaling multiplies it by the square: $\operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X)$. The **covariance** is $\operatorname{Cov}(X, Y) = E[XY] - E[X]E[Y]$, and the variance of a sum is

$$
\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y).
$$

So variances add for independent (or merely uncorrelated) variables, and the variance of a difference of independent variables is also the sum: $\operatorname{Var}(X - Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$. That one fact gives the standard error of a difference in means in every A/B test. For $n$ variables, $\operatorname{Var}(\sum_i X_i) = \sum_i \sum_j \operatorname{Cov}(X_i, X_j)$, and for a vector $a^\top X$ it is $a^\top \Sigma a$.

**Correlation** rescales covariance to $[-1, 1]$: $\rho = \operatorname{Cov}(X, Y) / (\sigma_X \sigma_Y)$. The bound is the Cauchy Schwarz inequality, and $|\rho| = 1$ exactly when $Y$ is a linear function of $X$. Correlation measures linear association only. A correlation of 0.3 means the best linear predictor of standardised $Y$ is $0.3$ times standardised $X$, and in simple regression $\rho^2 = 0.09$ is the share of variance explained.

> **Interview trap:** Zero correlation does not imply independence. Take $X$ uniform on $[-1, 1]$ and $Y = X^2$. $Y$ is a function of $X$, yet $\operatorname{Cov}(X, Y) = E[X^3] = 0$ by symmetry. The CS229 review uses this exact counterexample. Independence implies zero covariance, but the converse holds only in special cases such as jointly Gaussian variables.

Pearson correlation is sensitive to outliers and to nonlinearity. **Spearman** correlation is Pearson on ranks and detects any monotone relation; Kendall's tau counts concordant minus discordant pairs. To test independence of two continuous variables beyond linear dependence, use a permutation test on a dependence measure such as distance correlation or mutual information.

### Law of total variance

The **law of total variance** decomposes spread into within group and between group parts:

$$
\operatorname{Var}(X) = E[\operatorname{Var}(X \mid Y)] + \operatorname{Var}(E[X \mid Y]).
$$

Two checked examples. Heights from a 50/50 mix of groups with means 178 and 164 and standard deviations 7 and 6: the within part is $0.5 \cdot 49 + 0.5 \cdot 36 = 42.5$, the between part is $0.25 \cdot 14^2 = 49$, total 91.5. A simulation of a million draws gave 91.50. For a random sum with $N \sim \text{Poisson}(10)$ orders a day, each worth mean 30 and standard deviation 10, $\operatorname{Var}(S) = E[N]\operatorname{Var}(X) + \operatorname{Var}(N)E[X]^2 = 1000 + 9000 = 10000$, so the daily total has standard deviation 100. Notice that 90 percent of the variance comes from not knowing how many orders arrive, not from order sizes. This decomposition is the idea behind ANOVA, random effects models, and variance reduction methods such as CUPED and stratification.

## Discrete distributions

Huyen's interview book has a standing question: pick any common distribution, write its equation and sketch it. Know the generative story for each, because the story is how you recognise it in a word problem.

| Distribution | Story | PMF | Mean | Variance |
| --- | --- | --- | --- | --- |
| Bernoulli($p$) | One trial, success with probability $p$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| Binomial($n, p$) | Successes in $n$ independent trials | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Geometric($p$) | Trials until the first success | $(1-p)^{k-1}p$, $k \ge 1$ | $1/p$ | $(1-p)/p^2$ |
| Negative binomial($r, p$) | Trials until the $r$th success | $\binom{k-1}{r-1}p^r(1-p)^{k-r}$ | $r/p$ | $r(1-p)/p^2$ |
| Poisson($\lambda$) | Count of rare events in a fixed window | $e^{-\lambda}\lambda^k/k!$, $k \ge 0$ | $\lambda$ | $\lambda$ |
| Hypergeometric($N, K, n$) | Successes in $n$ draws without replacement | $\binom{K}{k}\binom{N-K}{n-k}/\binom{N}{n}$ | $nK/N$ | $n\frac{K}{N}\frac{N-K}{N}\frac{N-n}{N-1}$ |
| Multinomial($n, \mathbf{p}$) | Counts in $k$ categories over $n$ trials | $\frac{n!}{\prod x_i!}\prod p_i^{x_i}$ | $np_i$ | $np_i(1-p_i)$, $\operatorname{Cov} = -np_ip_j$ |

Notes that interviewers probe:

- **Binomial** is a sum of $n$ independent Bernoullis, which is why its mean and variance are $n$ times the Bernoulli ones. Lyft has asked for the probability that exactly one of $N$ riders uses a coupon with use probability $P$: that is $NP(1-P)^{N-1}$, and the expected cost of USD 5 coupons is $5NP$.
- **Geometric** has two conventions: trials until the first success (support $1, 2, \dots$, mean $1/p$) or failures before it (support $0, 1, \dots$, mean $(1-p)/p$). scipy's `geom` uses the first; `nbinom` counts failures. State which one you mean. Expected rolls until a 6 is $1/p = 6$. A family that keeps having children until the first girl has 2 children on average, and the population sex ratio stays 1:1 because every birth is still a fair coin.
- The geometric distribution is **memoryless**: $P(X > s + t \mid X > s) = P(X > t)$. It is the only discrete distribution with this property.
- **Negative binomial** is a sum of $r$ independent geometrics, so trials to get 3 successes at $p = 0.2$ has mean 15 and variance 60. It is also the standard model for overdispersed counts (variance greater than mean), as a Poisson whose rate is gamma distributed.
- **Poisson** arises as the limit of Binomial($n, p$) with $np = \lambda$ fixed, $n \to \infty$. Binomial(1000, 0.003) gives $P(0) = 0.0496$ against Poisson(3)'s 0.0498. Sums of independent Poissons are Poisson with the rates added. Mean equals variance; real count data usually violate this (overdispersion), which is the first thing to check.
- **Hypergeometric** is the binomial without replacement; the factor $\frac{N-n}{N-1}$ is the finite population correction. It is the null distribution behind Fisher's exact test.
- **Multinomial** components are negatively correlated, because the counts must sum to $n$. A categorical variable is a multinomial with $n = 1$, and its MLE is the empirical frequency, which is exactly what Karpathy's bigram model computes by counting.

> **Worked example:** Huyen's file storage question: 5 independent crashes last year. Model crashes as a Poisson process with rate 5 per year. The chance of at least one crash next month is $1 - e^{-5/12} \approx 0.341$. The chance of a crash "at any given moment" is zero for a continuous time process; the useful answer is a rate.

## Continuous distributions

| Distribution | Story | Density | Mean | Variance |
| --- | --- | --- | --- | --- |
| Uniform($a, b$) | No location preferred | $\frac{1}{b-a}$ on $[a, b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| Normal($\mu, \sigma^2$) | Sum of many small independent effects | $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ |
| Exponential($\lambda$) | Waiting time to the next Poisson event | $\lambda e^{-\lambda x}$, $x \ge 0$ | $1/\lambda$ | $1/\lambda^2$ |
| Gamma($\alpha, \lambda$) | Waiting time to the $\alpha$th event | $\frac{\lambda^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\lambda x}$ | $\alpha/\lambda$ | $\alpha/\lambda^2$ |
| Beta($\alpha, \beta$) | A probability, or an order statistic of uniforms | $\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ on $[0, 1]$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

The **normal** matters because of the central limit theorem and because it is closed under linear maps: $aX + b$ is normal, sums of independent normals are normal, and marginals and conditionals of a multivariate normal are normal. Memorise 68, 95 and 99.7 percent within 1, 2 and 3 standard deviations (0.6827, 0.9545, 0.9973), and the quantiles 1.645 (90 percent two sided), 1.96 (95 percent) and 2.576 (99 percent). A z score $z = (x - \mu)/\sigma$ locates a value in standard deviations; on a skewed distribution it still standardises but no longer maps to normal percentiles.

The **exponential** is the continuous memoryless distribution: $P(X > s + t \mid X > s) = e^{-\lambda t} = P(X > t)$. A simulation with mean 5 gave $P(X > 8 \mid X > 3) = 0.3676$ against $P(X > 5) = 0.3677$. Two consequences come up constantly. The minimum of independent exponentials with rates $\lambda_1, \lambda_2$ is exponential with rate $\lambda_1 + \lambda_2$, and $P(X_1 < X_2) = \lambda_1 / (\lambda_1 + \lambda_2)$. And memorylessness drives the **inspection paradox**: if buses arrive as a Poisson process every 10 minutes on average, a passenger arriving at a random time waits 10 minutes on average, not 5, because long gaps are more likely to contain your arrival.

Parameterisation is a trap. The CS229 review writes the exponential with rate $\lambda$ (mean $1/\lambda$), numpy's `exponential(scale)` and scipy's `expon(scale=...)` take the mean. Gamma has the same split between rate and scale.

The **gamma** with integer shape $\alpha$ is a sum of $\alpha$ independent exponentials: four exponentials of mean 2 sum to mean 8 and variance 16 (simulation: 8.02 and 15.99). The chi square with $k$ degrees of freedom is Gamma($k/2$, rate $1/2$), the sum of $k$ squared standard normals. The **beta** lives on $[0, 1]$, which makes it the natural prior for a probability; the $k$th smallest of $n$ uniforms is Beta($k, n - k + 1$), so the second smallest of four uniforms has mean $2/5$.

The **t distribution** with $\nu$ degrees of freedom is $Z / \sqrt{V/\nu}$ for independent standard normal $Z$ and chi square $V$. It has heavier tails than the normal, which is the price of estimating $\sigma$ from data, and approaches the normal as $\nu$ grows. The **F distribution** is a ratio of scaled chi squares and is the null distribution of the ANOVA statistic.

### The exponential family

Most of these distributions share one form, which CS229 writes as $p(y; \eta) = b(y)\exp(\eta^\top T(y) - a(\eta))$, with natural parameter $\eta$, sufficient statistic $T(y)$ and log partition function $a(\eta)$. The Bernoulli has $\eta = \log\frac{\phi}{1-\phi}$, and inverting gives $\phi = 1/(1 + e^{-\eta})$: the sigmoid falls out of the distribution rather than being chosen. The Gaussian with unit variance has $\eta = \mu$ and $a(\eta) = \eta^2/2$. Derivatives of $a$ give the moments ($a'(\eta) = E[T(y)]$), the log likelihood is concave in $\eta$, and conjugate priors exist. Generalised linear models set $\eta = \theta^\top x$, which yields linear regression for the Gaussian, logistic regression for the Bernoulli and Poisson regression for counts.

> **Watch:** [CS229 Lecture 4: Exponential Family, GLMs Classification](https://www.youtube.com/watch?v=8gVi4Rk21Eg) builds GLMs from this form; section 3 of the CS229 notes is the written version.

## Law of large numbers and the CLT

The **law of large numbers** says the sample mean $\bar X_n$ of i.i.d. draws converges to $\mu$. The weak law (convergence in probability) follows from Chebyshev in one line when the variance is finite: $P(|\bar X_n - \mu| \ge \epsilon) \le \sigma^2/(n\epsilon^2) \to 0$. The strong law (almost sure convergence) needs only $E|X| < \infty$, which is how the CS229 review states it.

The **central limit theorem** says that for i.i.d. draws with mean $\mu$ and finite variance $\sigma^2$,

$$
\frac{\sqrt{n}(\bar X_n - \mu)}{\sigma} \xrightarrow{d} N(0, 1).
$$

The sample mean is approximately $N(\mu, \sigma^2/n)$ whatever the shape of the data, which is why normal based tests and intervals work for means of non normal metrics such as revenue. The CS229 review summarises this as holding "irrespective of what distribution X follows", but its own theorem assumes a finite variance, and that assumption is the one that fails in practice.

What breaks them:

| Failure | What happens | Example |
| --- | --- | --- |
| No finite mean | LLN fails; the sample mean never settles | Cauchy: the mean of $n$ draws is again standard Cauchy |
| Infinite variance | LLN may hold, CLT fails; sums converge to a stable law, not a normal | Pareto with tail index below 2, some city sizes and file sizes |
| Strong dependence | Effective sample size is far below $n$; standard errors are too small | Time series, users with many sessions, clustered data |
| Not identically distributed | Lindeberg conditions may fail if one term dominates | One whale customer in a revenue sum |
| Finite $n$ with skew | CLT holds but slowly; tails of $\bar X$ stay skewed | Exponential: skewness of $\bar X$ is $2/\sqrt{n}$ |

```python
import numpy as np

rng = np.random.default_rng(0)

def iqr_of_means(draw, n, reps=2_000):
    means = draw((reps, n)).mean(axis=1)
    q75, q25 = np.percentile(means, [75, 25])
    return q75 - q25

for n in (10, 1_000):
    e = iqr_of_means(lambda s: rng.exponential(1.0, s), n)
    c = iqr_of_means(rng.standard_cauchy, n)
    print(f"n={n:>5}  spread of sample means: exponential {e:.3f}   cauchy {c:.3f}")

# skewness of the mean of n exponentials is 2 / sqrt(n): the CLT is slow for skewed data
for n in (5, 30, 200):
    means = rng.exponential(1.0, size=(50_000, n)).mean(axis=1)
    z = (means - means.mean()) / means.std()
    print(f"n={n:>3}  skew {float((z ** 3).mean()):.3f}  theory {2 / n ** 0.5:.3f}")
```

Output: the spread of exponential sample means fell from 0.408 to 0.044 (a factor of 10 for 100 times the data, as $1/\sqrt{n}$ predicts), while the Cauchy spread stayed at 2.04 and 1.99. The skewness of the exponential mean was 0.901, 0.369 and 0.135 at $n = 5, 30, 200$ against theory 0.894, 0.365 and 0.141. The "n of 30 is enough" rule is a folk rule; for heavy right skew such as revenue per user, you need far more.

> **Interview trap:** The CLT is about the distribution of the sample mean, not the data. A histogram of a million revenue values stays skewed forever; the distribution of the average of those values across repeated samples becomes normal.

## Sampling distributions and standard error

A **statistic** is any function of the sample; its **sampling distribution** is the distribution it would have across repeated samples. The **standard error** is the standard deviation of that sampling distribution. For the mean of $n$ i.i.d. draws,

$$
\operatorname{SE}(\bar X) = \frac{\sigma}{\sqrt{n}}, \qquad \widehat{\operatorname{SE}} = \frac{s}{\sqrt{n}}, \qquad s^2 = \frac{1}{n-1}\sum_i (x_i - \bar x)^2.
$$

The standard deviation describes the spread of individual values; the standard error describes the uncertainty in an estimate, and it shrinks with $\sqrt{n}$. Halving the standard error takes four times the data. Google has asked how many more samples it takes to cut a margin of error from 3 to 0.3: a factor of $(3/0.3)^2 = 100$.

Useful special cases:

- **Proportion:** $\operatorname{SE}(\hat p) = \sqrt{p(1-p)/n}$, largest at $p = 0.5$. With $p = 0.3$ and $n = 400$, SE is 0.023.
- **Difference of independent means:** $\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}$, from the variance of a difference.
- **Sample size for a margin $\delta$ on a proportion at 95 percent:** $n \ge 1.96^2 p(1-p)/\delta^2$. Facebook has asked this for a click through rate; at $p = 0.05$ and $\delta = 0.005$ it gives about 7,300.

**Bessel's correction.** Dividing by $n - 1$ makes $s^2$ unbiased because the deviations are measured from $\bar x$, which is itself fitted to the data and sits closer to the points than $\mu$ does. Formally, $E\left[\sum_i (X_i - \bar X)^2\right] = (n-1)\sigma^2$, because $\sum_i (X_i - \mu)^2 = \sum_i (X_i - \bar X)^2 + n(\bar X - \mu)^2$ and the last term has expectation $\sigma^2$. Note that $s$ itself is still biased low (Jensen, since square root is concave).

**The delta method** gives the standard error of a smooth function of an estimate: if $\hat\theta \approx N(\theta, \sigma^2/n)$ then $g(\hat\theta) \approx N(g(\theta), g'(\theta)^2\sigma^2/n)$. It is how you get standard errors for ratio metrics like revenue per session, where both numerator and denominator are random and the sessions of one user are correlated. The bootstrap is the computational alternative.

> **Interview trap:** Treating sessions as independent when users are the randomisation unit. If each user contributes many correlated sessions, the naive $s/\sqrt{n}$ over sessions understates the standard error. Cluster by user or use the delta method.

## Confidence intervals

A 95 percent **confidence interval** is the output of a procedure that, over repeated samples, produces intervals containing the true parameter 95 percent of the time. For a mean with unknown variance,

$$
\bar x \pm t_{n-1, 0.975} \frac{s}{\sqrt{n}}.
$$

Huyen's puppies: 100 newborn puppies have mean weight 1 lb, population standard deviation 0.12 lb. The 95 percent interval for the mean is $1 \pm 1.96 \times 0.012 = (0.976, 1.024)$ lb.

The correct reading is about the method. Once you compute $(0.976, 1.024)$, the true mean is either in it or not; the 95 percent describes how often intervals built this way cover the truth. Greenland and coauthors list the probability reading of a single interval as a standard misinterpretation, along with two others interviewers use as distractors: that 95 percent of individual puppies fall in the interval (that would be a prediction interval, much wider at roughly $1 \pm 1.96 \times 0.12$), and that the next sample's mean has a 95 percent chance of landing inside it (the actual chance is lower, about 83 percent for an exact replication, because both means are noisy).

Coverage checks in simulation (20,000 repetitions, normal data):

| Interval | n = 30 | n = 5 |
| --- | --- | --- |
| $t$ quantile with $s$ | 0.950 | 0.950 |
| 1.96 with $s$ | 0.939 | 0.877 |

Using 1.96 when you estimated $\sigma$ undercovers, badly at small $n$. That is the reason the $t$ distribution exists.

Other intervals worth naming:

- **Proportion:** the Wald interval $\hat p \pm 1.96\sqrt{\hat p(1 - \hat p)/n}$ fails near 0 or 1 and at small $n$ (it can even have zero width at $\hat p = 0$). Wilson or Clopper Pearson intervals fix this. Tesla has asked for an interval on 25 defects out of 100: Wald gives $0.25 \pm 0.085$.
- **Duality with tests:** a 95 percent interval contains exactly the parameter values a two sided test at 5 percent would not reject. If the interval for a difference excludes zero, $p < 0.05$.
- **Overlap rule:** two 95 percent intervals can overlap and the difference can still be significant, because the standard error of a difference is $\sqrt{SE_1^2 + SE_2^2}$, smaller than $SE_1 + SE_2$. Test the difference directly.
- **Credible interval:** the Bayesian analogue, where "95 percent probability the parameter is in this interval" is a correct statement, conditional on the prior.

> **Say it like this:** "A 95 percent confidence interval comes from a procedure that captures the true value in 95 percent of repeated samples. For the one interval I computed, I can't say there's a 95 percent chance the truth is inside; it either is or isn't. If I want that probability statement, I need a Bayesian credible interval and a prior."

## Hypothesis testing

A test compares the data with what a **null hypothesis** $H_0$ predicts. You pick a test statistic $T$ whose distribution under $H_0$ is known, compute its observed value, and ask how surprising it is. The **alternative** $H_1$ fixes which direction counts as surprising: two sided ($\mu \ne \mu_0$) or one sided ($\mu > \mu_0$), chosen before seeing the data.

The **p value** is the probability, computed assuming $H_0$ and every other model assumption holds, of a test statistic at least as extreme as the one observed. It is not $P(H_0 \mid \text{data})$, not the probability that chance produced the result, and not the probability that you are making an error. Greenland and coauthors (European Journal of Epidemiology, 2016) catalogue 25 such misreadings, and the first two on their list are the first two here. Their own summary: a p value measures how well the data fit what the test hypothesis predicts, under all the assumptions used to compute it.

Under a true null with a continuous statistic, the p value is uniformly distributed on $[0, 1]$, because $P(p \le u) = P(F(T) \ge 1 - u) = u$. That is what makes "reject when $p \le \alpha$" a test with false positive rate $\alpha$. In 20,000 simulated null t-tests, 5.05 percent had $p < 0.05$ and each tenth of $[0, 1]$ held about 2,000 p values. Huyen's book asks for this distribution directly.

**Errors and power.**

| | $H_0$ true | $H_0$ false |
| --- | --- | --- |
| Reject | Type I error, rate $\alpha$ | Correct, rate $1 - \beta$ = power |
| Fail to reject | Correct, rate $1 - \alpha$ | Type II error, rate $\beta$ |

**Significance level** $\alpha$ is chosen in advance and is the Type I rate. **Power** is the probability of rejecting when a specific alternative is true. A test with 80 percent power at a 1 point lift misses that lift one time in five. Power rises with the effect size, the sample size and $\alpha$, and falls with the noise. The standard sample size formula for comparing two means with equal groups is

$$
n_{\text{per group}} = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot 2\sigma^2}{\delta^2} \approx \frac{16\sigma^2}{\delta^2} \text{ at } \alpha = 0.05,\ 1-\beta = 0.8,
$$

since $2(1.96 + 0.84)^2 = 15.7$. For proportions, replace $2\sigma^2$ by $p_1(1-p_1) + p_2(1-p_2)$: detecting 10 percent versus 11 percent needs about 14,748 users per arm, and a simulation at that size rejected 80.7 percent of the time.

Points interviewers push on:

- **Failing to reject is not accepting.** A large p value from a small study is compatible with large effects. Report the interval.
- **Significance is not importance.** With a million users per arm, a 0.01 percent lift can be significant and worthless. With 50 users, a 20 percent lift can be insignificant.
- **The 5 percent false positive rate is long run.** It does not mean a given significant finding has a 5 percent chance of being false. That chance depends on the prior share of true effects and the power, the same base rate logic as the medical test.
- **Peeking inflates Type I error.** Checking a running test every day and stopping at the first $p < 0.05$ produces far more than 5 percent false positives: in a simulation of A/A tests with 10 interim looks, 18.9 percent crossed 0.05 at some look. Fix the sample size, or use sequential methods built for continuous monitoring.
- **Frequentist versus Bayesian.** Frequentists treat the parameter as fixed and the data as random, so probability statements are about procedures. Bayesians treat the parameter as random, put a prior on it, and report a posterior. Yelp and Huyen both ask for this contrast.

> **Say it like this:** "The p value is the chance of seeing data at least this extreme if the null were true. It's a statement about the data assuming the null, not about the null given the data. So p equals 0.03 doesn't mean a 3 percent chance the null is true; to get that I'd need a prior, which is the Bayesian route."

## Choosing a test

Pick by three questions: what kind of outcome (continuous, binary or count, ranked), how many groups, and whether the observations are paired.

| Situation | Test | Statistic and null distribution | Assumptions and caveats |
| --- | --- | --- | --- |
| Mean versus a known value, $\sigma$ known or $n$ large | z-test | $\frac{\bar x - \mu_0}{\sigma/\sqrt n} \sim N(0, 1)$ | Proportions in big A/B tests are z-tests |
| Mean versus a value, $\sigma$ unknown | One sample t-test | $\frac{\bar x - \mu_0}{s/\sqrt n} \sim t_{n-1}$ | Normal data or large $n$ |
| Two independent means, equal variances | Student two sample t-test | Pooled variance, $t_{n_1+n_2-2}$ | Breaks when variances and sizes both differ |
| Two independent means, any variances | Welch t-test | $\frac{\bar x_1 - \bar x_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$, Welch Satterthwaite df | Safe default for two groups |
| Same units measured twice | Paired t-test | One sample t-test on the differences | Pairing removes between unit variance |
| Three or more means | One way ANOVA | $F$ = between group over within group mean square | Equal variances; follow with corrected pairwise tests |
| Two categorical variables, large counts | Chi square test of independence | $\sum (O - E)^2 / E \sim \chi^2_{(r-1)(c-1)}$ | Expected counts of at least 5 per cell |
| 2 by 2 table, small counts | Fisher exact test | Hypergeometric, conditional on margins | Exact; conservative |
| Two independent groups, skewed or ordinal | Mann Whitney U | Rank sum | Tests stochastic ordering, not medians |
| Any statistic, exchangeable under the null | Permutation test | Recompute the statistic on shuffled labels | Needs exchangeability, which randomisation gives |

**Welch by default.** Student's pooled test assumes equal variances. When the smaller group has the larger variance, it underestimates the standard error and its false positive rate explodes; when the larger group has the larger variance, it becomes conservative and loses power. Welch's test costs almost nothing when variances are equal. scipy's `ttest_ind` still defaults to `equal_var=True`, so you must ask for Welch.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
reps = 20_000
small_noisy = rng.normal(0, 3, size=(reps, 10))   # n = 10, sd = 3
large_quiet = rng.normal(0, 1, size=(reps, 40))   # n = 40, sd = 1, same mean

student = stats.ttest_ind(small_noisy, large_quiet, axis=1).pvalue
welch = stats.ttest_ind(small_noisy, large_quiet, axis=1, equal_var=False).pvalue
print("false positive rate, Student:", (student < 0.05).mean())
print("false positive rate, Welch:  ", (welch < 0.05).mean())
```

Both groups have the same mean, so every rejection is a false positive. Student rejected 25.4 percent of the time; Welch rejected 4.9 percent. With the variances swapped (large group noisy), Student's rate fell to 0.1 percent, which is the power loss.

**Paired designs** analyse the differences. Before and after measurements on the same 20 people with a mean drop of 1.5 and individual spread of 8 are hopeless as two independent groups but clear as differences, because the between person variance cancels.

**ANOVA** tests whether all group means are equal. A significant $F$ says some differ, not which; then run pairwise tests with a correction (Tukey's HSD or Holm). Running all pairwise t-tests without correction is the multiple comparisons error.

**Chi square** compares observed counts with the counts expected under independence, $E_{ij} = \text{row}_i \times \text{col}_j / N$. For a 2 by 2 table, scipy's `chi2_contingency` applies Yates' continuity correction by default. On a conversion table of 120 of 1000 against 150 of 1000 it gives $p = 0.058$. For small expected counts, use **Fisher's exact test**, which conditions on both margins and uses the hypergeometric distribution: on the table $\begin{pmatrix} 3 & 7 \\ 9 & 1 \end{pmatrix}$ it gives $p = 0.020$.

**Mann Whitney U** ranks all observations together and compares rank sums. scipy's documentation states its null as equality of the two underlying distributions; it is a test of location only under extra assumptions such as a pure shift. Calling it "a test of medians" is a common error: two distributions with the same median but different shapes can give a significant U.

**Permutation tests** need no distributional assumption. Under the null of no treatment effect, labels are exchangeable, so the observed difference should look like the differences you get by shuffling labels.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
control = rng.exponential(10.0, size=200)
treatment = rng.exponential(11.5, size=200)
observed = treatment.mean() - control.mean()

pooled = np.concatenate([control, treatment])
diffs = np.empty(10_000)
for i in range(diffs.size):
    perm = rng.permutation(pooled)
    diffs[i] = perm[200:].mean() - perm[:200].mean()
p_manual = (np.sum(np.abs(diffs) >= abs(observed)) + 1) / (diffs.size + 1)

res = stats.permutation_test(
    (treatment, control), lambda a, b, axis: a.mean(axis=axis) - b.mean(axis=axis),
    n_resamples=9_999, vectorized=True, rng=rng)
print(f"observed diff {observed:.2f}  manual p {p_manual:.4f}  scipy p {res.pvalue:.4f}")
```

It printed an observed difference of 1.63 with p values of 0.169 (manual) and 0.162 (scipy): a real 15 percent lift in a skewed metric, missed at this sample size. The $+1$ in numerator and denominator counts the observed labelling as one of the permutations and keeps the p value away from an impossible zero.

> **Interview trap:** "My data aren't normal so I need a nonparametric test." For comparing means with moderate samples, the t-test is already justified by the CLT on the mean. The nonparametric test answers a different question (ranks, not means), so switching changes the hypothesis.

The runnable menu, on simulated data:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
a, b, c = rng.normal(10, 2, 30), rng.normal(11, 2, 35), rng.normal(10.5, 2, 25)
before = rng.normal(70, 8, 20)
after = before - rng.normal(1.5, 2, 20)

print("one sample t   ", stats.ttest_1samp(a, popmean=10).pvalue.round(4))
print("Welch t        ", stats.ttest_ind(a, b, equal_var=False).pvalue.round(4))
print("paired t       ", stats.ttest_rel(before, after).pvalue.round(4))
print("one way ANOVA  ", stats.f_oneway(a, b, c).pvalue.round(4))
print("Mann Whitney U ", stats.mannwhitneyu(a, b).pvalue.round(4))
table = np.array([[120, 880], [150, 850]])           # converted / not, per arm
chi2 = stats.chi2_contingency(table)
print("chi square     ", round(chi2.pvalue, 4), "expected min", chi2.expected_freq.min().round(1))
print("Fisher exact   ", round(stats.fisher_exact([[3, 7], [9, 1]]).pvalue, 4))
```

## Multiple comparisons

Run $m$ independent tests at $\alpha = 0.05$ when every null is true and the chance of at least one false positive is $1 - 0.95^m$: 64 percent for $m = 20$. Huyen's stock correlation question is the extreme version: among about 50 million pairs of stocks, some pair will correlate above 0.8 over 24 months by chance alone.

Two error rates to control:

- **Family wise error rate (FWER):** $P(V \ge 1)$, the chance of any false rejection. Right when one false claim is costly, such as a confirmatory medical trial or a launch decision.
- **False discovery rate (FDR):** $E[V / \max(R, 1)]$, the expected share of false rejections among rejections. Right when you screen many hypotheses and will follow up the hits, such as many metrics, genes, or features.

The procedures:

| Method | Rule | Controls | Notes |
| --- | --- | --- | --- |
| Bonferroni | Reject if $p_i \le \alpha/m$ | FWER, any dependence | From the union bound; conservative |
| Šidák | Reject if $p_i \le 1 - (1-\alpha)^{1/m}$ | FWER, independent tests | Barely different from Bonferroni (0.00256 against 0.0025 at $m = 20$) |
| Holm | Sort; reject $p_{(k)}$ while $p_{(k)} \le \alpha/(m - k + 1)$, stop at the first failure | FWER, any dependence | Uniformly more powerful than Bonferroni; no reason to use plain Bonferroni |
| Benjamini Hochberg | Sort; find the largest $k$ with $p_{(k)} \le k\alpha/m$; reject the first $k$ | FDR at $\frac{m_0}{m}\alpha$ under independence or positive dependence | Step up: one small p value can rescue earlier ones |
| Benjamini Yekutieli | BH with $\alpha$ divided by $\sum_{i=1}^m 1/i$ | FDR, any dependence | Conservative |

Holm is a step down procedure (start from the smallest p value, stop at the first failure); BH is step up (start from the largest $k$ that passes and take everything below it). The FDR article on Wikipedia gives the BH bound and the BY harmonic correction.

```python
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

p = np.array([0.001, 0.0065, 0.0075, 0.020, 0.028, 0.20, 0.35, 0.62])
for method in ("bonferroni", "holm", "fdr_bh"):
    reject, p_adj, _, _ = multipletests(p, alpha=0.05, method=method)
    print(f"{method:>10}: rejects {reject.sum()}  adjusted {np.round(p_adj, 3)}")
print("scipy BH:  ", np.round(stats.false_discovery_control(p), 3))
```

Bonferroni rejected 1, Holm 3, BH 5. Walk the BH thresholds $k \times 0.05/8$: 0.00625, 0.0125, 0.01875, 0.025, 0.03125, 0.0375. The fifth p value, 0.028, is under 0.03125, and the sixth, 0.20, fails, so the first five are rejected.

> **Interview trap:** statsmodels' `multipletests` defaults to `method='hs'` (Holm Šidák), not Bonferroni and not BH. Always pass the method. scipy's `false_discovery_control` returns BH adjusted p values by default.

Multiplicity also hides in forms that are not a list of tests: many metrics in one experiment, many segments examined after the fact, many stopping times (peeking), many model variants tried on one test set. The fix is either a correction or a pre registered primary metric with the rest labelled exploratory.

## Maximum likelihood and MAP

The **likelihood** is the probability of the observed data viewed as a function of the parameter: $L(\theta) = p(x_1, \dots, x_n \mid \theta) = \prod_i p(x_i \mid \theta)$ for i.i.d. data. The **maximum likelihood estimate** maximises it, and since $\log$ is increasing, you maximise the log likelihood $\ell(\theta) = \sum_i \log p(x_i \mid \theta)$, which turns products into sums and avoids numerical underflow. Minimising the negative log likelihood is the same thing as minimising cross entropy against the empirical distribution, which is the loss of every classifier and language model.

### Bernoulli derivation

With $k$ successes in $n$ trials, $\ell(p) = k\log p + (n - k)\log(1 - p)$. Setting the derivative to zero:

$$
\frac{k}{p} - \frac{n - k}{1 - p} = 0 \quad\Rightarrow\quad \hat p = \frac{k}{n}.
$$

The second derivative $-k/p^2 - (n-k)/(1-p)^2 < 0$ confirms a maximum. The **Fisher information** per observation is $I(p) = 1/(p(1-p))$, so the asymptotic variance of $\hat p$ is $p(1-p)/n$, matching the binomial standard error.

### Gaussian derivation

For $x_i \sim N(\mu, \sigma^2)$,

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2.
$$

$\partial \ell / \partial \mu = \frac{1}{\sigma^2}\sum_i (x_i - \mu) = 0$ gives $\hat\mu = \bar x$. $\partial \ell / \partial \sigma^2 = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_i (x_i - \mu)^2 = 0$ gives $\hat\sigma^2 = \frac{1}{n}\sum_i (x_i - \bar x)^2$. That estimate is biased: its expectation is $\frac{n-1}{n}\sigma^2$. With $n = 5$ and $\sigma^2 = 4$, the average MLE over 200,000 samples was 3.207 against the theoretical 3.2, while the $n - 1$ version averaged 4.009.

```python
import numpy as np

rng = np.random.default_rng(0)
n, sigma2 = 5, 4.0
samples = rng.normal(0, np.sqrt(sigma2), size=(200_000, n))
mle_var = samples.var(axis=1, ddof=0)        # divide by n
unbiased_var = samples.var(axis=1, ddof=1)   # divide by n - 1
print("E[MLE]", mle_var.mean().round(3), "theory", sigma2 * (n - 1) / n)
print("E[unbiased]", unbiased_var.mean().round(3))
```

Two more that get asked. Exponential with rate $\lambda$: $\ell = n\log\lambda - \lambda\sum x_i$, so $\hat\lambda = 1/\bar x$ (Huyen lists this one). Linear regression with Gaussian noise: CS229 section 1.3 shows the log likelihood is a constant minus $\frac{1}{2\sigma^2}\sum_i (y_i - \theta^\top x_i)^2$, so maximum likelihood is least squares.

### Properties

Under regularity conditions the MLE is consistent, asymptotically normal with variance $1/(nI(\theta))$, and asymptotically efficient (it reaches the Cramér Rao lower bound). It is invariant: the MLE of $g(\theta)$ is $g(\hat\theta)$. It can be biased in finite samples (the Gaussian variance), it can overfit (a bigram model that never saw a pair assigns it probability zero), it can fail to exist or be degenerate (a Gaussian mixture component collapsing onto one point has unbounded likelihood), and it can be unattainable in closed form (logistic regression, mixtures), which is where Newton's method, gradient descent and EM come in.

### MAP

**Maximum a posteriori** estimation adds a prior: $\hat\theta_{\text{MAP}} = \arg\max_\theta \left[\log p(x \mid \theta) + \log p(\theta)\right]$. CS229 section 9.4 frames it as the practical approximation to full Bayesian prediction. The prior becomes a regulariser:

- A Gaussian prior $\theta \sim N(0, \tau^2 I)$ adds $-\frac{1}{2\tau^2}\lVert\theta\rVert^2$, which is L2 regularisation (ridge), with strength $\lambda = \sigma^2/\tau^2$ in regression.
- A Laplace prior adds $-\frac{1}{b}\lVert\theta\rVert_1$, which is L1 (lasso).
- A Beta($\alpha, \beta$) prior on a Bernoulli gives $\hat p_{\text{MAP}} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}$. With $\alpha = \beta = 2$ this is $\frac{k+1}{n+2}$, Laplace smoothing, the fix CS229 uses for words never seen in training. Karpathy's "fake counts" in the bigram model are the same move.

MAP with a flat prior equals the MLE. As $n$ grows the likelihood dominates and MAP converges to the MLE. MAP is not invariant to reparameterisation (the mode of a density moves under a nonlinear change of variables), and it ignores posterior uncertainty, which is what full Bayesian inference keeps.

### Generative models and EM

**Gaussian discriminant analysis** models $p(y)$ as Bernoulli and $p(x \mid y)$ as Gaussian with a shared covariance, fits everything by MLE in closed form (class frequencies, class means, pooled covariance), and classifies with Bayes' rule. CS229 shows the resulting $p(y = 1 \mid x)$ is a logistic function of $x$: GDA implies logistic regression, not the reverse, so GDA is more data efficient when the Gaussian assumption holds and less reliable when it does not.

A **Gaussian mixture** is GDA with the labels hidden. The log likelihood $\sum_i \log \sum_j \phi_j N(x_i; \mu_j, \Sigma_j)$ has a sum inside the log and no closed form. **EM** alternates an E step (compute responsibilities $w_{ij} = p(z_i = j \mid x_i)$ by Bayes' rule) and an M step (the GDA formulas with soft counts). Each iteration does not decrease the likelihood, because the E step makes Jensen's lower bound tight, but EM finds a local maximum and can collapse a component onto a point, so you use several restarts and floor the variances.

> **Watch:** [CS229 Lecture 5: Gaussian Discriminant Analysis](https://www.youtube.com/watch?v=zRdE8A4UZes), [Lecture 9: K-Means and GMM (non-EM)](https://www.youtube.com/watch?v=bSmIGBCoffA) and [Lecture 10: GMM (EM), PCA](https://www.youtube.com/watch?v=sUS-eTa0l6s). For MLE by counting and smoothing as a prior, Karpathy's [makemore part 1 at the negative log likelihood chapter](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3014s) and the [smoothing with fake counts chapter](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3650s); the timestamps are the chapter markers in the video description.

## Bayesian inference and conjugate priors

Bayesian inference treats the parameter as a random variable and updates a prior into a posterior:

$$
p(\theta \mid x) = \frac{p(x \mid \theta)\,p(\theta)}{\int p(x \mid \theta')\,p(\theta')\,d\theta'} \propto p(x \mid \theta)\,p(\theta).
$$

The denominator (the evidence) is the hard part; in general it is an integral with no closed form, which is why practitioners use MCMC, variational inference or a point estimate. A **conjugate prior** is one where the posterior stays in the prior's family, so the update is arithmetic on hyperparameters.

| Likelihood | Prior | Posterior | Prior read as |
| --- | --- | --- | --- |
| Bernoulli or binomial ($k$ of $n$) | Beta($\alpha, \beta$) | Beta($\alpha + k, \beta + n - k$) | $\alpha$ prior successes, $\beta$ prior failures |
| Poisson (counts $x_i$, $n$ windows) | Gamma(shape $a$, rate $b$) | Gamma($a + \sum x_i$, $b + n$) | $a$ events seen in $b$ windows |
| Exponential (times $x_i$) | Gamma($a$, rate $b$) | Gamma($a + n$, $b + \sum x_i$) | $a$ waits summing to $b$ |
| Multinomial (counts $c_j$) | Dirichlet($\boldsymbol\alpha$) | Dirichlet($\boldsymbol\alpha + \mathbf{c}$) | $\alpha_j$ pseudo counts per category |
| Normal, known $\sigma^2$ | Normal($\mu_0, \tau_0^2$) | Precision weighted, below | A prior "sample" with precision $1/\tau_0^2$ |

For the normal with known variance, precisions add and the posterior mean is a precision weighted average:

$$
\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}, \qquad \mu_n = \tau_n^2\left(\frac{\mu_0}{\tau_0^2} + \frac{n\bar x}{\sigma^2}\right).
$$

The posterior mean shrinks the sample mean toward the prior mean, more when $n$ is small or the data are noisy. This is the logic behind empirical Bayes shrinkage of noisy rates, such as ranking sellers by ratings when some have three reviews.

> **Worked example:** A coin of unknown bias, flat prior Beta(1, 1), lands heads 8 times in 12. The posterior is Beta(9, 5): mean $9/14 = 0.643$, mode (MAP) $8/12 = 0.667$, 95 percent credible interval $(0.386, 0.861)$, and $P(p > 0.5 \mid \text{data}) = 0.867$.

```python
from scipy import stats

a0, b0 = 1, 1                 # flat prior
heads, flips = 8, 12
post = stats.beta(a0 + heads, b0 + flips - heads)
lo, hi = post.interval(0.95)
print("posterior mean", round(post.mean(), 3), "MAP", round((a0 + heads - 1) / (a0 + b0 + flips - 2), 3))
print("95% credible interval", round(float(lo), 3), round(float(hi), 3))
print("P(p > 0.5 | data)", round(float(post.sf(0.5)), 3))
```

The posterior mean $\frac{\alpha + k}{\alpha + \beta + n}$ is a weighted average of the prior mean $\frac{\alpha}{\alpha + \beta}$ and the sample proportion $\frac{k}{n}$, with weights $\alpha + \beta$ and $n$. That is the cleanest way to explain what a prior "is worth": Beta(10, 90) is worth 100 observations at a 10 percent rate.

**Posterior predictive.** To predict the next observation, average over the posterior instead of plugging in a point estimate: $p(\tilde x \mid x) = \int p(\tilde x \mid \theta)p(\theta \mid x)\,d\theta$. For the beta binomial, the probability the next flip is heads is the posterior mean. Under a uniform prior after $k$ successes in $n$ trials this gives Laplace's rule of succession, $\frac{k+1}{n+2}$.

**Which coin?** A jar holds one fair coin and one double headed coin; you draw one and see two heads. $P(\text{fair} \mid HH) = \frac{0.5 \cdot 0.25}{0.5 \cdot 0.25 + 0.5 \cdot 1} = 0.2$. With 1 double headed coin among 1000 and 10 heads in a row, the posterior on the double headed coin is $\frac{0.001}{0.001 + 0.999/1024} \approx 0.506$: ten heads only roughly cancel prior odds of 999 to 1.

**Priors matter most when data are scarce.** With enough data any reasonable prior washes out (Bernstein von Mises). A prior that puts zero mass on a region can never be overturned, which is the argument for weakly informative priors over hard constraints.

## The bootstrap

The **bootstrap** estimates the sampling distribution of a statistic by resampling the observed data with replacement. The empirical distribution of the sample stands in for the unknown population; draw $B$ samples of size $n$ from it, compute the statistic on each, and use the spread of those $B$ values as the standard error or to form an interval. Efron introduced it in 1979. It works for statistics with no tidy formula: medians, percentiles, ratios, AUC, the difference in p90 latency between two builds.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = rng.lognormal(mean=3.0, sigma=1.0, size=80)   # skewed, like order values

boot = np.array([np.median(rng.choice(x, size=x.size, replace=True)) for _ in range(10_000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"median {np.median(x):.1f}, percentile CI ({lo:.1f}, {hi:.1f}), bootstrap SE {boot.std(ddof=1):.1f}")

res = stats.bootstrap((x,), np.median, n_resamples=10_000, method="BCa", rng=rng)
print("scipy BCa CI:", np.round(res.confidence_interval, 1))
```

On 80 lognormal order values it gave a median of 22.2, a percentile interval of (16.9, 28.7), a bootstrap standard error of 3.3, and a BCa interval of (17.1, 28.8).

Interval flavours:

- **Percentile:** take the 2.5th and 97.5th percentiles of the bootstrap statistics. Simple, transformation invariant, but biased when the statistic's distribution is skewed or the estimator is biased.
- **Basic (reverse percentile):** $(2\hat\theta - q_{0.975},\ 2\hat\theta - q_{0.025})$, which pivots around the estimate.
- **BCa (bias corrected and accelerated):** adjusts the percentiles for bias and skewness. It is scipy's default (`method='BCa'`, `n_resamples=9999`), and scipy warns and may return NaN when the bootstrap distribution is degenerate.

Where the bootstrap fails or needs care:

- **Dependent data.** Resampling individual rows destroys correlation. Resample whole users (cluster bootstrap) when users have many sessions, or blocks of time for a time series.
- **Extremes and non smooth statistics.** The bootstrap distribution of the sample maximum is wrong: the resampled maximum equals the observed one about 63 percent of the time ($1 - (1 - 1/n)^n$).
- **Very small samples.** The empirical distribution is a poor stand in for the population with 10 points; no resampling scheme invents information.
- **Heavy tails.** For infinite variance data, the bootstrap of the mean is inconsistent, the same failure as the CLT.

The bootstrap resamples to estimate variability around the observed data; a permutation test shuffles labels to simulate the null. Use the bootstrap for intervals and the permutation test for p values. A **parametric bootstrap** simulates from the fitted model instead of the empirical distribution, for example resampling from Geometric($\hat p = 1/\bar x$).

> **Say it like this:** "The bootstrap treats my sample as a stand in for the population. I resample it with replacement many times, recompute the statistic each time, and the spread of those values estimates its standard error. I use it for things like medians or ratio metrics where there's no clean formula, and I resample at the user level if users contribute many rows."

## Causation and data traps

**Correlation versus causation.** An association between $X$ and $Y$ can come from $X$ causing $Y$, $Y$ causing $X$, a common cause (confounder) $Z$ driving both, selection on a common effect, or chance (multiple comparisons). Randomisation breaks every path except the causal one, which is why an experiment answers questions observational data cannot. Without one, you need a causal model and an adjustment strategy, plus a story for why there are no unmeasured confounders. Ice cream sales and drownings both rise in summer; users who use a feature more retain better, partly because engaged users do everything more.

### Simpson's paradox

An association can reverse when you aggregate over a confounder. The kidney stone data reported on Wikipedia:

| | Treatment A | Treatment B |
| --- | --- | --- |
| Small stones | 81/87 = 93.1% | 234/270 = 86.7% |
| Large stones | 192/263 = 73.0% | 55/80 = 68.8% |
| All | 273/350 = 78.0% | 289/350 = 82.6% |

A wins in both strata and loses overall, because doctors gave A to the harder large stone cases. Stone size affects both treatment choice and success. The UC Berkeley admissions case is the same shape: women were admitted at a lower overall rate because they applied more to departments with low admission rates for everyone.

```python
import pandas as pd

df = pd.DataFrame({
    "treatment": ["A", "A", "B", "B"],
    "stone": ["small", "large", "small", "large"],
    "success": [81, 192, 234, 55],
    "total": [87, 263, 270, 80],
})
by_stone = df.assign(rate=df.success / df.total).pivot(index="stone", columns="treatment", values="rate")
overall = df.groupby("treatment")[["success", "total"]].sum()
print(by_stone.round(3))
print((overall.success / overall.total).round(3))
```

Which table is right depends on the causal structure, not on the numbers. If the stratifying variable is a confounder (a cause of treatment and outcome), condition on it. If it is a mediator or a collider (caused by the treatment, or a common effect), conditioning on it introduces bias. Pearl's back door criterion formalises this. In A/B tests, Simpson's paradox appears when the traffic split changes over time or differs across platforms and you pool days with different allocation ratios.

### Regression to the mean

When a measurement is part signal and part noise, units selected for an extreme first measurement will on average be less extreme on the second. For standardised variables with correlation $\rho$, $E[Z_Y \mid Z_X = z] = \rho z$. In a simulation where each score is skill plus equal noise ($\rho = 0.5$), the top decile on test 1 averaged 2.49 and the same people averaged 1.24 on test 2: exactly half, as $\rho$ predicts.

```python
import numpy as np

rng = np.random.default_rng(0)
skill = rng.normal(0, 1, 100_000)
test1 = skill + rng.normal(0, 1, skill.size)
test2 = skill + rng.normal(0, 1, skill.size)
top = test1 > np.quantile(test1, 0.9)
print("top decile on test 1:", test1[top].mean().round(2), "same people on test 2:", test2[top].mean().round(2))
print("correlation of tests:", np.corrcoef(test1, test2)[0, 1].round(2))
```

It explains Kahneman's flight instructors (praise followed by worse landings), the sophomore slump, Galton's tall parents with shorter children, and most "the treatment worked on our worst users" claims. Huyen's kidney failure question is the sibling effect: both the highest and the lowest rates show up in small cities because small samples have high variance. The remedy for evaluation is a control group selected the same way.

### Selection and survivorship bias

**Selection bias** is any process that makes the analysed sample unrepresentative of the population you want to talk about, in a way related to the outcome. **Survivorship bias** is the special case where only units that passed a filter are observed. Abraham Wald's analysis of returning bombers: the holes in planes that came back mark where a plane can be hit and survive, so armour goes where the returning planes were not hit. Funds that closed drop out of performance databases, so surviving funds overstate average returns; Elton, Gruber and Blake (1996) estimated the bias at about 0.9 percent a year for US mutual funds. In product analytics: analysing only users who completed onboarding, training a churn model on users who stayed long enough to be labelled, or backtesting on today's index members.

**Collider bias** (Berkson's paradox) is selection on a common effect. If a company hires people who are either strong coders or strong communicators, the two traits look negatively correlated among employees even when they are independent in the applicant pool.

> **Interview trap:** "Users who saw the recommendation module bought more, so the module works." Exposure was not random: the module shows up for engaged users on long sessions. That is selection, and the honest answer is to propose an experiment or at least a comparison that conditions on engagement.

## Classic probability puzzles

The pattern behind most of them: define states, condition on the first step, and use linearity or symmetry. Every answer below was checked by simulation.

```python
import numpy as np

rng = np.random.default_rng(0)

def flips_until(pattern, trials=20_000):
    total = 0
    for _ in range(trials):
        seq = ""
        while not seq.endswith(pattern):
            seq += "H" if rng.random() < 0.5 else "T"
        total += len(seq)
    return total / trials

print("HH", flips_until("HH"), "HT", flips_until("HT"))

n = 1_000_000
car, pick = rng.integers(0, 3, n), rng.integers(0, 3, n)
print("Monty Hall: stay", (car == pick).mean(), "switch", (car != pick).mean())

days = np.sort(rng.integers(0, 365, (200_000, 23)), axis=1)
print("birthday, 23 people:", (np.diff(days, axis=1) == 0).any(axis=1).mean())

def rolls_to_see_all(faces=6):
    seen, t = set(), 0
    while len(seen) < faces:
        seen.add(int(rng.integers(faces)))
        t += 1
    return t

print("coupon collector, die:", np.mean([rolls_to_see_all() for _ in range(50_000)]))

cuts = np.sort(rng.random((1_000_000, 2)), axis=1)
pieces = np.column_stack([cuts[:, 0], cuts[:, 1] - cuts[:, 0], 1 - cuts[:, 1]])
print("stick makes a triangle:", (pieces.max(axis=1) < 0.5).mean())
```

It printed HH 5.96, HT 3.98, stay 0.333, switch 0.667, birthday 0.508, coupon 14.67 and triangle 0.249.

### Expected flips until HH versus HT

Let $a$ be the expected flips to finish from scratch and $b$ from a state where the last flip was H.

For **HH**: $a = 1 + \frac12 b + \frac12 a$ and $b = 1 + \frac12 \cdot 0 + \frac12 a$, because a T after an H sends you back to the start. Solving gives $a = 6$.

For **HT**: $a = 1 + \frac12 b + \frac12 a$ and $b = 1 + \frac12 \cdot 0 + \frac12 b$, because an H after an H keeps you in the H state. So $b = 2$ and $a = 4$.

Both patterns have probability $1/4$ at any position, yet the waits differ because HH overlaps itself: a failed HH attempt (HT) destroys progress, a failed HT attempt (HH) keeps it. The general rule (Conway's leading numbers, or the martingale argument): for a fair coin, the expected wait is the sum of $2^k$ over every $k$ where the length $k$ prefix equals the length $k$ suffix. HH: $2 + 4 = 6$. HT: $4$. HTH: $2 + 8 = 10$. HHH: $2 + 4 + 8 = 14$. A longer simulation gave 6.00, 4.00, 10.00 and 14.07.

### Monty Hall

You pick one of three doors; the host, who knows where the car is and always opens a goat door you did not pick, opens one and offers a switch. Switching wins with probability $2/3$. Your first pick is right with probability $1/3$ and the host's action does not change that, so the remaining door carries the other $2/3$. With Bayes: if you pick door 1 and the host opens door 3, $P(\text{car at 2} \mid \text{opens 3}) = \frac{1 \cdot \frac13}{\frac12 \cdot \frac13 + 1 \cdot \frac13 + 0} = \frac23$. The host's knowledge is load bearing: if the host opens a random other door and it happens to show a goat, switching wins only $1/2$ (simulated 0.4997).

### Birthday problem

With $n$ people and 365 equally likely birthdays, $P(\text{no match}) = \prod_{i=0}^{n-1}\frac{365 - i}{365}$. This drops below one half at $n = 23$, where $P(\text{match}) = 0.507$ (0.476 at 22, 0.970 at 50). It is low because there are $\binom{23}{2} = 253$ pairs, and $1 - e^{-253/365} \approx 0.50$. The different question "someone shares my birthday" needs $1 - (364/365)^n \ge 0.5$, which takes $n = 253$. The same arithmetic sizes hash collisions: with $m$ buckets, a collision becomes likely around $\sqrt{m}$ keys.

### Coupon collector

Expected draws to see all $n$ types: once you have $i$ types, each draw is new with probability $\frac{n-i}{n}$, so the wait is geometric with mean $\frac{n}{n-i}$. Summing by linearity, $E[T] = n\sum_{k=1}^{n}\frac1k = nH_n \approx n\ln n + 0.577n + \frac12$. A die needs $6H_6 = 14.7$ rolls; 50 coupons need 224.96. The variance is $n^2\sum 1/k^2 - nH_n$, 38.99 for a die, so the wait is very spread out.

### Drawing balls

An urn has $r$ red and $b$ blue balls, drawn without replacement.

- Probability the $k$th draw is red: $r/(r+b)$ for every $k$, by exchangeability.
- Expected number of blue balls before the first red: each blue ball is equally likely to fall in any of the $r + 1$ gaps the red balls create, and it precedes all reds only in the first gap, so the answer is $b/(r+1)$. With 3 red and 5 blue: 1.25 (simulated 1.24).
- Two queens from a deck, same suit, two aces: count with combinations as in the counting section.

### Random walks

A symmetric walk starting at $k$ with absorbing barriers at 0 and $N$ (gambler's ruin) reaches $N$ first with probability $k/N$, because the position is a martingale, and the expected duration is $k(N - k)$. Starting at 3 with $N = 10$: probability 0.3 and 21 steps (simulated 0.299 and 20.98). For a biased walk with up probability $p \ne \frac12$ and $q = 1 - p$, $P(\text{reach } N) = \frac{1 - (q/p)^k}{1 - (q/p)^N}$. After $n$ unbiased steps $E[S_n^2] = n$ and $E|S_n| \approx \sqrt{2n/\pi}$ (7.98 at $n = 100$, simulated 7.94). The one dimensional walk returns to the origin with probability 1, but the expected return time is infinite.

### A stick broken in three

Break a unit stick at two independent uniform points. The pieces form a triangle when every piece is shorter than $1/2$ (the triangle inequality). In the unit square of cut points, the three conditions carve out a region of area $1/4$: by symmetry assume $x < y$; you need $x < \frac12$, $y > \frac12$ and $y - x < \frac12$, a triangle of area $1/8$ in the half square of area $1/2$. Cut the Knot's equilateral triangle argument (Viviani's theorem) gives the same $1/4$. The procedure matters: break once, then break the longer piece at a uniform point, and the answer is $2\ln 2 - 1 \approx 0.386$ (simulated 0.387).

### More that come up

- **Fair coin from a biased coin** (von Neumann): flip twice, call HT heads and TH tails, repeat on HH or TT. Each round succeeds with probability $2p(1-p)$, so expected flips are $1/(p(1-p))$, 4.76 at $p = 0.3$.
- **Expected maximum of two dice**: $\sum_k P(\max \ge k) = 161/36 \approx 4.47$. The maximum of $n$ uniforms on $[0,1]$ has mean $n/(n+1)$.
- **German tank problem**: from $k$ serial numbers with maximum $m$, the minimum variance unbiased estimate of the total is $m + m/k - 1$. With 300 tanks and 5 observed, the raw maximum averaged 251; the corrected estimator averaged 300.2.
- **Ants on a triangle**: three ants each pick a direction at random; no collision only if all go the same way, $2/8 = 1/4$.
- **Days until a standard normal draw exceeds 0.5**: geometric with $p = P(Z > 0.5) = 0.3085$, mean 3.24 days.
- **Classifier wrong 10 of 100 times, next 20 all correct**: $0.9^{20} = 0.12$, assuming independence, which a real classifier violates.

> **Say it like this:** "For HH the trick is that a tail after a head throws away my progress, so I set up two states, start and just saw a head, and condition on the next flip. That gives 6. For HT a second head doesn't hurt me, I'm still one tail away, so it's only 4. Same per position probability, different overlap structure."

## What interviewers probe

- **Bayes.** You give $P(D \mid +)$ for the base rate example. Then: what about two positive tests? What assumption does that need? What if the tests are run on the same blood sample? What happens to the answer as prevalence rises to 10 percent?
- **Expectation.** You solve a puzzle by linearity. Then: are the indicators independent? Does it matter? What is the variance (which does need the covariances)?
- **Distributions.** You name the Poisson for counts. Then: why does mean equal variance? Your data have variance three times the mean, now what (negative binomial, or a gamma mixed Poisson)? What is the waiting time distribution between events, and why is it memoryless?
- **CLT.** You justify a t-test on revenue by the CLT. Then: how big does $n$ need to be with a skewness of 10? What if one customer is 30 percent of revenue? What breaks it completely?
- **Confidence intervals.** You give the procedure reading. Then: so what does the interval tell me about this experiment? How is a credible interval different? Two intervals overlap; is the difference insignificant?
- **p values.** You define it correctly. Then: what is its distribution under the null? Under the alternative? Why does peeking break it? Why can a huge sample make everything significant?
- **Tests.** You pick Welch. Then: why not Student? When would you prefer Mann Whitney, and what does it actually test? How would you test independence of two categorical variables, and what if a cell has 2 expected counts?
- **Multiplicity.** You correct with Bonferroni. Then: why is Holm always at least as good? When would you want FDR instead? Does BH hold under dependence?
- **MLE.** You derive $\hat p = k/n$. Then: what if $k = 0$? Add a prior; which prior gives Laplace smoothing? Why is the Gaussian variance MLE biased and does it matter? What is the asymptotic variance?
- **Bayesian.** You update a beta prior. Then: what is the prior worth in observations? What is the posterior predictive? When does the prior stop mattering?
- **Bootstrap.** You bootstrap a median. Then: how would you bootstrap with repeated users? Why does it fail for the maximum? Bootstrap versus permutation test?
- **Causal traps.** You spot Simpson's paradox. Then: which table do you trust and why? How would you design the analysis so the paradox cannot happen? What is the difference between conditioning on a confounder and on a collider?
- **Puzzles.** You get the answer. Then: generalise (HTH, $n$ coupons, $N$ doors, biased walk), and check it with a quick simulation in code.

## Sources

- [CS229 Probability Theory Review and Reference (2022)](https://cs229.stanford.edu/notes2022fall/cs229-probability_review.pdf)
- [CS229 Lecture Notes, Ma and Ng (Spring 2026)](https://cs229.stanford.edu/main_notes.pdf)
- [Huyen, Machine Learning Interviews: statistics questions](https://huyenchip.com/ml-interviews-book/contents/5.2.2-stats.html)
- [Huyen, Machine Learning Interviews: probability questions](https://huyenchip.com/ml-interviews-book/contents/5.2.1.2-questions.html)
- [Huyen, Machine Learning Interviews: probability concepts](https://huyenchip.com/ml-interviews-book/contents/5.2.1-probability.html)
- [120 Data Science Interview Questions: probability](https://github.com/JifuZhao/120-DS-Interview-Questions/blob/master/probability.md)
- [120 Data Science Interview Questions: statistical inference](https://github.com/JifuZhao/120-DS-Interview-Questions/blob/master/statistical-inference.md)
- [StrataScratch: 30 probability and statistics interview questions](https://www.stratascratch.com/blog/30-probability-and-statistics-interview-questions-for-data-scientists)
- [TechInterview: 12 classic probability brainteasers](https://www.techinterview.org/post/3233474735/probability-brainteaser-walkthroughs-12-classics/)
- [QuantVault: top 50 quant interview questions](https://quantvault.org/top-50-quant-interview-questions.html)
- [Greenland et al. 2016, Statistical tests, P values, confidence intervals, and power: a guide to misinterpretations](https://pmc.ncbi.nlm.nih.gov/articles/PMC4877414/)
- [scipy.stats.ttest_ind documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
- [scipy.stats.bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [scipy.stats.mannwhitneyu documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
- [statsmodels multipletests documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)
- [Wikipedia: Simpson's paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox)
- [Wikipedia: Coupon collector's problem](https://en.wikipedia.org/wiki/Coupon_collector%27s_problem)
- [Wikipedia: False discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate)
- [Wikipedia: Conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)
- [Wikipedia: Regression toward the mean](https://en.wikipedia.org/wiki/Regression_toward_the_mean)
- [Wikipedia: Survivorship bias](https://en.wikipedia.org/wiki/Survivorship_bias)
- [Cut the Knot: stick broken into three pieces](https://www.cut-the-knot.org/Curriculum/Probability/TriProbability.shtml)
