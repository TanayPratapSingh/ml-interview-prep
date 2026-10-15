# Classical machine learning: the deep version

Everything an interviewer can ask about machine learning before neural networks: how to frame a problem, the bias variance decomposition and its derivation, regularisation and its Bayesian reading, linear and logistic regression from first principles, generalised linear models, generative classifiers, kNN, SVMs and kernels, trees and their ensembles, clustering, dimensionality reduction, evaluation, validation and leakage, imbalance, features, interpretability, tuning and time series. Derivations follow the Stanford CS229 Spring 2026 notes. Every number in a worked example came from running the code shown, with numpy, scipy, scikit-learn and statsmodels. The perceptron is the last model covered here; neural networks have their own module.

## Framing a problem as ML

A machine learning problem is defined by four choices: what you predict (the target), for which unit (a user, a transaction, a house), at what moment (what information exists when the prediction is made), and how the prediction will be used (the decision it feeds and the cost of each kind of error). Get these four wrong and no model choice rescues you.

### The learning settings

- **Supervised learning.** You have pairs $(x^{(i)}, y^{(i)})$ and learn a hypothesis $h: \mathcal{X} \to \mathcal{Y}$. When $y$ is continuous it is regression; when $y$ takes a small number of discrete values it is classification. Training minimises an empirical risk, the average loss over the training set, as a stand in for the population risk $L(\theta) = \mathbb{E}_{(x,y)\sim D}[\ell(h_\theta(x), y)]$. It is called empirical because the expectation over the true distribution $D$ is replaced by an average over the sample.
- **Unsupervised learning.** Only $x$ is observed. The tasks are clustering (k-means, GMMs, DBSCAN), density estimation (GMM via EM) and dimensionality reduction (PCA). There is no ground truth to score against, which is why choosing k and validating clusters is hard.
- **Self supervised learning.** The labels are manufactured from the input itself: predict the next token, fill in a masked patch, or pull two augmented views of the same image together while pushing other images apart (contrastive learning). The resulting representation is then reused by a linear probe or by fine tuning. CS229 Spring 2026 covers this under representation learning and foundation models.
- **Semi supervised, weakly supervised and active learning.** Semi supervised uses a few labels plus many unlabelled points. Weak supervision uses noisy, programmatic or partial labels. Active learning lets the model choose which points a human labels next, usually the most uncertain ones.
- **Reinforcement learning.** An agent picks actions in states, receives rewards, and learns a policy that maximises expected discounted return. The feedback is evaluative (a reward, never the right answer) and often delayed, and the data distribution depends on the policy itself, so the iid assumption behind everything else in this module breaks.

> **Watch:** [CS229 Lecture 1](https://www.youtube.com/watch?v=DATnpGoGhM8) for the course map, [CS229 Lecture 2](https://www.youtube.com/watch?v=cmNIMjPYdgM) for the supervised setup, [CS229 Lecture 12](https://www.youtube.com/watch?v=_kREM2UAiJ8) for representation learning, and [CS229 Lecture 16](https://www.youtube.com/watch?v=hHC-SF3utxg) for RL and policy gradient.

### Parametric and nonparametric

A parametric model has a fixed number of parameters regardless of $n$ (linear and logistic regression). A nonparametric model keeps an amount of information that grows with the training set: kNN and locally weighted regression must store the data, and a fully grown tree can have up to $n$ leaves. CS229 introduces the term with locally weighted linear regression, where every prediction refits a weighted least squares problem around the query point.

### A framing checklist you can say out loud

1. Write the label definition precisely, including the time window ("churn" means no purchase in the 30 days after the prediction date).
2. Fix the prediction time, then list only features known before it. This one step prevents most leakage.
3. Choose the metric from the decision: ranking quality (AUC), a top k budget (precision at k), calibrated risk (log loss), or a cost matrix.
4. Build the dumbest baseline first: the majority class, the mean, last value for a time series, or a logistic regression.
5. Decide the split from the deployment pattern: random for iid rows, by group for repeated entities, by time for anything that forecasts.

> **Interview trap:** Classification and regression are interchangeable more often than candidates admit. You can bin a continuous target into classes, and you can regress on a probability. The question is which loss matches the decision. Predicting days until churn as a regression ignores censoring (customers who have not churned yet); a survival model or a fixed horizon classifier handles it.

## Bias and variance

Test error for squared loss splits exactly into irreducible noise, squared bias and variance. Bias is the error you would still have with infinite data from the same model family; variance is how much the fitted model moves when you redraw the training set.

### The derivation

Follow the CS229 setup. Data come from $y = h^\star(x) + \xi$ with $\xi \sim \mathcal{N}(0, \sigma^2)$ independent of everything. Draw a training set $S$, fit $\hat h_S$, and evaluate at a fixed test input $x$ with a fresh noise draw. The expected test error is

$$
\mathrm{MSE}(x) = \mathbb{E}_{S,\xi}\big[(y - \hat h_S(x))^2\big].
$$

The only tool needed: if $A$ and $B$ are independent and $\mathbb{E}[A] = 0$, then $\mathbb{E}[(A+B)^2] = \mathbb{E}[A^2] + \mathbb{E}[B^2]$ because the cross term $2\mathbb{E}[A]\mathbb{E}[B]$ vanishes.

Step 1. Take $A = \xi$ and $B = h^\star(x) - \hat h_S(x)$. The test noise is independent of $S$, so

$$
\mathrm{MSE}(x) = \sigma^2 + \mathbb{E}_S\big[(h^\star(x) - \hat h_S(x))^2\big].
$$

Step 2. Define the average model $h_{\mathrm{avg}}(x) = \mathbb{E}_S[\hat h_S(x)]$. Take $c = h^\star(x) - h_{\mathrm{avg}}(x)$, a constant, and $A = h_{\mathrm{avg}}(x) - \hat h_S(x)$, which has mean zero over $S$. Then

$$
\mathrm{MSE}(x) = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{\big(h^\star(x) - h_{\mathrm{avg}}(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_S\big[(\hat h_S(x) - h_{\mathrm{avg}}(x))^2\big]}_{\text{variance}}.
$$

Averaging over the test distribution of $x$ gives the same decomposition for expected test MSE. Three things interviewers check: the expectation is over training sets, not over parameters; the noise term is a floor no model can beat; and the decomposition is exact only for squared loss. CS229 notes that for classification there is still no agreed decomposition.

> **Worked example:** Fit polynomials to $y = \sin(2\pi x) + \xi$ with $\sigma = 0.3$ and $n = 30$, then measure everything at $x_0 = 0.25$ across 4,000 redrawn training sets. Degree 1: bias squared 0.2647, variance 0.0196. Degree 3: bias squared 0.0001, variance 0.0108. Degree 9: bias squared about 0, variance 0.0402. Adding noise 0.09 to each sum gives 0.3743, 0.1009 and 0.1302, matching the directly simulated test MSE of 0.3706, 0.1014 and 0.1341 to Monte Carlo error.

```python
import numpy as np

rng = np.random.default_rng(2)
f = lambda x: np.sin(2 * np.pi * x)
sigma, n, x0, R = 0.3, 30, 0.25, 4000
for deg in [1, 3, 9]:
    preds, errs = np.empty(R), np.empty(R)
    for r in range(R):
        x = rng.uniform(0, 1, n)
        y = f(x) + rng.normal(scale=sigma, size=n)
        preds[r] = np.polyval(np.polyfit(x, y, deg), x0)
        errs[r] = (f(x0) + rng.normal(scale=sigma) - preds[r]) ** 2
    bias2, var = (preds.mean() - f(x0)) ** 2, preds.var()
    print(f"deg {deg}: bias2 {bias2:.4f} var {var:.4f} "
          f"sum {bias2 + var + sigma**2:.4f} simulated {errs.mean():.4f}")
```

### Underfitting, overfitting and learning curves

Underfitting means training error is high (well above the noise level), and validation error is similar. Overfitting means training error is low and validation error is much higher. A learning curve plots both errors against training set size:

| Pattern | Training error | Validation error | Gap | What helps |
| --- | --- | --- | --- | --- |
| High bias | High, rises slightly with n | High, flattens quickly | Small | Richer features, less regularisation, a more flexible model |
| High variance | Low | High, keeps falling with n | Large | More data, stronger regularisation, simpler model, bagging, early stopping |
| Good fit | Near noise floor | Near training error | Small | Stop |

More data only cures variance. If both curves have converged to the same high value, collecting data is wasted money. `sklearn.model_selection.learning_curve` produces these curves with cross validation at each size.

### Double descent

The U shaped test error curve is the classical regime. CS229 Spring 2026 also covers double descent: as parameters grow past the number of examples, test error can peak near $n \approx d$ and then fall again. The peak comes from a nearly singular fit at the interpolation threshold, where the minimum norm solution has a huge norm. Tuned ridge regularisation largely removes the peak, and gradient descent from zero initialisation is itself an implicit regulariser because it converges to the minimum norm interpolating solution. Measured by the norm of the fitted parameters instead of their count, the curve looks classical again.

> **Watch:** [CS229 Lecture 6](https://www.youtube.com/watch?v=llnEgyyuYkQ) on dataset splits and ML advice, which is the practical side of this section.

> **Say it like this:** "Expected squared error at a point is noise plus squared bias plus variance. Bias is how far the average fitted model sits from the truth, variance is how much the fit wobbles across training sets. I diagnose which one dominates from the learning curve: a big train validation gap is variance, two high converged curves is bias."

## Linear regression

Linear regression models $\mathbb{E}[y \mid x] = \theta^\top x$ (with $x_0 = 1$ for the intercept) and fits $\theta$ by minimising squared error. It has a closed form solution, a probabilistic justification, a geometric meaning and a set of assumptions, and interviewers ask about all four.

### Normal equations

Stack the inputs as rows of the design matrix $X \in \mathbb{R}^{n \times (d+1)}$ and the targets as $\vec y$. The cost is $J(\theta) = \tfrac12 (X\theta - \vec y)^\top (X\theta - \vec y)$. Expand and differentiate, using $\nabla_\theta\, b^\top\theta = b$ and $\nabla_\theta\, \theta^\top A \theta = 2A\theta$ for symmetric $A$:

$$
\nabla_\theta J = \tfrac12 \nabla_\theta\big(\theta^\top X^\top X \theta - 2 (X^\top \vec y)^\top \theta + \vec y^\top \vec y\big) = X^\top X \theta - X^\top \vec y.
$$

Setting this to zero gives the normal equations $X^\top X\theta = X^\top \vec y$, so $\hat\theta = (X^\top X)^{-1} X^\top \vec y$ when $X^\top X$ is invertible. $J$ is a convex quadratic with Hessian $X^\top X \succeq 0$, so this stationary point is the global minimum.

Geometrically, the normal equations say $X^\top(\vec y - X\hat\theta) = 0$: the residual vector is orthogonal to every column of $X$. The fitted values $\hat y = H\vec y$ with $H = X(X^\top X)^{-1}X^\top$ are the orthogonal projection of $\vec y$ onto the column space. $H$ is called the hat matrix; it is symmetric, idempotent, and its trace equals the number of parameters $p$.

### Why squared error: the Gaussian likelihood

Assume $y^{(i)} = \theta^\top x^{(i)} + \epsilon^{(i)}$ with $\epsilon^{(i)}$ iid $\mathcal{N}(0, \sigma^2)$. The log likelihood is

$$
\ell(\theta) = n \log \frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2}\sum_{i=1}^n \big(y^{(i)} - \theta^\top x^{(i)}\big)^2,
$$

so maximising likelihood is minimising squared error, and the answer does not depend on $\sigma^2$. CS229 stresses that the Gaussian story is one justification, not a requirement: least squares is a reasonable procedure without it. Laplace noise would give least absolute deviations instead.

### Gradient descent and when to use it

Batch gradient descent updates $\theta := \theta - \alpha X^\top(X\theta - \vec y)/n$; stochastic gradient descent uses one example (the LMS or Widrow Hoff rule $\theta := \theta + \alpha (y^{(i)} - \theta^\top x^{(i)}) x^{(i)}$). Because $J$ is convex there are no bad local minima; the step size must stay below $2/\lambda_{\max}$ of the Hessian or the iterates diverge. Cost comparison: the normal equations cost $O(nd^2 + d^3)$ once; gradient descent costs $O(nd)$ per pass. Use the closed form (via QR or SVD, never an explicit inverse) for $d$ up to a few thousand, and iterative methods when $d$ is huge, data stream in, or $X$ is sparse.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 200
X = np.c_[np.ones(n), rng.normal(size=(n, 3))]
y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(scale=0.5, size=n)

b_normal = np.linalg.solve(X.T @ X, X.T @ y)       # normal equations
b_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]     # QR or SVD, preferred in practice
b_gd = np.zeros(4)
for _ in range(5000):                               # batch gradient descent on MSE / 2
    b_gd -= 0.1 * X.T @ (X @ b_gd - y) / n
print(b_normal.round(4), b_lstsq.round(4), b_gd.round(4))
print("X'(y - Xb) = 0:", np.allclose(X.T @ (y - X @ b_normal), 0))
H = X @ np.linalg.solve(X.T @ X, X.T)               # hat matrix
print("trace(H) =", round(np.trace(H), 6), " max leverage", round(H.diagonal().max(), 4))
```

All three solvers return $[0.9819, 1.955, -1.0481, 0.5265]$, the residuals are orthogonal to $X$, and the trace of $H$ is exactly 4.

### OLS assumptions and what breaks when each fails

| Assumption | If it fails |
| --- | --- |
| Linear in parameters (features may be transformed) | Systematic bias; residual plots show curvature |
| Exogeneity: $\mathbb{E}[\epsilon \mid X] = 0$ | Coefficients are biased and inconsistent (omitted variable bias, reverse causality, measurement error in $x$) |
| No perfect multicollinearity (full column rank) | $X^\top X$ is singular, the solution is not unique |
| Homoscedasticity: constant $\mathrm{Var}(\epsilon \mid X)$ | Coefficients stay unbiased, but standard errors and tests are wrong; OLS is no longer the most efficient |
| No autocorrelation of errors | Same as above, common in time series and panels |
| Normal errors | Only needed for exact small sample t and F tests; the central limit theorem covers large $n$ |

Under the first five (the Gauss Markov conditions) OLS is the best linear unbiased estimator. Normality is the assumption candidates name first and the one that matters least.

> **Interview trap:** "Linear regression assumes the features are normally distributed." It does not. It makes no distributional assumption about $x$ at all, and the normality assumption, when used, is about the errors given $x$.

### Multicollinearity

Interviewers ask: you have columns $x$, $z$ and $w = x + z$; what happens? $X^\top X$ is rank deficient (rank 3 instead of 4 with an intercept, condition number around $10^{16}$ when computed), so infinitely many coefficient vectors give the same fit. Add a little noise to $w$ and the matrix becomes invertible but ill conditioned: coefficients get huge variances, flip sign between bootstrap samples, and the variance inflation factors $\mathrm{VIF}_j = 1/(1 - R_j^2)$, where $R_j^2$ comes from regressing feature $j$ on the others, run into the hundreds. Predictions inside the data range are usually fine; the individual coefficients are what become meaningless. Remedies: drop or combine features, ridge regression (which adds $\lambda I$ and makes the matrix well conditioned), or PCA regression.

### Heteroscedasticity and weighted least squares

When error variance changes with $x$, OLS coefficients remain unbiased but the usual standard errors are wrong, usually too small where variance is high. Diagnose with a residual versus fitted plot (a funnel) or a Breusch Pagan test. Fix the inference with heteroscedasticity consistent (sandwich) standard errors such as HC3, or fix the estimator with weighted least squares.

Weighted least squares minimises $\sum_i w_i (y^{(i)} - \theta^\top x^{(i)})^2$, giving $\hat\theta = (X^\top W X)^{-1} X^\top W \vec y$ with $W = \mathrm{diag}(w)$. It is the maximum likelihood estimator when $\mathrm{Var}(\epsilon_i) = \sigma^2 / w_i$, so the right weight is the inverse variance.

```python
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

rng = np.random.default_rng(1)
x = rng.uniform(1, 10, 500)
y = 2 + 3 * x + rng.normal(scale=0.5 * x)          # noise sd grows with x
X = sm.add_constant(x)
ols = sm.OLS(y, X).fit()
print("Breusch-Pagan p:", f"{het_breuschpagan(ols.resid, X)[1]:.1e}")
print("OLS slope", ols.params[1].round(3), "naive se", ols.bse[1].round(4),
      "HC3 se", ols.get_robustcov_results("HC3").bse[1].round(4))
wls = sm.WLS(y, X, weights=1 / x**2).fit()         # weight = 1 / variance
print("WLS slope", wls.params[1].round(3), "se", wls.bse[1].round(4))
```

The test rejects homoscedasticity with p around $2 \times 10^{-22}$. OLS gives slope 3.054 with a naive standard error of 0.0556 that is too small (HC3 says 0.0635). WLS with the correct weights gives slope 3.052 and a genuinely smaller standard error of 0.0406, because it stops trusting the noisy high $x$ points as much.

Locally weighted linear regression, the CS229 Lecture 3 topic, is weighted least squares refitted for every query point $x$ with $w^{(i)} = \exp\!\big(-\lVert x^{(i)} - x\rVert^2 / (2\tau^2)\big)$. The bandwidth $\tau$ is a bias variance knob: small $\tau$ follows the data closely (low bias, high variance), large $\tau$ approaches a single global line. It is nonparametric because the whole training set must be kept.

> **Watch:** [CS229 Lecture 3](https://www.youtube.com/watch?v=uJF_gL3jhxI) on weighted least squares and locally weighted regression.

### Interpreting coefficients

$\hat\theta_j$ is the change in the predicted mean of $y$ for a one unit increase in $x_j$, holding the other features fixed. Translations interviewers expect: with $\log y$ as the target, $100\,\theta_j$ is roughly the percent change in $y$ per unit of $x_j$ (exactly $100(e^{\theta_j} - 1)$); with both logged, $\theta_j$ is an elasticity; a dummy coefficient is the difference from the reference category; with an interaction $x_1 x_2$, the effect of $x_1$ depends on $x_2$. Coefficient size is not importance unless features share a scale, and none of this is causal unless exogeneity holds.

### Leverage points, outliers and influence

The leverage score of point $i$ is $h_{ii}$, the $i$th diagonal entry of the hat matrix. It depends only on $x$: it measures how far $x^{(i)}$ sits from the centre of the data. Leverage values lie in $[0, 1]$, sum to $p$, and average $p/n$; a common flag is $h_{ii} > 2p/n$. An outlier has a large residual; a high leverage point has an unusual $x$; an influential point is both, and moves the fit when removed. Cook's distance combines them:

$$
D_i = \frac{e_i^2}{p\, s^2} \cdot \frac{h_{ii}}{(1 - h_{ii})^2},
$$

with $e_i$ the residual and $s^2$ the residual variance. Because $\mathrm{Var}(e_i) = \sigma^2(1 - h_{ii})$, high leverage points have small residuals by construction, which is why you check studentised residuals and Cook's distance instead of raw residuals.

### Goodness of fit

$R^2 = 1 - \mathrm{SSR}/\mathrm{SST}$ is the share of variance explained relative to predicting the mean. It never decreases when you add a feature, so adjusted $R^2 = 1 - (1 - R^2)\frac{n-1}{n-p}$ (with $p$ counting the intercept) penalises extra parameters. On a test set $R^2$ can be negative, which just means the model is worse than the mean.

## Regularisation

Regularisation adds a penalty to the training loss, $J_\lambda(\theta) = J(\theta) + \lambda R(\theta)$, trading a little bias for a large cut in variance. Standardise features first (the penalty is not scale invariant) and never penalise the intercept.

### Ridge, lasso and elastic net

- **Ridge (L2):** $R = \lVert\theta\rVert_2^2$, closed form $\hat\theta = (X^\top X + \lambda I)^{-1}X^\top \vec y$, invertible for any $\lambda > 0$ even when $d > n$ or features are collinear. In the SVD basis it multiplies the OLS component along singular direction $j$ by $s_j^2/(s_j^2 + \lambda)$, shrinking low variance directions most. In gradient descent it is weight decay: each step multiplies $\theta$ by $(1 - \eta\lambda)$ before the usual gradient step.
- **Lasso (L1):** $R = \lVert\theta\rVert_1$, no closed form in general (solved by coordinate descent), and it sets coefficients exactly to zero, so it does feature selection.
- **Elastic net:** $\lambda\big(\rho\lVert\theta\rVert_1 + \tfrac{1-\rho}{2}\lVert\theta\rVert_2^2\big)$. With a group of correlated useful features, lasso tends to keep one arbitrarily; the L2 part makes elastic net keep and share weight across the group, and it stays well defined when $d > n$.

### Why L1 gives exact zeros

Three arguments, from most visual to most rigorous.

1. **Geometry.** The penalised problem equals the constrained problem $\min J(\theta)$ subject to $\lVert\theta\rVert_1 \le t$. The L1 ball is a diamond (a cross polytope) whose corners lie on the axes. The elliptical contours of $J$ usually first touch it at a corner or an edge, where some coordinates are exactly zero. The L2 ball is round, the touching point is generic, and no coordinate is zero.
2. **Orthonormal design.** If $X^\top X = I$, the problem separates by coordinate. With OLS estimate $z_j$ and objective $\tfrac12\lVert \vec y - X\theta\rVert^2 + \lambda\lVert\theta\rVert_1$, the lasso solution is soft thresholding, $\hat\theta_j = \mathrm{sign}(z_j)\max(|z_j| - \lambda, 0)$, while ridge with $\lambda\lVert\theta\rVert_2^2/2$ gives $z_j/(1+\lambda)$: proportional shrinkage that never reaches zero.
3. **Subgradients.** At $\theta_j = 0$ the subdifferential of $\lambda|\theta_j|$ is the whole interval $[-\lambda, \lambda]$, so zero stays optimal whenever the loss gradient there has magnitude at most $\lambda$. The L2 penalty has gradient zero at zero, so only an exactly zero loss gradient would keep a coefficient at zero.

```python
import numpy as np
from sklearn.linear_model import Lasso

rng = np.random.default_rng(0)
n = 200
Q, _ = np.linalg.qr(rng.normal(size=(n, 4)))       # orthonormal columns
y = Q @ np.array([3.0, -0.4, 0.05, 1.2]) + rng.normal(scale=0.1, size=n)
z = Q.T @ y                                         # OLS coefficients when Q'Q = I
lam = 0.5
soft = np.sign(z) * np.maximum(np.abs(z) - lam, 0)  # closed form lasso
# sklearn minimises (1/2n)||y - Xb||^2 + alpha ||b||_1, so alpha = lam / n
lasso = Lasso(alpha=lam / n, fit_intercept=False, tol=1e-10).fit(Q, y)
print("OLS   ", z.round(3))
print("soft  ", soft.round(3))
print("sklearn lasso", lasso.coef_.round(3))
print("ridge, lambda 1:", (z / (1 + 1.0)).round(3))
```

OLS gives $[3.099, -0.429, 0.129, 1.176]$. Soft thresholding at 0.5 and sklearn's lasso agree on $[2.599, 0, 0, 0.676]$: the two small coefficients are exactly zero and the large ones move down by exactly 0.5. Ridge halves everything and zeroes nothing. Note the scaling trap in the comment: sklearn divides the squared error by $2n$.

### The Bayesian view

Put a prior on the parameters and take the posterior mode (MAP). With Gaussian noise of variance $\sigma^2$ and prior $\theta \sim \mathcal{N}(0, \tau^2 I)$,

$$
\hat\theta_{\mathrm{MAP}} = \arg\min_\theta \frac{1}{2\sigma^2}\lVert \vec y - X\theta\rVert^2 + \frac{1}{2\tau^2}\lVert\theta\rVert^2,
$$

which is ridge with $\lambda = \sigma^2/\tau^2$: a tighter prior means stronger shrinkage. A Laplace prior $p(\theta_j) \propto e^{-|\theta_j|/b}$ gives the lasso. CS229 section 9.4 presents the Gaussian case and notes that MAP estimates have smaller norm than the MLE. The expert follow up: sparsity is a property of the mode. The posterior mean under a Laplace prior is not sparse, which is why the fully Bayesian lasso does not zero out coefficients.

### Choosing lambda

Use cross validation over a log spaced grid (`RidgeCV`, `LassoCV`, `ElasticNetCV`). The one standard error rule picks the largest $\lambda$ whose CV error is within one standard error of the minimum, giving a simpler model at a statistically indistinguishable error.

## Logistic regression

Logistic regression models the log odds as linear, $\log\frac{p}{1-p} = \theta^\top x$, so $p(y=1 \mid x) = \sigma(\theta^\top x) = 1/(1 + e^{-\theta^\top x})$, and fits $\theta$ by maximum likelihood. It is a linear classifier because the decision boundary $\theta^\top x = \text{threshold logit}$ is a hyperplane in feature space, even though the probability is a nonlinear function of $x$.

### Log loss from the likelihood

Write the Bernoulli likelihood compactly as $p(y \mid x) = h^y(1-h)^{1-y}$ with $h = \sigma(\theta^\top x)$. The log likelihood over $n$ iid examples is

$$
\ell(\theta) = \sum_{i=1}^n y^{(i)}\log h(x^{(i)}) + (1 - y^{(i)})\log\big(1 - h(x^{(i)})\big).
$$

Its negative is the log loss, or binary cross entropy. Minimising cross entropy and maximising likelihood are the same thing; cross entropy between the empirical label distribution and the predicted distribution is exactly the average negative log likelihood.

### The gradient

Use $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. For one example,

$$
\frac{\partial \ell}{\partial \theta_j} = \Big(\frac{y}{h} - \frac{1-y}{1-h}\Big) h(1-h)\, x_j = (y - h)\, x_j,
$$

so the gradient of the loss is $X^\top(\sigma(X\theta) - \vec y)$. It has the same form as the least squares gradient, which is not a coincidence: both are GLMs with canonical links. The Hessian of the loss is $X^\top S X$ with $S = \mathrm{diag}(h_i(1-h_i))$, positive semidefinite, so the loss is convex. Newton's method $\theta := \theta - H^{-1}\nabla$ (also called iteratively reweighted least squares, or Fisher scoring here) converges in a handful of iterations for moderate $d$. A finite difference check matched the analytic gradient to five decimals, and eight Newton steps matched statsmodels' `Logit` exactly.

### Why not MSE

1. **Convexity.** Squared error through a sigmoid is not convex in $\theta$: in a three point example its second difference changed sign along a line, while log loss stayed convex.
2. **Vanishing gradient when confidently wrong.** With true label 1 and logit $z = -6$, the gradient of squared error with respect to $z$ is $2(p-1)p(1-p) \approx -0.005$, while the log loss gradient is $p - 1 \approx -0.998$. Squared error barely corrects the worst mistakes.
3. **Probabilistic fit.** Log loss is the Bernoulli likelihood, and it is a strictly proper scoring rule, which is why unpenalised logistic regression tends to be well calibrated.

### Interpreting coefficients

$e^{\theta_j}$ is the odds ratio for a one unit increase in $x_j$ holding the others fixed. In the verification fit, $\theta_1 = 1.86$ means the odds multiply by $e^{1.86} = 6.42$ per unit. The effect on the probability itself is not constant: it is largest near $p = 0.5$, with slope $\theta_j p(1-p)$.

### Separation, regularisation and multiclass

If the classes are perfectly linearly separable, the likelihood keeps increasing as $\lVert\theta\rVert \to \infty$ and the MLE does not exist. Regularisation fixes it, which is one reason sklearn's `LogisticRegression` applies an L2 penalty by default (`C` is the inverse penalty strength). For $k$ classes, softmax regression uses $p(y = i \mid x) \propto e^{\theta_i^\top x}$ with gradient $(\phi_i - \mathbb{1}\{y = i\})\,x$ for class $i$, the same residual times input form.

### The perceptron, where this module stops

Replace the sigmoid by a hard threshold and keep the update $\theta := \theta + \alpha(y - h_\theta(x))x$ and you have the perceptron. It converges if the data are separable, but it has no probabilistic interpretation and is not a maximum likelihood method. Stacking such units with smooth activations is the subject of the neural networks module ([CS229 Lecture 7](https://www.youtube.com/watch?v=fRM41w9jzQo)).

> **Watch:** [CS229 Lecture 4](https://www.youtube.com/watch?v=8gVi4Rk21Eg) covers the exponential family, GLMs and classification.

> **Interview trap:** Saying logistic regression is "regression, so not a classifier". It is a probability model; the classifier appears when you choose a threshold, and 0.5 is rarely the right one.

## GLMs and the exponential family

A generalised linear model makes three choices: the response $y$ given $x$ follows an exponential family distribution, the prediction is $h(x) = \mathbb{E}[y \mid x]$, and the natural parameter is linear in the input, $\eta = \theta^\top x$. Linear regression, logistic regression, softmax and Poisson regression all fall out of this one recipe.

### The exponential family

A distribution is in the exponential family if

$$
p(y; \eta) = b(y)\exp\big(\eta^\top T(y) - a(\eta)\big),
$$

with natural parameter $\eta$, sufficient statistic $T(y)$ (usually $y$), and log partition function $a(\eta)$, which normalises the density. Two derivatives of $a$ do the heavy lifting: $a'(\eta) = \mathbb{E}[T(y)]$ and $a''(\eta) = \mathrm{Var}(T(y))$. Since a variance is nonnegative, $a$ is convex, and that is why GLM negative log likelihoods are convex in $\theta$.

| Distribution | $\eta$ | $a(\eta)$ | Mean $a'(\eta)$ | GLM |
| --- | --- | --- | --- | --- |
| Gaussian, $\sigma^2 = 1$ | $\mu$ | $\eta^2/2$ | $\eta$ | Linear regression |
| Bernoulli($\phi$) | $\log\frac{\phi}{1-\phi}$ | $\log(1 + e^\eta)$ | $\sigma(\eta)$ | Logistic regression |
| Poisson($\lambda$) | $\log\lambda$ | $e^\eta$ | $e^\eta$ | Poisson regression |
| Categorical | log ratios to a reference class | log sum exp | softmax | Softmax regression |

Derivation for the Bernoulli: $\phi^y(1-\phi)^{1-y} = \exp\big(y\log\frac{\phi}{1-\phi} + \log(1-\phi)\big)$, so $\eta$ is the log odds and inverting gives $\phi = 1/(1+e^{-\eta})$. The sigmoid is not an arbitrary squashing function; it is what the Bernoulli exponential family hands you. Numerically, $a'(0.7)$ equals $\sigma(0.7) = 0.668$ and $a''(0.7)$ equals $p(1-p) = 0.2217$.

### Links and the shared gradient

The function mapping $\eta$ to the mean, $g(\eta) = \mathbb{E}[T(y); \eta]$, is the canonical response function; its inverse is the canonical link (identity for Gaussian, logit for Bernoulli, log for Poisson). With the canonical link, the log likelihood gradient for every GLM is $\sum_i (y^{(i)} - h_\theta(x^{(i)}))\,x^{(i)}$. That is why least squares and logistic regression share the update rule.

### Poisson regression in practice

For counts (arrivals per hour, claims per policy) use a log link, $\mathbb{E}[y \mid x] = e^{\theta^\top x}$. Coefficients are log rate ratios: on simulated data with true slope 0.8, statsmodels' Poisson GLM estimated 0.826, a rate ratio of $e^{0.826} \approx 2.285$ per unit of $x$. If the variance exceeds the mean (overdispersion), switch to negative binomial or quasi Poisson; for different exposure times add $\log(\text{exposure})$ as an offset.

> **Interview trap:** Fitting OLS to counts or to log of counts plus one. OLS can predict negative counts, assumes constant variance where count variance grows with the mean, and the plus one fudge biases the estimates.

## Generative classifiers: GDA and naive Bayes

A discriminative model learns $p(y \mid x)$ or a decision boundary directly. A generative model learns the class prior $p(y)$ and the class conditional $p(x \mid y)$, then classifies with Bayes' rule, $\arg\max_y p(x \mid y)p(y)$ (the evidence $p(x)$ cancels).

### Gaussian discriminant analysis

GDA assumes $y \sim \mathrm{Bernoulli}(\phi)$ and $x \mid y = k \sim \mathcal{N}(\mu_k, \Sigma)$ with a shared covariance. The maximum likelihood estimates are counts and averages: $\phi$ is the fraction of positives, $\mu_k$ the class means, $\Sigma$ the pooled within class covariance. Expanding the log posterior ratio, the quadratic terms in $x$ cancel because $\Sigma$ is shared, leaving

$$
p(y = 1 \mid x) = \sigma(w^\top x + b), \quad w = \Sigma^{-1}(\mu_1 - \mu_0), \quad b = -\tfrac12\mu_1^\top\Sigma^{-1}\mu_1 + \tfrac12\mu_0^\top\Sigma^{-1}\mu_0 + \log\tfrac{\phi}{1-\phi}.
$$

A numeric check at one query point gave 0.141445 from Bayes' rule with Gaussian densities and 0.141445 from the sigmoid formula. So GDA's posterior has exactly the logistic form, but the converse fails: many class conditionals (Poisson ones, for instance) also give a logistic posterior. Hence the CS229 trade off. GDA makes stronger assumptions; when they hold it is asymptotically efficient and needs less data. Logistic regression is less sensitive to a wrong model and wins with plenty of non Gaussian data. With a separate covariance per class you get quadratic discriminant analysis and curved boundaries; with shared covariance and the sklearn name, `LinearDiscriminantAnalysis`.

### Naive Bayes

Naive Bayes assumes features are conditionally independent given the class, $p(x \mid y) = \prod_j p(x_j \mid y)$. That cuts the parameters for 50,000 binary words from $2^{50000}$ to 100,000 per class pair. The assumption is wrong for text, yet the classifier often ranks well, because classification only needs the argmax to be right, not the probabilities. The cost shows up in calibration: correlated evidence is counted repeatedly, pushing posteriors toward 0 and 1 (the sklearn calibration guide shows exactly this pattern for `GaussianNB`).

The zero count problem: a word never seen with a class gets probability zero, and one zero wipes out the whole product. Laplace smoothing adds one pseudo count to each outcome, $\phi_{j \mid y} = (1 + \text{count}_{j,y}) / (2 + n_y)$ for binary features ($|V|$ instead of 2 in the multinomial event model).

```python
import numpy as np
from sklearn.naive_bayes import BernoulliNB

# rows are emails, columns are words ["free", "meeting", "neurips"]
X = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
y = np.array([1, 1, 1, 0, 0, 0])                   # 1 = spam
new = np.array([[1, 1, 0]])                        # "free" + "meeting"; "free" never seen in ham
for a in [1e-10, 1.0]:
    nb = BernoulliNB(alpha=a).fit(X, y)
    print(f"alpha={a:g}  P(word=1 | spam) = {np.exp(nb.feature_log_prob_[1]).round(3)}"
          f"  P(spam | new) = {nb.predict_proba(new)[0, 1]:.3f}")
```

Without smoothing, one unseen word makes the email certainly spam (posterior 1.000) regardless of the other evidence. With Laplace smoothing the posterior is 0.667. Multinomial naive Bayes, which models word counts, usually beats the Bernoulli version on text.

> **Watch:** [CS229 Lecture 5](https://www.youtube.com/watch?v=zRdE8A4UZes) on Gaussian discriminant analysis and the generative versus discriminative comparison.

## kNN and the curse of dimensionality

k nearest neighbours predicts by majority vote (classification) or average (regression) of the $k$ training points closest to the query. There is no training step, prediction costs $O(nd)$ per query without an index, and the model is the data.

### k as the bias variance knob

$k = 1$ has zero training error and high variance: the boundary follows every noisy point. Large $k$ averages over a wide neighbourhood, lowering variance and raising bias; $k = n$ predicts the majority class everywhere. Choose $k$ by cross validation, prefer odd $k$ for binary votes, and consider distance weighted votes. Scale features first, because the distance metric decides everything: a feature in dollars swamps one in years.

### The curse of dimensionality

In high dimensions, distance stops being informative and neighbourhoods stop being local.

```python
import numpy as np

rng = np.random.default_rng(0)
for d in [2, 10, 100, 1000]:
    P = rng.uniform(size=(500, d))          # 500 points in the unit cube
    q = rng.uniform(size=d)                 # one query point
    dist = np.linalg.norm(P - q, axis=1)
    print(f"d={d:4d}  nearest/farthest {dist.min() / dist.max():.3f}  "
          f"edge of cube holding 10% of volume {0.1 ** (1 / d):.3f}")
```

The ratio of nearest to farthest distance climbs from 0.024 in 2 dimensions to 0.904 in 1,000: every point is roughly equally far away. To capture 10 percent of uniformly spread data you need a sub cube with edge 0.316 in 2 dimensions but 0.977 in 100, so the "neighbourhood" spans almost the whole range of every feature. Data needed for a fixed density grows exponentially with $d$. The escape is that real data usually sit near a low dimensional manifold, which is why feature selection, PCA and learned embeddings make kNN usable, and why approximate nearest neighbour indexes work for embeddings at all.

## SVMs and kernels

A support vector machine finds the separating hyperplane with the largest margin, allows violations at a cost $C$, and through the dual and the kernel trick fits linear boundaries in feature spaces it never constructs. No CS229 Spring 2026 lecture video in the playlist covers it; the notes do (chapters 5 and 6).

### Margin to quadratic program

Use labels $y \in \{-1, 1\}$ and classifier $\mathrm{sign}(w^\top x + b)$. The functional margin $y^{(i)}(w^\top x^{(i)} + b)$ can be inflated by scaling $w$; the geometric margin $y^{(i)}(w^\top x^{(i)} + b)/\lVert w\rVert$ cannot. Fix the scale by requiring the smallest functional margin to equal 1; maximising $1/\lVert w\rVert$ is then minimising $\tfrac12\lVert w\rVert^2$ subject to $y^{(i)}(w^\top x^{(i)} + b) \ge 1$, a convex quadratic program. The margin width between the two supporting hyperplanes is $2/\lVert w\rVert$.

### Soft margin and hinge loss

Slack variables $\xi_i \ge 0$ give $\min \tfrac12\lVert w\rVert^2 + C\sum_i \xi_i$ subject to $y^{(i)}(w^\top x^{(i)} + b) \ge 1 - \xi_i$. Eliminating the slacks gives the unconstrained form $\tfrac12\lVert w\rVert^2 + C\sum_i \max(0, 1 - y^{(i)} f(x^{(i)}))$: hinge loss plus an L2 penalty. Compared with log loss, hinge loss is exactly zero beyond the margin, so well classified points stop influencing the solution; that is where sparsity in support vectors comes from, and also why SVM scores are not probabilities (Platt scaling fits a sigmoid on top). Small $C$ means a wide margin and many violations (more bias); large $C$ means a narrow margin that chases every point (more variance). On one dataset, $C = 0.01$ gave 134 support vectors and margin width 2.62, $C = 1$ gave 58 and 1.15.

### The dual and support vectors

The Lagrangian gives $w = \sum_i \alpha_i y^{(i)} x^{(i)}$ and $\sum_i \alpha_i y^{(i)} = 0$; substituting yields

$$
\max_\alpha \sum_i \alpha_i - \tfrac12 \sum_{i,j} y^{(i)}y^{(j)}\alpha_i\alpha_j \langle x^{(i)}, x^{(j)}\rangle \quad \text{s.t. } 0 \le \alpha_i \le C,\ \sum_i \alpha_i y^{(i)} = 0.
$$

KKT complementary slackness says $\alpha_i > 0$ only for points on or inside the margin: the support vectors. Points with $\alpha_i = C$ violate the margin. In sklearn the largest $|\alpha_i y_i|$ equalled $C$ exactly in every fit, as the box constraint requires. The SMO algorithm solves the dual by optimising two $\alpha$s at a time, because the equality constraint makes a single $\alpha$ immovable.

### The kernel trick

The dual and the prediction $\sum_i \alpha_i y^{(i)}\langle x^{(i)}, x\rangle + b$ touch the data only through inner products. Replace $\langle x, z\rangle$ by $K(x, z) = \langle\phi(x), \phi(z)\rangle$ and you fit a linear separator in $\phi$ space at the cost of evaluating $K$. For $(x^\top z + 1)^2$ in 3 dimensions the explicit map has 13 coordinates; for degree $k$ in $d$ dimensions it has $O(d^k)$, while $K$ costs $O(d)$. The RBF kernel $\exp(-\gamma\lVert x - z\rVert^2)$ corresponds to an infinite dimensional map. Mercer's theorem: $K$ is a valid kernel exactly when every Gram matrix it produces is symmetric positive semidefinite.

```python
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

rng = np.random.default_rng(0)
x, z = rng.normal(size=3), rng.normal(size=3)
phi = lambda v: np.r_[1, np.sqrt(2) * v, np.outer(v, v).ravel()]  # 13 features
print("kernel (x.z + 1)^2:", round((x @ z + 1) ** 2, 6))
print("explicit phi(x).phi(z):", round(phi(x) @ phi(z), 6))
K = rbf_kernel(rng.normal(size=(50, 4)), gamma=0.5)
print("RBF Gram matrix smallest eigenvalue:", f"{np.linalg.eigvalsh(K).min():.2e}")
```

Both lines print 1.730612, and the RBF Gram matrix has smallest eigenvalue $3.2 \times 10^{-2} > 0$. In RBF SVMs, large $\gamma$ makes each point's influence local (wiggly boundary, variance), small $\gamma$ makes it nearly linear; tune $C$ and $\gamma$ jointly on a log grid. Kernel SVMs scale between $O(n^2)$ and $O(n^3)$ in training and store the support vectors, so past roughly a hundred thousand rows use a linear SVM, random Fourier features, or boosted trees. Feature scaling matters for both linear and RBF kernels.

## Decision trees

A decision tree partitions feature space with axis aligned splits chosen greedily, top down, to reduce impurity, and predicts the majority class or mean target in each leaf. CART, the algorithm behind sklearn, makes binary splits only.

### Impurity and information gain

For class proportions $p_k$ in a node: Gini impurity $1 - \sum_k p_k^2$ (the chance two random draws disagree), entropy $-\sum_k p_k\log_2 p_k$, and misclassification rate $1 - \max_k p_k$. A split's information gain is the parent impurity minus the size weighted impurity of the children. Regression trees use variance (MSE) reduction instead.

> **Worked example:** A node with 10 positives and 10 negatives splits into children of 8/2 and 2/8. Gini falls from 0.5 to 0.32. Entropy falls from 1.0 bit to 0.7219 in each child, an information gain of 0.2781 bits.

Gini and entropy almost always pick the same split; Gini is cheaper (no logarithm). Misclassification rate is too flat to grow trees with, because many splits leave it unchanged even when they make the children purer. Entropy based gain favours attributes with many values, which is why C4.5 uses gain ratio.

### Finding splits

For a numeric feature, sort the values once and scan the $n - 1$ thresholds while updating class counts, so a node costs $O(n \log n)$ per feature, or $O(n)$ with presorting or histograms. For a categorical feature with a binary target, sorting the categories by their positive rate and treating them as ordered finds the optimal binary partition without trying all $2^{k-1}$ subsets. Plain sklearn trees need numeric input; `HistGradientBoosting` supports categoricals natively.

### Pruning

An unconstrained tree grows until leaves are pure, memorising the training set. Pre pruning stops early with `max_depth`, `min_samples_leaf` or `min_impurity_decrease`. Post pruning grows the full tree and then removes subtrees by cost complexity, minimising $R_\alpha(T) = R(T) + \alpha |T|$ where $|T|$ counts leaves; each $\alpha$ gives the best subtree of that size, and CV chooses $\alpha$.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
best = max(path.ccp_alphas[:-1], key=lambda a: cross_val_score(
    DecisionTreeClassifier(ccp_alpha=a, random_state=0), X_tr, y_tr, cv=5).mean())
full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
pruned = DecisionTreeClassifier(ccp_alpha=best, random_state=0).fit(X_tr, y_tr)
print("full tree:   leaves", full.get_n_leaves(), " test acc", round(full.score(X_te, y_te), 3))
print("pruned tree: leaves", pruned.get_n_leaves(), " test acc", round(pruned.score(X_te, y_te), 3))
```

Pruning cut the tree from 16 leaves to 9 and raised test accuracy from 0.881 to 0.916.

### Strengths and failure modes

Trees need no feature scaling, handle mixed types and interactions, ignore monotone transforms of features, and are readable when shallow. They have high variance (a small data change can flip the root split), produce staircase boundaries for diagonal structure, are greedy rather than optimal, and cannot extrapolate: a regression tree predicts a constant outside the training range of $x$, which matters for trending time series.

## Bagging and random forests

Bagging trains many high variance models on bootstrap resamples and averages them. Random forests add a second source of randomness, a random subset of features at each split, to make the trees less correlated, because correlation is what limits how much averaging helps.

### Why decorrelation matters

If $B$ predictors each have variance $\sigma^2$ and pairwise correlation $\rho$, their average has variance

$$
\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2.
$$

Adding trees kills the second term but never the first, so beyond a few hundred trees only lower $\rho$ helps. A simulation with $\rho = 0.3$, $B = 50$ gave variance 0.3154 against the formula's 0.314. Breiman's 2001 paper makes the same point for classification with the bound $PE^* \le \bar\rho(1 - s^2)/s^2$, where $s$ is the strength of individual trees and $\bar\rho$ their mean correlation: you want strong trees that disagree. `max_features` sets the trade off; smaller values decorrelate more but weaken each tree. Common defaults are $\sqrt{d}$ for classification and a larger share for regression. Deep, unpruned trees are used on purpose: low bias, with the variance averaged away.

### Out of bag error

A bootstrap sample of size $n$ leaves out each row with probability $(1 - 1/n)^n \to e^{-1} \approx 0.368$. Score each row using only the trees that did not see it and you get the out of bag estimate, a nearly free validation score. Breiman reports it is as accurate as a test set of the same size; it is slightly pessimistic early on because each row is scored by only about a third of the trees.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

n = 1000
print("P(row left out of a bootstrap sample):", round((1 - 1 / n) ** n, 4), " 1/e:", round(np.exp(-1), 4))

# variance of an average of B correlated predictors: rho*s2 + (1 - rho)*s2/B
rng = np.random.default_rng(0)
s2, rho, B = 1.0, 0.3, 50
cov = np.full((B, B), rho * s2) + np.eye(B) * (1 - rho) * s2
avg = rng.multivariate_normal(np.zeros(B), cov, size=20000).mean(axis=1)
print("simulated var", avg.var().round(4), " formula", rho * s2 + (1 - rho) * s2 / B)

X, y = make_classification(n_samples=3000, n_features=20, n_informative=6, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
for m in [1.0, "sqrt"]:                     # max_features=1.0 is plain bagging of trees
    rf = RandomForestClassifier(n_estimators=300, max_features=m, oob_score=True,
                                random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    print(f"max_features={m}: OOB acc {rf.oob_score_:.3f}  test acc {rf.score(X_te, y_te):.3f}")
```

Plain bagging reached test accuracy 0.909 and the random forest 0.920; both had OOB accuracy 0.906, close to the test numbers.

### What interviewers add

Trees train independently, so forests parallelise trivially. Adding trees does not overfit in the sense of increasing test error; it converges. Correlated features share importance, so each looks less important than it is. Extremely randomised trees go further and draw split thresholds at random, trading a little bias for lower variance and faster training.

## Boosting

Boosting fits weak learners sequentially, each one correcting the current ensemble, and so reduces bias as well as variance. Bagging averages independent strong learners; boosting adds dependent weak ones.

### AdaBoost

Start with uniform weights on the examples. At round $m$, fit a weak classifier $h_m$ to the weighted data, compute its weighted error $\epsilon_m$, give it vote $\alpha_m = \tfrac12\ln\frac{1-\epsilon_m}{\epsilon_m}$, and reweight $w_i \leftarrow w_i e^{-\alpha_m y_i h_m(x_i)}$ then normalise, so misclassified points gain weight. Error 0.1 earns a vote of 1.099, error 0.3 earns 0.424, and error 0.5 (coin flipping) earns 0. AdaBoost is forward stagewise additive modelling under exponential loss $e^{-yF(x)}$, which explains both its power and its sensitivity to label noise: the loss grows exponentially on badly misclassified points.

### Gradient boosting as gradient descent in function space

Friedman's gradient boosting treats the model's predictions on the training set as the parameters. Start from the best constant $F_0$. At step $m$ compute pseudo residuals $r_i = -\partial L(y_i, F)/\partial F\,\big|_{F = F_{m-1}(x_i)}$, fit a small regression tree $h_m$ to them, and update $F_m = F_{m-1} + \nu\,h_m$ with shrinkage $\nu$. For squared loss the pseudo residual is the ordinary residual $y - F$; for log loss it is $y - p$; for absolute loss it is the sign of the residual. Each tree is one gradient step, projected onto what a tree can represent.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
X = rng.uniform(0, 1, (300, 1))
y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(scale=0.2, size=300)

F = np.full(300, y.mean())                  # F_0: the constant that minimises squared loss
nu = 0.1                                    # shrinkage (learning rate)
for m in range(200):
    residual = y - F                        # negative gradient of 0.5 * (y - F)^2
    tree = DecisionTreeRegressor(max_depth=2).fit(X, residual)
    F += nu * tree.predict(X)
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2).fit(X, y)
print("scratch train MSE", round(np.mean((y - F) ** 2), 4),
      " sklearn train MSE", round(np.mean((y - gb.predict(X)) ** 2), 4))
```

The ten line loop and sklearn agree exactly (training MSE 0.0203 each). Smaller $\nu$ needs more trees but generalises better; choose the number of trees by early stopping on a validation set. Row subsampling per tree (stochastic gradient boosting) adds variance reduction.

### XGBoost

XGBoost (Chen and Guestrin, 2016) adds a regularised objective $\sum_i l(y_i, \hat y_i) + \sum_k\big(\gamma T_k + \tfrac12\lambda\lVert w_k\rVert^2\big)$, with $T$ leaves and leaf weights $w$, and a second order Taylor expansion with gradients $g_i$ and Hessians $h_i$. For a fixed tree structure the optimal leaf weight and split gain are

$$
w_j^\star = -\frac{G_j}{H_j + \lambda}, \qquad \mathrm{Gain} = \frac12\Big[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\Big] - \gamma,
$$

where $G$ and $H$ sum $g_i$ and $h_i$ over a leaf. For squared loss $h_i = 1$, so a leaf holding residuals 3, 5 and 4 gets weight $12/(3 + 1) = 3$ instead of the mean 4: $\lambda$ shrinks leaves toward zero and $\gamma$ is the minimum gain needed to split. The paper also introduces a weighted quantile sketch for approximate split finding and a sparsity aware algorithm that learns a default direction for missing values at each split, reported as more than 50 times faster than a naive version on sparse data.

### LightGBM and CatBoost

| | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- |
| Split finding | Exact or histogram | Histogram, plus GOSS (keep large gradient rows, sample small ones) and EFB (bundle mutually exclusive sparse features) | Histogram on oblivious trees |
| Tree growth | Level wise by default | Leaf wise (best first), controlled by `num_leaves` | Symmetric: one split per depth level |
| Categoricals | Encode yourself (newer versions add native support) | Native, gradient statistics per category | Ordered target statistics |
| Signature idea | Second order regularised objective | Speed: the paper reports training up to over 20 times faster at similar accuracy | Removing target leakage: ordered TS and ordered boosting |

CatBoost's argument (Prokhorenkova et al.) is worth knowing in detail. Replacing a category by the mean target of its rows leaks the label: if every category is unique, the encoded value is a smoothed copy of $y$, one split classifies the training set perfectly, and test accuracy is 0.5. Even leave one out encoding leaks. CatBoost orders rows by a random permutation and encodes each row using only rows before it, $\hat x_k = (\sum_{j<k} \mathbb{1}\{x_j = x_k\}y_j + a p)/(\sum_{j<k}\mathbb{1}\{x_j = x_k\} + a)$ with prior $p$. The same idea fixes prediction shift: residuals for a row should come from a model that never trained on that row, which ordered boosting approximates with supporting models over permutation prefixes.

> **Interview trap:** "Random forest and boosting are both tree ensembles, so they behave the same." Forests use deep trees and reduce variance; boosting uses shallow trees and reduces bias. Boosting overfits if you add trees without early stopping; a forest does not.

## Clustering

Clustering groups unlabelled points so that points within a group are similar. Each algorithm encodes a different notion of "cluster", so the choice of algorithm is a choice of assumption.

### k-means and k-means++

k-means minimises the distortion $J(c, \mu) = \sum_i \lVert x^{(i)} - \mu_{c^{(i)}}\rVert^2$. Lloyd's algorithm alternates two exact minimisations: assign each point to its nearest centroid (minimise over $c$), then move each centroid to the mean of its points (minimise over $\mu$). This is coordinate descent, so $J$ never increases and the algorithm converges, but $J$ is nonconvex and the answer depends on initialisation. Each iteration costs $O(nkd)$. k-means assumes roughly spherical, similarly sized clusters in a space where Euclidean distance means something, so scale features first.

k-means++ seeds the first centroid uniformly at random and each later one with probability proportional to $D(x)^2$, the squared distance to the nearest chosen centroid. Arthur and Vassilvitskii (2007) prove the seeding alone is $\Theta(\log k)$ competitive with the optimal clustering in expectation. In a 15 cluster test, single runs from k-means++ seeds averaged inertia 1259.9 against 1853.4 from random seeds. Still run several initialisations (`n_init`) and keep the lowest $J$.

### Choosing k

```python
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=0)
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    bic = GaussianMixture(n_components=k, random_state=0).fit(X).bic(X)
    print(f"k={k} inertia {km.inertia_:7.1f} silhouette {silhouette_score(X, km.labels_):.3f} GMM BIC {bic:.1f}")

Xm, ym = make_moons(n_samples=500, noise=0.06, random_state=0)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Xm)
db = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xm)
print("moons ARI: k-means", round(adjusted_rand_score(ym, km), 3), " DBSCAN", round(adjusted_rand_score(ym, db), 3))
```

Inertia always falls as $k$ grows (2922 at $k = 2$, 810 at $k = 6$), so the elbow is a judgement call. The silhouette, $s = (b - a)/\max(a, b)$ with $a$ the mean distance to the point's own cluster and $b$ to the nearest other cluster, peaks at the true $k = 4$ (0.500). The GMM's BIC prefers $k = 3$ (4942.2 against 4953.0), because two of the generated blobs overlap. Criteria disagree on real data; report several and use domain knowledge and stability across resamples.

### Gaussian mixtures and EM

A GMM says $z^{(i)} \sim \mathrm{Multinomial}(\phi)$ and $x^{(i)} \mid z^{(i)} = j \sim \mathcal{N}(\mu_j, \Sigma_j)$. With $z$ unobserved there is no closed form MLE. EM alternates an E step, computing responsibilities $w_j^{(i)} = p(z^{(i)} = j \mid x^{(i)})$ by Bayes' rule, and an M step, which is GDA's estimator with soft counts: $\phi_j = \frac1n\sum_i w_j^{(i)}$, $\mu_j = \sum_i w_j^{(i)}x^{(i)} / \sum_i w_j^{(i)}$, and the weighted covariance.

Why it works (CS229 chapter 11): for any distribution $Q$ over $z$, Jensen's inequality on the concave log gives

$$
\log p(x; \theta) = \log \sum_z Q(z)\frac{p(x, z; \theta)}{Q(z)} \ge \sum_z Q(z)\log\frac{p(x, z; \theta)}{Q(z)} = \mathrm{ELBO}(Q, \theta),
$$

with equality when $Q(z) = p(z \mid x; \theta)$. The E step makes the bound tight at the current $\theta$; the M step maximises the bound over $\theta$; so $\ell(\theta^{(t+1)}) \ge \mathrm{ELBO}(Q^{(t)}, \theta^{(t+1)}) \ge \mathrm{ELBO}(Q^{(t)}, \theta^{(t)}) = \ell(\theta^{(t)})$. The likelihood is monotone (checked numerically over 30 iterations) but only a local optimum is guaranteed. A component can collapse onto one point and send the likelihood to infinity, which is why sklearn adds `reg_covar` to the diagonal. k-means is the limit of EM with shared spherical covariances shrinking to zero: soft assignments become hard.

### DBSCAN and hierarchical clustering

DBSCAN calls a point core if at least `min_samples` points lie within `eps`, grows clusters through chains of core points, attaches border points, and labels the rest noise ($-1$). It finds arbitrary shapes and needs no $k$: on two moons it scored ARI 1.0 where k-means scored 0.253. It fails when densities vary, because one `eps` cannot fit all clusters; HDBSCAN removes `eps` by building the hierarchy over all density levels. Pick `eps` from the knee of a sorted k distance plot.

Agglomerative clustering merges the two closest clusters repeatedly, giving a dendrogram you cut at any height. Ward linkage minimises the increase in within cluster variance and behaves like k-means; complete linkage uses the farthest pair; average the mean pair; single the closest pair, which chains clusters together. It needs $O(n^2)$ memory for the distance matrix.

> **Watch:** [CS229 Lecture 9](https://www.youtube.com/watch?v=bSmIGBCoffA) on k-means and mixtures, and [CS229 Lecture 10](https://www.youtube.com/watch?v=sUS-eTa0l6s) on EM and PCA.

## Dimensionality reduction

PCA finds the orthogonal directions of maximum variance and projects onto the top $k$; t-SNE and UMAP build nonlinear 2D or 3D maps for looking at data. Only PCA is a safe preprocessing step for a model.

### PCA by eigendecomposition and by SVD

Centre the data (and scale it if features have different units). The variance of the projection onto a unit vector $u$ is $u^\top\Sigma u$ with $\Sigma$ the sample covariance. Maximise it subject to $u^\top u = 1$: the Lagrangian $u^\top\Sigma u - \lambda(u^\top u - 1)$ has stationary points $\Sigma u = \lambda u$, and the variance achieved is $u^\top\Sigma u = \lambda$, so the best direction is the top eigenvector and the top $k$ eigenvectors span the best $k$ dimensional subspace. The same subspace minimises squared reconstruction error, and the error equals the sum of the discarded eigenvalues.

In practice compute the SVD of the centred data, $X_c = USV^\top$. The columns of $V$ are the principal directions, the eigenvalues are $s_j^2/(n-1)$, and the scores are $US$. The SVD avoids forming $X^\top X$, which squares the condition number.

```python
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)
Z = rng.normal(size=(400, 5)) @ rng.normal(size=(5, 5))     # correlated features
Zc = Z - Z.mean(axis=0)
eigvals = np.linalg.eigvalsh(np.cov(Zc, rowvar=False))[::-1]
s = np.linalg.svd(Zc, compute_uv=False)
print("eigenvalues      ", eigvals.round(3))
print("s^2 / (n - 1)    ", (s**2 / (len(Z) - 1)).round(3))
pca = PCA(n_components=2).fit(Z)
print("explained variance ratio", pca.explained_variance_ratio_.round(3))
recon = pca.inverse_transform(pca.transform(Z))
err = ((Z - recon) ** 2).sum(axis=1).sum() / (len(Z) - 1)
print("reconstruction error", round(err, 4), " sum of dropped eigenvalues", round(eigvals[2:].sum(), 4))
```

Both routes give eigenvalues $[11.601, 6.308, 5.597, 0.21, 0.023]$. Two components explain 0.489 and 0.266 of the variance, and the reconstruction error 5.8303 equals the sum of the three dropped eigenvalues. The explained variance ratio $\lambda_j/\sum_l\lambda_l$ drives the choice of $k$: a scree plot elbow, or the smallest $k$ reaching 90 to 95 percent.

Pitfalls: PCA is scale sensitive; it is linear; it ignores $y$, so a low variance direction can be the predictive one; component signs are arbitrary; and in a pipeline it must be fit on the training fold only.

### t-SNE and UMAP

t-SNE matches neighbour probabilities in the original space (Gaussian kernels whose widths are set by the perplexity) with those in 2D (a heavy tailed Student t kernel), by minimising KL divergence. Wattenberg, Viégas and Johnson's Distill article (2016) lists the traps: results change with perplexity (typical range 5 to 50, and it must be below the number of points); stopping early leaves pinched shapes; cluster sizes mean nothing because t-SNE expands dense regions and contracts sparse ones; distances between clusters may mean nothing; and random Gaussian noise can produce apparent clusters at low perplexity. Compare several perplexities and seeds before reading anything into a plot.

UMAP builds a weighted k nearest neighbour graph and optimises a low dimensional layout to match it. `n_neighbors` (default 15) trades local detail for global structure, `min_dist` (default 0.1) controls how tightly points pack. It is faster than t-SNE, scales to more output dimensions, and can embed new points with `transform`. It is often described as keeping more global structure, but the same rule holds: sizes and gaps in a UMAP plot are not measurements. Never cluster on a t-SNE embedding and report the clusters as findings without checking them in the original space.

## Evaluation metrics

Pick the metric from the decision the model supports, then check it against a trivial baseline. Most metric mistakes in interviews are about imbalance, thresholds and probabilities.

### Confusion matrix metrics

Precision $= TP/(TP + FP)$ is the share of flagged cases that are real. Recall (sensitivity, TPR) $= TP/(TP + FN)$ is the share of real cases caught. Specificity $= TN/(TN + FP)$, and the false positive rate is one minus it. F1 is the harmonic mean $2PR/(P + R)$, dominated by the smaller of the two; $F_\beta$ weights recall $\beta$ times as much. For multiclass, macro averaging treats classes equally, micro averaging pools counts (and equals accuracy for single label problems), weighted averaging weights by support.

> **Worked example:** 100 cases, 10 positive. The model flags 20, catching 8. TP 8, FP 12, FN 2, TN 78. Precision 0.4, recall 0.8, F1 0.533, accuracy 0.86, which is worse than the 0.90 of predicting "negative" for everyone. Predicting "positive" for everyone at 1 percent prevalence gives precision 0.01, recall 1 and F1 0.0198.

### ROC AUC versus PR AUC

The ROC curve plots TPR against FPR over all thresholds. Its area equals the probability that a random positive scores above a random negative, the Mann Whitney statistic, so it measures ranking, ignores calibration, and does not depend on prevalence. That last property is the problem under heavy imbalance: an FPR of 1 percent sounds small but means 99 false alarms per 9,900 negatives. Precision recall curves use precision, which counts false positives against true positives, so they react to prevalence; a random classifier's PR AUC equals the positive rate.

```python
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

rng = np.random.default_rng(0)
for n_pos, n_neg in [(1000, 1000), (100, 9900)]:
    pos, neg = rng.normal(1.5, 1, n_pos), rng.normal(0, 1, n_neg)   # same score distributions
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
    s = np.r_[pos, neg]
    pairwise = (pos[:, None] > neg[None, :]).mean()                # P(random pos > random neg)
    print(f"prevalence {n_pos / (n_pos + n_neg):.2f}: ROC AUC {roc_auc_score(y, s):.3f} "
          f"(pairwise {pairwise:.3f})  PR AUC (AP) {average_precision_score(y, s):.3f}")
```

Identical score distributions give ROC AUC 0.850 at 50 percent prevalence and 0.853 at 1 percent, each equal to the pairwise probability, while average precision falls from 0.846 to 0.138. Saito and Rehmsmeier (PLOS ONE, 2015) show the same with one operating point: 500 true positives cost 160 false positives on balanced data but 1,600 on data with ten times as many negatives, and precision at 50 percent recall drops from 75 to 25 percent while the ROC curve does not move. Use ROC AUC to compare rankers across populations, PR AUC or precision at a fixed recall when positives are rare and false alarms cost money. Summarise PR curves with average precision, not linear interpolation, which is optimistic in PR space.

### Probabilities: log loss, Brier score and calibration

Log loss $-\frac1n\sum_i[y_i\log p_i + (1 - y_i)\log(1 - p_i)]$ is 0.693 ($\ln 2$) for a constant 0.5 and 4.605 for a single confident miss at $p = 0.01$. The Brier score is the mean squared error of probabilities; predicting the base rate 0.1 on a 10 percent problem scores 0.09. Both are strictly proper scoring rules: they are optimised in expectation only by the true probabilities. A lower Brier score does not prove better calibration, because it also rewards discrimination (Murphy's decomposition).

A model is calibrated if, among cases scored 0.8, about 80 percent are positive. The reliability diagram bins predictions and plots observed frequency against mean predicted probability. Typical shapes from the sklearn guide: logistic regression is usually close to the diagonal; naive Bayes is overconfident; random forests are pulled away from 0 and 1 (a sigmoid curve) because averaging trees rarely produces extreme votes; max margin methods are more sigmoidal still. Fix it with `CalibratedClassifierCV` on data the model did not train on: Platt scaling (a sigmoid, good for small calibration sets and preserves ranking) or isotonic regression (any monotone shape, better above roughly 1,000 calibration samples, can introduce ties that change AUC). The current guide also lists temperature scaling for multiclass.

### Regression metrics

RMSE is minimised by the conditional mean, matches Gaussian likelihood, and punishes large errors; MAE is minimised by the median, matches Laplace likelihood, and is more resistant to outliers; MAPE is scale free but undefined at zero and penalises over prediction more than under prediction; quantile (pinball) loss targets a chosen quantile; $R^2$ compares against predicting the mean. If one 10x error matters more than ten 1x errors, use RMSE.

### Choosing a threshold

A score is not a decision. With calibrated probabilities and costs $c_{FP}$, $c_{FN}$, predict positive when $p > c_{FP}/(c_{FP} + c_{FN})$: if a missed fraud costs ten times a false alarm, the threshold is $1/11 \approx 0.091$, not 0.5. Without costs, pick the threshold on a validation set to hit a recall target, a review capacity (top k per day), or maximum F1, and report test metrics at that frozen threshold. sklearn's `TunedThresholdClassifierCV` does the search with cross validation.

## Cross validation and leakage

Validation estimates how the whole modelling procedure performs on data it has never influenced. Leakage is anything that lets information from the evaluation data, or from the future, into training, and it produces scores the model cannot reproduce in production.

### Splits and k fold

Use three roles: train to fit, validation (or CV inside the training data) to choose models and hyperparameters, test once at the end. k fold CV trains on $k - 1$ folds and scores the remaining one, rotating through all $k$; 5 or 10 folds is the usual choice and sklearn's guide prefers them to leave one out, whose estimate has high variance and costs $n$ fits. Larger $k$ means less pessimistic bias (each model sees more data) but more compute and more correlated folds. Repeated k fold reduces the variance of the estimate. Deep learning uses CV less because each fit is expensive and datasets are large enough that one validation set is stable.

### Pick the splitter from the dependence structure

| Data | Splitter | Why |
| --- | --- | --- |
| iid rows, regression | `KFold(shuffle=True)` | Default |
| Classification, especially imbalanced | `StratifiedKFold` | Every fold keeps the class ratio; avoids folds with no positives |
| Repeated entities (patients, users, sessions) | `GroupKFold`, `StratifiedGroupKFold` | No group in both train and test; measures generalisation to new entities |
| Anything ordered in time | `TimeSeriesSplit`, with `gap` | Train on the past, test on the future, leave a buffer for lagged features |

### Leakage you should be able to name

- **Preprocessing fit on all the data.** Scalers, imputers, PCA, feature selection and target encoders learned before the split. Wrap them in a `Pipeline` so they refit inside each fold.
- **Duplicates or near duplicates** across train and test, including oversampled copies of the same row.
- **Group leakage.** The same patient, user or device on both sides.
- **Temporal leakage.** Random splits of time ordered data; features computed with future information (a 30 day average centred on the prediction date); labels defined over a window that overlaps the features.
- **Target proxies.** Features created after or because of the outcome: "account closed reason" in a churn model, "treatment given" in a diagnosis model.
- **Test set reuse.** Tuning against the test set, or picking among many models by test score.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(4)
# 1. feature selection outside the CV loop, on pure noise
X = rng.normal(size=(200, 10000))
y = rng.integers(0, 2, 200)
leaky = cross_val_score(LogisticRegression(max_iter=1000), SelectKBest(k=25).fit_transform(X, y), y, cv=5)
honest = cross_val_score(make_pipeline(SelectKBest(k=25), LogisticRegression(max_iter=1000)), X, y, cv=5)
print("select then CV:", leaky.mean().round(3), "  select inside CV:", honest.mean().round(3))

# 2. 40 patients with 25 rows each; the target depends only on the patient
groups = np.repeat(np.arange(40), 25)
Xg = rng.normal(size=(40, 3))[groups] + 0.1 * rng.normal(size=(1000, 3))
yg = rng.normal(size=40)[groups] + 0.1 * rng.normal(size=1000)
rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
print("KFold R2:", cross_val_score(rf, Xg, yg, cv=KFold(5, shuffle=True, random_state=0)).mean().round(3),
      "  GroupKFold R2:", cross_val_score(rf, Xg, yg, cv=GroupKFold(5), groups=groups).mean().round(3))
```

On labels that are pure coin flips, selecting 25 of 10,000 features before CV gives accuracy 0.855; doing it inside each fold gives 0.475, chance level (the sklearn pitfalls page reports the same pattern, 0.76 versus 0.5). On the patient data, random folds report $R^2$ 0.953 because the forest memorises each patient's feature signature; grouped folds report $-1.003$, the truth for new patients. Kapoor and Narayanan (Patterns, 2023) found leakage errors in 41 papers across 30 fields, which together affected 648 papers; in their civil war prediction case study, the complex models' advantage over logistic regression disappeared once leakage was fixed.

### Nested cross validation

If CV both selects hyperparameters and reports the score, the score is optimistic, because you picked the best of many noisy estimates. Nested CV runs the whole search inside each outer training fold and scores the chosen model on the outer fold. The outer score estimates the performance of the procedure, not of one particular tuned model.

## Class imbalance

Imbalance is a problem of too few positive examples and of metrics that hide failure, more than of the ratio itself. A model trained on 1 percent positives with plain log loss is not broken: its probabilities can be well calibrated, it just rarely crosses 0.5.

### What to do, in order

1. **Fix the metric.** PR AUC, recall at a fixed precision, precision at k, or expected cost. Never accuracy.
2. **Fix the threshold.** Move it using costs or a validation target before touching the training data.
3. **Reweight the loss.** `class_weight="balanced"` sets class weights to $n/(k\,n_c)$: with 900 negatives and 100 positives, 0.556 and 5.0. It is cheap and keeps all data, but it distorts predicted probabilities, so recalibrate afterwards if you need them.
4. **Resample.** Random undersampling of the majority (fast, throws data away), random oversampling of the minority (risk of memorising duplicates), SMOTE (interpolates between minority neighbours; can blur the boundary in noisy or high dimensional data). Resample the training fold only.
5. **Change the loss.** Focal loss, $-\alpha_t(1 - p_t)^\gamma\log p_t$, down weights easy examples so training concentrates on hard ones. Lin et al. (2017) found $\gamma = 2$ with $\alpha = 0.25$ worked best for dense object detection. It matters most in deep learning where easy negatives number in the hundreds of thousands per image.
6. **Get more positives.** Labelling more, active learning, or reframing as anomaly detection when positives are essentially absent.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold

rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 5))                       # features carry no signal at all
y = (rng.uniform(size=2000) < 0.05).astype(int)      # 5 percent positives
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)

def oversample(X, y, times=15):
    idx = np.r_[np.arange(len(y)), np.repeat(np.where(y == 1)[0], times)]
    return X[idx], y[idx]

Xo, yo = oversample(X, y)                            # wrong: duplicate before splitting
cv = StratifiedKFold(5, shuffle=True, random_state=0)
wrong = [recall_score(yo[te], rf.fit(Xo[tr], yo[tr]).predict(Xo[te])) for tr, te in cv.split(Xo, yo)]
right = []
for tr, te in cv.split(X, y):                        # right: duplicate inside the training fold only
    Xtr, ytr = oversample(X[tr], y[tr])
    right.append(recall_score(y[te], rf.fit(Xtr, ytr).predict(X[te])))
print("oversample then split, recall:", np.mean(wrong).round(3))
print("split then oversample, recall:", np.mean(right).round(3))
```

With features that carry no signal, oversampling before the split reports recall 1.0, because copies of each positive land in both train and test; oversampling inside the training fold reports 0.0. This is the scenario in Chip Huyen's question about a model that excels on the test set and fails in production.

## Features and missing data

Feature work decides more outcomes than model choice, and most of it is about encoding information the model cannot infer and removing information it should not have.

### Feature engineering

- **Scaling.** Needed for distance and margin methods (kNN, SVM, k-means, PCA) and for penalised or gradient trained linear models. Not needed for trees.
- **Categoricals.** One hot for low cardinality (drop one level for unpenalised linear models to avoid the dummy trap); target encoding for high cardinality, fit out of fold or ordered to avoid leakage; hashing for huge vocabularies; trees in LightGBM, CatBoost and `HistGradientBoosting` take categoricals natively.
- **Numeric transforms.** Logs for skewed positive variables such as prices, splines or bins for nonlinearity in linear models, ratios and differences that encode domain knowledge.
- **Interactions.** Linear models need them explicitly; trees find them, but only through sequential splits.
- **Time.** Cyclical encodings (sine and cosine of hour), lags and rolling statistics computed with only past data.

### Feature selection

Filter methods score features one at a time (correlation, mutual information, chi squared): fast, blind to interactions. Wrapper methods search subsets with a model (recursive feature elimination, forward selection): better, expensive, prone to overfitting the selection. Embedded methods select during fitting (L1 penalties, tree importances). L1 selects; L2 does not, because it never produces exact zeros. Every selection step belongs inside the CV loop, as the leakage example above shows.

### Missing data

Ask why values are missing. MCAR (missing completely at random) loses only precision; MAR (missingness depends on observed variables) is handled by models that condition on those variables; MNAR (missingness depends on the missing value or the outcome) carries signal and biases naive imputation. Options: drop rows (only for small MCAR fractions), mean or median imputation (shrinks variance and correlations), model based imputation (`IterativeImputer`, kNN), adding a missingness indicator, or a model that handles NaN natively.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=2000, n_features=8, n_informative=2, n_redundant=0,
                           class_sep=0.5, flip_y=0.1, random_state=0)
# missing not at random: feature 0 goes missing more often when y == 1
X[(y == 1) & (rng.uniform(size=len(y)) < 0.5), 0] = np.nan
X[(y == 0) & (rng.uniform(size=len(y)) < 0.1), 0] = np.nan

plain = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000))
flagged = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), LogisticRegression(max_iter=1000))
hgb = HistGradientBoostingClassifier(random_state=0)   # handles NaN natively
for name, m in [("median impute", plain), ("impute + indicator", flagged), ("HGB native NaN", hgb)]:
    print(f"{name:20s} AUC {cross_val_score(m, X, y, cv=5, scoring='roc_auc').mean():.3f}")
```

When missingness depends on the label, the indicator column lifts AUC from 0.925 to 0.941, and gradient boosting with native NaN handling reaches 0.954 (it also captures nonlinearity). Fit imputers inside the pipeline, and check that the missingness pattern at serving time matches training.

## Interpretability

Global interpretability asks which features drive the model overall; local interpretability asks why this prediction came out this way. Every method answers a narrower question than people assume.

### Coefficients

Linear and logistic coefficients are exact descriptions of the model, conditional on the other features. They are comparable only on standardised features, unstable under multicollinearity, and causal only under exogeneity.

### Impurity importance versus permutation importance

Tree impurity importance (MDI) sums the impurity decrease from every split on a feature. It is computed on training data and is biased toward high cardinality features, which offer many candidate thresholds and can split on noise. Permutation importance shuffles one column of held out data and measures the score drop; it is model agnostic and reflects generalisation.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n = 3000
signal = rng.integers(0, 2, n)                        # binary feature that matters
noise_id = rng.integers(0, 1000, n)                   # high cardinality noise, like a user id
y = np.where(rng.uniform(size=n) < 0.75, signal, 1 - signal)
X = np.c_[signal, noise_id]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)
print("MDI  [signal, noise_id]:", rf.feature_importances_.round(3))
perm = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0)
print("perm [signal, noise_id]:", perm.importances_mean.round(3))
```

MDI ranks the random ID first (0.692 against 0.308). Permutation importance on the test set gives the real signal 0.155 and the ID 0.003. Both methods share one weakness: with correlated features, permuting one leaves its twin available, so both look unimportant. The sklearn guide suggests clustering correlated features and keeping one per cluster.

### SHAP

SHAP (Lundberg and Lee, NIPS 2017) explains one prediction as $f(x) = \phi_0 + \sum_i \phi_i$, where $\phi_0$ is the average prediction and $\phi_i$ is feature $i$'s Shapley value: its marginal contribution averaged over all orderings in which features could be added. The paper shows it is the unique additive attribution satisfying local accuracy (the attributions sum to the prediction), missingness (absent features get zero) and consistency (if a model relies more on a feature, its attribution does not drop), and that LIME, DeepLIFT, layer wise relevance propagation and three Shapley based methods fit the same additive form. Kernel SHAP estimates it for any model by weighted linear regression; TreeSHAP computes it exactly and fast for tree ensembles. Caveats: approximations often treat features as independent when filling in "missing" ones, which creates unrealistic inputs when features are correlated; and SHAP explains the model, not the world.

## Hyperparameter search

Hyperparameters are set before training (regularisation strength, tree depth, learning rate); parameters are learned. Tune them on validation data or inner CV, never on the test set.

- **Grid search** tries every combination: fine for one or two parameters, wasteful beyond.
- **Random search** samples combinations: with the same budget it tries many more distinct values of each parameter, which matters because usually only a few parameters affect the score much. Sample scale parameters such as $C$, $\lambda$ and learning rate log uniformly.
- **Bayesian optimisation** (Gaussian processes, TPE as in Optuna) fits a surrogate of score versus hyperparameters and picks the next trial by expected improvement.
- **Successive halving and Hyperband** give many configurations a small budget (rows, trees, epochs) and promote the best; sklearn has `HalvingRandomSearchCV`.

```python
from scipy.stats import loguniform
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=30, n_informative=5, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
search = RandomizedSearchCV(pipe, {"logisticregression__C": loguniform(1e-3, 1e2)},
                            n_iter=20, cv=5, scoring="roc_auc", random_state=0)
# nested CV: the outer loop scores the whole search procedure, not one tuned model
outer = cross_val_score(search, X, y, cv=5, scoring="roc_auc")
search.fit(X, y)
print("best C:", round(search.best_params_["logisticregression__C"], 4))
print("inner best AUC:", round(search.best_score_, 3), " nested AUC:", outer.mean().round(3))
```

Here the best $C$ is 0.0826, and the inner best and nested AUC agree at 0.964 because only one hyperparameter was searched; the gap between them grows with the size of the search. For gradient boosting, a practical order: fix a modest learning rate (0.05 to 0.1) and use early stopping for the number of trees, then tune tree complexity (`max_depth` or `num_leaves`, `min_child_weight`), then row and column subsampling, then the L1 and L2 leaf penalties, then lower the learning rate and raise the tree count for the final fit.

## Time series basics

Time series observations are dependent and ordered, so the iid machinery above needs three changes: model the dependence, validate forward in time, and never let a feature see the future.

### Stationarity and autocorrelation

A series is weakly stationary if its mean and variance are constant and its autocovariance depends only on the lag. Trends, seasonality and changing variance break it. The autocorrelation function (ACF) is the correlation of the series with itself at lag $k$; the partial autocorrelation (PACF) removes the effect of shorter lags. For an AR(1) process $x_t = \phi x_{t-1} + \epsilon_t$ with $|\phi| < 1$ the ACF is $\phi^k$, decaying geometrically. A random walk ($\phi = 1$) has a unit root: its variance grows with time and its ACF decays very slowly. Differencing, $x_t - x_{t-1}$, removes a unit root; seasonal differencing, $x_t - x_{t-s}$, removes a stable seasonal pattern; a log first stabilises growing variance.

The two standard tests point in opposite directions. ADF's null hypothesis is a unit root (non stationary); KPSS's null is stationarity. Run both: if ADF fails to reject and KPSS rejects, difference.

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import acf, adfuller, kpss

rng = np.random.default_rng(0)
phi, T = 0.7, 5000
ar = np.zeros(T)
for t in range(1, T):                                # AR(1): x_t = 0.7 x_{t-1} + noise
    ar[t] = phi * ar[t - 1] + rng.normal()
print("AR(1) sample ACF", acf(ar, nlags=3).round(3), " theory", [round(phi**k, 3) for k in range(4)])

walk = np.cumsum(rng.normal(size=1000))              # random walk: unit root
print("random walk:  ADF p", round(adfuller(walk)[1], 3), " KPSS p", kpss(walk, nlags="auto")[1])
print("differenced:  ADF p", f"{adfuller(np.diff(walk))[1]:.1e}", " KPSS p", kpss(np.diff(walk), nlags="auto")[1])

for tr, te in TimeSeriesSplit(n_splits=3, gap=1).split(np.arange(10)):
    print("train", tr, "test", te)
```

The AR(1) sample ACF is 0.698, 0.488, 0.338 against theory 0.7, 0.49, 0.343. The random walk gives ADF p 0.664 (cannot reject a unit root) and KPSS p 0.01 (reject stationarity); after differencing, ADF p is essentially 0 and KPSS p is 0.1. statsmodels warns that 0.01 and 0.1 are the ends of its lookup table, so read them as "below 0.01" and "above 0.1".

### ARIMA at a high level

ARIMA($p, d, q$) differences the series $d$ times, then models it with $p$ autoregressive terms (lags of the series) and $q$ moving average terms (lags of the forecast errors). Special cases worth knowing, from Duke's Robert Nau notes: ARIMA(0,1,0) is a random walk, ARIMA(1,0,0) is a mean reverting AR(1), and ARIMA(0,1,1) without a constant is simple exponential smoothing. Box Jenkins identification reads the ACF and PACF of the stationary series: a PACF that cuts off after lag $p$ with a decaying ACF suggests AR($p$); an ACF that cuts off after lag $q$ suggests MA($q$). Compare candidates by AIC and check that residuals look like white noise. SARIMA adds seasonal terms; exogenous regressors give ARIMAX.

### Validation and features without leakage

Validate on a rolling or expanding forecast origin (`TimeSeriesSplit`), with a `gap` at least as long as your longest lag or label window, and score at the horizon you will deploy. Baselines first: last value (naive), the value one season ago (seasonal naive), and a moving average; many complex models fail to beat seasonal naive. For gradient boosting on time series, build lag and rolling features from strictly past values, remember that trees cannot extrapolate a trend (difference or detrend first, or model the trend separately), and never shuffle.

## What interviewers probe

These chains come from the question banks in the sources (Chip Huyen's ML interviews book, Alexey Grigorev's theory questions) and from where candidates usually stall.

1. **Bias and variance.** Define them, then derive the decomposition. Why does the cross term vanish? How do you tell which dominates from learning curves? Does more data help a high bias model? How does $k$ in kNN, depth in trees, or $\lambda$ in ridge move you along the trade off? What is double descent and does it contradict the trade off?
2. **Linear regression.** Derive the normal equations. When is $X^\top X$ not invertible? What happens with $z = x + y$ as a column, and with noise added? Which assumptions matter for unbiasedness and which only for standard errors? What does heteroscedasticity break and how do you fix it? Leverage points versus outliers versus influential points?
3. **Regularisation.** Why does L1 give zeros, three ways? What prior is L2? Why standardise first? Why not penalise the intercept? When elastic net over lasso? Can you use L2 for feature selection?
4. **Logistic regression.** Derive the log loss and its gradient. Is it a linear model? Why not MSE? What happens with perfectly separable data? Interpret a coefficient of 0.7. Why is it usually well calibrated?
5. **GLMs and generative models.** Put the Bernoulli in exponential family form. Why do linear and logistic regression share an update rule? Show that GDA implies a logistic posterior; why is the converse false? When does naive Bayes beat logistic regression? Why are its probabilities extreme?
6. **SVMs.** Derive the margin $2/\lVert w\rVert$. What do $C$ and $\gamma$ do? What are support vectors, in KKT terms? Why does the dual make kernels possible? Is any similarity function a kernel? Why scale features?
7. **Trees and ensembles.** Compute Gini and entropy gain by hand. Why do forests use deep trees and boosting shallow ones? Derive the variance of an average of correlated predictors. What is the OOB fraction and why? Write gradient boosting as gradient descent. XGBoost leaf weight? LightGBM leaf wise growth risk? What leakage does CatBoost fix?
8. **Clustering and PCA.** Why does k-means converge and why not to the global optimum? What does k-means++ guarantee? How do you choose k, and what if the criteria disagree? Derive EM's monotonicity. PCA from variance maximisation; relation to SVD; why scale; why t-SNE distances mislead.
9. **Evaluation.** Compute precision, recall and F1 from a confusion matrix. Why can 99.99 percent accuracy be useless? ROC AUC as a probability. When PR AUC instead? What is calibration and how do you fix it? Pick a threshold from costs. RMSE or MAE?
10. **Validation and leakage.** Why train, validation and test? Which splitter for patients, for time? Name five kinds of leakage and how you would detect one (suspiciously high single feature importance, train and production gap, adversarial validation). Why nested CV? What goes wrong when you oversample before splitting?
11. **Practice.** Two of twenty features are 30 percent missing: what do you do? Which importance would you trust and why? How would you tune XGBoost on a budget? How is time series validation different?

## Sources

- [Stanford CS229 main notes, Spring 2026 (Ma and Ng)](https://cs229.stanford.edu/main_notes.pdf)
- [Stanford CS229 Spring 2026 course page](https://cs229.stanford.edu/index.html-spr26)
- [Chip Huyen, ML Interviews Book: classic ML questions](https://huyenchip.com/ml-interviews-book/contents/8.1.2-questions.html)
- [Chip Huyen, ML Interviews Book: ML basics](https://huyenchip.com/ml-interviews-book/contents/7.1-basics.html)
- [Chip Huyen, ML Interviews Book: sampling and training data](https://huyenchip.com/ml-interviews-book/contents/7.2-sampling-and-creating-training-data.html)
- [Chip Huyen, ML Interviews Book: objectives, metrics and evaluation](https://huyenchip.com/ml-interviews-book/contents/7.3-objective-functions,-metrics,-and-evaluation.html)
- [Alexey Grigorev, data science interview theory questions](https://github.com/alexeygrigorev/data-science-interviews/blob/master/theory.md)
- [scikit-learn: common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: cross validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn: permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [scikit-learn: clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Breiman (2001), Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [Chen and Guestrin (2016), XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Ke et al. (2017), LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [Prokhorenkova et al., CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
- [Arthur and Vassilvitskii (2007), k-means++: The Advantages of Careful Seeding](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- [Saito and Rehmsmeier (2015), The precision recall plot is more informative than the ROC plot on imbalanced datasets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
- [Lin et al. (2017), Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [Kapoor and Narayanan (2023), Leakage and the Reproducibility Crisis in ML based Science](https://reproducible.cs.princeton.edu/)
- [Wattenberg, Viégas and Johnson (2016), How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- [UMAP documentation: basic parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html)
- [statsmodels: stationarity and detrending with ADF and KPSS](https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html)
- [Robert Nau, Duke: introduction to ARIMA models](https://people.duke.edu/~rnau/411arim.htm)
