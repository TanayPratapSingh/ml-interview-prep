# Classical ML cheatsheet

One page for the ten minutes before a classical ML interview: the formulas you may be asked to derive, the decision tables, the numbers worth remembering, and short "if asked X, say Y" lines. Details and derivations are in the deep version.

## Core formulas

| Thing | Formula |
| --- | --- |
| Bias variance (squared loss) | $\mathbb{E}[(y - \hat h(x))^2] = \sigma^2 + (h^\star - h_{\mathrm{avg}})^2 + \mathrm{Var}(\hat h)$ |
| OLS | $\hat\theta = (X^\top X)^{-1}X^\top y$; residuals $\perp$ columns of $X$; $H = X(X^\top X)^{-1}X^\top$, $\mathrm{tr}(H) = p$ |
| Ridge | $(X^\top X + \lambda I)^{-1}X^\top y$; MAP with prior $\mathcal{N}(0, \tau^2)$, $\lambda = \sigma^2/\tau^2$ |
| Lasso, orthonormal $X$ | $\mathrm{sign}(z)\max(\lvert z\rvert - \lambda, 0)$ (soft threshold); Laplace prior |
| WLS | $(X^\top W X)^{-1}X^\top W y$, weights $= 1/\mathrm{variance}$ |
| Logistic | $p = \sigma(\theta^\top x)$, log odds linear; loss gradient $X^\top(p - y)$; Hessian $X^\top S X \succeq 0$ |
| Exponential family | $p(y;\eta) = b(y)e^{\eta T(y) - a(\eta)}$, $a' = $ mean, $a'' = $ variance |
| GDA posterior | $\sigma(w^\top x + b)$ with $w = \Sigma^{-1}(\mu_1 - \mu_0)$ |
| SVM primal | $\min \tfrac12\lVert w\rVert^2 + C\sum\max(0, 1 - y f(x))$; margin $2/\lVert w\rVert$ |
| SVM dual | $\max \sum\alpha_i - \tfrac12\sum\alpha_i\alpha_j y_i y_j K(x_i, x_j)$, $0 \le \alpha_i \le C$ |
| Gini, entropy | $1 - \sum p_k^2$; $-\sum p_k\log_2 p_k$; gain $=$ parent minus weighted children |
| Averaged predictors | $\mathrm{Var} = \rho\sigma^2 + (1-\rho)\sigma^2/B$ |
| AdaBoost vote | $\alpha = \tfrac12\ln\frac{1 - \epsilon}{\epsilon}$ |
| Gradient boosting | fit tree to $-\partial L/\partial F$, $F \leftarrow F + \nu h$ |
| XGBoost leaf | $w^\star = -G/(H + \lambda)$; gain $\tfrac12[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}] - \gamma$ |
| EM bound | $\log p(x) \ge \sum_z Q(z)\log\frac{p(x,z)}{Q(z)}$, tight at $Q = p(z \mid x)$ |
| PCA | top eigenvectors of covariance $=$ right singular vectors of centred $X$; $\lambda_j = s_j^2/(n-1)$ |
| Precision, recall, F1 | $\frac{TP}{TP+FP}$, $\frac{TP}{TP+FN}$, $\frac{2PR}{P+R}$ |
| Cost threshold | predict positive if $p > c_{FP}/(c_{FP} + c_{FN})$ |
| VIF | $1/(1 - R_j^2)$ |
| Cook's distance | $\frac{e_i^2}{p s^2}\frac{h_{ii}}{(1-h_{ii})^2}$ |

## Numbers worth remembering

- Out of bag share: $(1 - 1/n)^n \to e^{-1} \approx 0.368$ of rows per tree.
- Log loss of a constant 0.5: $\ln 2 = 0.693$. One confident miss at $p = 0.01$: 4.6.
- Random classifier: ROC AUC 0.5; PR AUC equals prevalence.
- Always predicting positive at 1 percent prevalence: F1 0.0198.
- Balanced class weight: $n/(k\,n_c)$; 900 versus 100 gives 0.556 and 5.0.
- k-means++ seeding: $\Theta(\log k)$ competitive in expectation (Arthur and Vassilvitskii, 2007).
- Focal loss best setting in the paper: $\gamma = 2$, $\alpha = 0.25$ (Lin et al., 2017).
- AR(1) autocorrelation at lag $k$: $\phi^k$.
- Isotonic calibration needs roughly 1,000 or more calibration samples (sklearn guide); otherwise Platt.
- t-SNE perplexity: typically 5 to 50, below the number of points.
- Leverage values: sum to $p$, mean $p/n$, flag above $2p/n$.

## Which model when

| Situation | First choice | Why |
| --- | --- | --- |
| Tabular, mixed types, nonlinear | Gradient boosting (LightGBM, XGBoost, CatBoost) | Best default accuracy; handles missing values |
| Need a quick, stable baseline | Random forest | Few knobs, OOB score, hard to break |
| Need coefficients or probabilities you can explain | Logistic or linear regression, regularised | Convex, calibrated, readable |
| Many features, few rows, sparse text | Linear model with L2 or L1; naive Bayes | Low variance |
| Counts | Poisson GLM (negative binomial if overdispersed) | Correct mean variance relation |
| Small n, clear margin, moderate d | SVM with RBF kernel | Max margin; scales badly past about 100k rows |
| Clusters spherical, k known | k-means with k-means++ | Fast |
| Clusters of arbitrary shape, noise present | DBSCAN or HDBSCAN | No k, labels outliers |
| Soft membership, elliptical clusters | Gaussian mixture via EM | Probabilities, BIC for k |
| Visualise high dimensional data | PCA first, then UMAP or t-SNE | Sizes and gaps in the plot are not meaningful |

## Which splitter and metric

| Data | Split | Metric |
| --- | --- | --- |
| iid, balanced | `KFold` / `StratifiedKFold` | Accuracy, ROC AUC, log loss |
| Rare positives | `StratifiedKFold` | PR AUC, recall at fixed precision, cost |
| Repeated entities | `GroupKFold` | Same, scored on unseen groups |
| Forecasting | `TimeSeriesSplit` with `gap` | MAE, RMSE, MAPE at the deployed horizon, versus seasonal naive |
| Need probabilities | any of the above | Log loss, Brier, reliability diagram |

## Bagging, boosting and the three libraries

| | Random forest | Gradient boosting |
| --- | --- | --- |
| Trees | Deep, independent, parallel | Shallow, sequential |
| Reduces | Variance | Bias (and variance with shrinkage, subsampling) |
| More trees | Converges, does not overfit | Overfits without early stopping |
| Key knobs | `max_features`, `min_samples_leaf` | learning rate, number of trees, depth or leaves, subsampling |

XGBoost: second order objective with $\gamma T + \tfrac12\lambda\lVert w\rVert^2$, learned default direction for missing values. LightGBM: histograms, leaf wise growth (watch `num_leaves`), GOSS and EFB. CatBoost: ordered target statistics and ordered boosting against target leakage, symmetric trees.

## If asked X, say Y

- **"Why L1 sparse, L2 not?"** The L1 ball has corners on the axes, and in an orthonormal design L1 soft thresholds while L2 rescales; the subgradient at zero is an interval of width $2\lambda$.
- **"Why not MSE for logistic?"** Nonconvex through the sigmoid, and its gradient vanishes when confidently wrong (0.005 versus 0.998 at logit $-6$); log loss is the Bernoulli likelihood.
- **"Is logistic regression linear?"** Its decision boundary is linear in the features; the probability is not.
- **"OLS assumptions?"** Linearity, exogeneity, full rank, homoscedastic and uncorrelated errors for BLUE; normal errors only for small sample tests. Normal features are never required.
- **"Heteroscedasticity?"** Coefficients still unbiased, standard errors wrong; use HC3 errors or WLS.
- **"Multicollinearity?"** Unstable coefficients and inflated variances, predictions fine; check VIF, use ridge or drop features.
- **"Bias or variance?"** Big train validation gap means variance; both high and converged means bias; more data only fixes variance.
- **"Why random features in a forest?"** To lower tree correlation $\rho$, the floor on ensemble variance.
- **"Gradient boosting in one sentence?"** Gradient descent in function space: each tree fits the negative gradient of the loss at the current predictions.
- **"ROC or PR AUC?"** ROC for ranking quality independent of prevalence; PR when positives are rare and false alarms cost money.
- **"99.99 percent accuracy on cancer detection?"** Compare with the majority baseline and report recall, precision and PR AUC.
- **"How do you pick k?"** Elbow, silhouette, BIC for GMMs, stability, domain sense; say when they disagree.
- **"PCA steps?"** Centre, scale, SVD, keep top k by explained variance, fit on training data only.
- **"Leakage examples?"** Preprocessing on all data, oversampling before split, same patient in train and test, future information in features, target proxies.
- **"Imbalance?"** Metric, then threshold, then class weights, then resampling inside folds; recalibrate if weights distort probabilities.
- **"Feature importance?"** Permutation importance on held out data over impurity importance, which favours high cardinality features; cluster correlated features.
- **"Time series validation?"** Forward chaining, a gap for lags, no shuffling, beat seasonal naive.
- **"Stationarity tests?"** ADF null is a unit root; KPSS null is stationarity; run both, then difference or detrend.

## Traps

- Fitting scalers, imputers, PCA or feature selection before the split.
- Tuning the threshold or hyperparameters on the test set.
- Reading t-SNE cluster sizes or distances as measurements.
- Reporting impurity importance as "what drives the business".
- Treating SVM scores or naive Bayes outputs as calibrated probabilities.
- Calling a tree ensemble able to extrapolate a trend.
