# ML coding cheatsheet

The ten minute version: round formats, the narration script, complexities, the formulas you must be able to write from memory, and the one line answers to the follow ups that come after every implementation.

## Before the round

| Round | Practise | Graded on |
| --- | --- | --- |
| DS pandas screen | groupby, merge, lag per user, sessions, rolling, pivot | Grain, denominators, row counts after joins |
| LeetCode style | hash map, two pointers, window, heap, binary search, intervals, BFS, DP | Optimal approach, complexity, edge cases |
| ML from scratch | the implementations below, in NumPy, timed | Correct math, vectorised, stable, tested |
| ML debugging | plant bugs in a small model and find them | Method, internals, verifying the fix |
| Take home | write up first, notebook runs from a fresh kernel | Answers the question asked |
| AI assisted | plan, delegate small pieces, review everything | Judgment and ownership |

Ask the recruiter: format, editor, allowed libraries, whether code is run, whether docs or AI are allowed. Time budgets from one public practice list: about 15 minutes easy, 20 to 35 medium, 40 to 60 or more hard.

## The narration script

1. Restate: inputs, outputs, shapes, sizes, ties, empty input.
2. Brute force in one sentence, then the better plan with time and space.
3. Tests out loud: normal, tie, empty, boundary, adversarial.
4. Code in named steps, narrating why.
5. Run tests, trace one by hand, state final complexity and one change at scale.

Stuck: narrate the gap, shrink to a tiny case, solve by hand, try hash map, heap, sort, two pointers, then fall back to a correct brute force.

## Python costs

| Operation | Cost |
| --- | --- |
| list index, append, pop() | O(1), append amortised |
| list insert(0), pop(0), `x in list` | O(n) |
| dict or set get, set, `in` | O(1) average, O(n) worst |
| deque append and pop at both ends | O(1) |
| heapify, heappush, heappop | O(n), O(log n), O(log n) |
| bisect search, insort | O(log n), O(n) |
| sorted | O(n log n), stable |
| top k from n with a heap | O(n log k) time, O(k) memory |

Gotchas: mutable defaults (use None), late binding lambdas (`i=i`), `[[0] * 3] * 3` aliasing, float equality (`math.isclose`), `-7 // 2 == -4`, defaultdict inserts on `d[key]` but not `.get`, generators are single use, heap tuples need a counter tiebreak, Python 3.14 added `heappush_max`.

## NumPy and pandas

- Broadcasting: align shapes from the right; sizes must be equal or 1. (n, 1) minus (n,) is (n, n), silently.
- Pairwise squared distances: row norms plus row norms minus 2 A Bᵀ, clip at 0. Memory O(nm), not O(nmd).
- Slices are views; integer and boolean indexing copy on read; mask assignment writes in place.
- `np.sum` is pairwise; a naive float32 running sum of a million values near 1000.5 gave a mean near 1010.
- agg = one row per group; transform = one row per input row; filter = whole groups.
- Merge with `validate=` and `indicator=True`; count rows before and after.
- Sessions: sort, `groupby().diff()`, flag gap > threshold, `groupby().cumsum()`.
- `rolling("7D")` counts days, `rolling(7)` counts rows.
- pandas 3.0 (January 2026): Copy on Write, chained assignment never works, use one `.loc`; strings are `str` dtype; datetimes default to microseconds; `how="left_anti"`.

## Formulas to write from memory

- Linear regression: $w = (A^\top A)^{-1}A^\top y$ with a ones column; solve, do not invert; ridge adds $\lambda I$ except the intercept.
- MSE gradient: $\frac{2}{n}X^\top(Xw + b - y)$. Converges if the learning rate is below $2/\lambda_{\max}$.
- Logistic: $\nabla_w = \frac{1}{n}X^\top(p - y) + \lambda w$. scikit-learn's $C = 1/(n\lambda)$.
- Stable BCE: $\log(1 + e^z) - yz$ via `np.logaddexp(0, z)`.
- Softmax: subtract the row max. Log softmax: $z_y - \operatorname{logsumexp}(z)$. Gradient: $(p - \text{onehot})/n$.
- Linear layer backward: $dW = X^\top G$, $db = \sum_{\text{rows}} G$, $dX = GW^\top$. Tanh: times $1 - h^2$.
- Gradient check: $(f(\theta+h) - f(\theta-h))/2h$, $h = 10^{-5}$, float64, relative error near $10^{-7}$ is good, above $10^{-2}$ is a bug.
- Attention: $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$, mask with $-\infty$ before the softmax, causal mask is lower triangular.
- KV cache bytes: $2 \times \text{layers} \times \text{kv heads} \times \text{head dim} \times \text{tokens} \times \text{bytes}$.
- Gini: $1 - \sum_c p_c^2$; split score is the size weighted child Gini.
- k-means++: next centre with probability proportional to $D(x)^2$; $O(\log k)$ competitive in expectation.
- Precision $TP/(TP+FP)$, recall $TP/(TP+FN)$, F1 $= 2PR/(P+R)$.
- AUC from ranks: $(\sum_{\text{pos}} r_i - P(P+1)/2)/(PN)$ with average ranks for ties.
- DCG@k: $\sum_{i \le k} g(\text{rel}_i)/\log_2(i+1)$; scikit-learn uses linear gain, many papers $2^{\text{rel}} - 1$.
- Welford: $\delta = x - \bar{x}$, $\bar{x} \mathrel{+}= \delta/n$, $M_2 \mathrel{+}= \delta(x - \bar{x}_{\text{new}})$, variance $M_2/(n-1)$.
- Chan merge: $M_2 = M_{2,A} + M_{2,B} + \delta^2 n_A n_B / n$.
- Reservoir: keep item i with probability $k/i$, replacing a uniform slot.
- EMA: $m = \alpha x + (1 - \alpha)m$, effective window about $2/\alpha - 1$.

## Complexities of the implementations

| Implementation | Time | Memory |
| --- | --- | --- |
| Closed form linear regression | $O(nd^2 + d^3)$ | $O(d^2)$ |
| GD epoch (linear or logistic) | $O(nd)$ | $O(d)$ |
| k-means iteration | $O(nkd)$ | $O(nk)$ |
| kNN query | $O(nd)$ | $O(nd)$ stored |
| Best tree split, one feature | $O(n \log n)$ | $O(n)$ |
| Attention per head | $O(T^2 d)$ | $O(T^2)$ scores |
| Decode step with KV cache | $O(tD)$ | grows with t |
| BPE training (naive) | $O(\text{merges} \times \text{corpus})$ | corpus |
| ROC AUC via ranks | $O(n \log n)$ | $O(n)$ |
| Bootstrap | $O(Bn)$ | $O(Bn)$ vectorised |
| Reservoir, Welford, moving average | $O(1)$ per item | $O(k)$, $O(1)$, $O(w)$ |

## If asked X, say Y

- **Why divide by root d_k?** Dot products of unit variance vectors have variance d_k; large logits saturate softmax and kill gradients. Use the head size.
- **Why subtract the max in softmax?** Softmax is shift invariant; the largest exponent becomes 1, so no overflow and the denominator is at least 1.
- **Why not zero initialisation?** Symmetry: identical units get identical gradients forever.
- **Does k-means converge?** Yes, to a local optimum; inertia never increases; use k-means++ and restarts.
- **Logistic regression on separable data?** Weights diverge; add L2 or stop early.
- **Why solve instead of inv?** Cheaper and more accurate; normal equations square the condition number; use lstsq or ridge if singular.
- **Why is heapify O(n)?** Cost is node height and most nodes are near the leaves.
- **Greedy versus beam?** Beam keeps B partial sequences; the blog example finds 0.36 where greedy finds 0.2; beam repeats in open ended text.
- **Top k versus top p?** Fixed count versus cumulative mass; nucleus adapts to confidence.
- **ROC AUC under heavy imbalance?** Prefer PR AUC; report prevalence as its baseline.
- **Is model A better than B?** Paired bootstrap of the metric difference on the same rows; resample users if rows cluster.
- **What is leakage?** Test information in training; fit every statistic on train only, use a Pipeline; split by group or time when rows are related.
- **Debugging a model?** Initial loss versus log C, overfit 8 examples, assert shapes, inspect inputs with labels, bisect.
- **Using AI in the round?** Plan first, delegate small specified pieces, read and run all output, narrate, never paste the whole problem.

## Numbers worth remembering

- Uniform loss: $\log 10 = 2.3026$, $\log 2 = 0.6931$.
- Gradient check: about $10^{-7}$ good; this module's MLP check gave $6.3 \times 10^{-8}$.
- Greedy 0.2 against beam 0.36 (Hugging Face example).
- BPE toy merges: ug (20), un (16), hug (15); minbpe "aaabdaaabac" gives [258, 100, 258, 97, 99].
- Leakage demo: 0.834 with leaky feature selection against 0.496 honest; scikit-learn's version is 0.76 against 0.5.
- KV cache example: 32 layers, 8 kv heads of 128, 16 bit: 128 KiB per token, 1 GiB at 8,192 tokens.
- Meta's AI enabled round started October 2025; Canva's AI assisted round was announced June 2025; pandas 3.0 shipped January 2026.
