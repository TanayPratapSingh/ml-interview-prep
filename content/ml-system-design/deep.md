# ML System Design: the deep version

This is the technical version of the ML system design round. It states the answer framework with time boxes, the back of the envelope arithmetic (checked by running it), eight designs at interview depth (recommendation, search, ads click through rate, fraud, feed, ETA, moderation, an LLM support assistant), and the MLOps practices that keep such systems correct after launch: feature stores, skew, registries, rollout, drift monitoring, retraining and technical debt. Figures from papers name the paper. Assumed numbers are marked as assumptions.

## What the ML system design round tests

The round is an open ended design conversation of typically 45 to 60 minutes. The prompt is one line ("design a feed ranking system") and the interviewer answers clarifying questions. Chip Huyen's interviews book reports that almost every company she spoke with asks at least one such question and that candidates often find it the hardest. The alirezadir interview repo describes the aim as seeing whether you can design a production level system that fits a company's ML infrastructure, with no single correct answer.

Seven signals recur in write ups: problem framing, data and labels, metrics, justified model choice, serving and scale, operations (experiment, rollout, monitoring, retraining), and communication. The grid below is a synthesis, not a published rubric.

| Signal | Good | Red flag |
| --- | --- | --- |
| Framing | States goal, scale, latency, data, assumptions | Draws a network in minute one |
| Data | Names label source, bias, delay | "We have labeled data" |
| Metrics | Offline, online, guardrail, with reasons | Accuracy only |
| Modeling | Baseline then upgrade with evidence | Deep net by default |
| Systems | Latency budget, capacity numbers | "Scale horizontally" |
| Operations | A/B plan, drift, retraining | Stops at deployment |

Level calibration: junior answers need a correct end to end pipeline; senior answers add failure modes, trade offs, numbers and rollback; staff answers add ambiguity handling, cost, platform reuse and a plan for how the system improves. Missing an entire area costs more than shallow depth in one.

## The answer framework and time boxes

The eleven steps are: clarify, frame, metrics, data and labels, features, model from baseline up, training, offline evaluation, serving, online test, monitor and iterate. The alirezadir framework has nine steps (problem formulation, metrics, architectural components, data, features, model development and offline evaluation, prediction service, online testing and deployment, scaling, monitoring and updates) and gives no per step times.

| Minutes | Block |
| --- | --- |
| 0 to 5 | Clarify goals, scale, constraints |
| 5 to 10 | Framing and metrics |
| 10 to 18 | Data, labels, features |
| 18 to 28 | Baseline, model, training, offline evaluation |
| 28 to 37 | Serving, latency budget, estimation |
| 37 to 43 | Online experiment, monitoring, iteration |
| 43 to 45 | Summary and risks |

The blocks add to 45 (5 + 5 + 8 + 10 + 9 + 6 + 2). The split is a suggestion. Scale by about 1.3 for 60 minutes and merge blocks for a 30 minute screen.

**Baselines.** Google's Rules of Machine Learning: rule 1, do not be afraid to launch without ML; rule 3, move from a complex heuristic to ML when you have data; rule 4, keep the first model simple and get the infrastructure right; rule 7, turn existing heuristics into features. A baseline exercises logging, joins, evaluation and serving, sets a floor, and gives a fallback.

**The funnel.** Cost is items in times cost per item. A 0.05 ms model over 1,000 candidates costs 50 ms, over 100 million items about 5,000 s. YouTube's paper describes a candidate generator taking millions of videos to hundreds, and a ranker scoring those to dozens. Facebook's ads paper describes a cascade of classifiers of increasing cost. Final quality is capped by candidate recall.

**Pattern matching.** A ranked list from a catalog maps to the funnel; a risk score on events maps to imbalanced classification with thresholds and review; a number maps to regression with a baseline residual; text maps to an LLM system with retrieval and guardrails.

## Clarifying and framing the problem

Clarify six areas in 3 to 5 minutes: goal, users and surface, scale, latency and freshness, data and labels, constraints (privacy, regulation, cost, on device, explainability). Write the assumptions down and refer back to them.

Framing is one line: input, output, label. Granularity can be pointwise, pairwise or listwise. Output type: a calibrated probability when a downstream decision uses it (ads, fraud thresholds), an ordering score when only order matters.

**Proxy objectives.** Google's rules 12 and 13: do not overthink the objective early, pick a simple observable attributable proxy. Rule 38: if objectives are misaligned, change the objective rather than adding features. The risk is Goodhart's law. YouTube's paper states that ranking by click through rate often promotes deceptive videos users do not finish, and that expected watch time better captures engagement; it also notes the surrogate problem chosen for candidate generation had an outsized effect in A/B testing and was hard to measure offline. A common pattern predicts several outcomes (click, watch, like, hide) and combines them with a tunable value function.

## Offline and online metrics

Offline: precision, recall, F1, ROC AUC, PR AUC, log loss for classification; precision at k, recall at k, MAP, MRR, NDCG for ranking; MSE and MAE for regression; latency and cost for systems. Online: click through rate, session success, watch time, engagement, conversion, revenue lift, with counter metrics such as hides and reports. YouTube's paper says live A/B results are not always correlated with offline results.

**Ranking metrics.** With binary relevance $rel_i$ and $R$ relevant items: $P@k = \frac1k\sum_{i\le k} rel_i$, $R@k = \frac1R\sum_{i\le k} rel_i$, $RR = 1/\text{rank of first relevant}$, $AP = \frac1R\sum_k P@k\cdot rel_k$. For ranked relevance 1,0,1,0,0,1,0,0,0,1 with $R=5$: $P@3 = 0.667$, $R@10 = 0.8$, $RR = 1$, $AP = 0.5133$.

**NDCG.** $DCG@k = \sum_{i\le k}(2^{rel_i}-1)/\log_2(i+1)$ and $NDCG@k = DCG@k/IDCG@k$.

```python
import math

def dcg(rels, k=None):
    rels = rels[:k] if k else rels
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

def ndcg(rels, k=None):
    ideal = dcg(sorted(rels, reverse=True), k)
    return dcg(rels, k) / ideal if ideal > 0 else 0.0

print(round(ndcg([3, 2, 3, 0, 1, 2]), 4))       # 0.9488
```

For grades 3, 2, 3, 0, 1, 2: DCG@6 = 13.848, IDCG@6 = 14.595, NDCG@6 = 0.9488.

**Choosing metrics.** ROC AUC ignores prevalence and calibration. At a 0.1 percent base rate, 90 percent recall and a 1 percent false positive rate give precision 8.3 percent. Log loss and calibration matter when the probability is consumed. Offline gains fail online because logs reflect the old policy, proxies mismatch, skew, latency, position bias, novelty and interference. Rule 37 splits skew into train versus holdout, holdout versus next day, and next day versus live. A launch rule: primary metric up, no guardrail breached; agree the exchange rate beforehand.

## Data, labels and features

Label sources: implicit logs (volume, noise, exposure bias, missing negatives), explicit feedback (sparse), human annotation (costly), programmatic or weak labels (cheap, noisy). YouTube trains on implicit watches because there are orders of magnitude more of them. Log impressions so negatives exist (rule 6 warns about dropped data).

**Splits and leakage.** Split by time. Leaks: target (post outcome features), temporal (random split, future aggregates), group (same user in both sets), preprocessing (statistics fitted on all data), duplicates. YouTube holds out the next watch by rolling back history, because a random held out watch leaks future information and ignores asymmetric consumption such as episodes.

**Imbalance.** Class weights, majority downsampling (recalibrate), minority oversampling, focal loss, threshold tuning. Never resample the test set. Rule 30: importance weight sampled data rather than drop it. Facebook tried negative downsampling rates from 0.1 to 0.0001 and found the best at 0.025.

**Features.** Per actor (user, item, context, query) static and windowed features, plus crosses. Facebook (He et al., 2014): the top 10 features were all historical; among the top 20 only 2 were contextual. Normalized entropy relative to a contextual only model: all features 95.65 percent, historical only 96.32 percent; contextual features are indispensable for cold start and depend more on freshness. YouTube: the strongest ranking signals describe the user's past interaction with the item and similar items.

**High cardinality inputs.** One embedding per ID space, dimension about proportional to the log of the vocabulary, top N by frequency with out of vocabulary mapped to zero, averaged multivalent features, shared tables. A million IDs at 32 dimensions have about 7 times the parameters of a 2048 wide layer. Continuous features are quantile normalised, with powers fed as extra inputs (YouTube).

> **Watch:** [CS229 Lecture 6, Dataset Split, ML Advice](https://www.youtube.com/watch?v=llnEgyyuYkQ) covers dataset splits and practical ML advice.

## Models from baseline up

Ladder: heuristic, logistic regression, gradient boosted trees, deep network, ensemble. Decision factors in the alirezadir framework: task complexity, data type and size, training speed, inference compute, latency and memory, continual learning, interpretability. Rules 14 (interpretable models are easier to debug and calibrate), 17 (observed features first), 26 (mine errors), 40 (simple ensembles), 41 (plateau means new information).

Trees are invariant to scaling of single features and need little tuning; networks need normalisation but learn embeddings and share layers across heads. YouTube reports its deep ranker beat earlier linear and tree based methods for watch time prediction.

**Hybrid trees plus logistic regression (He et al., 2014).** Each boosted tree is a categorical feature whose value is the leaf index, one hot coded, feeding a sparse linear classifier. Normalized entropy relative to trees only: hybrid 96.58 percent, logistic regression only 99.43 percent, trees only 100 percent. Trees had at most 12 leaves, nearly all the gain came from the first 500 trees, and the last 1,000 trees improved normalized entropy by under 0.1 percent.

**Offline evaluation checklist.** Later test data, sliced metrics, calibration per slice, error analysis, comparison with the old model's output (rule 24), leave one feature out ablations, latency and memory, sanity gate before export (rule 9). For logged data from a past policy $\mu$, the inverse propensity estimate is $\hat V(\pi) = \frac1n\sum_i r_i\,\mathbb 1[\pi(x_i)=a_i]/\mu(a_i\mid x_i)$, unbiased with correct propensities and support but high variance.

## Serving, latency and experiments

**Latency budget.** Plan at p99 because a request touches many services. An illustrative split of 200 ms: network 20, gateway 15, feature fetch 30, retrieval 25, ranking 60, reranking 10, response 10 (total 170, slack 30). Parallelise independent calls, set per stage timeouts and fallbacks, use circuit breakers and hedged requests. YouTube's ranker table shows weighted per user loss falling from 41.6 percent with no hidden layers to 34.6 percent with 1024, 512, 256 ReLU layers, and the paper chose the best configuration that stayed within its serving CPU budget.

**Caching.** Feature, embedding, prediction, result and CDN caches. With a 40 percent hit rate at 5 ms and 120 ms misses, the mean is $0.4 \cdot 5 + 0.6 \cdot 120 = 74$ ms; the tail is unchanged. Always filter cached lists against a fresh seen set.

**A/B design.** Randomise by user. For a proportion with baseline $p$ and absolute difference $\delta$: $n \approx 2(z_{1-\alpha/2}+z_{1-\beta})^2 p(1-p)/\delta^2$ per arm. With $p = 0.05$ and a 2 percent relative lift ($\delta = 0.001$) this gives about 745,600 users per arm. Run one to two full weeks, check sample ratio mismatch, avoid peeking, use the delta method or a user level bootstrap for ratio metrics, ramp traffic.

**Exploration.** A model influences its own training data (a direct feedback loop in the Hidden Technical Debt paper). Contextual bandits are theoretically right but may not scale to large action spaces; mitigate with randomisation or isolating data. Rule 34 holds out a slice for unbiased data; rule 35 notes ranking systems create skew. Options: random slice, epsilon greedy, UCB, Thompson sampling. Facebook's Bayesian model gives a predictive distribution usable for exploration.

## Back of the envelope estimation

All numbers below were computed by running Python.

**Traffic.** 50M daily users x 10 requests = 500M per day. Average QPS = 500M / 86,400 = 5,787. Peak at 3 times: 17,361. With 500 candidates: 2.89M item scores per second on average and 8.68M at peak. At 500 QPS per server (assumed): 34.7 servers, 52 with 50 percent headroom.

**Storage.** 10 billion impressions per day at 200 bytes: 2 TB per day, 180 TB for 90 days. Keeping all positives and 10 percent of negatives at a 2 percent click rate keeps 11.8 percent of rows (1.18 billion per day, 236 GB). 100M items x 128 x 4 bytes = 51.2 GB (25.6 GB in float16, 1.6 GB with 16 byte product quantisation codes). An HNSW graph with $M = 32$ adds about 256 bytes per vector, 25.6 GB.

**Streaming state.** 10 billion events per day is 115,741 per second. 20 counters per user for 50M users at 8 bytes is 8 GB; with 60 one minute buckets each it is 480 GB; a decayed counter (value plus timestamp, 16 bytes) is 16 GB.

**Compute.** A dense forward pass is about $2 \times$ weights FLOPs per example. 3M weights x 500 candidates = 3 GFLOPs per request; at 17,361 QPS that is 52 TFLOPs per second, about 2.6 GPUs at 20 percent of 100 TFLOPs (assumed) or about 2,600 CPU cores at 20 GFLOPs (assumed). Training FLOPs are about $6 \times$ parameters $\times$ tokens.

**LLM memory.** Inference weights: 7B parameters at 2 bytes is 14 GB (7 GB int8, 3.5 GB 4 bit). Mixed precision Adam training: 2 + 2 + 4 + 4 + 4 = 16 bytes per parameter, 112 GB before activations. KV cache bytes per token $= 2 L H_{kv} d_{head} b$: 32 layers, 32 KV heads, 128 dimensions, 2 bytes gives 524,288 bytes (0.5 MB), 2 GiB per 4,096 tokens, about 30 sequences on an 80 GB GPU after 14 GB of weights; with 8 KV heads about 123.

## Recommendation system

Assumptions: 50M daily users, 100M videos, 200 ms, about 5,800 average and 17,400 peak QPS. Objective: expected watch time with hide rate, creator diversity and p99 latency as guardrails.

```text
 request -> user features -> candidate generation (~25 ms)
     two tower ANN (500) + co visitation (300) + trending/fresh (200) -> ~1,000
 -> ranking (~60 ms): multi head P(watch), P(like), P(hide), value = w1*watch + w2*like - w3*hide -> top 100
 -> reranking (~10 ms): MMR diversity, dedupe, seen filter, creator cap, freshness slot -> top 20
```

**Two tower retrieval.** User tower $u$, item tower $v$, score $u\cdot v$. Item vectors are precomputed and indexed. YouTube poses candidate generation as extreme multiclass classification, $P(w_t=i\mid U,C) = e^{v_i\cdot u}/\sum_j e^{v_j\cdot u}$, trained with several thousand sampled negatives and importance weighting (over a 100 times speedup versus full softmax), served by nearest neighbour search in dot product space; A/B results were not sensitive to the search algorithm. In batch negatives need the logQ correction, $\text{logit}_{ij} = u_i\cdot v_j/\tau - \log q_j$, because popular items appear as negatives more often (Yi et al., RecSys 2019, cited from memory).

**ANN.** Exact search over 100M x 128 floats is 12.8 billion multiply adds per query (about 444 TFLOPs per second at 17,361 QPS). HNSW, IVF and product quantisation trade recall for latency and memory. Keep a delta index for new items and version index and model together.

**Ranking.** Hundreds of features, impression frequency features for churn, retrieval source and score as features. YouTube's weighted logistic regression weights clicked impressions by watch time $T_i$ and others by 1, so odds $= \sum T_i/(N-k) = E[T]/(1-P) \approx E[T](1+P)$; inference uses $e^x$. In a simulation (200,000 impressions, 5 percent clicks, mean watch 300 s) learned odds were 15.84 versus true $E[T]$ of 15.07.

**Example age.** Feed the training example's age, set to zero at serving, so the model does not average popularity over the training window; YouTube reports better holdout precision and a large watch time gain on fresh videos.

**Reranking.** MMR: $\arg\max_i\,[\lambda\, rel_i - (1-\lambda)\max_{j\in S} \text{sim}(i,j)]$, then a policy layer (rule 15, rule 42).

**Cold start and loops.** Content towers, demographic and context priors, exploration slots; feedback loops from popularity and exposure bias countered by randomised slices, training on organic interactions, propensity weighting, diversity and holdouts.

## Search ranking

Pipeline: query understanding (spelling, intent, entities, rewrite), retrieval (BM25 plus dense, about 1,000 to 10,000 candidates), light ranker (GBDT, keep about 100), heavy ranker (keep about 20), blending. The stage counts are illustrative. Hybrid retrieval and cross encoders are covered in the RAG module.

**Learning to rank.** Pointwise: regress or classify a label. Pairwise (RankNet): $\mathcal L_{ij} = \log(1+e^{-(s_i-s_j)})$ with gradient magnitude $1/(1+e^{s_i-s_j})$ (0.731 when misordered by one unit). Listwise: ListNet or metric aware gradients. LambdaRank scales pair gradients by $|\Delta NDCG_{ij}|$, and LambdaMART fits boosted trees to those lambdas. For ranked grades 3, 2, 3, 0, 1, 2, swapping ranks 1 and 4 changes NDCG by $-0.273$, ranks 3 and 4 by $-0.033$.

**Position bias.** $P(\text{click}) = P(\text{examined}\mid k)\,P(\text{relevant})$. With illustrative examination 1.0, 0.62, 0.45, 0.33, 0.25 and clicks 0.10, 0.07, 0.05, 0.04, 0.03 the corrected relevance is 0.100, 0.113, 0.111, 0.121, 0.120. Fixes: inverse propensity weights (clip them), position as a feature pinned at serving, a bias tower, randomised slices. Rule 36: keep position features separate.

**Evaluation.** Raters grade 0 to 4, NDCG@10; online success rate, reciprocal rank of first click, abandonment, reformulation, zero results, latency. Team draft interleaving gives a sensitive preference signal with far less traffic than an A/B test, but no magnitude.

**Marketplace design (100M listings, 5,000 peak QPS, 300 ms).** Query understanding 5 ms, retrieval 40 ms, L1 30 ms, L2 60 ms, blending 10 ms. Labels: graded relevance plus corrected clicks and purchases. Cold start: content embeddings and an exploration boost.

## Ads click through rate prediction

Rank by $eCPM = 1000 \times bid \times pCTR$. A 1 dollar bid at a true CTR of 1 percent is worth 10 dollars per thousand; a model predicting 2 percent makes it 20. Calibration, the ratio of average predicted to observed CTR, must be near 1 overall and per slice.

**Normalized entropy** (He et al., 2014): $NE = \text{average log loss} / H(\bar p)$, $RIG = 1-NE$. Doubling predictions raised NE from 0.864 to 0.956 in a simulation with identical AUC (0.7914).

**Downsampling.** Keeping negatives with probability $w$ gives $p = q/(q+(1-q)w)$, so $q = p/(p+(1-p)/w)$, equivalent to $\text{logit}(q) = \text{logit}(p) + \ln w$. With $w = 0.1$ and a true rate of 2 percent the sampled rate is 16.95 percent and $p = 0.17$ maps to $q = 0.0201$. Facebook's example: 0.1 percent with $w = 0.01$ becomes about 9.1 percent (the paper says roughly 10 percent).

**Position bias.** $P(click\mid ad, slot) \approx P(examined\mid slot)\,P(attractive\mid ad)$; 4 percent attractiveness at examination 0.6 gives 2.4 percent. Train with slot as a feature and default it at serving, or factorise.

**Delayed feedback.** Fake negatives bias the rate down. Facebook's online joiner labels impressions negative after a fixed window, trading recency against click coverage. Delay modelling (Chapelle, KDD 2014, cited from memory), correction of early labels, and evaluation on mature cohorts.

**Freshness.** Daily instead of weekly retraining cut NE by about 1 percent. The linear layer is trained online with per coordinate learning rates $\eta_{t,i} = \alpha/(\beta+\sqrt{\sum_j g_{j,i}^2})$ ($\alpha = 0.1$, $\beta = 1$), and a circuit breaker disconnects the trainer if the joined stream's distribution jumps.

## Fraud detection

Constraints: base rate near 0.1 percent, asymmetric and variable cost, adversaries, label delay, decision in tens of milliseconds, selection bias on declines.

**Precision at low prevalence.** Precision $= \pi\,TPR/(\pi\,TPR + (1-\pi)FPR)$. With $\pi = 0.001$, $TPR = 0.9$: FPR 1 percent gives 8.26 percent precision (11.1 false alerts per fraud), 0.5 percent gives 15.3 percent, 0.1 percent gives 47.4 percent. On 2M transactions per day: 2,000 frauds, 1,800 caught, 19,980 false positives, 21,780 alerts, about 109 reviewers at 200 per day.

**Cost based threshold.** Block when $p\,c_{fn} > (1-p)\,c_{fp}$, so $p^* = c_{fp}/(c_{fp}+c_{fn})$. With 8 and 120: 0.0625. With $c_{fn}$ equal to the amount: 0.286 at 20 dollars, 0.0625 at 120, 0.0079 at 1,000. Needs calibrated probabilities and a correction after downsampling. The Hidden Technical Debt paper warns that fixed thresholds in dynamic systems go stale.

**System.** Hard rules, a GBDT with streaming velocity features (decayed counters), device and graph features, a policy layer by amount and segment, human review of the uncertain band, outcomes feeding labels. Drift: recency weighted retraining, rules as the fast lever, label free monitoring (score PSI, confirmed rate among reviewed alerts), a holdout slice, champion challenger.

## Feed ranking

Score $= \sum_k w_k \hat p_k$ over like, comment, share, dwell and negative heads. With 0.10, 0.02, 0.01, 0.005 and weights 1, 5, 10, -50 the score is $0.10+0.10+0.10-0.25 = 0.05$. Calibrate each head; weights are a product lever validated in A/B tests.

**Candidates and fan out.** In network and out of network pools. Fan out on write inserts a post into every follower's cached timeline (a 10M follower account causes 10M writes per post); fan out on read merges at request time; hybrids push for normal accounts and pull for very large ones.

**Policy layer and long term value.** Integrity demotion, author and topic caps, quality models, negative feedback. Rule 28 (short term behaviour does not guarantee long term behaviour), rule 38 (revisit objectives), rule 39 (launch decisions are proxies). Keep a 1 to 5 percent long term holdout.

## ETA prediction

Regression on duration. Loss: absolute error or Huber; avoid MAPE. Pinball loss $L_\tau = \max(\tau d, (\tau-1)d)$ with $d = y-\hat y$; at $\tau = 0.9$ a 5 minute under prediction costs 4.5 and an over prediction 0.5.

**Design.** Predict the residual over a routing engine baseline: $\hat y = b + \hat r$. Features: route, time, live traffic versus free flow, historical segment speed, weather, and for delivery restaurant preparation time and courier state. Log features at serving. Fall back to the baseline.

**Evaluation.** Predictions 10, 20, 30, 60 against actuals 12, 19, 45, 58: absolute errors 2, 1, 15, 2; MAE 5.0, RMSE 7.65, bias +3.5, MAPE 14.7 percent. Also p90 error, quantile coverage by slice, late rate, cancellations and conversion online. Padding trades lateness for abandonment, so choose it by experiment.

## Content moderation

Multi label classification per policy with exact hash matching and rules first, context features, and a policy layer.

**Bands (illustrative).** 10M posts per day, 0.2 percent prevalence: 20,000 violations. Score at least 0.95: 8,000 caught, 998 good removed, precision 88.9 percent. Score 0.5 to 0.95: 10,000 caught, 19,960 good, precision 33.4 percent, about 150 reviewers at 200 items. Recall 90 percent.

**Queues.** Priority $= P(\text{violation}) \times \text{reach} \times \text{severity}$ (0.9 x 100 = 90 versus 0.6 x 50,000 = 30,000). Double review a sample and track Cohen's kappa. Review labels are selection biased, so add a random sample.

**Measurement.** $n = z^2 p(1-p)/E^2$: prevalence 0.2 percent within 0.05 points needs 30,671 posts; recall within 5 points needs about 138 violations, about 69,000 random posts. Stratified sampling with inverse rate weights cuts cost.

## LLM support assistant

Chain: input guardrails, router, retrieval (RAG module), tools with least privilege, generation with citations, output guardrails, human fallback, logging.

**Latency.** Time to first token is driven by prompt length, retrieval and queueing; decode time by output length. 300 tokens at 50 tokens per second take 6 s, with first words at about 0.65 s when streaming. Prefix caching, speculative decoding, parallel guardrails.

**Cost (assumed prices 1 and 4 dollars per million input and output tokens).** 100,000 conversations x 4 turns x (3,000 in, 300 out) is 1.2B input and 120M output tokens, 1,680 dollars per day. Routing a fraction $f$ to a model costing $r$ times as much gives $C(1-f+fr)$: $f = 0.7$, $r = 0.1$ gives 621.60 dollars (63 percent less). A further 30 percent cache hit rate gives 435.12 dollars. Load: 4.63 turns per second average, 18.5 at 4 times peak, 1,389 decode tokens per second average.

**Caching risk.** A semantic cache can return a wrong answer for a near duplicate with different entities, so restrict it to non personal stable questions.

**Evaluation.** Golden set with citations (retrieval recall, faithfulness, correctness, refusal, red team) as a CI gate; online resolution with no re contact, CSAT, escalation, cost per resolution. A 1 percent wrong rate within 0.5 points needs $1.96^2 \times 0.01 \times 0.99/0.005^2 \approx 1{,}521$ reviews.

> **Watch:** [CME295 Lecture 8, LLM Evaluation](https://www.youtube.com/watch?v=8fNP4N46RRo) covers evaluating language model systems.

## Feature stores and point in time correctness

A feature store has a registry, an offline store (history), an online store (latest per key), materialisation jobs and a serving API. Michelangelo (Uber, 2017) reported about 10,000 features refreshed daily, Cassandra backed online features fed by batch or streaming jobs, a DSL shared by training and serving, and P95 latency under 5 ms (10 ms with lookups).

For label time $t$ use the latest feature value with feature time $\le t$.

```python
import pandas as pd

labels = pd.DataFrame({
    "user_id": [1, 1, 2],
    "event_time": pd.to_datetime(["2026-03-10 12:00", "2026-03-20 12:00", "2026-03-15 09:00"]),
    "label": [0, 1, 0],
}).sort_values("event_time")
features = pd.DataFrame({
    "user_id": [1, 1, 1, 2],
    "feature_time": pd.to_datetime(["2026-03-01", "2026-03-12", "2026-03-25", "2026-03-01"]),
    "orders_30d": [2, 5, 9, 1],
}).sort_values("feature_time")
train = pd.merge_asof(labels, features, left_on="event_time",
                      right_on="feature_time", by="user_id", direction="backward")
print(train[["user_id", "event_time", "orders_30d", "label"]])
```

On the example data user 1's 10 March label gets `orders_30d = 2` instead of the leaky 9 from 25 March. Distinguish event time from knowledge time for late data. Logging features at serving (rule 29) is point in time correct by construction. Backfill after definition changes, set TTLs, monitor freshness, and rule 10 warns about silent staleness.

## Training serving skew and inference modes

Sources: different code (rule 32), different data, different timing (rule 31), different traffic, missing or stale values. Rule 37's three gaps locate it. Tests: feature parity on logged requests, per feature PSI between training and serving, prediction parity by replaying the model on logged features.

Batch scores everyone on a schedule (50M users at 5,000 per second is 10,000 s, about 2.8 hours); online scores each request; streaming updates on events; hybrids mix them. Log features at serving (rule 29), keep new feature backfills, then compare logged and backfilled values for parity before shipping.

## Model registry, CI and CD for ML

A registry stores artifact, version, code commit, data reference, configuration, feature definitions, metrics, validation results, owner and stage. Michelangelo identifies models by UUID and tag and runs several side by side.

CI/CD: code tests, data validation, pipeline tests, a model gate (metric at least the champion's, slices, calibration, latency and memory, parity), packaging, staging, shadow, canary, A/B, automatic rollback, and a continuous training loop. Rule 5: test infrastructure independently of the model; rule 9: sanity checks before export. Reproducibility debt (non determinism, initial conditions) means aiming for statistical reproducibility: gains must exceed seed to seed spread.

## Shadow, canary and A/B rollout

Shadow copies requests and discards outputs (health, latency, score agreement, no behaviour effect). Canary serves 1 to 5 percent for real. A/B randomises and measures causal effect. Gates: errors, p99, PSI of scores versus the old model, guardrails. Rollback is a configuration change to a warm previous version; expand then contract for features; kill switch to a heuristic; keep a 1 to 2 percent long term holdout. A 10 versus 10 percent test on 100M users has 10M per arm against 745,600 needed in the earlier example.

## Monitoring and drift

Covariate shift changes $P(x)$, label shift $P(y)$, concept drift $P(y\mid x)$. Upstream data changes are bugs.

$PSI = \sum_i (a_i-e_i)\ln(a_i/e_i) = KL(a\|e)+KL(e\|a)$. Expected 0.20 in five bins against 0.10, 0.15, 0.20, 0.25, 0.30 gives per bin 0.0693, 0.0144, 0, 0.0112, 0.0405, total 0.1354. Rule of thumb: under 0.1 stable, 0.1 to 0.25 moderate, above 0.25 large; a convention from credit scoring.

```python
import numpy as np

def psi(expected, actual, bins=10, eps=1e-6):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))
```

Simulated results: same distribution 0.0003, mean shift 0.3 with scale 1.1 gives 0.0897, shift of one sd 0.9261. The 0.3 shift gives 0.079, 0.090, 0.096, 0.101 for 5, 10, 20, 50 bins, and a KS statistic of 0.12 with p near zero at 100,000 points, so alert on effect size and importance. With label delay monitor inputs and predictions first, then proxies, then mature cohorts. Prediction bias (predicted versus observed rate) and action limits come from the Hidden Technical Debt paper.

## Retraining, labelling and active learning

Measure the decay curve by training at day $T$ and scoring $T+1,\dots,T+k$ (rule 8). Facebook: daily versus weekly training cut NE by about 1 percent. Warm start daily, full retrain periodically. Online learning needs protections: stream distribution checks, learning rate floors (0.00001 in the paper), holdout validation before publishing, snapshots.

Cohen's kappa $= (p_o-p_e)/(1-p_e)$. For 40 yes yes, 50 no no, 6 and 4 disagreements: $p_o = 0.90$, $p_e = 0.5048$, $\kappa = 0.798$.

Active learning: uncertainty (least confidence, margin, entropy), committee disagreement, diversity. Pitfalls: sampling bias, redundancy, noise, cold start. In a 30 seed toy run (logistic regression, 200 labels), uncertainty sampling reached accuracy 0.7585 versus 0.7465 for random, within a seed spread of about 0.08.

## ML technical debt

Sculley et al. (2015): debt sits at the system level. CACE: changing anything changes everything. Correction cascades add dependencies. Undeclared consumers create visibility debt and hidden loops. Unstable data dependencies need versioned signals. Underutilised inputs (legacy, bundled, epsilon, correlated features) are found by leave one out tests. Direct and hidden feedback loops. Glue code (a mature system might be at most 5 percent ML code), pipeline jungles, dead experimental codepaths (the paper cites Knight Capital losing 465 million dollars in 45 minutes), abstraction debt, configuration debt (small diffs, review, verification). Five debt questions: new algorithm at full scale, transitive closure of data dependencies, measuring the impact of change, whether improving one signal degrades others, and onboarding time.

## What interviewers probe

- Why a baseline first, and what would make you move up the model ladder?
- What exactly is the label, how biased and how late is it?
- Which metric, which guardrail, and why can offline and online disagree?
- Walk through the latency budget. What happens when the ranker times out?
- How do you handle cold start, position bias, delayed feedback and feedback loops?
- Derive the recalibration after negative downsampling, or the cost based threshold.
- How would you detect and prevent training serving skew?
- How would you roll out safely and roll back in under a minute?
- Compute PSI by hand, and say what threshold you would alert on and why.
- What is your retraining cadence and how did you decide it?
- Which technical debt patterns would you expect in this system?

## Sources

- [alirezadir Machine Learning Interviews: ML system design framework and sample questions](https://raw.githubusercontent.com/alirezadir/Machine-Learning-Interviews/main/src/MLSD/ml-system-design.md)
- [Chip Huyen: Machine Learning Interviews book](https://huyenchip.com/ml-interviews-book/)
- [Google: Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Hidden Technical Debt in Machine Learning Systems (Sculley et al., NIPS 2015)](https://papers.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
- [Deep Neural Networks for YouTube Recommendations (Covington, Adams, Sargin, RecSys 2016)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
- [Practical Lessons from Predicting Clicks on Ads at Facebook (He et al., ADKDD 2014)](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
- [Meet Michelangelo: Uber's Machine Learning Platform](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
