# ML System Design: cheatsheet

One page for the ten minutes before the interview. Numbers marked "assumed" are example inputs for the arithmetic, not facts about any company. Everything else was computed by running it or comes from a source listed in `sources.json`.

## The 11 steps and 45 minute boxes

| Min | Step | Say |
| --- | --- | --- |
| 0 to 5 | 1 Clarify | goal, users, scale, latency, data, limits; write assumptions |
| 5 to 10 | 2 Frame, 3 Metrics | input, output, label in one line; offline, online, guardrail |
| 10 to 18 | 4 Data and labels, 5 Features | label source, bias, delay; user, item, context, cross |
| 18 to 28 | 6 Model, 7 Training, 8 Offline eval | baseline first; time split; slices, calibration |
| 28 to 37 | 9 Serving | funnel, latency budget at p99, batch vs online, cache |
| 37 to 43 | 10 Online test, 11 Monitor | shadow, canary, A/B; drift, label delay, retrain |
| 43 to 45 | Wrap | two risks, next step |

5 + 5 + 8 + 10 + 9 + 6 + 2 = 45. Multiply by about 1.3 for 60 minutes.

## Pattern to design

| Output | Pattern | Watch for |
| --- | --- | --- |
| Ranked list from a catalog | Funnel: retrieval, ranking, reranking | position bias, cold start, feedback loops |
| Risk score on events | Imbalanced classifier, thresholds, review | base rate, cost, label delay, adversaries |
| A number | Regression, residual over a baseline | loss shape, late vs early |
| Text | LLM system: retrieval, guardrails, routing | latency, cost, grounding, evaluation |

## Metrics

- P@k = relevant in top k over k. R@k = relevant in top k over all relevant. RR = 1 over rank of first hit.
- DCG@k = sum of (2^rel - 1) / log2(i + 1). NDCG = DCG / ideal DCG. Grades 3,2,3,0,1,2: 13.85 / 14.60 = 0.949.
- At 0.1 percent base rate, TPR 90 percent, FPR 1 percent: precision 8.3 percent, 11 false alerts per fraud.
- AUC ignores calibration. Normalized entropy = log loss / entropy of base rate (1.0 means no better than base rate).
- Calibration ratio = average predicted / observed (aim for 1), per slice.
- Rule 37 (Google): train vs holdout, holdout vs next day, next day vs live.

## Estimation numbers

| Quantity | Value |
| --- | --- |
| QPS | requests per day / 86,400, peak 2 to 5 times |
| 50M users x 10 requests | 500M per day, 5,787 QPS, 17,361 peak (3 times) |
| 500 candidates at peak | 8.7M item scores per second |
| Servers (500 QPS each, assumed, 50 percent headroom) | 35, so 52 |
| 10B impressions x 200 bytes | 2 TB per day, 180 TB for 90 days |
| 100M items x 128 dims x 4 bytes | 51.2 GB (float16 25.6, PQ 16 bytes 1.6) |
| Streaming 10B events per day | 115,741 per second |
| Counters, 50M x 20 | 8 GB; 60 minute buckets 480 GB; decayed 16 GB |
| Dense model FLOPs | 2 x weights per example; 3M weights x 500 = 3 GFLOPs |
| 7B model weights | 14 GB fp16, 7 GB int8, 3.5 GB 4 bit |
| Adam mixed precision training | 16 bytes per parameter, 112 GB for 7B |
| KV cache per token | 2 x layers x KV heads x head dim x bytes; 0.5 MB for 32 x 32 x 128 fp16 |

Latency split for 200 ms: 20 network, 15 gateway, 30 features, 25 retrieval, 60 ranking, 10 rules, 10 response = 170, slack 30.

## Formulas to reproduce

| Topic | Formula |
| --- | --- |
| Sample size per arm | 2 (z_a/2 + z_b)^2 p(1-p) / delta^2; p 0.05, 2 percent lift gives 745,600 |
| Recalibrate after keeping negatives at rate w | q = p / (p + (1-p)/w); logit q = logit p + ln w |
| Cost threshold | block if p > c_fp / (c_fp + c_fn); 8 and 120 gives 0.0625 |
| Pinball loss | max(tau d, (tau - 1) d); tau 0.9: 4.5 under, 0.5 over |
| eCPM | 1000 x bid x pCTR |
| Cache mean latency | h x t_hit + (1-h) x t_miss; 0.4 x 5 + 0.6 x 120 = 74 |
| Routing cost | C (1 - f + f r); f 0.7, r 0.1 on 1,680 gives 621.60 |
| MMR | argmax lambda rel - (1 - lambda) max sim to chosen |
| RankNet | log(1 + exp(-(s_i - s_j))); LambdaRank times abs delta NDCG |
| Two tower in batch loss | softmax of u.v / tau - log q_j (logQ correction) |
| YouTube watch time | weight clicks by T, others by 1; odds = sum T / (N - k) = E[T] / (1 - P) |
| PSI | sum (a - e) ln(a / e); 20 percent x 5 vs 10,15,20,25,30 gives 0.135 |
| Cohen kappa | (p_o - p_e) / (1 - p_e); 0.90 and 0.505 gives 0.80 |
| Sample for a rate | n = z^2 p(1-p) / E^2; 0.2 percent within 0.05 points: 30,671 |

PSI rule of thumb: under 0.1 stable, 0.1 to 0.25 moderate, above 0.25 large. It is a convention, not a test.

## Designs in one line

- **Recommendation:** ANN two tower + other sources (about 1,000), multi head ranker with watch time, MMR plus rules, cold start by content and priors, exploration slice, guardrail hide rate.
- **Search:** query understanding, BM25 plus dense, LambdaMART then neural, NDCG@10, position bias correction, interleaving then A/B.
- **Ads CTR:** calibrated pCTR, downsampling fix, position as feature pinned at serving, delay window, daily retrain, online linear layer.
- **Fraud:** rules + GBDT + policy layer, streaming velocity features, cost thresholds by amount, holdout slice, label delay.
- **Feed:** per outcome heads (like, comment, share, hide), weighted value, hybrid fan out, policy layer, long term holdout.
- **ETA:** residual over routing engine, quantile loss, MAE, bias, coverage, late rate.
- **Moderation:** multi label per policy, bands (remove, review, allow), priority = P x reach x severity, random audit sample.
- **LLM support:** guardrails, router, RAG with citations, small model routing, caches, golden set, resolution not deflection.

## MLOps checklist

- Feature store: one definition, point in time join, TTL, backfill, freshness alert.
- Skew: log features at serving (rule 29), parity tests, replay the model on logged features.
- Registry: version, code, data, config, features, metrics, stage; rollback by re pointing.
- CI/CD gate: beats champion, no slice worse than 1 percent, calibration, p99, parity.
- Rollout: shadow, 1 percent canary, A/B, ramp, holdout, kill switch, previous model warm.
- Monitoring: data quality, drift (PSI), prediction mean vs observed, proxies, mature cohorts, action limits.
- Retrain: measure the decay curve (rule 8); warm start daily, full retrain periodically.
- Debt: CACE, undeclared consumers, unstable inputs, glue code, dead branches, config review.

## If asked X, say Y

- "Why not a bigger model?" Show the latency budget and the error analysis that justifies it.
- "Offline up, online flat?" Logs reflect the old policy, proxy mismatch, skew, novelty; A/B decides.
- "How do you know the labels are good?" Source, bias, delay, noise, double labelling and kappa.
- "Why is accuracy useless here?" 0.1 percent base rate; use precision at recall and cost.
- "What if labels arrive late?" Wait window, mature cohorts, proxy signals, delay modelling.
- "How do you avoid leakage?" Time split, point in time joins, log features at serving.
- "What would you monitor?" Data quality, feature drift, score distribution, proxies, mature metrics.
- "How do you roll back?" Previous version warm, a config change, kill switch to a heuristic.
