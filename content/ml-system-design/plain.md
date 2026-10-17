# ML System Design: the plain version

This is the easy version of the machine learning (ML) system design round. In this interview you design a whole ML product from a vague prompt, such as a recommender, a search engine, a fraud detector or a support chatbot. The guide follows one reusable framework, then walks through the classic designs, then covers how ML systems are run safely after launch. Each section gives the idea in everyday words, one example with real numbers, and the place where it usually breaks.

## What the ML system design round tests

The round is a 45 to 60 minute conversation. You get a one line prompt like "design a video recommendation system", and you must turn it into a working plan. Think of designing a restaurant kitchen, not cooking one dish. The recipe (the model) matters, but so do the ingredients (data), the order system (serving) and the health inspections (monitoring).

Chip Huyen's interview book says almost every company asks at least one such question, and that candidates often find it the hardest. Interviewers look at framing, data, metrics, model choice, serving, monitoring and how clearly you talk.

> **Interview trap:** spending 40 of the 45 minutes on the model and never mentioning labels or monitoring.

## The answer framework and time boxes

Use the same eleven steps every time and say them out loud. Clarify the goal, frame it as an ML task, choose metrics, design data and labels, design features, pick a model starting from a simple baseline, plan training, evaluate offline, design serving, run an online experiment, then monitor and iterate.

A workable 45 minute split is 5, 5, 8, 10, 9, 6 and 2 minutes across seven blocks. Start simple: Google's Rules of Machine Learning says to launch without ML if a rule will do, because a rule gives you real data and a score to beat.

Large systems use a funnel. Cheap steps cut millions of items to a few hundred, and a heavier model then ranks those few.

> **Remember:** a later step can only show items the earlier step kept.

## Clarifying and framing the problem

Spend three to five minutes asking about the goal, the users, the scale, the speed needed, the data and the limits. Write your assumptions down, for example 50 million daily users and 100 million videos.

Then say in one line what the model takes in, gives out and learns from. The thing it learns from is the label, the known right answer for past examples.

You usually optimise a proxy, a stand in for the real goal. Clicks are easy to count but reward clickbait. YouTube reports that ranking by click rate promotes videos people do not finish, so it predicts expected watch time instead.

## Offline and online metrics

Offline metrics are measured on stored logs before launch. Online metrics are measured on live users in an experiment, and they are the final judge. They do not always agree.

For a ranked list, precision at k is the share of the top k results that are relevant. A score called NDCG (normalized discounted cumulative gain) rewards putting the best results first. For grades 3, 2, 3, 0, 1, 2 the total is 13.85, the best possible is 14.60, so NDCG is 0.949.

A guardrail is a measure that must not get worse, such as hide rate or response time. Name one offline metric, one online metric and one guardrail for every design.

## Data, labels and features

Labels come from behaviour logs, explicit feedback, human annotation or rules. Logs are plentiful but only show what the old system displayed, and a missing click does not mean dislike.

Split data by time, not randomly. Leakage means the model sees information in training that will not exist at prediction time, such as an order count that includes the next 30 days.

For rare events, accuracy is meaningless. Use precision at a fixed recall, and fix training by weighting or dropping common examples, then correct the probabilities.

Features describe the user, the item, the moment, and the pair. Facebook found the top 10 features in its ads model were all about history.

> **Watch:** [CS229 Lecture 6, Dataset Split, ML Advice](https://www.youtube.com/watch?v=llnEgyyuYkQ) covers splitting data and practical advice for ML projects.

## Models from baseline up

Climb a ladder: a rule, logistic regression, boosted decision trees, then neural networks. Move up only when studying the model's mistakes shows what the next step would fix.

Boosted trees are the default for table data. Neural networks win on text, images, sequences and huge lists of IDs. Facebook's hybrid, where trees prepare features for a simple linear model, beat either alone by over 3 percent.

Evaluate beyond one number. Test on later data, split results by group such as new users, and check calibration, which asks whether a predicted 10 percent really happens 10 percent of the time.

> **Say it like this:** "I'll start with a popularity baseline, then a gradient boosted model, and move to a neural ranker only if error analysis justifies it."

## Serving, latency and experiments

Split the time limit across steps and plan for the slowest requests. For 200 milliseconds, one split is 20, 15, 30, 25, 60, 10 and 10, which totals 170 and leaves 30 spare. The slowest 1 percent of requests is called p99 latency.

Precompute what changes slowly, such as item embeddings, and compute what depends on the request. A cache trades freshness for speed. If 40 percent of requests hit a 5 millisecond cache and the rest take 120, the average is 74.

For an A/B test, assign users at random. With a 5 percent click rate and a 2 percent relative gain, you need about 745,000 users per group. A model also needs some exploration, because it only learns about items it shows.

## Back of the envelope estimation

Estimation means quick, checked arithmetic with stated assumptions. Queries per second is requests per day divided by 86,400, times a peak factor of 2 to 5. For 50 million users making 10 requests, that is 5,787 per second on average and about 17,400 at 3 times peak.

Storage is rows times bytes times retention. Ten billion impressions at 200 bytes is 2 terabytes a day. Embeddings for 100 million items at 128 numbers of 4 bytes take 51.2 gigabytes.

A 7 billion parameter model needs 14 gigabytes at 2 bytes each. Training with a standard setup needs about 16 bytes per parameter, which is 112 gigabytes.

## Recommendation system

Build three stages. Retrieval finds about 1,000 candidates, using a two tower model: one network for the user, one for the item, scored by how well their vectors line up. Item vectors are computed ahead and searched with approximate nearest neighbour search.

A ranker with many features then predicts outcomes such as watch, like and hide, and a weighted sum gives one score. YouTube weights clicked examples by watch time so one simple model predicts expected watch time.

A reranker adds variety and rules. Plan for cold start, which means new users and items with no history, and for feedback loops, where the model shapes its own training data.

## Search ranking

A search request goes through query understanding, retrieval, ranking and blending. Query understanding fixes spelling and finds intent, for example brand, category and size in "nike running shoes mens 10".

Learning to rank has three styles. Pointwise scores each document alone, pairwise learns which of two should rank higher, and listwise looks at the whole list. A method called LambdaMART focuses effort on mistakes near the top.

Clicks are a biased label, because top positions get seen more. If clicks fall from 10 percent at position 1 to 7 percent at position 2, but position 2 is seen only 62 percent as often, the true relevance is about 11 percent in both.

## Ads click through rate prediction

An ad auction multiplies the bid by the predicted click probability, so the probability must be right, not only ordered. If a model predicts 2 percent when the truth is 1 percent, a 1 dollar bid acts like 2 dollars.

If you drop most negative examples to save compute, the probabilities come out too high. With 10 percent of negatives kept and a 2 percent true rate, an output of 0.17 maps back to about 0.02.

Some labels arrive late, so wait before labelling and judge only on older data. Facebook found daily retraining beat weekly by about 1 percent.

## Fraud detection

Fraud is rare, costly, adversarial and slow to confirm. At a 0.1 percent fraud rate, catching 90 percent while flagging 1 percent of good payments gives only 8.3 percent precision, about 11 false alerts per real fraud.

Choose the block threshold from costs. With 8 dollars of friction and 120 dollars of fraud loss, block when the probability exceeds 6.25 percent. The threshold should fall as the amount rises.

Use rules for known patterns and fast fixes, a model for the rest, and people for the uncertain middle. Fraudsters adapt, so watch signals that need no labels.

## Feed ranking

A feed ranker predicts several outcomes for each post, such as like, comment, share and hide, and a weighted sum gives one score. With chances of 0.10, 0.02, 0.01 and 0.005, and weights 1, 5, 10 and minus 50, the score is 0.05. The hide term dominates on purpose.

Candidates come from followed accounts and recommended posts. Pushing each post into every follower's timeline is fast to read, but one post from a 10 million follower account means 10 million writes. Many systems mix push and pull.

A rules layer adds integrity and variety. Keep a long term holdout, because short term engagement is only a proxy.

## ETA prediction

Estimated time of arrival (ETA) is a regression problem: predict minutes. A routing engine already knows the map, so the model learns the correction, such as rush hour, weather and restaurant preparation time.

Being late hurts more than being early, so many systems predict a high quantile. A 90th percentile loss charges 4.5 for being 5 minutes too low and 0.5 for being 5 minutes too high.

For predictions of 10, 20, 30 and 60 minutes against actuals of 12, 19, 45 and 58, the average miss is 5.0 minutes and the bias is 3.5 minutes late.

## Content moderation

Moderation scores each post for each policy, such as hate, nudity or spam. The written policy defines the labels, and exact matching of known bad content runs first.

Use several thresholds. Very high scores are removed automatically, the middle band goes to human review, and low scores are allowed. With 10 million posts a day and 0.2 percent violations, one setup catches 8,000 automatically and 10,000 by review, needing about 150 reviewers.

To measure prevalence to within 0.05 percentage points you must label about 30,700 random posts, because the model only shows what it flagged.

## LLM support assistant

A support assistant is a chain of checks around a language model. Input checks catch hidden instructions and personal data, a router picks between help articles, tools and a person, and retrieval grounds the answer in current text with citations.

For 100,000 conversations a day, 4 turns each, 3,000 input and 300 output tokens per turn, assumed prices give about 1,680 dollars a day. Sending 70 percent of traffic to a model at one tenth the price cuts it to about 622 dollars.

Evaluate with a golden set offline and with resolution, not just deflection, online. About 1,500 reviewed conversations estimate a 1 percent wrong answer rate within half a point.

> **Watch:** [CME295 Lecture 8, LLM Evaluation](https://www.youtube.com/watch?v=8fNP4N46RRo) covers how to evaluate language model systems.

## Feature stores and point in time correctness

A feature store keeps one definition of each feature and serves it two ways: full history for training, and the latest value for fast lookups. It stops teams rebuilding the same signal and keeps training and serving consistent.

Point in time correctness means each training example uses only values known at its moment. A user's order count was 2 on 10 March, but a naive join attaches the latest value, 9 from 25 March, which leaks the future.

> **Worked example:** join each label to the latest feature row at or before its timestamp, for instance with the pandas function merge_asof.

## Training serving skew and inference modes

Training serving skew means the model sees different inputs in production than in training. Causes are different code, different data, different timing and different traffic. Nothing errors. The model just quietly gets worse.

Prevent it with one feature definition, by logging the features used at serving, and with parity tests that compare logged and recomputed values.

Batch inference scores everyone on a schedule and stores results. Online inference scores each request. Scoring 50 million users nightly at 5,000 per second takes about 2.8 hours, which fits an overnight window.

## Model registry, CI and CD for ML

A model registry is a versioned catalogue of trained models with code, data, settings, metrics and a stage. Deploying means pointing serving at a version, and rollback means pointing back.

Continuous integration tests code, data and models before they merge. A gate says the new model must match or beat the current one, no user group may drop more than 1 percent, and the slowest requests must stay within the limit.

Record the code, data snapshot, settings and seeds so a model can be rebuilt. Exact repeats are often impossible, so aim for results within normal noise.

## Shadow, canary and A/B rollout

A shadow deployment sends copies of live requests to the new model and hides the answers. It tests speed and errors but not user behaviour. A canary serves a small share, say 1 percent, of real traffic. An A/B test splits users at random and measures the real effect.

Roll out in stages with automatic checks, keep the old model warm for a rollback in under a minute, and keep a kill switch that falls back to a simple rule.

> **Interview trap:** flipping 100 percent of traffic to a new model because offline metrics improved.

## Monitoring and drift

Drift means the data or world changes after training. Covariate shift changes the inputs, label shift changes the outcome rate, and concept drift changes the relationship between them.

PSI (population stability index) compares a feature's training and live distributions. With 20 percent in each of 5 bins against live shares of 10, 15, 20, 25 and 30 percent, PSI is 0.135, a moderate shift. Under 0.1 is usually stable and over 0.25 is large.

When labels arrive late, watch inputs and predictions first, then fast proxy signals, then true labels on older groups.

## Retraining, labelling and active learning

Measure how fast a model gets worse as it ages, then retrain before the loss costs more than the retrain. Facebook found retraining daily instead of weekly cut a click model's error by about 1 percent.

Check label quality by having two people label the same items. If they agree on 90 of 100 and chance agreement is about 0.505, Cohen's kappa is about 0.80.

Active learning lets the model pick which items a person should label next. Gains are not guaranteed, so evaluate on a random test set.

## ML technical debt

Technical debt is the future cost of a shortcut. A Google paper argues ML systems build it fast, because the model is a small part of a larger system. A mature system might be at most 5 percent ML code, with the rest glue code and pipelines.

CACE means changing anything changes everything: changing one input can change how the model uses all the others. Undeclared consumers are systems that quietly use your model's output, so you cannot change it safely.

> **Remember:** reward deleting features and reducing complexity as much as raising accuracy.
