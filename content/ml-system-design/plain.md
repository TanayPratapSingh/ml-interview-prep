# ML System Design: the plain version

This is the easy version of the machine learning (ML) system design round. In this interview you design a whole ML product from a vague prompt. Examples are a recommender (a system that suggests items to a user), a search engine, a fraud detector or a support chatbot. The guide follows one reusable framework. Then it walks through the classic designs. Last, it covers how teams run ML systems safely after launch. Each section gives the idea in everyday words, one example with real numbers, and the place where it usually breaks.

## What the ML system design round tests

The round is a conversation of 45 to 60 minutes. You get a one line prompt such as "design a video recommendation system". You must turn it into a working plan.

Think of designing a restaurant kitchen, not cooking one dish. The recipe is the model, a program that learns patterns from data. It matters, but so do three other parts. The ingredients are the data. The order system is serving, which means running the model for real users. The health inspections are monitoring, which means watching the system after launch.

Chip Huyen's interview book says almost every company asks at least one such question. It also says candidates often find it the hardest. Interviewers look at framing (stating the problem as an ML task), data and metrics (the scores that tell you if it works). They also look at model choice, serving, monitoring and how clearly you talk.

> **Interview trap:** spending 40 of the 45 minutes on the model and never mentioning labels (the known right answers the model learns from) or monitoring.

## The answer framework and time boxes

Use the same eleven steps every time, and say them out loud.

1. Clarify the goal.
2. Frame it as an ML task.
3. Choose metrics.
4. Design the data and labels.
5. Design features (the facts the model reads about each example).
6. Pick a model, starting from a simple baseline (a basic version that later versions must beat).
7. Plan training (how the model learns).
8. Evaluate offline (test on stored data before launch).
9. Design serving.
10. Run an online experiment (test on live users).
11. Monitor and iterate.

A workable 45 minute split is 5, 5, 8, 10, 9, 6 and 2 minutes across seven blocks.

Start simple. Google's Rules of Machine Learning says to launch without ML if a rule will do. A rule is a hand written instruction. It gives you real data and a score to beat.

Large systems use a funnel. Cheap steps cut millions of items to a few hundred. A heavier model then ranks those few, which means it puts them in order.

> **Remember:** a later step can only show items the earlier step kept.

## Clarifying and framing the problem

Start by asking questions for three to five minutes. Ask about the goal, the users, the scale (how many users and items), the speed needed, the data you have and the limits. Write your assumptions down, for example 50 million daily users and 100 million videos.

Then say in one line what the model takes in, what it gives out and what it learns from. The thing it learns from is the label, the known right answer for past examples.

You usually optimise a proxy, which is a stand in for the real goal. Clicks are easy to count, but they reward clickbait (eye catching titles that disappoint). YouTube reports that ranking by click rate promotes videos people do not finish. So it predicts expected watch time instead.

## Offline and online metrics

Offline metrics are scores measured on stored logs (records of past behaviour) before launch. Online metrics are scores measured on live users in an experiment. Online metrics are the final judge. The two do not always agree.

Ranking metrics score an ordered list. Precision at k is the share of the top k results that are relevant. NDCG (normalized discounted cumulative gain) gives each result a grade for how useful it is. It then counts good results for more when they sit higher in the list.

Take grades 3, 2, 3, 0, 1, 2 for six results in that order. The total score is 13.85. The best possible order would score 14.60. NDCG divides the first by the second, so it is 0.949.

A guardrail is a measure that must not get worse, such as hide rate (how often users hide a result) or response time. Name one offline metric, one online metric and one guardrail for every design.

## Data, labels and features

Labels come from behaviour logs, explicit feedback (ratings users give), human annotation (people marking examples by hand) or rules. Logs are plentiful, but they only show what the old system displayed. A missing click does not mean dislike.

Split data by time, not randomly. Train on the past and test on the later period. A random split can cause leakage. Leakage means the model sees information in training that will not exist at prediction time. An example is an order count that includes the next 30 days.

For rare events, accuracy is meaningless, because a model that always says "no" is almost always right. Use precision at a fixed recall instead. Precision is the share of flagged cases that are truly positive. Recall is the share of all true positives you caught. To train on rare events, give rare examples more weight or drop many common ones. Then correct the probabilities, because the training mix no longer matches the real world.

Features describe the user, the item, the moment and the pair (a user and an item together). Facebook found the top 10 features in its ads model were all about history, meaning past behaviour.

> **Watch:** [CS229 Lecture 6, Dataset Split, ML Advice](https://www.youtube.com/watch?v=llnEgyyuYkQ) covers splitting data and practical advice for ML projects.

## Models from baseline up

Climb a ladder of models. Start with a rule. Next is logistic regression, a simple model that adds up weighted features to predict a probability. Then come boosted decision trees, many small trees built one after another, each fixing the last one's mistakes. The top step is neural networks, models built from layers of simple connected units. Move up only when studying the model's mistakes shows what the next step would fix.

Boosted trees are the default for table data (rows and columns, like a spreadsheet). Neural networks win on text, images, sequences and huge lists of IDs (identifiers that name each user or item). Facebook's hybrid used trees to prepare features for a simple linear model (one that adds up weighted inputs, like logistic regression). It beat either method alone by over 3 percent.

Evaluate beyond one number. Test on later data. Split results by group, such as new users. Check calibration, which asks whether a predicted 10 percent really happens 10 percent of the time.

> **Say it like this:** "I'll start with a popularity baseline, then a gradient boosted model, and move to a neural ranker only if error analysis justifies it." Gradient boosted models are boosted trees. A neural ranker is a neural network that orders items. Error analysis means reading the model's mistakes.

## Serving, latency and experiments

Serving means running the model on real requests, and time is the main limit. Latency is the delay a user waits. Split the time limit across the steps, and plan for the slowest requests.

For 200 milliseconds, one split is 20, 15, 30, 25, 60, 10 and 10, which totals 170 and leaves 30 spare. The slowest 1 percent of requests is called p99 latency, the wait that 99 percent of requests beat.

Precompute what changes slowly, such as item embeddings (lists of numbers that describe each item). Compute only what depends on the request. A cache stores ready answers. It trades freshness for speed, because stored answers can be out of date. If 40 percent of requests hit a 5 millisecond cache and the rest take 120, the average is 74.

An A/B test splits users at random into two groups, A and B, and compares them. With a 5 percent click rate and a 2 percent relative gain, you need about 745,000 users per group. A relative gain of 2 percent means 5.0 percent becomes 5.1.

A model also needs some exploration. It only learns about items it shows, so sometimes it must show items it would not normally pick.

## Back of the envelope estimation

Estimation means quick arithmetic with stated assumptions. Check each step as you go.

Queries per second is the number of requests that arrive each second. Divide requests per day by 86,400, the seconds in a day. Then multiply by a peak factor of 2 to 5 for the busiest times. For 50 million users making 10 requests, that is 5,787 per second on average and about 17,400 at 3 times peak.

Storage is rows times bytes per row times retention (how long you keep data). A byte is a basic unit of computer memory. An impression is one time an item is shown to a user. Ten billion impressions at 200 bytes is 2 terabytes a day. Embeddings for 100 million items at 128 numbers of 4 bytes take 51.2 gigabytes.

A parameter is one learned number inside a model. A 7 billion parameter model needs 14 gigabytes at 2 bytes each. Training with a standard setup needs about 16 bytes per parameter, because it also stores extra working numbers. That is 112 gigabytes.

## Recommendation system

Build a recommender in three stages. Retrieval finds about 1,000 candidates (items that might fit). It uses a two tower model: one neural network for the user, one for the item. Each turns its input into a vector, a list of numbers. The score is how well the two vectors line up.

Item vectors are computed ahead and searched with approximate nearest neighbour search, a fast method that finds close vectors and accepts small misses.

A ranker with many features then predicts outcomes such as watch, like and hide. A weighted sum gives one score. YouTube weights each clicked example by its watch time, so one simple model predicts expected watch time.

A reranker adds variety and rules.

Plan for cold start, which means new users and items with no history. Plan for feedback loops, where the model shapes its own training data: it shows items, people click them, and it learns to show more of the same.

## Search ranking

A search request goes through four steps: query understanding, retrieval, ranking and blending (mixing result types into one page). Query understanding fixes spelling and finds the intent. In "nike running shoes mens 10" it finds the brand, the category and the size.

Learning to rank means training a model to order results. It has three styles. Pointwise scores each document alone. Pairwise learns which of two documents should rank higher. Listwise looks at the whole list. A method called LambdaMART focuses effort on mistakes near the top.

Clicks are a biased label, because top positions get seen more. This is called position bias. Suppose clicks fall from 10 percent at position 1 to 7 percent at position 2. But position 2 is seen only 62 percent as often.

Divide each click rate by how often the position is seen. That gives 10 percent for position 1 and about 11 percent for position 2. So the true relevance is about the same at both positions, close to 11 percent.

## Ads click through rate prediction

Click through rate (CTR) is the share of ad views that end in a click. An ad auction (a bidding contest that picks the ad to show) multiplies the bid by the predicted click probability. So the probability must be right in size, not only in order. If a model predicts 2 percent when the truth is 1 percent, a 1 dollar bid acts like 2 dollars.

Teams often drop most negative examples (ad views with no click) to save compute. That makes the probabilities come out too high. Map the output back to the true scale. With 10 percent of negatives kept and a 2 percent true rate, an output of 0.17 maps back to about 0.02.

Some labels arrive late, because a click can come long after the ad is shown. So wait before labelling, and judge only on older data. Retraining means training again on fresh data. Facebook found daily retraining beat weekly by about 1 percent.

## Fraud detection

Fraud is rare, costly, adversarial (the other side fights back) and slow to confirm.

Rare events make precision low. At a 0.1 percent fraud rate, catching 90 percent while flagging 1 percent of good payments gives only 8.3 percent precision. That is about 11 false alerts per real fraud.

Choose the block threshold, the score above which you act, from costs. Friction is the cost of annoying a good customer. Block when the chance of fraud times the loss beats the chance of a good payment times the friction. With 8 dollars of friction and 120 dollars of fraud loss, block when the probability exceeds 6.25 percent. The threshold should fall as the amount rises.

Use rules for known patterns and fast fixes, a model for the rest, and people for the uncertain middle. Fraudsters adapt, so also watch signals that need no labels, such as shifts in inputs and scores.

## Feed ranking

A feed ranker predicts several outcomes for each post, such as like, comment, share and hide. A weighted sum (each chance times its weight, added up) gives one score. With chances of 0.10, 0.02, 0.01 and 0.005 and weights 1, 5, 10 and minus 50, the score is 0.05. The hide term dominates on purpose.

Candidates come from followed accounts and recommended posts. Pushing each post into every follower's timeline at write time makes reading fast. But one post from an account with 10 million followers means 10 million writes. Pulling means building the timeline when the user opens the app. Many systems mix push and pull.

A rules layer adds integrity (keeping harmful content out) and variety. Keep a long term holdout, a group that stays on the old system, because short term engagement is only a proxy.

## ETA prediction

Estimated time of arrival (ETA) is a regression problem, which means predicting a number: the minutes. A routing engine already knows the map. So the model learns the correction, such as rush hour, weather and restaurant preparation time.

Being late hurts more than being early, so many systems predict a high quantile. A quantile is a cut point. The 90th percentile is the value that 90 percent of outcomes fall below. A 90th percentile loss charges 4.5 for being 5 minutes too low and 0.5 for being 5 minutes too high. A loss is the penalty the model learns to shrink.

Take predictions of 10, 20, 30 and 60 minutes against actuals of 12, 19, 45 and 58. The average miss is 5.0 minutes. The bias is 3.5 minutes late, meaning predictions run low on average.

## Content moderation

Moderation scores each post for each policy, such as hate, nudity or spam. The written policy defines the labels. Exact matching of known bad content (a lookup against a list of items already known to be bad) runs first.

Use several thresholds. Very high scores are removed automatically. The middle band goes to human review. Low scores are allowed. With 10 million posts a day and 0.2 percent violations, one setup catches 8,000 automatically and 10,000 by review. That needs about 150 reviewers.

Prevalence is the share of all posts that break the rules. To measure it to within 0.05 percentage points, you must label about 30,700 random posts. The model only shows what it flagged, so a random sample is the only way to see what it missed.

## LLM support assistant

A support assistant is a chain of checks around a large language model (LLM), a program trained on huge amounts of text to write replies. Input checks catch hidden instructions (text that tries to trick the model) and personal data. A router picks between help articles, tools and a person. Retrieval (looking up relevant documents) grounds the answer in current text with citations, so each claim points to its source.

Take 100,000 conversations a day, 4 turns each, with 3,000 input and 300 output tokens per turn. A turn is one user message and one reply. A token is a small chunk of text, often a word or part of one. Assumed prices of 1 and 4 dollars per million input and output tokens give about 1,680 dollars a day. Sending 70 percent of traffic to a model at one tenth the price cuts it to about 622 dollars.

Offline, evaluate with a golden set, a fixed list of questions with checked right answers. Online, measure resolution, not just deflection. Resolution means the problem was solved. Deflection only means no person was needed. About 1,500 reviewed conversations estimate a 1 percent wrong answer rate within half a point.

> **Watch:** [CME295 Lecture 8, LLM Evaluation](https://www.youtube.com/watch?v=8fNP4N46RRo) covers how to evaluate language model systems.

## Feature stores and point in time correctness

A feature store keeps one definition of each feature and serves it two ways. One way is full history, for training. The other is the latest value, for fast lookups. It stops teams from rebuilding the same signal, and it keeps training and serving consistent.

Point in time correctness means each training example uses only values known at its moment. A user's order count was 2 on 10 March. A naive join (a simple match of rows from two tables) attaches the latest value, which is 9 from 25 March. That leaks the future into training.

> **Worked example:** join each label to the latest feature row at or before its timestamp, for instance with the pandas function merge_asof. A timestamp is the recorded time of an event. Pandas is a Python library for working with tables.

## Training serving skew and inference modes

Training serving skew means the model sees different inputs in production (the live system) than it saw in training. The causes are different code, different data, different timing and different traffic. Nothing errors. The model just quietly gets worse.

Prevent it in three ways. Use one feature definition. Log the features used at serving. Run parity tests, which compare logged and recomputed values.

Inference means running the trained model to get predictions. Batch inference scores everyone on a schedule and stores the results. Online inference scores each request as it arrives. Scoring 50 million users nightly at 5,000 per second takes about 2.8 hours, which fits an overnight window.

## Model registry, CI and CD for ML

A model registry is a versioned catalogue of trained models. Each entry holds the code, data, settings, metrics and a stage, such as testing or live. Deploying means pointing serving at a version. Rollback means pointing back to the old one.

Continuous integration (CI) runs tests on code, data and models before changes merge. Continuous delivery (CD) then ships changes that pass. A gate is a rule the new model must pass. For example, the new model must match or beat the current one. No user group may drop more than 1 percent. The slowest requests must stay within the limit.

Record the code, data snapshot, settings and seeds so a model can be rebuilt. A seed is the starting number for random choices. Exact repeats are often impossible, so aim for results within normal noise.

## Shadow, canary and A/B rollout

A rollout is how a new model reaches users. A shadow deployment sends copies of live requests to the new model and hides its answers. It tests speed and errors but not user behaviour. A canary serves a small share of real traffic, say 1 percent. An A/B test splits users at random and measures the real effect.

Roll out in stages with automatic checks. Keep the old model warm (loaded and ready) so you can roll back in under a minute. Keep a kill switch, a control that turns the new model off and falls back to a simple rule.

> **Interview trap:** flipping 100 percent of traffic to a new model because offline metrics improved.

## Monitoring and drift

Drift means the data or the world changes after training. Covariate shift changes the inputs. Label shift changes the outcome rate. Concept drift changes the link between inputs and outcomes.

PSI (population stability index) compares how a feature's values are spread in training and in live traffic. Split the values into bins, and compare the share in each bin. Take 5 bins that each hold 20 percent in training. Live shares of 10, 15, 20, 25 and 30 percent give a PSI of 0.135, a moderate shift. Under 0.1 is usually stable and over 0.25 is large.

When labels arrive late, watch inputs and predictions first. Then watch fast proxy signals, which are early stand ins for the true outcome. Last, check true labels on older groups, whose labels have arrived.

## Retraining, labelling and active learning

Measure how fast a model gets worse as it ages. Then retrain before the loss costs more than the retrain. Facebook found retraining daily instead of weekly cut a click model's error by about 1 percent.

Check label quality by having two people label the same items. If they agree on 90 of 100 and chance agreement is about 0.505, Cohen's kappa is about 0.80. Kappa measures agreement beyond chance. A kappa of 1 is perfect and 0 is no better than chance.

Active learning lets the model pick which items a person should label next, such as the ones it is least sure about. Gains are not guaranteed. Evaluate on a random test set, because the picked items are not typical.

## ML technical debt

Technical debt is the future cost of a shortcut. A Google paper argues ML systems build it fast, because the model is a small part of a larger system. A mature system might be at most 5 percent ML code. The rest is glue code (code that connects parts) and pipelines (chains of automated data steps).

CACE means changing anything changes everything: changing one input can change how the model uses all the others. Undeclared consumers are systems that quietly use your model's output, so you cannot change it safely.

> **Remember:** reward deleting features and reducing complexity as much as raising accuracy.
