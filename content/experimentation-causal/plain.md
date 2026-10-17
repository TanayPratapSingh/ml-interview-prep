# Experimentation and causal inference

This is the easy version of the experimentation and causal inference module. An A/B test answers the question "did my change cause this?" by letting a coin flip decide who gets the change. Everything else in the module is either a way to run that test well (sizing, timing, checks, noise reduction) or a way to answer the same question when you cannot flip the coin (difference in differences, synthetic control, instruments, discontinuities, propensity scores). Each section gives the intuition, a number you can check, and where the idea breaks. The deep version has the derivations and code, and every number here came from a simulation or a formula that was run.

## Experiment design basics

An A/B test is a fair race between two versions of your product. A coin flip sends each user to control (the old experience) or treatment (the new one), and you compare an outcome. The coin matters because it makes the two groups alike in everything, including traits nobody measured, so a gap beyond noise belongs to the change. If users opt in to the new feature themselves, keen users opt in and you credit the feature for their keenness.

A good hypothesis names the change, the users, the metric, the direction, and the reason. Write the smallest effect worth shipping and the decision rule before launch. Choosing the metric after seeing the data lets you find a win in anything.

> **Say it like this:** "I work backwards from the decision: hypothesis, one primary metric, guardrails, unit, sample size, duration in full weeks, checks, then one analysis at the planned end."

## Metrics for experiments

The primary metric decides the launch. Secondary metrics explain why it moved. Guardrails are things that must not get worse, such as latency, crashes, refunds or unsubscribes. Pick one primary metric before launch, because every extra "primary" is another chance for a false win.

A good metric is measured per user, sensitive enough to move in a few weeks, and clearly good or bad in one direction. A lift inside a small group is diluted: a 10 percent gain for users who are 1 percent of traffic moves the overall metric by about 0.1 percent. For guardrails you ask "is it worse by more than we can tolerate?" with a margin chosen up front, not "is it significantly different?", because a small noisy test passes that check by seeing nothing.

## Randomisation and the unit

The unit is the thing you flip the coin for, usually the user. It should give each person a consistent experience and keep units from affecting each other. The most common mistake is randomising users and then analysing sessions as if each were independent. Ten sessions from one user are not ten pieces of evidence. In a simulation of experiments with no real effect, the session level test called 42 percent of them significant at a nominal 5 percent, and the user level test called 4.9 percent.

Assignment usually hashes the user id with a salt specific to the experiment, so the same person always lands in the same arm and different experiments are independent. Analyse everyone you assigned, in the arm you assigned them to, even if some never opened the feature. If only a fraction of users can see the change, a triggered analysis on those users gives the same answer with far fewer people.

## Sample size and power

Power is the chance of seeing a real effect when it exists. Four things set the sample size: how big an effect you care about, how noisy the metric is, how often you accept a false alarm (usually 5 percent), and the power you want (usually 80 percent). The formula multiplies the noise by about 7.85 and divides by the effect squared. Halving the effect needs about four times the users.

> **Worked example:** Conversion is 10 percent and you want to detect 11 percent. You need about 14,750 users per arm. A simulation at that size found the lift 80 percent of the time, and statsmodels gave 14,744.

The smallest effect your test can detect is about 2.8 standard errors. Observed power after the fact is just the p value in disguise, so report the confidence interval instead. Significant results from small tests are exaggerated: at 17 percent power, the significant estimates averaged about 2.5 times the truth.

## Duration and weekly cycles

Divide the users you need by the eligible users who arrive per day, then round up to whole weeks. Needing 29,500 users with 2,000 arriving daily is about 15 days, so run three weeks. People behave differently on weekdays and weekends, so a partial week misweights some days. With a 2 percent weekday lift and a 10 percent weekend lift, the true weekly lift is 4.7 percent, but five weekdays alone show 2.0 percent.

If you hit the sample size on day three, keep going. Three days is a partial week, shows nothing about novelty, and stopping because the numbers look good breaks the error guarantee.

## Peeking and sequential testing

Every look at the data gives noise another chance to cross the significance line. If you stop at the first significant look, the real false positive rate grows with the number of looks: 5 percent with one look, about 14 percent with five, 19 percent with ten, 37 percent with a hundred, when nothing is going on.

The fixes are to fix the sample size and look once, or to use a method built for looking. Group sequential designs use stricter thresholds at early looks. Always valid tests stay correct however often you look, at some cost in power. A Bayesian posterior does not rescue you: stopping at "95 percent probability B is better" falsely stopped 43 percent of the time over 20 looks in a simulation. Watching guardrails and sample ratio is fine; stopping on the primary metric is the problem.

> **Interview trap:** "I'll just check daily and stop when it is significant." Say that you will look but not act, or use a planned sequential design.

## Trust checks and sample ratio mismatch

Before reading any metric, check the counts. If you planned 50/50 and got 821,588 versus 815,482, it looks close, but the chance of that by luck is about 1 in 550,000. That is a sample ratio mismatch (SRM), and it means users were lost or added unevenly, usually in a way tied to their behaviour. Then the groups are no longer comparable and the result cannot be trusted, even if it looks great.

SRMs come from assignment bugs, late variants, bot filters that react to the treatment, wrong filters in analysis, and people forcing a variant. Diagnose by asking where it appears: one segment, one day, one pipeline, many experiments at once. Use a strict threshold (many teams use 0.001) because the check runs on every experiment. An A/A test, where both arms get the identical experience, should show about 5 percent significant results. More means your variance is understated.

## Novelty, primacy and long term effects

A novelty effect is a lift that fades because the change is new. A primacy effect is a dip while users learn. Plot the effect by days since each user's first exposure; a real lasting effect goes flat. If the lift is 8 percent on day 1 and settles near 2 percent, the first week average of about 5 percent overstates what users will see later. Report the steady state.

Two weeks can also miss slow effects like retention, habit, or sellers adapting to a ranking change. For important launches, keep a small long term holdout. A 2 percent holdout of a million users can only detect about 0.6 points on a 10 percent baseline, so size it for the question.

## Interference and network effects

A standard A/B test assumes one user's treatment does not change another user's outcome. In social networks, treated users influence friends. In marketplaces, treated buyers compete with control buyers for the same supply. When this fails, the test measures the wrong thing. In a simulation with a direct effect of 1 and a spillover of 1, user level randomisation estimated 1.0 while treating everyone gives 2.0. In a toy marketplace the A/B test showed a 21.8 percent purchase lift while a full rollout gave 9.5 percent, because treated buyers took items control buyers would have bought.

Fixes randomise bigger units. Clusters keep friends together, at the cost of power: clusters of 50 with within group correlation 0.1 inflate variance 5.9 times. Switchback tests flip a whole market between arms by time slice, with slices longer than the system's adjustment time. With only a few cities, there are few independent units, so six cities with three treated can never give p below 0.10.

## Variance reduction and CUPED

Sample size is proportional to variance, so removing noise is like buying traffic. Much of the noise is that some users are always heavy and others light. CUPED uses each user's behaviour before the experiment to subtract that predictable part. The pre period is the same in both arms, so the effect estimate stays unbiased, but the variance drops by the squared correlation between the pre period and the outcome. With a correlation of 0.7, variance falls by about half: the standard error went from 0.141 to 0.101 in a simulation.

Only use covariates measured before the experiment. A covariate the treatment moves absorbs real effect: in a simulation a post treatment covariate turned a true effect of 0.5 into minus 0.45. Stratification, capping extreme values, and more sensitive metrics also help. Revenue per user is heavily skewed, so capping at a high percentile can cut variance sharply, at the price of measuring the capped metric.

## Ratio metrics and the delta method

A ratio metric such as clicks per view divides two user level totals. If you randomise users, the user is the independent unit, not the view. Treating each view as independent understates the noise: in an A/A simulation it flagged 11.7 percent of experiments at a nominal 5 percent, while the correct method flagged 5.4 percent.

The delta method fixes this with a straight line approximation. For a ratio of averages it gives a closed form variance using the spread of the numerator, the spread of the denominator, and how they move together. Equivalently, compute for each user the value "clicks minus ratio times views" and run an ordinary test on it. On a toy table with a ratio of 0.18 the standard error is 0.0432. The same trick gives a confidence interval for a percent lift.

## Heterogeneous effects and multiple testing

The average effect can hide opposite effects in different groups. Looking for them is useful and dangerous: with no real effect anywhere, 65 percent of simulated experiments had at least one of 20 subgroups significant. Pre specify a short list, use one interaction test, and confirm anything you discover in a new test.

The same problem applies to metrics. With 10 independent metrics and no effect, there is a 40 percent chance that at least one is significant. Let one primary metric carry the significance level, give guardrails their own margins, and treat secondary metrics as clues. With several variants against one control, give the control more traffic, and remember the winner's curse: the best of 10 variants with no true effect showed a lift of 0.77 points on average.

## Ranking, bandits and when not to test

Interleaving compares two rankers by merging their results into one list for each user and crediting clicks to the ranker that supplied the clicked item. It is a paired comparison, so in a toy simulation it reached 88 percent power with about 2,000 queries where an A/B test had 31 percent.

A bandit sends more traffic to the better option as it learns, which earns more during the test but breaks the usual analysis: in a simulation a naive test on bandit data rejected 14 percent of the time with equal arms. Use bandits for short lived choices with many options and A/B tests for permanent launches that need trustworthy estimates.

Do not run a test when you cannot reach power, when the outcome takes too long, when interference has no feasible design, when randomising is unethical, or when the change is an obvious fix. Use a quasi experiment, careful observational analysis, or research instead.

## Potential outcomes and confounding

Every user has two possible outcomes, with and without the treatment, and you only see one. The effect for one person is never observed, so every causal method needs a stand in for the missing outcome: a control group, a matched group, a trend, or a model. The naive comparison of adopters and non adopters equals the true effect plus selection bias. In a simulation it was 3.12, made of a true effect of 1.00 and selection bias of 2.12, because motivated users adopt and do better anyway.

A confounder is a common cause of treatment and outcome. Randomisation removes confounding by construction. Without it, adjust for measured confounders, not for mediators or colliders, and accept that unmeasured ones remain. Different methods estimate different averages: the whole population, the treated, or compliers.

## Difference in differences and synthetic control

Difference in differences compares the change in a treated group with the change in an untreated group over the same period. In John Snow's cholera study, deaths per 10,000 households fell by 48 where the water changed and rose by 49 elsewhere, so the estimate is minus 97. It works if both groups would have moved in parallel without the treatment. Check parallel trends before the treatment with plots and an event study, and run placebo tests. In a simulation a trend gap of 0.3 per period turned a true effect of 2.0 into 3.82.

Synthetic control builds a counterfactual for one treated unit as a weighted mix of donor units that matches its history. Inference comes from placebo runs on each donor. With adoption at different times, plain two way fixed effects can mislead, so use estimators built for staggered timing.

## Instrumental variables and regression discontinuity

An instrument is a nudge that shifts who gets the treatment and affects the outcome only through it, such as a random invitation. The effect equals the nudge's effect on the outcome divided by its effect on adoption. In a simulation where motivated users adopt, ordinary regression gave 5.30 for a true effect of 2.0 and the instrument gave 2.08. It estimates the effect for compliers only, and weak instruments make it unstable.

Regression discontinuity uses a cutoff rule, such as a score above 600. Users just above and below are nearly alike, so the jump at the cutoff is the effect there. Check that nobody can manipulate the score and that other variables do not jump at the cutoff.

## Propensity scores and uplift

The propensity score is the probability of getting the treatment given your characteristics. Matching or weighting on it balances measured characteristics. In a simulation the naive difference was 3.73 for a true effect of 2.0, and weighting gave 2.00. Extreme weights blow up the variance, so check overlap. A doubly robust estimator is right if either the outcome model or the propensity model is right. None of these fix unmeasured confounding; sensitivity measures such as the E value say how strong a hidden confounder would have to be.

Uplift modelling targets the people whose behaviour the treatment changes. Response models pick people who would have converted anyway: the top 20 percent by response had an observed uplift of 0.9 percent, against 4.9 percent for an uplift model.

## Decision questions

A tiny effect can be significant with millions of users, so compare the interval with the smallest worthwhile effect and with the cost. If the primary metric wins and a guardrail loses, check the loss is real, compare it with the agreed margin, convert both to value, and then fix, segment, ramp or reject. If one metric rises and another falls, use the metric hierarchy.

If a result is flat, check the experiment, read what the interval rules out, and look at dilution and sensitivity. If a result looks too good to be true, it probably is: with a prior of 1 in 3 that an idea works, a significant result is real 89 percent of the time, but with a prior of 1 in 500 only 3.1 percent. Explain results with the decision first, then the effect and its range in business terms.
