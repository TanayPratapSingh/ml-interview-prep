# Experimentation and causal inference

This is the easy version of the experimentation and causal inference module. Causal inference means working out whether one thing caused another. An A/B test answers the question "did my change cause this?" by letting a coin flip decide who gets the change. Everything else in the module is one of two things. It is either a way to run that test well (sizing, timing, checks, noise reduction). Or it is a way to answer the same question when you cannot flip the coin.

The methods that work without a coin are these:

- **Difference in differences:** compare how a treated group and an untreated group each changed.
- **Synthetic control:** build a stand in comparison group from a mix of other units.
- **Instruments:** use a nudge that shifts who gets the change.
- **Discontinuities:** use a cutoff rule that splits similar people.
- **Propensity scores:** use each person's chance of getting the change.

Each section gives the intuition, a number you can check, and where the idea breaks. The deep version has the derivations and code. Every number here came from a simulation (a computer experiment on made up data) or a formula that was run.

## Experiment design basics

An A/B test is a fair race between two versions of your product. A coin flip sends each user to control (the old experience) or treatment (the new one), and you compare an outcome. The coin matters because it makes the two groups alike in everything, including traits nobody measured. So a gap beyond noise belongs to the change. Noise is random variation that has nothing to do with the change. If users opt in to the new feature themselves, keen users opt in and you credit the feature for their keenness.

A good hypothesis (a clear prediction you can test) names the change, the users, the metric, the direction, and the reason. The metric is the number you measure. Write the smallest effect worth shipping and the decision rule before launch. Choosing the metric after seeing the data lets you find a win in anything.

> **Say it like this:** "I work backwards from the decision: hypothesis, one primary metric, guardrails, unit, sample size, duration in full weeks, checks, then one analysis at the planned end." Guardrails are measures that must not get worse. The unit is the thing you randomise, such as the user.

## Metrics for experiments

The primary metric decides the launch. Secondary metrics explain why it moved. Guardrails are things that must not get worse, such as latency (how long pages take to respond), crashes, refunds or unsubscribes. Pick one primary metric before launch, because every extra "primary" is another chance for a false win. A false win is a result that looks like a win but is only luck.

A good metric is measured per user, sensitive enough to move in a few weeks, and clearly good or bad in one direction. A lift, which is a rise in the metric, inside a small group is diluted, meaning spread thin across everyone. A 10 percent gain for users who are 1 percent of traffic moves the overall metric by about 0.1 percent.

For guardrails you ask "is it worse by more than we can tolerate?" with a margin chosen up front. You do not ask "is it significantly different?" (unlikely to be luck alone), because a small noisy test passes that check by seeing nothing.

## Randomisation and the unit

The unit is the thing you flip the coin for, usually the user. Randomisation means assigning units by chance. It should give each person a consistent experience and keep units from affecting each other. The most common mistake is randomising users and then analysing sessions as if each were independent. Ten sessions from one user are not ten pieces of evidence.

In a simulation of experiments with no real effect, the session level test called 42 percent of them significant at a nominal 5 percent. A test is called significant when the result is unlikely to be luck alone. Nominal 5 percent means it should do this by luck only 5 percent of the time. The user level test called 4.9 percent.

Assignment usually hashes the user id (identifier) with a salt specific to the experiment. This way the same person always lands in the same arm, and different experiments are independent. Hashing scrambles the id into a number, and the salt is extra text so each experiment scrambles differently. An arm is one group of the test. Analyse everyone you assigned, in the arm you assigned them to, even if some never opened the feature. If only a fraction of users can see the change, a triggered analysis on those users gives the same answer with far fewer people.

## Sample size and power

Power is the chance of seeing a real effect when it exists. Four things set the sample size. The first is how big an effect you care about. The second is how noisy the metric is, which means its variance, the spread of its values. The third is how often you accept a false alarm (usually 5 percent), which means calling luck a real effect. The fourth is the power you want (usually 80 percent). The formula multiplies the noise by about 7.85 and divides by the effect squared. Halving the effect needs about four times the users.

> **Worked example:** Conversion is 10 percent and you want to detect 11 percent. You need about 14,750 users per arm. A simulation at that size found the lift 80 percent of the time, and statsmodels gave 14,744. Conversion is the share of users who complete the goal, and statsmodels is a Python statistics library.

The smallest effect your test can detect is about 2.8 standard errors. A standard error measures how much an estimate would wobble from sample to sample. Observed power after the fact is just the p value in disguise. The p value is the chance of a result at least this extreme if the change did nothing. So report the confidence interval instead, which is a range of effects consistent with the data. Significant results from small tests are exaggerated: at 17 percent power, the significant estimates averaged about 2.5 times the truth.

## Duration and weekly cycles

Divide the users you need by the eligible users who arrive per day, then round up to whole weeks. Needing 29,500 users with 2,000 arriving daily is about 15 days, so run three weeks. People behave differently on weekdays and weekends, so a partial week misweights some days. With a 2 percent weekday lift and a 10 percent weekend lift, the true weekly lift is 4.7 percent. But five weekdays alone show 2.0 percent.

If you hit the sample size on day three, keep going. Three days is a partial week and shows nothing about novelty, the effect of a change being new. Stopping because the numbers look good also breaks the error guarantee, which is the promise that false alarms stay at 5 percent.

## Peeking and sequential testing

Peeking means checking results before the planned end. Every look at the data gives noise another chance to cross the significance line. If you stop at the first significant look, the real false positive rate grows with the number of looks. A false positive is a result that looks real but is not. When nothing is going on, the rate is 5 percent with one look and about 14 percent with five. It is 19 percent with ten looks and 37 percent with a hundred.

The fixes are to fix the sample size and look once, or to use a method built for looking. Group sequential designs plan several looks in advance and use stricter thresholds at early looks. Always valid tests stay correct however often you look, at some cost in power. A Bayesian posterior (the probability that B is better, given the data) does not rescue you. Stopping at "95 percent probability B is better" falsely stopped 43 percent of the time over 20 looks in a simulation. Watching guardrails and sample ratio (the split of users between arms) is fine. Stopping on the primary metric is the problem.

> **Interview trap:** "I'll just check daily and stop when it is significant." Say that you will look but not act, or use a planned sequential design.

## Trust checks and sample ratio mismatch

Before reading any metric, check the counts. Suppose you planned 50/50 and got 821,588 versus 815,482. It looks close, but the chance of that by luck is about 1 in 550,000. That is a sample ratio mismatch (SRM), which means the split of users differs from the plan by more than luck allows. It means users were lost or added unevenly, usually in a way tied to their behaviour. Then the groups are no longer comparable and the result cannot be trusted, even if it looks great.

SRMs come from assignment bugs, late variants, bot filters that react to the treatment, wrong filters in analysis, and people forcing a variant. A variant is one version in the test, and a bot filter removes automated traffic. Diagnose by asking where it appears: one segment, one day, one pipeline (a chain of data steps), many experiments at once.

Use a strict threshold (many teams use 0.001) because the check runs on every experiment. An A/A test, where both arms get the identical experience, should show about 5 percent significant results. More means your variance is understated, which means the test is too confident.

## Novelty, primacy and long term effects

A novelty effect is a lift that fades because the change is new. A primacy effect is a dip while users learn. Plot the effect by days since each user's first exposure. A real lasting effect goes flat. Say the lift is 8 percent on day 1 and settles near 2 percent. Then the first week average of about 5 percent overstates what users will see later. Report the steady state, which is the level the effect settles at.

Two weeks can also miss slow effects like retention (users coming back), habit, or sellers adapting to a ranking change. For important launches, keep a small long term holdout, a group that never gets the change. A 2 percent holdout of a million users can only detect about 0.6 points on a 10 percent baseline. The baseline is the metric's current level. So size the holdout for the question.

## Interference and network effects

A standard A/B test assumes one user's treatment does not change another user's outcome. This assumption is called no interference. In social networks, treated users influence friends. In marketplaces, treated buyers compete with control buyers for the same supply. When this fails, the test measures the wrong thing.

Take a simulation with a direct effect of 1 and a spillover of 1. The spillover is the effect that treated neighbours have on a user. User level randomisation estimated 1.0, while treating everyone gives 2.0. In a toy marketplace the A/B test showed a 21.8 percent purchase lift. A full rollout gave 9.5 percent, because treated buyers took items control buyers would have bought.

Fixes randomise bigger units. Clusters keep friends together, at the cost of power. Clusters of 50 with within group correlation 0.1 inflate variance 5.9 times. Within group correlation means how alike users in one cluster are, and inflated variance means a noisier estimate. Switchback tests flip a whole market between arms by time slice, with slices longer than the system's adjustment time. A time slice is a block of time, such as an hour. With only a few cities, there are few independent units. So six cities with three treated can never give p below 0.10.

## Variance reduction and CUPED

Sample size is proportional to variance, so removing noise is like buying traffic. Much of the noise is that some users are always heavy and others light. CUPED (controlled experiment using pre experiment data) uses each user's behaviour before the experiment to subtract that predictable part. The pre period is the time before the experiment began. It is the same in both arms, so the effect estimate stays unbiased, which means right on average. The variance drops by the squared correlation between the pre period and the outcome. Correlation measures how closely two numbers move together. With a correlation of 0.7, variance falls by about half. The standard error went from 0.141 to 0.101 in a simulation.

Only use covariates measured before the experiment. A covariate is an extra measured fact about each user. One that the treatment moves absorbs real effect: in a simulation a post treatment covariate turned a true effect of 0.5 into minus 0.45.

Other helpers also work. Stratification splits users into similar groups before comparing. Capping cuts extreme values down to a limit. More sensitive metrics help too. Revenue per user is heavily skewed, which means a few huge values pull the average. So capping at a high percentile can cut variance sharply, at the price of measuring the capped metric.

## Ratio metrics and the delta method

A ratio metric such as clicks per view divides two user level totals. If you randomise users, the user is the independent unit, not the view. Treating each view as independent understates the noise. Take an A/A simulation, where both arms are identical so any flag is false. Treating views as independent flagged 11.7 percent of experiments at a nominal 5 percent. The correct method flagged 5.4 percent.

The delta method fixes this with a straight line approximation, which treats a curve as straight near the point of interest. For a ratio of averages it gives a closed form variance, which is an exact formula. The formula uses the spread of the numerator (the top of the ratio), the spread of the denominator (the bottom), and how they move together.

Equivalently, compute for each user the value "clicks minus ratio times views" and run an ordinary test on it. On a toy table with a ratio of 0.18 the standard error is 0.0432. The same trick gives a confidence interval for a percent lift.

## Heterogeneous effects and multiple testing

The average effect can hide opposite effects in different groups. This is called heterogeneity, and the groups are subgroups. Looking for them is useful and dangerous. With no real effect anywhere, 65 percent of simulated experiments had at least one of 20 subgroups significant. Pre specify a short list, which means write it down before looking. Use one interaction test, which asks whether the effect differs between groups. Confirm anything you discover in a new test.

The same problem applies to metrics. It is called multiple testing: more comparisons give more chances for a false alarm. With 10 independent metrics and no effect, there is a 40 percent chance that at least one is significant. Let one primary metric carry the significance level, which is the false alarm rate you allow. Give guardrails their own margins, and treat secondary metrics as clues.

With several variants against one control, give the control more traffic. Remember the winner's curse, which means the best looking result is usually overstated. The best of 10 variants with no true effect showed a lift of 0.77 points on average.

## Ranking, bandits and when not to test

Interleaving compares two rankers (systems that order results) by merging their results into one list for each user. It credits clicks to the ranker that supplied the clicked item. It is a paired comparison, since every user sees both rankers, so user to user noise cancels. In a toy simulation it reached 88 percent power with about 2,000 queries where an A/B test had 31 percent.

A bandit sends more traffic to the better option as it learns. This earns more during the test but breaks the usual analysis. In a simulation a naive test on bandit data rejected 14 percent of the time with equal arms. Rejected here means it falsely called a difference. Use bandits for short lived choices with many options. Use A/B tests for permanent launches that need trustworthy estimates.

Do not run a test in these cases:

- You cannot reach power.
- The outcome takes too long.
- Interference has no feasible design.
- Randomising is unethical.
- The change is an obvious fix.

Use a quasi experiment (a natural comparison that nobody set up with a coin flip), careful observational analysis (studying data without randomising), or research instead.

## Potential outcomes and confounding

Every user has two possible outcomes, with and without the treatment, and you only see one. These are the potential outcomes. The effect for one person is never observed. So every causal method needs a stand in for the missing outcome: a control group, a matched group, a trend, or a model.

The naive comparison of adopters and non adopters equals the true effect plus selection bias. Selection bias means the groups differed before the treatment because people chose. In a simulation the comparison was 3.12. It was made of a true effect of 1.00 and selection bias of 2.12, because motivated users adopt and do better anyway.

A confounder is a common cause of treatment and outcome. Randomisation removes confounding by construction. Without it, adjust for measured confounders, not for mediators or colliders, and accept that unmeasured ones remain. A mediator is a step on the path from treatment to outcome. A collider is a variable that both treatment and outcome cause. Different methods estimate different averages: the whole population, the treated, or compliers. Compliers are users who take the treatment only when nudged.

## Difference in differences and synthetic control

Difference in differences compares the change in a treated group with the change in an untreated group over the same period. In John Snow's cholera study, deaths per 10,000 households fell by 48 where the water changed. They rose by 49 elsewhere, so the estimate is minus 97.

It works if both groups would have moved in parallel without the treatment. Check parallel trends before the treatment with plots and an event study, which plots the gap period by period around the treatment date. Also run placebo tests, which apply the method where nothing happened and should find nothing. In a simulation a trend gap of 0.3 per period turned a true effect of 2.0 into 3.82.

Synthetic control builds a counterfactual for one treated unit. A counterfactual is what would have happened without the treatment. It is a weighted mix of donor units, which are untreated units used as building blocks, that matches the treated unit's history. Inference, meaning judging whether the result is more than luck, comes from placebo runs on each donor. With adoption at different times, plain two way fixed effects can mislead. Two way fixed effects is a standard regression that adjusts for each unit and each time period. So use estimators built for staggered timing, where units adopt on different dates.

## Instrumental variables and regression discontinuity

An instrument is a nudge that shifts who gets the treatment and affects the outcome only through it, such as a random invitation. The effect equals the nudge's effect on the outcome divided by its effect on adoption. In a simulation where motivated users adopt, ordinary regression gave 5.30 for a true effect of 2.0. Ordinary regression is the standard line fitting method. The instrument gave 2.08. It estimates the effect for compliers only. Weak instruments make it unstable, because a nudge that barely changes adoption means dividing by a tiny number.

Regression discontinuity uses a cutoff rule, such as a score above 600. Users just above and below are nearly alike, so the jump at the cutoff is the effect there. Check that nobody can manipulate the score and that other variables do not jump at the cutoff.

## Propensity scores and uplift

The propensity score is the probability of getting the treatment given your characteristics. Matching or weighting on it balances measured characteristics. Matching pairs each treated person with a similar untreated one. Weighting gives extra weight to people who look unlike their own group. In a simulation the naive difference was 3.73 for a true effect of 2.0, and weighting gave 2.00.

Extreme weights blow up the variance, so check overlap, which means both treated and untreated people exist for every kind of person. A doubly robust estimator is right if either the outcome model or the propensity model is right. None of these fix unmeasured confounding. Sensitivity measures such as the E value say how strong a hidden confounder would have to be.

Uplift modelling targets the people whose behaviour the treatment changes. Uplift is the extra conversion that the treatment causes. Response models pick people who would have converted anyway. The top 20 percent by response had an observed uplift of 0.9 percent, against 4.9 percent for an uplift model.

## Decision questions

A tiny effect can be significant with millions of users, so compare the interval with the smallest worthwhile effect and with the cost. If the primary metric wins and a guardrail loses, check the loss is real and compare it with the agreed margin. Then convert both to value, and fix, segment, ramp or reject. To ramp is to raise exposure step by step. To segment is to ship to a subgroup only. If one metric rises and another falls, use the metric hierarchy, the agreed ranking of which metrics matter most.

If a result is flat, check the experiment, read what the interval rules out, and look at dilution and sensitivity. If a result looks too good to be true, it probably is. A prior is how likely you thought the idea was to work before the test. With a prior of 1 in 3, a significant result is real 89 percent of the time. With a prior of 1 in 500 it is real only 3.1 percent of the time. Explain results with the decision first, then the effect and its range in business terms.
