# Probability and statistics

This is the easy version of the probability and statistics module. Probability is the arithmetic of uncertainty: given how the world works, how likely is what we see? Statistics runs the other way: given what we saw, what can we say about how the world works, and how sure can we be? Each section below gives the intuition, one example with real numbers, and where the idea breaks. The deep version covers the same sections with derivations and code, and every number here was checked by simulation.

## Probability rules

A probability is a number from 0 to 1 attached to an event, where an event is a set of possible outcomes ("the die shows an even number"). Three rules define everything: probabilities are never negative, something must happen (all outcomes together have probability 1), and for events that cannot happen together, the probability of either one is the sum.

Two tricks come from those rules and solve a surprising number of interview questions.

**Count the opposite.** "At least one" is hard to count directly but easy to subtract. The chance of at least one 4 when rolling two dice is 1 minus the chance of no 4s: 1 minus (5/6 times 5/6) = 11/36, about 31 percent.

**Subtract the overlap.** When two groups overlap, adding them counts the overlap twice. Of the numbers 1 to 300, 100 are divisible by 3 and 60 by 5, but the 20 multiples of 15 are in both lists, so the answer is 100 + 60 - 20 = 140.

> **Interview trap:** Adding probabilities of events that can happen together. One sixth plus one sixth for "a 4 on either die" counts the double 4 twice.

## Conditional probability and Bayes

Conditional probability is the chance of something once you know something else happened. You throw away the outcomes that are now impossible and rescale. A die shows 6 with chance 1/6, but if you're told the roll is even, only 2, 4 and 6 remain, so the chance is 1/3.

Two events are independent when knowing one tells you nothing about the other, like a coin and a die. That is different from events that can't happen together: rolling a 2 and rolling a 5 on the same die can't both happen, so knowing one happened tells you the other didn't. That's strong dependence, not independence.

Bayes' theorem turns "how likely is this evidence if my idea is true" into "how likely is my idea now that I've seen the evidence". It needs a starting belief (the prior), and that's where most people go wrong.

> **Worked example:** A disease affects 1 in 10,000 people. A test catches 99 percent of sick people and wrongly flags 1 percent of healthy people. You test positive. Picture a million people: 100 are sick and 99 of them test positive. Of the 999,900 healthy people, 1 percent, about 9,999, also test positive. So of roughly 10,100 positive results, only 99 are real. Your chance of being sick is under 1 percent.

The test is good; the disease is just rare, so false alarms from the huge healthy group swamp the true cases. Ignoring how rare something is to start with is called base rate neglect. A second, independent positive test raises the chance to about 50 percent, but only if the two tests make mistakes independently. Rerunning the same test on the same blood sample doesn't count as independent.

The odds version is the mental math shortcut: new odds equal old odds times how much more likely the evidence is if the idea is true. A test that's 99 times more likely to be positive for a sick person multiplies the odds of disease by 99.

Puzzles test whether you condition on the right event. If a family has two children and at least one is a boy, the chance both are boys is 1/3, because the equally likely families boy boy, boy girl and girl boy remain. If you meet one of the children and it's a boy, the answer is 1/2. Same family, different information.

> **Say it like this:** "Bayes is prior odds times the likelihood ratio. A 99 percent accurate test for a 1 in 10,000 disease turns odds of 1 to 9,999 into about 1 to 100, so a positive means roughly a 1 percent chance."

## Counting and combinatorics

When every outcome is equally likely, probability is counting: favourable outcomes divided by total outcomes. The counting formulas come from two questions: does order matter, and can things repeat?

Ranking 3 of 10 runners as gold, silver and bronze: order matters, so 10 × 9 × 8 = 720 ways. Picking a committee of 3 from 10: order doesn't matter, so divide by the 6 orderings of each group, giving 120. A 4 digit PIN allows repeats and order matters: 10,000.

> **Worked example:** Two cards from a deck. The first can be anything; 12 of the remaining 51 cards share its suit, so the chance of the same suit is 12/51 = 4/17. Two queens: 4/52 for the first, 3/51 for the second, so 1/221.

Look for symmetry before counting. Shuffle 3 red and 5 blue balls: the chance the last ball is red is 3/8, the same as the first, because a shuffle treats every position alike.

> **Interview trap:** Splitting 12 people into 3 unnamed teams of 4 and forgetting that the teams have no labels. Divide by the 6 ways to order the teams.

## Random variables and expectation

A random variable is a number produced by chance, like the number of heads in 10 flips. For whole number outcomes you list the chance of each value. For measurements like height, you use a density curve, and probabilities are areas under the curve, not its height. That's why a density can be bigger than 1: a value spread evenly between 0 and 0.5 has density 2, and the area is still 0.5 × 2 = 1.

The expected value is the long run average: the probability weighted average of the possible values. A die's expected value is 3.5, even though you can never roll 3.5.

The most useful fact in this whole module: **the average of a sum is the sum of the averages**, even when the pieces depend on each other. Shuffle 10 letters into 10 envelopes; how many land in the right envelope on average? Each letter has a 1 in 10 chance, 10 letters, so 1 on average. The letters aren't independent, and it doesn't matter.

The second most useful fact: **condition on the first step**. How many die rolls until a 6? Call the answer m. The first roll costs one; with chance 5/6 it isn't a 6 and you're back where you started. So m = 1 + (5/6) m, which gives m = 6.

> **Worked example:** You're paid the face of a die roll and may reroll once. A fresh roll is worth 3.5, so keep 4, 5 or 6 and reroll 1, 2 or 3. The game is worth half of 5 (the average of 4, 5, 6) plus half of 3.5, which is 4.25.

Averages don't pass through curves. The average of squared values is bigger than the square of the average value, by exactly the variance. This rule (Jensen's inequality) matters later for machine learning.

## Variance, covariance and correlation

Variance measures spread: the average squared distance from the average. The standard deviation is its square root, back in the original units. Shifting all values by a constant doesn't change the spread; doubling them doubles the standard deviation and quadruples the variance.

When you add two independent noisy quantities, their variances add. Surprisingly, they also add when you subtract: the gap between two noisy numbers is noisier than either one. That's why the difference between an A/B test's two groups is harder to pin down than either group's average. When the two quantities move together, as with the same users measured before and after, subtracting cancels much of the noise. That's the reason paired comparisons are so powerful.

Correlation squeezes "moving together" onto a scale from -1 to 1. A correlation of 0.3 is a weak straight line trend: a straight line explains only about 9 percent (0.3 squared) of the variation.

> **Interview trap:** Zero correlation does not mean unrelated. Spread X evenly from -1 to 1 and set Y to X squared. Y is completely determined by X, yet the correlation is zero, because the U shape slopes down on the left and up on the right, and the two halves cancel.

Total spread splits into "spread within groups" plus "spread between group averages". Heights in a population that's half a group averaging 178 cm and half a group averaging 164 cm have both kinds, and the between group part is surprisingly large. This split is the idea behind ANOVA and behind variance reduction tricks in experiments.

## Discrete distributions

A few distributions cover most count questions. Learn the story behind each, because the story is how you recognise it.

| Name | Story | Average |
| --- | --- | --- |
| Bernoulli | One yes or no trial | p |
| Binomial | Number of yeses in n independent trials | n × p |
| Geometric | Trials until the first yes | 1/p |
| Negative binomial | Trials until the r-th yes | r/p |
| Poisson | Count of rare, independent events in a window | Its rate |
| Multinomial | Counts across several categories | n × each share |

Geometric waits have no memory: after 10 misses, the expected number of extra tries is the same as at the start. Families that keep having children until the first girl average 2 children, and the overall ratio of boys to girls stays even.

The Poisson's average equals its variance. Real counts usually vary more than that, because some users are far more active than others; when that happens, the negative binomial fits better.

> **Worked example:** A storage system crashed 5 times last year. Treat crashes as a steady random rate of 5 per year. The chance of at least one crash next month is 1 minus e to the power of minus 5/12, about 34 percent. Saying 5/12, about 42 percent, mistakes an expected count for a probability.

## Continuous distributions

**Uniform:** every value in a range equally likely; probabilities are lengths divided by the total width.

**Normal:** the bell curve. About 68 percent of values fall within one standard deviation of the average, 95 percent within two, 99.7 percent within three. It appears when many small independent effects add up, and averages of large samples follow it even when the data don't.

**Exponential:** the waiting time until the next random event, such as the next customer. It has no memory: if buses arrive at random every 10 minutes on average, you wait 10 minutes on average whenever you arrive, not 5. You're more likely to arrive during a long gap than a short one.

**Gamma, beta and t:** the gamma is the wait for several events; the beta is a distribution over probabilities, perfect for describing uncertainty about a conversion rate; the t looks like a normal with fatter tails and is what you use when you estimate the spread from a small sample.

> **Worked example:** Puppy weights are roughly normal with average 1 lb and standard deviation 0.12 lb. A 1.1 lb puppy is 0.1/0.12 = 0.83 standard deviations above average. The top 10 percent start at 1 + 1.28 × 0.12, about 1.15 lb.

Most of these belong to one family (the exponential family), and that's where the sigmoid in logistic regression comes from: it's what the coin flip distribution looks like when written in that shared form.

> **Watch:** [CS229 Lecture 4: Exponential Family, GLMs Classification](https://www.youtube.com/watch?v=8gVi4Rk21Eg).

## Law of large numbers and the CLT

The law of large numbers says averages settle down: flip a fair coin 10 times and you might see 70 percent heads; flip it 10,000 times and you'll be very close to 50 percent. It does not say luck evens out in the short run. After 10 heads, the next flip is still 50/50.

The central limit theorem says the average of many independent draws follows a bell curve, whatever the shape of the data, as long as the data's spread is finite. Revenue per user can be wildly lopsided, but the average revenue over 10,000 users, across imagined repeat experiments, is close to a bell curve. That's why ordinary tests work on business metrics.

Where it breaks:

- **No finite spread.** Some distributions (Cauchy, some power laws) have tails so heavy that averages never stabilise. A simulation showed the spread of Cauchy averages stayed the same from 10 to 1,000 data points, while exponential averages tightened tenfold.
- **Lopsided data converge slowly.** "30 is enough" is folklore. Averages of skewed waiting times are still visibly lopsided at 30; revenue with big spenders may need thousands.
- **Linked data.** A million sessions from 50,000 users carry far less information than a million independent observations.

## Sampling distributions and standard error

Imagine running your survey again and again. Your estimate would come out a bit different each time; the standard error is the typical size of that wobble. The standard deviation describes how different individuals are; the standard error describes how uncertain your estimate is. For an average, the standard error is the standard deviation divided by the square root of the sample size.

That square root is why precision is expensive. To cut a margin of error from 3 points to 0.3 points, you need 100 times the sample, not 10 times.

> **Worked example:** To estimate a 5 percent click through rate within half a percentage point at 95 percent confidence, you need about 1.96 squared × 0.05 × 0.95 ÷ 0.005 squared, roughly 7,300 impressions, assuming impressions are independent. They usually aren't, because the same users see many ads, so count users or use a user level method.

The sample variance divides by n minus 1 instead of n. The sample's own average sits closer to the data than the true average does, so distances from it run a little small, and dividing by n minus 1 corrects that exactly.

## Confidence intervals

A 95 percent confidence interval comes from a method that captures the true value in 95 percent of repeated studies. Think of a ring toss machine that rings the peg 95 percent of the time: once a ring has landed, it's either on the peg or not. The 95 percent describes the machine, not this toss.

> **Worked example:** 100 puppies average 1 lb, and individual weights have a standard deviation of 0.12 lb. The interval for the average weight is 1 ± 1.96 × 0.12 / 10, about 0.976 to 1.024 lb. That's very narrow, so most individual puppies fall outside it; it describes the average, not individuals. Another litter's average would land inside only about 83 percent of the time.

Two intervals that overlap can still hide a real difference, because the uncertainty in a difference is smaller than the two widths added together. Compute an interval for the difference itself.

If you want to say "there's a 95 percent chance the value is in this range", you need the Bayesian version, a credible interval, which requires stating a prior.

## Hypothesis testing

A test starts from a default claim, the null hypothesis, usually "no effect", and asks whether the data would be surprising if it were true.

The p value is how often you'd see data at least this extreme if there were truly no effect. A p value of 0.03 does not mean a 3 percent chance the null is true, or a 97 percent chance your feature works. It's a measure of surprise assuming no effect.

Two ways to be wrong: a false alarm (Type I error), whose rate you set in advance, usually 5 percent; and a miss (Type II error). Power is the chance of catching a real effect of a given size. A test with 80 percent power for a 1 point lift misses a real 1 point lift one time in five.

> **Worked example:** To tell a 10 percent conversion rate from 11 percent with 5 percent false alarms and 80 percent power, you need about 14,750 users in each group. A simulated test of that size caught the difference about 81 percent of the time.

Things that go wrong in practice:

- **Peeking.** Checking a test daily and stopping at the first p below 0.05 gives chance many tries. In a simulation with 10 looks, false alarms rose to about 19 percent.
- **Huge samples.** With millions of users, trivial effects become "significant". Ask whether the effect is big enough to matter.
- **Rare true effects.** If only 10 percent of tested ideas work and power is 80 percent, about 36 percent of significant results are false alarms. It's the medical test logic again.

> **Say it like this:** "The p value is the chance of data at least this extreme if there were no effect. It's about the data assuming the null, not about the null given the data."

## Choosing a test

Three questions pick the test: what kind of outcome (a measurement, a yes or no, a ranking), how many groups, and whether the same units are measured twice.

| Situation | Test |
| --- | --- |
| Two groups, a measurement like revenue | Welch's t-test |
| Same people before and after | Paired t-test |
| Three or more groups | ANOVA |
| Yes or no outcome, two groups, big samples | Two proportion z-test or chi square |
| Yes or no outcome, small counts | Fisher's exact test |
| Ratings or rankings | Mann Whitney |
| Anything, with randomised groups | Permutation test |

Use Welch's version of the t-test by default. The older Student version assumes both groups are equally noisy; when the smaller group is noisier, its false alarm rate in a simulation was about 25 percent instead of 5. Welch's stays near 5.

Pairing is powerful: people differ a lot from each other, but each person is consistent, so comparing each person with themselves cancels the person to person noise.

A permutation test is the most intuitive: shuffle the "treatment" and "control" labels thousands of times and see how often chance alone produces a difference as big as the real one.

> **Interview trap:** "The data aren't normal, so use a nonparametric test." A rank test asks a different question (does one group tend to have larger values?) than a t-test (do the averages differ?). Pick the question first.

## Multiple comparisons

Run 20 tests where nothing is really going on, and the chance at least one looks significant is about 64 percent. Checking many metrics, many user segments or many model versions all count as running many tests.

Two kinds of protection:

- **No false claims at all** (family wise error rate). Bonferroni divides the threshold by the number of tests. Holm does the same for the smallest p value and relaxes step by step; it's never worse than Bonferroni.
- **A known share of false claims** (false discovery rate). Benjamini Hochberg lets about 5 percent of your discoveries be false leads, which is the right deal when screening many candidates you'll follow up anyway.

> **Worked example:** On eight p values (0.001, 0.0065, 0.0075, 0.020, 0.028, 0.20, 0.35, 0.62), Bonferroni declares 1 significant, Holm 3, and Benjamini Hochberg 5.

The best protection is deciding on one main metric before the test starts.

## Maximum likelihood and MAP

Likelihood flips probability around. Probability asks "if the coin is fair, how likely is 7 heads in 10?" Likelihood asks "given 7 heads in 10, which coin best explains it?" Maximum likelihood estimation picks the parameter that makes the observed data most probable. For a coin, that's just the observed rate, 7/10. For a bell curve, it's the sample average, and the spread estimate turns out slightly too small.

Two connections make this central to machine learning. Least squares regression is maximum likelihood when the noise follows a bell curve. And the cross entropy loss used to train every classifier and language model is the negative log likelihood.

Maximum likelihood trusts the data completely, so it fails with thin data. A word never seen in training gets probability zero, which breaks a spam filter. MAP estimation adds a prior belief to the likelihood. A prior that says "weights are probably small" is exactly L2 regularisation; "add one to every count" (Laplace smoothing) is a prior that says nothing is impossible.

Generative models apply these ideas to classification. Gaussian discriminant analysis fits a bell curve to each class and classifies new points with Bayes' rule. A Gaussian mixture does the same when the class labels are hidden, using EM: guess each point's cluster membership, refit the clusters, repeat.

> **Watch:** [CS229 Lecture 5: Gaussian Discriminant Analysis](https://www.youtube.com/watch?v=zRdE8A4UZes), [Lecture 10: GMM (EM), PCA](https://www.youtube.com/watch?v=sUS-eTa0l6s), and Karpathy's [makemore part 1 on the negative log likelihood](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3014s).

## Bayesian inference and conjugate priors

Bayesian inference treats the unknown, like a coin's chance of heads, as uncertain and describes that uncertainty with a distribution. Start with a prior, multiply by the likelihood, and you get the posterior, your updated belief. Then you can say things directly, like "there's an 87 percent chance this coin favours heads."

For coin flips and conversion rates the update is just counting. The beta prior acts like imaginary past successes and failures; you add the real ones.

> **Worked example:** Start with a flat prior (one imaginary head and one imaginary tail) and observe 8 heads in 12 flips. The posterior is like 9 heads and 5 tails: average 9/14, about 0.64, with a 95 percent chance the true rate is between about 0.39 and 0.86.

That pull toward the prior is useful. A seller with 3 perfect reviews shouldn't outrank one with 190 positive out of 200; a prior worth 20 reviews at 90 percent ranks them at about 91 and 94 percent, the right order.

Bayes also explains puzzles. One coin in 1,000 is double headed; you pick a coin and flip 10 heads in a row. The chance it's the double headed coin is only about 51 percent, because the rarity of the trick coin almost cancels the rarity of 10 fair heads.

## The bootstrap

The bootstrap estimates uncertainty without a formula. Treat your sample as a mini population, draw a new sample of the same size from it with replacement, recompute your statistic, and repeat thousands of times. The spread of the results is your standard error.

> **Worked example:** For 80 skewed order values with a median of 22.2, 10,000 bootstrap medians put the 95 percent interval at about 16.9 to 28.7.

It's ideal for medians, percentiles, ratios and model metrics. It breaks when data points are linked (resample whole users, not single events), for extremes like the maximum, and with tiny samples. A permutation test shuffles labels to ask "could this be chance?"; the bootstrap resamples to ask "how precise is this estimate?"

## Causation and data traps

Two things moving together can mean one causes the other, the reverse, a third factor driving both, a biased sample, or luck. Randomised experiments rule out all but the first.

**Simpson's paradox:** a trend in every subgroup can reverse when you combine groups. In a kidney stone study, treatment A did better for small stones (93 versus 87 percent) and for large stones (73 versus 69 percent), yet worse overall (78 versus 83 percent), because A was given mostly to the harder large stone cases.

**Regression to the mean:** an extreme result is partly luck, and luck doesn't repeat. Kahneman's flight instructors thought criticism improved pilots and praise hurt them; really, unusually bad landings were followed by more normal ones, and so were unusually good ones. In a simulation, the top 10 percent on a test that was half skill and half luck scored about half as far above average on a retest.

**Small samples look extreme.** The highest and the lowest kidney failure rates both turn up in small cities, because small populations have noisy rates.

**Survivorship bias:** studying only what survived a filter. Abraham Wald saw bullet holes on bombers that returned and said to armour where the holes weren't, because planes hit there never came back.

> **Interview trap:** "Users who saw the new module bought 30 percent more." The module shows up for engaged users who buy more anyway. Propose an experiment.

## Classic probability puzzles

The pattern behind most puzzles: name the states, condition on the next step, and use averaging or symmetry.

- **Two heads in a row:** 6 flips on average. A tail after a head wipes out your progress. **Head then tail** takes only 4, because a second head doesn't hurt you.
- **Monty Hall:** switch, and you win 2/3 of the time, because the host knows where the car is and never opens it. If the host opened a door at random and happened to show a goat, it would be 50/50.
- **Birthdays:** 23 people give a better than even chance that two share a birthday, because 23 people form 253 pairs. Matching your birthday takes 253 other people.
- **Collecting all six die faces:** about 14.7 rolls, because the last new face takes 6 rolls on average by itself.
- **Gambler's ruin:** starting with 3 dollars, betting 1 dollar on fair flips until you reach 10 or 0, you reach 10 with chance 3/10, after 21 bets on average.
- **Stick broken in three:** the pieces form a triangle with chance 1/4, because every piece must be shorter than half.
- **Fair coin from a biased one:** flip twice; heads then tails means heads, tails then heads means tails, otherwise retry.
- **Tank serial numbers:** with 5 observed and a maximum of 250, estimate 250 + 250/5 - 1 = 299.

> **Say it like this:** "For two heads in a row I track two states, start and just saw a head, and condition on the next flip. A tail after a head throws away my progress, which is why it's 6 instead of 4."
