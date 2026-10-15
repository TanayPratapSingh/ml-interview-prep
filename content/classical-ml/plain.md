# Classical machine learning, explained plainly

This is the easy version of the classical machine learning module: the ideas behind every model and evaluation habit that came before neural networks, told with analogies and small worked examples. Each section matches a section of the deep version, so you can switch whenever you want the derivation. Every number below came from code that was run.

## Framing a problem as ML

Before any model, decide four things: what you are predicting, for whom, at what moment, and what decision the prediction feeds. A bank predicting fraud must say "for this transaction, at the moment it is authorised, so we can block it". That sentence tells you which data you are allowed to use (only what exists at authorisation) and how to judge the model (a missed fraud and a blocked honest customer cost different amounts).

The main settings differ by what feedback you get:

- **Supervised learning:** you have examples with answers, like houses with their sale prices. Predicting a number is regression; predicting a category is classification.
- **Unsupervised learning:** you have no answers and look for structure, like grouping customers by behaviour.
- **Self supervised learning:** you invent answers from the data itself, such as hiding a word and predicting it. This is how large language models learn before anyone labels anything.
- **Reinforcement learning:** an agent acts, gets a reward or a penalty, and learns which actions pay off over time, like learning a game by playing it.

A model is parametric if it keeps a fixed set of numbers no matter how much data you have (a straight line has two), and nonparametric if what it stores grows with the data (a nearest neighbour model keeps every example).

> **Watch:** [CS229 Lecture 1](https://www.youtube.com/watch?v=DATnpGoGhM8) and [CS229 Lecture 2](https://www.youtube.com/watch?v=cmNIMjPYdgM).

## Bias and variance

A model's error on new data has three sources. Noise is the randomness no model can predict. Bias is being systematically wrong because the model is too simple: a straight line through a curve misses in the same way whatever data you give it. Variance is being wrong because the model is too jumpy: it bends to fit the quirks of the particular sample it saw, so it changes a lot if you redraw the sample.

Think of darts. Bias is your average throw landing off centre. Variance is your throws scattering widely around wherever they land on average.

> **Worked example:** Fit curves to points scattered around a sine wave, 30 points at a time, 4,000 times over. A straight line has bias error 0.26 and variance 0.02: it is always wrong in the same way. A cubic has bias near 0 and variance 0.011, the best of the three. A degree 9 polynomial has bias near 0 but variance 0.040, four times the cubic's. Add the noise (0.09) and the totals match the measured test error.

Underfitting shows up as high error on the training data and similar error on new data. Overfitting shows up as low training error and much higher error on new data. A learning curve plots both against the amount of training data: a big gap that shrinks as data grows means variance, and more data helps; two curves meeting at a high error mean bias, and more data does nothing.

## Linear regression

Linear regression draws the best straight line (or flat plane, with more inputs) through the data, where "best" means the smallest total of squared vertical distances from points to the line. Squaring punishes big misses heavily and makes the answer easy to compute: there is a formula, the normal equations, that gives the best line in one step. When there are millions of inputs you instead walk downhill on the error surface step by step (gradient descent), and both routes land on the same line.

Why squared distance? If the scatter around the true line is bell shaped, the line that minimises squared error is the most likely line. That is the probabilistic justification.

Things that go wrong:

- **Two inputs that say the same thing** (house size in square feet and in square metres). The model cannot tell which deserves credit, so the coefficients become huge and unstable even though predictions stay fine. This is multicollinearity.
- **Scatter that widens** as the input grows (spending is more variable among rich households). The line is still fine but the stated uncertainty is wrong. Weighted least squares fixes it by trusting the noisy points less: in one test it cut the standard error of the slope from 0.056 to 0.041.
- **One unusual point far out on the input axis** can drag the line toward itself. That is a high leverage point; if it also sits off the line, it is influential.

Reading a coefficient: if the bedroom coefficient were 12,000, you would say "each extra bedroom adds 12,000 dollars to the predicted price, holding the other inputs fixed". That describes the model, not cause and effect.

> **Watch:** [CS229 Lecture 3](https://www.youtube.com/watch?v=uJF_gL3jhxI) on weighted least squares.

## Regularisation

Regularisation adds a penalty for large coefficients, so the model prefers simpler explanations unless the data strongly demand otherwise. It is like a budget: every unit of coefficient costs something, and only features that earn their cost keep weight.

- **Ridge (L2)** charges by the square of each coefficient. It shrinks everything toward zero smoothly but never all the way.
- **Lasso (L1)** charges by the absolute size. It pushes weak coefficients exactly to zero, so it also picks features.
- **Elastic net** mixes both, which helps when several useful features are correlated.

Why does lasso produce exact zeros? Picture the allowed region for the coefficients. For lasso it is a diamond with sharp corners sitting on the axes; for ridge it is a circle. The best fit tends to touch the diamond at a corner, where some coefficients are zero; a circle has no corners.

> **Worked example:** With unregularised coefficients of 3.10, 0.43, 0.13 and 1.18 (signs aside) and a lasso penalty of 0.5, lasso returns 2.60, 0, 0 and 0.68: it subtracts 0.5 from each and stops at zero. Ridge with penalty 1 halves every coefficient and zeroes none.

There is also a Bayesian reading. Ridge is what you get if you believed beforehand that coefficients are probably small and bell shaped around zero; lasso is that belief with a sharper peak at zero. Scale your features first, or the penalty treats a feature measured in grams differently from the same feature in kilograms.

## Logistic regression

Logistic regression predicts the probability of a yes or no outcome. It computes a weighted score of the inputs, just like linear regression, then squashes the score into the range 0 to 1 with the S shaped sigmoid curve. The weighted score is the log odds, so each coefficient says how much the odds multiply per unit of input: a coefficient of 1.86 means the odds become 6.4 times larger.

It is trained by making the observed outcomes as likely as possible, which is the same as minimising log loss. Log loss punishes confident wrong answers hard: predicting 1 percent for something that happened costs 4.6, while a shrug of 50 percent costs 0.69.

Why not use squared error, as in linear regression? Two reasons. Squared error through a sigmoid gives a surface that is not a single bowl, with flat stretches where training stalls, while log loss gives one smooth bowl. And squared error barely corrects a badly wrong confident prediction: in one example its push was 0.005 against log loss's 0.998.

The decision boundary is a straight line (or flat plane), so logistic regression is a linear classifier. If the two classes can be separated perfectly, the coefficients race to infinity, which is one reason libraries add regularisation by default. Replace the smooth S curve with a hard yes or no step and you get the perceptron, the ancestor of neural networks, covered in the next module.

> **Watch:** [CS229 Lecture 4](https://www.youtube.com/watch?v=8gVi4Rk21Eg).

## GLMs and the exponential family

Linear regression, logistic regression and Poisson regression are the same recipe with different ingredients. Generalised linear models say: pick a probability distribution that suits the outcome (bell curve for amounts, coin flip for yes or no, Poisson for counts), compute a weighted score of the inputs, and connect the score to the distribution's average through a link function (identity, sigmoid, exponential).

The distributions that fit this recipe form the exponential family, and they share a neat property: the same training update works for all of them, "nudge the weights in proportion to the prediction error times the input". That is why linear and logistic regression have identical looking update rules.

> **Worked example:** Counting visitors per hour is a count, so use Poisson regression. On simulated data its coefficient was 0.83, meaning each unit of the input multiplies the expected count by about 2.3. Ordinary least squares on counts can predict negative visitors.

## Generative classifiers: GDA and naive Bayes

There are two ways to tell elephants from dogs. A discriminative model learns the boundary between them directly. A generative model learns what elephants look like and what dogs look like, then asks which description fits a new animal better.

Gaussian discriminant analysis assumes each class forms a bell shaped cloud. If the clouds share the same shape, the resulting boundary is exactly a logistic regression boundary. The difference is in the assumptions: when the clouds really are bell shaped, GDA learns well from less data; when they are not, logistic regression is safer.

Naive Bayes assumes every feature is independent once you know the class, as if each word in a spam email were chosen without regard to the others. That is false, yet the classifier often works well, because getting the ranking right is easier than getting the probabilities right. Its probabilities tend to be too extreme.

> **Worked example:** A tiny spam filter has never seen the word "free" in a normal email. Without smoothing, any email containing "free" is spam with probability 1.000, whatever else it says. Laplace smoothing pretends each word appeared once more than it did, and the same email gets 0.667.

> **Watch:** [CS229 Lecture 5](https://www.youtube.com/watch?v=zRdE8A4UZes).

## kNN and the curse of dimensionality

k nearest neighbours classifies a new point by asking its k closest training points to vote. There is no training, only remembering. Small k follows every local wiggle (high variance); large k smooths everything into the majority (high bias). Because "closest" depends on distance, features must be on comparable scales.

The curse of dimensionality is what happens to "closest" when there are many features. In two dimensions, the nearest of 500 random points is about 40 times closer than the farthest. In 1,000 dimensions the nearest is 0.90 as far as the farthest: everyone is roughly equally distant, so "nearest" carries little information. To capture 10 percent of the data in 100 dimensions you need a box spanning 98 percent of each feature's range, which is hardly a neighbourhood. Real data usually lie near a much lower dimensional surface, which is why reducing dimensions first rescues kNN.

## SVMs and kernels

A support vector machine draws the boundary that leaves the widest possible empty street between the two classes. Only the points on the edge of the street, the support vectors, decide where it goes; every other point could move without changing anything. When classes overlap, a setting called C decides how much you pay for points that wander into the street: small C gives a wide, forgiving street, large C a narrow one that chases every point.

The kernel trick lets an SVM draw curved boundaries cheaply. Curved boundaries in the original space are straight boundaries in a bigger space of combined features (squares, products). The SVM's maths only ever needs similarity scores between pairs of points, and a kernel computes the similarity in that bigger space without building it. A degree 2 kernel on 3 features matched the explicit 13 feature calculation exactly (1.730612 both ways). The popular RBF kernel corresponds to an infinitely large feature space.

SVM scores are distances from the boundary, not probabilities, and kernel SVMs get slow beyond about a hundred thousand rows.

## Decision trees

A decision tree is a game of twenty questions: "is income above 50,000?", then "has the customer called support?", until it reaches a leaf with a prediction. It picks each question to make the groups below it as pure as possible, measured by Gini impurity or entropy.

> **Worked example:** A group of 10 buyers and 10 non buyers has Gini impurity 0.5 (maximally mixed). A question that splits it into 8 buyers with 2 non buyers, and 2 buyers with 8 non buyers, lowers the impurity to 0.32. Measured in entropy, the split gains 0.28 bits.

Trees need no scaling and capture interactions naturally, but a deep tree memorises the training data, and a small change in the data can produce a very different tree. Pruning cuts back branches that do not earn their keep: in one test it shrank a tree from 16 leaves to 9 and raised test accuracy from 0.881 to 0.916. Trees also cannot predict outside the range they saw, which hurts when a trend keeps rising.

## Bagging and random forests

Bagging builds many trees, each on a different random resample of the data, and averages them. Individual trees are jumpy; their average is steady, the way a crowd's average guess beats most individual guesses.

The catch: averaging only cancels errors that differ. If all trees make the same mistakes, averaging changes nothing. A random forest makes the trees more different by letting each split consider only a random subset of features, so one dominant feature cannot dictate every tree. With 50 trees whose errors are 30 percent correlated, averaging cuts the error variance to about 0.31 of a single tree's, and adding more trees cannot push it below 0.30.

Each resample leaves out about 37 percent of the rows. Scoring each row with only the trees that never saw it gives a free validation score, the out of bag estimate; in one test it said 0.906 when the true test accuracy was 0.920.

## Boosting

Boosting builds trees one after another, each fixing what the previous ones got wrong. Bagging is a committee voting independently; boosting is a relay where each runner starts where the last one stumbled.

AdaBoost does this by increasing the weight of misclassified examples, so the next model concentrates on them, and giving accurate models a bigger vote. Gradient boosting does it by fitting each new small tree to the current errors (the residuals) and adding a fraction of it to the ensemble. That loop takes ten lines of code and matched scikit-learn's implementation exactly in a test.

The three famous libraries are refinements. XGBoost adds penalties on tree size and leaf values and handles missing values by learning which branch they should take. LightGBM is built for speed, binning features and growing the most promising leaf first. CatBoost focuses on categorical features and avoids a subtle leak where a category's average outcome is computed from the very rows being predicted.

Boosting can overfit if you keep adding trees, so stop when validation error stops improving.

## Clustering

Clustering finds groups without labels. k-means places k centres, assigns each point to its nearest centre, moves each centre to the middle of its points, and repeats. Every round reduces the total distance, so it always settles, though not always in the best place, which is why you rerun it from different starts. k-means++ chooses starting centres spread far apart: in a 15 cluster test it cut the average error by about a third compared with random starts.

Choosing k has no single right answer. The total within cluster distance always falls as k grows, so look for an elbow; the silhouette score rewards tight, well separated clusters. On data built from four groups, the silhouette picked 4 while a Gaussian mixture's BIC picked 3, because two groups overlapped.

Other methods suit other shapes. Gaussian mixture models allow oval clusters and give each point a probability of belonging to each one, fitted by the EM algorithm (guess memberships, refit the shapes, repeat). DBSCAN grows clusters through dense regions and labels isolated points as noise: on two interlocking crescent shapes it found both perfectly where k-means scored 0.25 on the agreement scale. Hierarchical clustering builds a family tree of merges you can cut at any level.

> **Watch:** [CS229 Lecture 9](https://www.youtube.com/watch?v=bSmIGBCoffA) and [CS229 Lecture 10](https://www.youtube.com/watch?v=sUS-eTa0l6s).

## Dimensionality reduction

PCA finds the directions along which the data spread out most and keeps only the top few. Imagine photographing a flat dinner plate: one angle shows its full width, another only its edge. PCA picks the angles that show the most.

> **Worked example:** Five correlated measurements were reduced to two components that kept 49 percent and 27 percent of the total spread, 75 percent together. The information lost equals exactly the spread along the three dropped directions.

PCA is sensitive to units, so standardise first, and it keeps spread rather than usefulness: a quiet direction can be the one that predicts your target.

t-SNE and UMAP make 2D maps for looking at high dimensional data. They keep neighbours together but distort everything else: cluster sizes and the gaps between clusters in these pictures mean nothing, and pure noise can look like clusters. Use them to explore, never as evidence.

## Evaluation metrics

Accuracy misleads when one class is rare. With 10 positives in 100 cases, a model that flags 20 cases and catches 8 has accuracy 0.86, lower than the 0.90 you get by predicting "no" for everyone.

Precision asks "of the cases I flagged, how many were real?" (8 of 20, so 0.4). Recall asks "of the real cases, how many did I catch?" (8 of 10, so 0.8). F1 is their harmonic mean (0.53) and sits near the smaller one.

ROC AUC is the chance that a random positive gets a higher score than a random negative: 0.5 is guessing, 1 is perfect. It ignores how rare positives are. PR AUC does not: the same model scored ROC AUC 0.85 whether positives were 50 percent or 1 percent of the data, while its PR AUC fell from 0.85 to 0.14, because at 1 percent every true alarm comes with many false ones.

Probabilities should also be calibrated: of the cases given 80 percent, about 80 percent should happen. Naive Bayes tends to be overconfident and random forests underconfident at the extremes. Finally, a probability is not a decision. If a missed fraud costs ten times a false alarm, flag anything above about 9 percent risk, not 50.

## Cross validation and leakage

Cross validation estimates performance on unseen data by rotating which slice of the data is held back. Split into five folds, train on four, test on the fifth, and repeat five times. Keep a final test set aside and look at it once.

How you split must match how the model will be used. If the same patient appears many times, keep each patient entirely in one fold, or the model is graded on people it has already met: in one test, random folds reported a score of 0.95 and patient grouped folds reported negative 1.0, which was the truth for new patients. For anything over time, train on the past and test on the future.

Leakage is any way information from the test data or the future sneaks into training. The classic case: choosing the "best" 25 of 10,000 features using all the data, then cross validating. On labels that were pure coin flips this produced 0.86 accuracy; doing the selection inside each fold gave 0.48, the honest answer. Other leaks: scaling or imputing with statistics from the whole dataset, duplicates across the split, and features recorded after the outcome.

## Class imbalance

When positives are rare, fix the evaluation before the model: use precision, recall or PR AUC, and move the decision threshold to match the costs. Then consider reweighting, which makes each rare example count more during training (with 900 negatives and 100 positives, balanced weights are about 0.56 and 5.0). Resampling, duplicating rare examples or dropping common ones, also works but must happen only inside the training data.

> **Worked example:** With features that contain no signal at all, duplicating the rare positives before splitting reported a perfect recall of 1.0, because copies of the same rows landed in both training and test. Duplicating inside the training fold only reported 0.0, the truth.

Focal loss, used in deep learning, down weights easy examples so training focuses on the hard ones.

## Features and missing data

Good features often matter more than the choice of model. Scale numbers for distance based models, turn categories into columns (one hot) or into category averages computed without peeking at the row itself, take logs of skewed amounts like prices, and add interactions when a linear model needs them.

For missing values, first ask why they are missing. If they are missing at random, filling with the median is fine. If missingness itself tells you something (people with high incomes skip the income question), keep that signal with an "is missing" flag: in one test the flag lifted AUC from 0.925 to 0.941, and a boosting model that handles gaps natively reached 0.954. Feature selection and imputation both belong inside the cross validation loop.

## Interpretability

Interpretability asks which inputs drive the model's predictions, overall or for one case. Linear coefficients are the simplest answer, valid when features are on the same scale and not strongly correlated.

Tree models report an impurity based importance, but it favours features with many distinct values. In a test, a random ID column with 1,000 values was ranked most important (0.69 against 0.31 for the real signal). Permutation importance asks a better question, "how much worse does the model get on new data if I shuffle this column?", and gave the real signal 0.155 and the ID 0.003.

SHAP splits a single prediction into contributions from each feature that add up exactly to the prediction, using a fair share rule from game theory. All of these explain the model, not the world.

## Hyperparameter search

Hyperparameters are the settings you choose before training, like the penalty strength or tree depth. Grid search tries every combination; random search samples combinations and usually finds good settings faster, because only a few settings matter and random search tries more distinct values of each. Smarter methods (Bayesian optimisation, successive halving) learn from earlier trials or stop bad ones early.

Two rules: search penalty strengths and learning rates on a logarithmic scale (0.001, 0.01, 0.1, ...), and never tune on the test set. If you want an honest score for the whole tuning process, put the search inside an outer cross validation loop (nested cross validation).

## Time series basics

Time series data come in order, and yesterday tells you about today. Three habits change. First, check stationarity: whether the average and spread stay stable over time. A trend or a random walk breaks it, and differencing (modelling the change from one step to the next) usually restores it. Second, look at autocorrelation, how strongly the series correlates with its own past: for a series where each value is 0.7 times the last plus noise, the correlation with one step back was 0.70, two steps 0.49, three steps 0.34.

Third, validate forward in time: train on January to June, test on July, then train through July and test on August. Random shuffling lets the model peek at the future. ARIMA models combine differencing, dependence on past values, and dependence on past errors. Always compare against simple baselines like "same as last week".
