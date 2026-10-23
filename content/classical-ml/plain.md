# Classical machine learning, explained plainly

This guide explains classical machine learning (ML) in everyday words. Classical machine learning covers the models and evaluation habits that came before neural networks. A neural network is a model built from many simple connected units that learn from data. The guide uses analogies and small worked examples. Each section matches a section of the deep version, so you can switch whenever you want the derivation (the step by step maths). The lecture links point to CS229, Stanford University's machine learning course. Every number below came from code that was run.

## Framing a problem as ML

Framing a problem means deciding four things before you pick any model. Decide what you are predicting, for whom, at what moment, and what decision the prediction feeds. A bank predicting fraud must say "for this transaction, at the moment it is authorised, so we can block it". That one sentence tells you two things. It tells you which data you may use: only what exists at authorisation. It also tells you how to judge the model, because a missed fraud and a blocked honest customer cost different amounts.

The main settings differ by what feedback you get.

- **Supervised learning:** you have examples with answers, like houses with their sale prices. The inputs are called features and the answers are called labels. Predicting a number is regression. Predicting a category is classification.
- **Unsupervised learning:** you have no answers and look for structure, like grouping customers by behaviour.
- **Self supervised learning:** you invent answers from the data itself, such as hiding a word and predicting it. This is how large language models (programs trained on huge amounts of text) learn before anyone labels anything.
- **Reinforcement learning:** an agent (a program that takes actions) acts and gets a reward or a penalty. Over time it learns which actions pay off, like learning a game by playing it.

A model is parametric if it keeps a fixed set of numbers no matter how much data you have. A straight line has two such numbers. A model is nonparametric if what it stores grows with the data. A nearest neighbour model keeps every example.

> **Watch:** [CS229 Lecture 1](https://www.youtube.com/watch?v=DATnpGoGhM8) and [CS229 Lecture 2](https://www.youtube.com/watch?v=cmNIMjPYdgM).

## Bias and variance

A model's error on new data has three sources. Noise is the randomness that no model can predict. Bias means being systematically wrong because the model is too simple. A straight line through a curve misses in the same way whatever data you give it. Variance means being wrong because the model is too jumpy. It bends to fit the quirks of the particular sample it saw, so it changes a lot if you redraw the sample.

Think of darts. Bias is your average throw landing off centre. Variance is your throws scattering widely around wherever they land on average.

> **Worked example:** Fit curves to points scattered around a sine wave (a smooth up and down curve), 30 points at a time, 4,000 times over. A straight line has bias error 0.26 and variance 0.02. It is always wrong in the same way. A cubic (a curve with up to two bends) has bias near 0 and variance 0.011, the best of the three. A degree 9 polynomial (a very bendy curve) has bias near 0 but variance 0.040, four times the cubic's. Add the noise (0.09) and the totals match the measured test error.

Underfitting means the model is too simple. It shows up as high error on the training data (the examples the model learned from) and similar error on new data. Overfitting means the model has memorised the quirks of its training data. It shows up as low training error and much higher error on new data. A learning curve plots both errors against the amount of training data. A big gap that shrinks as data grows means variance, and more data helps. Two curves meeting at a high error mean bias, and more data does nothing.

## Linear regression

Linear regression draws the best straight line through the data. With more inputs, it draws a flat plane instead. "Best" means the smallest total of squared vertical distances from points to the line. Squaring punishes big misses heavily. It also makes the answer easy to compute. A formula called the normal equations gives the best line in one step. When there are millions of inputs, you instead walk downhill on the error surface step by step. This is called gradient descent. Both routes land on the same line.

Why squared distance? If the scatter around the true line is bell shaped, the line that minimises squared error is the most likely line. That is the probabilistic justification, meaning the reason from probability theory.

Things that go wrong:

- **Two inputs that say the same thing**, such as house size in square feet and in square metres. The model cannot tell which deserves credit. The coefficients (the numbers that multiply each input) become huge and unstable even though predictions stay fine. This is multicollinearity.
- **Scatter that widens** as the input grows. Spending is more variable among rich households, for example. The line is still fine, but the stated uncertainty is wrong. Weighted least squares fixes this by trusting the noisy points less. In one test it cut the standard error of the slope (the typical error in the estimated slope) from 0.056 to 0.041.
- **One unusual point far out on the input axis** can drag the line toward itself. That is a high leverage point. If it also sits off the line, it is influential.

Reading a coefficient needs care. Suppose the bedroom coefficient were 12,000. You would say "each extra bedroom adds 12,000 dollars to the predicted price, holding the other inputs fixed". That describes the model, not cause and effect.

> **Watch:** [CS229 Lecture 3](https://www.youtube.com/watch?v=uJF_gL3jhxI) on weighted least squares.

## Regularisation

Regularisation adds a penalty for large coefficients. The model then prefers simpler explanations unless the data strongly demand otherwise. It is like a budget. Every unit of coefficient costs something, and only features that earn their cost keep weight.

- **Ridge (L2)** charges by the square of each coefficient. L2 is the name of this squared size measure. Ridge shrinks everything toward zero smoothly but never all the way.
- **Lasso (L1)** charges by the absolute size. L1 is the name of this plain size measure. Lasso pushes weak coefficients exactly to zero, so it also picks features.
- **Elastic net** mixes both. This helps when several useful features are correlated.

Why does lasso produce exact zeros? Picture the allowed region for the coefficients. For lasso it is a diamond with sharp corners sitting on the axes. For ridge it is a circle. The best fit tends to touch the diamond at a corner, where some coefficients are zero. A circle has no corners.

> **Worked example:** Start with unregularised coefficients (coefficients fitted with no penalty): 3.10, 0.43, 0.13 and 1.18, signs aside. Apply a lasso penalty of 0.5. Lasso returns 2.60, 0, 0 and 0.68. It subtracts 0.5 from each and stops at zero. Ridge with penalty 1 halves every coefficient and zeroes none.

There is also a Bayesian reading, which treats beliefs held before seeing the data as part of the model. Ridge is what you get if you believed beforehand that coefficients are probably small and bell shaped around zero. Lasso is that belief with a sharper peak at zero. Scale your features first. Otherwise the penalty treats a feature measured in grams differently from the same feature in kilograms.

## Logistic regression

Logistic regression predicts the probability of a yes or no outcome. It computes a weighted score of the inputs, just like linear regression. Then it squashes the score into the range 0 to 1 with the S shaped sigmoid curve. The weighted score is the log odds. Odds are the chance of yes divided by the chance of no, and log odds is the logarithm of that. So each coefficient says how much the odds multiply per unit of input. A coefficient of 1.86 means the odds become 6.4 times larger. That is the constant e (about 2.72) raised to the power 1.86.

It is trained by making the observed outcomes as likely as possible. That is the same as minimising log loss. Log loss punishes confident wrong answers hard. Predicting 1 percent for something that happened costs 4.6, while a shrug of 50 percent costs 0.69.

Why not use squared error, as in linear regression? There are two reasons. First, squared error through a sigmoid gives a surface that is not a single bowl. It has flat stretches where training stalls. Log loss gives one smooth bowl. Second, squared error barely corrects a badly wrong confident prediction. In one example its push was 0.005 against log loss's 0.998.

The decision boundary is the line where the model switches from predicting no to predicting yes. For logistic regression it is a straight line (or flat plane), so logistic regression is a linear classifier. If the two classes can be separated perfectly, the coefficients race to infinity. This is one reason libraries add regularisation by default. Replace the smooth S curve with a hard yes or no step and you get the perceptron. The perceptron is the ancestor of neural networks, covered in the next module.

> **Watch:** [CS229 Lecture 4](https://www.youtube.com/watch?v=8gVi4Rk21Eg).

## GLMs and the exponential family

Linear regression, logistic regression and Poisson regression are the same recipe with different ingredients. Poisson regression predicts counts, such as visitors per hour. Generalised linear models (GLMs) follow the recipe in three steps. First, pick a probability distribution that suits the outcome. Use a bell curve for amounts, a coin flip for yes or no, and the Poisson distribution for counts. Second, compute a weighted score of the inputs. Third, connect the score to the distribution's average through a link function. The link function can be the identity (no change), the sigmoid or the exponential.

The distributions that fit this recipe form the exponential family. They share a neat property. The same training update works for all of them: "nudge the weights in proportion to the prediction error times the input". That is why linear and logistic regression have identical looking update rules.

> **Worked example:** Counting visitors per hour is a count, so use Poisson regression. On simulated data its coefficient was 0.83, meaning each unit of the input multiplies the expected count by about 2.3. Ordinary least squares (the standard straight line fit) on counts can predict negative visitors.

## Generative classifiers: GDA and naive Bayes

There are two ways to tell elephants from dogs. A discriminative model learns the boundary between them directly. A generative model learns what elephants look like and what dogs look like. It then asks which description fits a new animal better.

Gaussian discriminant analysis (GDA) assumes each class forms a bell shaped (Gaussian) cloud of points. If the clouds share the same shape, the resulting boundary is exactly a logistic regression boundary. The difference is in the assumptions. When the clouds really are bell shaped, GDA learns well from less data. When they are not, logistic regression is safer.

Naive Bayes assumes every feature is independent once you know the class. It is as if each word in a spam email were chosen without regard to the others. That is false, yet the classifier often works well. The reason is that getting the ranking right is easier than getting the probabilities right. Its probabilities tend to be too extreme.

> **Worked example:** A tiny spam filter has never seen the word "free" in a normal email. Without smoothing, any email containing "free" is spam with probability 1.000, whatever else it says. Laplace smoothing pretends each word appeared once more than it did, and the same email gets 0.667.

> **Watch:** [CS229 Lecture 5](https://www.youtube.com/watch?v=zRdE8A4UZes).

## kNN and the curse of dimensionality

k nearest neighbours (kNN) classifies a new point by asking its k closest training points to vote. The letter k is a number you choose, such as 5. There is no training, only remembering. Small k follows every local wiggle, which means high variance. Large k smooths everything into the majority, which means high bias. "Closest" depends on distance, so features must be on comparable scales.

The curse of dimensionality is what happens to "closest" when there are many features. Each feature adds one dimension. In two dimensions, the nearest of 500 random points is about 40 times closer than the farthest. In 1,000 dimensions the nearest is 0.90 as far as the farthest. Everyone is roughly equally distant, so "nearest" carries little information.

To capture 10 percent of the data in 100 dimensions, you need a box spanning 98 percent of each feature's range. That is hardly a neighbourhood. Real data usually lie near a much lower dimensional surface. That is why reducing dimensions first rescues kNN.

## SVMs and kernels

A support vector machine (SVM) draws the boundary that leaves the widest possible empty street between the two classes. Only the points on the edge of the street decide where it goes. These points are the support vectors. Every other point could move without changing anything. When classes overlap, a setting called C decides how much you pay for points that wander into the street. Small C gives a wide, forgiving street. Large C gives a narrow one that chases every point.

The kernel trick lets an SVM draw curved boundaries cheaply. Curved boundaries in the original space are straight boundaries in a bigger space of combined features, such as squares and products. The SVM's maths only ever needs similarity scores between pairs of points. A kernel computes the similarity in that bigger space without building it. A degree 2 kernel on 3 features matched the explicit 13 feature calculation exactly (1.730612 both ways). The popular RBF (radial basis function) kernel corresponds to an infinitely large feature space.

SVM scores are distances from the boundary, not probabilities. Kernel SVMs get slow beyond about a hundred thousand rows.

## Decision trees

A decision tree is a game of twenty questions. It asks "is income above 50,000?", then "has the customer called support?", and so on. It stops at a leaf, which is an end point that holds a prediction. It picks each question to make the groups below it as pure as possible. Purity is measured by Gini impurity or entropy. Both are scores that are low when a group is mostly one class and high when it is mixed.

> **Worked example:** A group of 10 buyers and 10 non buyers has Gini impurity 0.5, which is maximally mixed. A question splits it into 8 buyers with 2 non buyers, and 2 buyers with 8 non buyers. That lowers the impurity to 0.32. Measured in entropy, the split gains 0.28 bits. A bit is the unit for measuring entropy.

Trees need no scaling. They also capture interactions naturally, which means the combined effect of two features. But a deep tree memorises the training data. A small change in the data can also produce a very different tree. Pruning cuts back branches that do not earn their keep. In one test it shrank a tree from 16 leaves to 9 and raised test accuracy from 0.881 to 0.916. Trees also cannot predict outside the range they saw. That hurts when a trend keeps rising.

## Bagging and random forests

Bagging builds many trees, each on a different random resample of the data, and averages them. The name is short for bootstrap aggregating. A resample draws rows from the data with replacement, so some rows repeat and others drop out. Individual trees are jumpy. Their average is steady, the way a crowd's average guess beats most individual guesses.

The catch is that averaging only cancels errors that differ. If all trees make the same mistakes, averaging changes nothing. A random forest makes the trees more different. Each split considers only a random subset of features, so one dominant feature cannot dictate every tree. With 50 trees whose errors are 30 percent correlated, averaging cuts the error variance to about 0.31 of a single tree's. Adding more trees cannot push it below 0.30.

Each resample leaves out about 37 percent of the rows. Score each row with only the trees that never saw it. This gives a free validation score, called the out of bag estimate. In one test it said 0.906 when the true test accuracy was 0.920.

## Boosting

Boosting builds trees one after another, each fixing what the previous ones got wrong. Bagging is a committee voting independently. Boosting is a relay where each runner starts where the last one stumbled.

AdaBoost (adaptive boosting) increases the weight of misclassified examples, so the next model concentrates on them. It also gives accurate models a bigger vote. Gradient boosting fits each new small tree to the current errors, which are called residuals. It then adds a fraction of that tree to the ensemble, which is the combined group of trees. That loop takes ten lines of code. It matched the implementation in scikit-learn, a widely used Python machine learning library, exactly in a test.

The three famous libraries are refinements.

- **XGBoost** (extreme gradient boosting) adds penalties on tree size and leaf values. It handles missing values by learning which branch they should take.
- **LightGBM** (light gradient boosting machine) is built for speed. It bins features, which means grouping their values into ranges. It also grows the most promising leaf first.
- **CatBoost** (categorical boosting) focuses on categorical features. It avoids a subtle leak where a category's average outcome is computed from the very rows being predicted.

Boosting can overfit if you keep adding trees, so stop when validation error stops improving.

## Clustering

Clustering finds groups without labels. k-means places k centres and assigns each point to its nearest centre. Then it moves each centre to the middle of its points, and repeats. Every round reduces the total distance, so it always settles. It does not always settle in the best place. That is why you rerun it from different starts. k-means++ chooses starting centres spread far apart. In a 15 cluster test it cut the average error by about a third compared with random starts.

Choosing k has no single right answer. The total within cluster distance always falls as k grows, so look for an elbow, the point where the drop flattens. The silhouette score rewards tight, well separated clusters. On data built from four groups, the silhouette picked 4. A Gaussian mixture's BIC (Bayesian information criterion, a score that trades fit against complexity) picked 3, because two groups overlapped.

Other methods suit other shapes.

- **Gaussian mixture models** allow oval clusters and give each point a probability of belonging to each one. They are fitted by the EM (expectation maximisation) algorithm: guess memberships, refit the shapes, repeat.
- **DBSCAN** (density based spatial clustering of applications with noise) grows clusters through dense regions and labels isolated points as noise. On two interlocking crescent shapes it found both perfectly, where k-means scored 0.25 on the agreement scale. That scale is the adjusted Rand index. A score of 1 means perfect agreement with the true groups, and 0 means random labels.
- **Hierarchical clustering** builds a family tree of merges that you can cut at any level.

> **Watch:** [CS229 Lecture 9](https://www.youtube.com/watch?v=bSmIGBCoffA) and [CS229 Lecture 10](https://www.youtube.com/watch?v=sUS-eTa0l6s).

## Dimensionality reduction

PCA (principal component analysis) finds the directions along which the data spread out most and keeps only the top few. Imagine photographing a flat dinner plate. One angle shows its full width, and another shows only its edge. PCA picks the angles that show the most.

> **Worked example:** Five correlated measurements were reduced to two components that kept 49 percent and 27 percent of the total spread, 75 percent together. The information lost equals exactly the spread along the three dropped directions.

PCA is sensitive to units, so standardise first. Standardising means rescaling every feature to the same spread. PCA also keeps spread rather than usefulness. A quiet direction can be the one that predicts your target.

t-SNE (t distributed stochastic neighbour embedding) and UMAP (uniform manifold approximation and projection) make 2D (two dimensional) maps for looking at high dimensional data. They keep neighbours together but distort everything else. Cluster sizes and the gaps between clusters in these pictures mean nothing. Pure noise can look like clusters. Use them to explore, never as evidence.

## Evaluation metrics

Accuracy misleads when one class is rare. Accuracy is the share of cases the model gets right. With 10 positives in 100 cases, a model that flags 20 cases and catches 8 has accuracy 0.86. That is lower than the 0.90 you get by predicting "no" for everyone.

Precision asks "of the cases I flagged, how many were real?" That is 8 of 20, so 0.4. Recall asks "of the real cases, how many did I catch?" That is 8 of 10, so 0.8. The F1 score is their harmonic mean (0.53) and sits near the smaller one. A harmonic mean is a kind of average that is pulled toward the smaller number.

ROC AUC (receiver operating characteristic, area under the curve) is the chance that a random positive gets a higher score than a random negative. A score of 0.5 is guessing, and 1 is perfect. It ignores how rare positives are. PR AUC (precision recall, area under the curve) does not ignore it. The same model scored ROC AUC 0.85 whether positives were 50 percent or 1 percent of the data. Its PR AUC fell from 0.85 to 0.14. The reason is that at 1 percent every true alarm comes with many false ones.

Probabilities should also be calibrated. Of the cases given 80 percent, about 80 percent should happen. Naive Bayes tends to be overconfident and random forests underconfident at the extremes. Finally, a probability is not a decision. Suppose a missed fraud costs ten times a false alarm. Then flag anything above about 9 percent risk, not 50. That cutoff is 1 divided by (1 plus 10).

## Cross validation and leakage

Cross validation estimates performance on unseen data by rotating which slice of the data is held back. Split the data into five folds, which are equal slices. Train on four, test on the fifth, and repeat five times. Keep a final test set aside and look at it once.

How you split must match how the model will be used. Suppose the same patient appears many times. Then keep each patient entirely in one fold. Otherwise the model is graded on people it has already met. In one test, random folds reported a score of 0.95. Patient grouped folds reported negative 1.0, which was the truth for new patients. The score was R squared, where 0 means no better than always guessing the average. For anything over time, train on the past and test on the future.

Leakage is any way information from the test data or the future sneaks into training. The classic case is choosing the "best" 25 of 10,000 features using all the data, then cross validating. On labels that were pure coin flips this produced 0.86 accuracy. Doing the selection inside each fold gave 0.48, the honest answer. There are other leaks.

- Scaling or imputing with statistics from the whole dataset. Imputing means filling in missing values.
- Duplicates across the split.
- Features recorded after the outcome.

## Class imbalance

Class imbalance means one class is rare. When positives are rare, fix the evaluation before the model. Use precision, recall or PR AUC. Move the decision threshold, which is the probability cutoff for flagging a case, to match the costs. Then consider reweighting, which makes each rare example count more during training. With 900 negatives and 100 positives, balanced weights are about 0.56 and 5.0. Resampling also works. It means duplicating rare examples or dropping common ones. But it must happen only inside the training data.

> **Worked example:** With features that contain no signal at all, duplicating the rare positives before splitting reported a perfect recall of 1.0. The reason is that copies of the same rows landed in both training and test. Duplicating inside the training fold only reported 0.0, the truth.

Focal loss is used in deep learning, which means training large neural networks. It down weights easy examples so training focuses on the hard ones.

## Features and missing data

Good features often matter more than the choice of model. A feature is one input column that the model sees. Several habits help.

- Scale numbers for distance based models.
- Turn categories into columns, one yes or no column per category. This is called one hot encoding. You can also turn categories into category averages, computed without peeking at the row itself.
- Take logs of skewed amounts like prices. Logs squeeze very large values closer to the rest.
- Add interactions when a linear model needs them.

For missing values, first ask why they are missing. If they are missing at random, filling with the median (the middle value) is fine. Sometimes the missingness itself tells you something. For example, people with high incomes skip the income question. Then keep that signal with an "is missing" flag. In one test the flag lifted AUC from 0.925 to 0.941. A boosting model that handles gaps natively reached 0.954. Feature selection and imputation both belong inside the cross validation loop.

## Interpretability

Interpretability asks which inputs drive the model's predictions, overall or for one case. Linear coefficients are the simplest answer. They are valid when features are on the same scale and not strongly correlated.

Tree models report an impurity based importance. It favours features with many distinct values. In a test, a random ID (identifier) column with 1,000 values was ranked most important, at 0.69 against 0.31 for the real signal. Permutation importance asks a better question. How much worse does the model get on new data if you shuffle this column? It gave the real signal 0.155 and the ID 0.003.

SHAP (SHapley Additive exPlanations) splits a single prediction into contributions from each feature. The contributions add up exactly to the prediction. SHAP uses a fair share rule from game theory. All of these methods explain the model, not the world.

## Hyperparameter search

Hyperparameters are the settings you choose before training, like the penalty strength or the tree depth. Grid search tries every combination. Random search samples combinations and usually finds good settings faster. The reason is that only a few settings matter, and random search tries more distinct values of each. Smarter methods learn from earlier trials or stop bad ones early. Two examples are Bayesian optimisation and successive halving.

Two rules apply. First, search penalty strengths and learning rates on a logarithmic scale, such as 0.001, 0.01, 0.1, and so on. A learning rate is how big a step each training update takes. Second, never tune on the test set. If you want an honest score for the whole tuning process, put the search inside an outer cross validation loop. This is called nested cross validation.

## Time series basics

A time series is data that comes in order, and yesterday tells you about today. Three habits change. First, check stationarity. That means whether the average and spread stay stable over time. A trend or a random walk breaks it. A random walk is a series that takes random steps from wherever it is. Differencing usually restores stationarity. Differencing means modelling the change from one step to the next. Second, look at autocorrelation, which is how strongly the series correlates with its own past. Take a series where each value is 0.7 times the last plus noise. Its correlation with one step back was 0.70, two steps 0.49, three steps 0.34.

Third, validate forward in time. Train on January to June and test on July. Then train through July and test on August. Random shuffling lets the model peek at the future. ARIMA (autoregressive integrated moving average) models combine differencing, dependence on past values, and dependence on past errors. Always compare against simple baselines like "same as last week".
