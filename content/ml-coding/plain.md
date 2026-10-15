# ML coding interviews: the plain version

Coding interviews for machine learning and data jobs test three things: whether you can write ordinary Python quickly and correctly, whether you can build the core pieces of machine learning yourself instead of calling a library, and whether you can explain what you are doing while you do it. This version explains every topic with intuition and small worked examples. The deep version has the full code, tests and derivations under the same headings.

## Coding round types

There are six kinds of coding round, and each one rewards different practice.

- **Data science screen.** A small table of orders or events and a business question: "average revenue per user by country". You answer with pandas.
- **Algorithm round.** A classic puzzle such as "find the k most common words", solved with plain Python data structures.
- **ML from scratch.** Build a model or metric yourself in NumPy: k-means, logistic regression, attention, ROC AUC.
- **Debugging round.** You get training code that runs without errors but learns badly, and you find out why.
- **Take home.** A dataset, a question and a few days. You hand in a short write up and a notebook.
- **AI assisted round.** You may use a coding assistant. Since 2025 Canva has required it for some roles and Meta has run a version that replaces one of its coding rounds.

Think of it like a driving test that can be held on a highway, in a parking lot or on a test track. You still need to know how to drive, but you would practise differently for each. So the first thing to do is ask the recruiter which kinds you will get, which libraries you may use, and whether your code will actually be run.

> **Remember:** a take home is read write up first. Reviewers look for a number and a recommendation in your first paragraph before they open your notebook.

## Thinking aloud while coding

Interviewers can only give you credit for reasoning they can hear. A candidate who types in silence for ten minutes gives them nothing to grade, and nothing to help with if things go wrong. The fix is a routine you use every time, so talking becomes automatic.

1. Say the problem back in your own words, and ask about the edge cases: what if the list is empty, what if two items tie?
2. Describe the slow obvious solution in one sentence, then a better one, and say how much work each does.
3. List a few test cases before writing code.
4. Write the code in small pieces, explaining why, not which keys you are pressing.
5. Run the tests and walk through one by hand.

> **Worked example:** "Find the 2 most common words in 'the cat the dog the cat'. I'll count every word with a dictionary, which is one pass. Then I'll pick the top 2. If two words tie I'll break the tie alphabetically. I'll test a tie, an empty list and asking for more words than exist." The answer is "the" (3 times), then "cat" (2 times).

When you get stuck, say what you know and what is missing, try a tiny example by hand, and if nothing better comes, write the slow version correctly. A working slow solution beats an unfinished clever one.

## Python data structures and complexity

Picking the right container is most of a Python screen. "Complexity" just means how the work grows as the data grows.

- A **list** is a row of numbered boxes. Grabbing box number 5 is instant. But inserting at the front means shifting every other box along by one, so it gets slower as the list grows. Asking "is 42 in this list?" means looking in every box.
- A **set** or **dict** is like a coat check. You hand over a key, the attendant computes which hook it belongs on, and goes straight there. Checking whether something is present takes about the same time whether there are ten coats or ten million.
- A **deque** is a line of people where you can join or leave at either end instantly. It is the right tool for a queue.
- A **heap** always keeps the smallest item at the front, so you can repeatedly ask "what is the smallest?" cheaply.

> **Worked example:** checking whether a number is in a list of 100,000 numbers took thousands of times longer than checking a set of the same numbers in one test run. The list looks at every element; the set jumps to one spot.

Two counting tools save time in every screen. `Counter` counts things for you ("the" appears 3 times) and treats missing words as 0. `defaultdict` fills in a starting value the first time you use a new key, which makes grouping items easy.

> **Interview trap:** using a list as a queue and removing from the front. Every removal shifts the whole list, so the program slows down badly as the queue grows. Use a deque.

## Python idioms and gotchas

Python has a few behaviours that surprise people, and interviewers like to plant them.

**Sorting with several rules.** To sort runs by highest accuracy, then lowest latency, sort on a pair of values: minus the accuracy (so bigger comes first) and the latency. Python's sort is stable, meaning items that tie keep their original order, which lets you sort in two passes when one rule cannot be flipped with a minus sign.

**Lazy versus eager.** A list builds everything at once. A generator hands you one item at a time, like a tap instead of a bucket. A million squared numbers take megabytes as a list and a couple hundred bytes as a generator. The catch: once the tap has run, it is empty, so a second pass gives nothing.

**The shared default list.** If a function has a default argument that is a list, Python creates that list once, when the function is defined, and every call reuses it. Call it twice and the second call still sees what the first one added. Use `None` as the default and create a new list inside.

**The lambda in a loop.** Functions made in a loop remember the loop variable, not its value at that moment, so they all end up using the last value.

**The copied grid.** `[[0] * 3] * 3` makes three references to the same row, so changing one cell changes a whole column. It is like photocopying a page and finding that writing on one copy writes on all of them.

> **Remember:** in Python, assignment never copies. It just puts another name tag on the same object.

## Algorithm patterns for ML screens

Most algorithm questions are one of about eight patterns wearing a costume. Spotting the pattern is most of the work.

| If the question says | Reach for | Example |
| --- | --- | --- |
| count, seen before, pairs that add up to | a dictionary | how many day ranges have exactly 3 signups |
| two sorted lists | two pointers | documents that appear in both search results |
| longest stretch with at most k of something | a sliding window | longest run of requests with at most 1 failure |
| top k, a stream | a heap | the 5 documents most similar to a query |
| a yes or no answer that flips once | binary search | the smallest stage size that fits a model in 3 stages |
| start and end times | sort, then sweep | busy hours of a GPU cluster |
| connections or dependencies | BFS or DFS | friend suggestions, the order to run a calculation |
| minimum edits, number of ways | dynamic programming | word error rate of a speech system |

> **Worked example:** a sliding window works like a caterpillar. The front moves forward one step at a time; whenever the window holds more than one failure, the back catches up. For the requests ok, fail, ok, ok, fail, fail, ok, ok, ok, fail with at most one failure allowed, the longest stretch is 4.

Dynamic programming sounds fancy but means writing down the answers to small versions of the problem in a table and building bigger answers from them. Word error rate counts the fewest word insertions, deletions and swaps to turn what the system heard into what was said: "the cat sit on mat" against "the cat sat on the mat" needs 2 edits out of 6 words, a rate of one third.

## NumPy vectorisation and broadcasting

NumPy lets you do arithmetic on whole arrays at once instead of looping over each number in Python. Python loops are slow because Python checks types and looks things up on every single step; NumPy runs one fast loop in compiled code. In one test, computing an average squared error over a million numbers was about 90 times faster with NumPy than with a Python loop, with the same answer.

**Broadcasting** is how NumPy combines arrays of different shapes. Picture a spreadsheet with 4 rows and 3 columns, and a single row of 3 column averages. NumPy stretches that row down to cover all 4 rows, so you can subtract the averages from every row in one line. The rule: line the shapes up from the right; each pair of sizes must match or one of them must be 1.

> **Interview trap:** predictions shaped as a column of 3 minus labels shaped as a row of 3 gives a 3 by 3 table, not 3 differences. NumPy does not complain, so the model quietly trains on nonsense. Check shapes.

Some other ideas worth knowing:

- Taking a slice of an array gives a window onto the same data, not a copy. Change the slice and you change the original.
- Distances between every pair of points can be computed with one matrix multiplication instead of a loop inside a loop.
- Adding a million small numbers into one running total in low precision loses the small digits. In one test the naive total gave an average of about 1010 when the true average was 1000.5. NumPy's sum adds in pairs, which keeps the error tiny.

## Pandas in interview form

Pandas screens use a handful of moves. The trick is knowing what shape comes back.

- `groupby(...).agg(...)` squashes each group to one row, like a pivot table total.
- `groupby(...).transform(...)` computes a group number and writes it back on every original row, for things like "each order's share of its customer's total".
- `merge` joins two tables. It can add rows when a key appears more than once.
- `shift` and `diff` within a group look at the previous row for the same user.
- `rolling` makes moving averages.

> **Worked example:** sessions. Sort each user's events by time, compute the gap since their previous event, mark a new session whenever the gap is over 30 minutes, and keep a running count of those marks. A user active at 9:00, 9:10 and then 11:00 has two sessions: the 11:00 event comes after a gap of 1 hour 50 minutes.

Before joining two tables, say what one row means in each ("one row per user", "one row per order"), and count rows before and after. If 4 users become 5 rows, someone had two orders, which matters if the question asked about users.

pandas 3.0 (January 2026) changed one habit: writing `df[mask]["col"] = 0` no longer changes `df` at all. Write `df.loc[mask, "col"] = 0` in one step.

## Linear and logistic regression

**Linear regression** draws the straight line (or flat surface) that sits as close as possible to the data points, where "close" means the smallest total of squared misses. There is a direct formula for it, called the normal equations, which you solve like a small system of simultaneous equations. You can also find it by gradient descent: start with a guess, look at which direction makes the misses smaller, take a small step that way, and repeat.

> **Worked example:** with data generated as 4 + 2 times the first feature minus the second plus half the third, plus a little noise, the formula recovered weights of about 1.99, -1.01 and 0.51 and an intercept of 4.00, matching scikit-learn exactly.

Gradient descent needs the features on similar scales. If one feature is measured in tens and another in tenths, the same step size is too big for one and too small for the other; in the test the unscaled version blew up to an absurd error while the scaled one converged.

**Logistic regression** answers yes or no questions with a probability. It computes a score from the features and squeezes it between 0 and 1 with the S shaped sigmoid curve. It learns by the same step by step process, and the direction to step works out to something simple: the predicted probability minus the true answer, times the features. If the model says 0.8 and the answer is 1, it nudges the score up a little.

> **Interview trap:** writing the sigmoid in the obvious way. For a score of minus 1000 the computer tries to compute an enormous number and fails. The stable version uses a different formula for negative scores.

## k-means and kNN

**k-means** groups points into k clusters. Picture k flags dropped on a map. Every house joins its nearest flag, then each flag moves to the middle of its houses. Repeat until the flags stop moving. It always settles, but where it settles depends on where the flags started, so k-means++ places the first flags far apart: each new flag is more likely to land on a point far from the existing ones.

> **Worked example:** on 600 points in 4 clear blobs, the code's clusters matched scikit-learn's exactly, with the same total squared distance of 1127.402.

Watch for a flag that ends up with no houses (an empty cluster) and move it somewhere useful, such as the point that is worst served.

**kNN**, k nearest neighbours, classifies a new point by asking its k closest neighbours in the training data to vote. It is like guessing whether a house will sell above its asking price by checking what happened to the five most similar houses nearby. Training is just storing the data; all the work happens when predicting. It struggles when there are hundreds of features, because in very high dimensions everything is roughly the same distance from everything else.

## Decision tree splits

A decision tree asks a series of yes or no questions, like "is income above 50,000?", to sort examples into groups that are as pure as possible. **Gini impurity** measures how mixed a group is: the chance that two examples picked at random from it have different labels. A group that is all one class scores 0; a 50/50 mix scores 0.5.

> **Worked example:** a node with 6 yes and 4 no has Gini 0.48. Splitting it into (5 yes, 1 no) and (1 yes, 3 no) gives children with Gini 0.278 and 0.375; weighted by size (6 and 4 examples) that averages 0.317, an improvement of 0.163.

To find the best question for one feature efficiently, sort the examples by that feature and slide a dividing line from left to right, updating the counts on each side as one example moves across. That is much faster than recounting everything for every possible line. The test matched scikit-learn's choice of feature and threshold every time.

## Softmax, cross entropy and backprop

A classifier first produces raw scores, one per class. **Softmax** turns them into probabilities that add up to 1: bigger scores get bigger shares. Computers overflow when exponentiating large numbers, so the safe version first subtracts the biggest score from all of them, which does not change the answer.

**Cross entropy** is the training penalty: minus the log of the probability the model gave the right answer. Giving the right answer 0.9 costs about 0.1; giving it 0.01 costs about 4.6. A model that guesses evenly among 10 classes scores log 10, about 2.3, which is what a brand new model should show.

**Backpropagation** works out how much each weight contributed to the error, by going backwards through the calculation one step at a time. It is like tracing a wrong total on a receipt back through each line item. A two layer network is a few lines forward and a few lines backward.

> **Remember:** always check hand written gradients with a **gradient check**: nudge each weight up and down by a tiny amount, see how the error changes, and compare with your formula. In the test, the worst disagreement was 6 parts in 100 million, which means the gradients are right.

## Attention from scratch

Attention lets each word in a sentence gather information from the other words. Each word makes a **query** ("what am I looking for?"), each word offers a **key** ("what do I contain?") and a **value** ("what I will pass on"). A word compares its query with every key, turns the match scores into weights with softmax, and takes a weighted average of the values.

The scores are divided by the square root of the vector size. Without that, bigger vectors give bigger scores, softmax puts almost all the weight on one word, and learning stalls. In a test with 16 candidate words and vectors of size 1024, the top word got 96 percent of the weight without scaling and about 25 percent with it.

A **causal mask** stops a word from looking at words that come after it, so a model that predicts the next word cannot peek at the answer. **Multi head** attention runs several smaller attentions side by side, each free to notice different relationships.

When generating text one word at a time, earlier words' keys and values never change, so a **KV cache** stores them instead of recomputing them at every step. The test shows the cached version gives exactly the same output, at the cost of memory: about 1 GiB for one long conversation of 8,192 tokens in a typical medium size model setup.

## Tokenization and decoding

Language models read **tokens**, chunks of text that are often pieces of words. **Byte pair encoding** learns those chunks by repeatedly gluing together the most common neighbouring pair. Starting from letters in the words hug, pug, pun, bun and hugs (with how often each appears), the pair "u g" appears 20 times, so it merges first into "ug"; then "u n" (16 times); then "h ug" (15 times). A new word like "thug" can still be written as "t" plus "hug".

**Decoding** is how the model picks the next token:

- **Greedy** always takes the most likely token. It can walk into a dead end: "The nice woman" scores 0.2 overall, while "The dog has" scores 0.36 but starts with a less likely word.
- **Beam search** keeps the few best partial sentences at each step and finds "The dog has".
- **Temperature** sharpens (below 1) or flattens (above 1) the probabilities before sampling.
- **Top k** samples only from the k most likely tokens; **top p** samples from the smallest group of top tokens whose probabilities add up to p, so the group shrinks when the model is confident.

## Metrics from scratch

Metrics turn predictions into a score you can compare.

- **Precision:** of the items you flagged, how many were right.
- **Recall:** of the items that were really positive, how many you caught.
- **F1:** a balance of the two that is low if either is low.

> **Worked example:** 10 emails, 4 of them spam. At a threshold of 0.5 the filter flags 4 emails, 3 of them spam: precision 3/4, recall 3/4. Lower the threshold to 0.25 and it flags 6, catching all 4 spam: recall rises to 1.0, precision drops to 4/6.

**ROC AUC** is the chance that a random positive gets a higher score than a random negative. In that example, 21 of the 24 spam and not spam pairs are ordered correctly, so AUC is 0.875. It only cares about the order of scores, not their values. When positives are very rare, precision and recall tell you more.

For search and retrieval: **recall at k** asks how many of the relevant documents made the top k; **MRR** looks at the position of the first relevant result; **nDCG** rewards putting the most relevant results at the very top.

## Splitting and resampling

You judge a model on data it never saw during training, so you split the data first. **k fold cross validation** splits the data into k parts and lets each part be the test set once, which gives k scores instead of one lucky or unlucky one. **Stratifying** keeps the mix of classes the same in every part, which matters when one class is rare.

**Leakage** is when information from the test data sneaks into training, making results look better than they are. The classic way is preparing the data (scaling, filling gaps, choosing features) using all rows before splitting.

> **Worked example:** with 10,000 completely random features and random labels, the honest accuracy is a coin flip. Choosing the 25 "best" features using all rows gave about 83 percent test accuracy on average. Choosing them using only the training rows gave about 50 percent, the truth.

If one user, patient or document has many rows, keep all of them on the same side of the split. If you predict the future, train on the past and test on later data.

The **bootstrap** measures how uncertain a number is by resampling your own data many times with replacement and seeing how much the number moves. For comparing two models, resample the same rows for both.

## Streaming statistics

Sometimes data arrives as a stream too large to store.

- **Reservoir sampling** keeps a fair random sample of k items without knowing how many will come. Keep the first k; for each later item number i, keep it with probability k divided by i, replacing a random current member. Every item ends up equally likely to be in the sample.
- **Welford's method** keeps a running average and variance one value at a time. The textbook shortcut (sum of squares minus square of the sum) breaks down on large numbers: on four values around a billion with a true variance of 30, it gave 0 in one run, while Welford gave 30.
- A **moving average** of the last w values keeps a running total: add the new value, subtract the one that falls out of the window. Each update costs the same no matter how long the window is.

## Debugging and AI assisted rounds

In a debugging round the code runs fine but the model learns badly, and you find out why. One reported version gives you a small transformer with four planted bugs and then asks you to add a KV cache.

A good method:

1. Check the starting loss. A model guessing evenly should score about log of the number of classes.
2. Try to make the model memorise a tiny batch of 8 examples. A correct setup can; if it cannot, the bug is in the model or the training step.
3. Print the shapes of everything.
4. Look at a few inputs next to their labels.
5. Swap in pieces you know are correct until the problem moves.

> **Worked example:** in a planted bug demo, a flipped sign made accuracy drop to about 5 percent, shuffling the inputs without the labels left the model at chance, and a column shaped weight vector silently grew into a 3 by 400 table. None of them raised an error.

In an **AI assisted round**, treat the assistant like a fast junior colleague. Read the code and say your plan first, hand it small clear jobs, read and run everything it writes, and explain what you kept and what you changed. Pasting in the whole problem is the most common way to fail, because the interviewer is grading your judgment.
