# ML coding interviews: the exhaustive version

This guide covers every kind of coding round an ML engineer, AI engineer, research engineer, data scientist or analyst meets in 2025 and 2026. It explains what each round looks like and how to narrate while you code. It then covers the Python, NumPy and pandas that interviewers actually test, about thirty from scratch implementations, and the algorithm patterns behind most screens. Every implementation comes with a test that proves it correct (against scikit-learn, SciPy, NumPy or a brute force), its complexity, and the follow up questions interviewers chain onto it. Every code block was run in Python 3.12 with NumPy 2.5, pandas 3.0, scikit-learn 1.9 and SciPy 1.18, and the printed output shown is from that run.

## Coding round types

An ML coding loop is not one kind of round. You will meet some mix of six formats, and each is graded on something different. Ask the recruiter which ones you will get, which language and libraries are allowed, and whether the code is run or only read.

| Round | Typical roles | Length | What you do | What is graded |
| --- | --- | --- | --- | --- |
| DS Python and pandas screen | Data scientist, analyst, some MLE | 30 to 60 min | Clean, join, group and summarise a small table; sometimes a plain Python data task | Correct grain and denominators, clean pandas, checking your own output |
| LeetCode style DSA | MLE, AI engineer, SWE on ML teams | 45 to 60 min, one or two problems | Arrays, hash maps, heaps, graphs, simple DP | Correct optimal algorithm, complexity, edge cases, clean code |
| ML from scratch | MLE, research engineer, applied scientist | 45 to 60 min | Implement k-means, logistic regression, attention, a metric, in NumPy or PyTorch | Correct math, vectorisation, numerical stability, a test |
| ML debugging | Research engineer and MLE at AI labs | About 60 min | Fix a training script or a small transformer that runs but trains badly | Systematic diagnosis, transformer internals, verifying the fix |
| Take home | DS, analyst, some MLE at startups | A few hours to a few days | Analysis or model on a provided dataset, plus a write up | Answering the question asked, correctness, a readable write up |
| AI assisted coding | SWE and MLE at Meta, Canva and others | About 60 min | A multi file task with an AI assistant in the editor | Judgment about what to delegate, reviewing AI output, testing |

### DS Python and pandas screens

The data science screen is closest to day to day work: a CSV or a small DataFrame, a business question, and twenty to forty minutes. Typical asks are a groupby with several aggregates, a merge followed by a count, a "previous event per user" calculation, sessionisation, a rolling average, a pivot, or a funnel. Interviewers rarely care whether you remember the exact keyword for `pd.merge_asof`; they care whether you check the grain of each table before joining, whether you notice that a left join multiplied rows, and whether you can say what the denominator of a rate is. Some screens are plain Python instead (parse a log file, count words, deduplicate records), which is why the Python and pattern sections below matter even for analysts.

### LeetCode style rounds for MLE

MLE and AI engineer loops at larger companies still include one or two general algorithm rounds. The alirezadir Machine Learning Interviews repo lists fourteen pattern families (hashing, two pointers and sliding window, stacks, binary search, linked lists, intervals, trees, heaps and tries, backtracking, graphs, advanced graphs, one and two dimensional DP, bit tricks) and suggests about 350 reviewed problems, mostly medium, as a ceiling rather than a floor. For an ML candidate the highest yield subset is hash maps, two pointers, sliding windows, heaps for top k, binary search, intervals, BFS and DFS, and simple DP, which the patterns section covers with ML flavoured examples.

### ML from scratch rounds

The round that is specific to ML: implement a model, a layer or a metric with NumPy (sometimes PyTorch), usually without scikit-learn. The alirezadir ML coding chapter lists 41 practice problems; the classic set includes numerically stable softmax and cross entropy, linear and logistic regression by gradient descent, kNN, k-means, a decision tree split, ROC AUC, reservoir sampling, a feedforward network with backprop, leakage free splits and stratified sampling. The language model set adds scaled dot product attention, causal masks, byte pair encoding, temperature with top k and top p sampling, and a KV cache. The same chapter suggests time budgets of about 15 minutes for an easy problem, 20 to 35 for a medium and 40 to 60 or more for a hard one, which matches the pace of a real round: you are expected to finish a k-means or a logistic regression with time left for questions.

> **Watch:** [Karpathy, Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) is the model for this style: every component typed from an empty file, checked as it is built.

### Take homes

Chip Huyen's ML interviews book describes the coding challenge as one to three questions with 30 minutes to 3 hours, usually for new graduates and interns. Data take homes are longer: a dataset, a question, and a deadline of a few days. A grading guide from PracHub reports that reviewers read the write up before the notebook, look for a number and a decision in the first paragraph, treat analysis errors as a gate rather than a scale, and use craft (a notebook that runs top to bottom, stated assumptions) as a tiebreaker. Keep the scope tight, lead with the answer, and make the notebook run from a fresh kernel.

### Rounds that changed in 2025 and 2026

Two formats grew quickly. ML debugging rounds hand you broken training code; an aced.io guide to the OpenAI research engineer loop describes a transformer of about 300 lines with four planted bugs, followed by implementing a KV cache. AI assisted rounds let or require you to use a coding assistant: Canva announced in June 2025 that backend, ML and frontend candidates are expected to use tools such as Copilot, Cursor or Claude, and Meta started an AI enabled coding round in October 2025 that replaces one of two onsite coding rounds. Both are covered in the last section.

## Thinking aloud while coding

Narration is graded. An interviewer cannot give credit for reasoning they did not hear, and a silent candidate who gets stuck gives them nothing to hint at. The habit that works is a fixed script: restate, plan with complexity, list test cases, then code while narrating, then run the tests.

### The script

1. **Restate and clarify.** Say the input and output types and shapes, the sizes, and what happens at the edges. "So I get a list of tokens, possibly empty, and an integer k; I return the k most frequent tokens as a list. If two tokens tie, I will break ties alphabetically unless you want something else. Is k ever larger than the vocabulary?"
2. **Plan with complexity before typing.** Give the brute force first in one sentence, then the better approach and its time and space cost. The Tech Interview Handbook calls the lower bound the best theoretical time complexity; saying it ("I have to read every token, so O(n) is the floor") tells the interviewer you know when to stop optimising.
3. **List test cases out loud.** A normal case, a tie, an empty input, a boundary such as k equal to zero, and one adversarial case. Writing them first means you test against what you promised, not against what you happened to build.
4. **Code in small, named steps, narrating intent.** Narrate why, not keystrokes: "I count with a Counter so this is linear, then I take k with a heap so I never sort the whole vocabulary."
5. **Run the tests and walk one through by hand.** Then state the final complexity and one thing you would change at scale.

```python
from collections import Counter
import heapq

def top_k_tokens(tokens, k):
    """Return the k most frequent tokens, ties broken alphabetically.

    Plan said out loud before typing:
      1. count with a hash map, O(n)
      2. pick k with a heap, O(m log k) for m distinct tokens
      3. return in descending count order
    Edge cases: k <= 0, k larger than the vocabulary, empty input, ties.
    """
    if k <= 0:
        return []
    counts = Counter(tokens)
    best = heapq.nsmallest(k, counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [tok for tok, _ in best]

# tests: normal case, tie, k too big, empty input, k = 0
assert top_k_tokens("the cat the dog the cat".split(), 2) == ["the", "cat"]
assert top_k_tokens(["b", "a", "b", "a", "c"], 2) == ["a", "b"]
assert top_k_tokens(["x"], 5) == ["x"]
assert top_k_tokens([], 3) == []
assert top_k_tokens(["x", "y"], 0) == []
print("all tests pass")
```

### When you get stuck

Say what you know and what you are missing: "I know the answer is monotone in the capacity, so binary search should work, but I am not sure how to check feasibility." Shrink the problem (two elements, one class), solve it by hand and turn the steps into code, or write the brute force and test against it. A correct brute force with a clear plan to improve it scores better than an unfinished clever solution.

> **Interview trap:** Typing for five minutes in silence and then announcing a solution. Even when the code is right, the interviewer has no evidence about how you reason, and that is half of what they score.

## Python data structures and complexity

Most Python screens reduce to choosing the right container. The numbers below are the CPython average costs from the Python wiki TimeComplexity page, where n is the container size and k the size of the other argument.

| Operation | list | dict or set | collections.deque | heapq on a list |
| --- | --- | --- | --- | --- |
| Index or lookup by key | O(1) | O(1) average, O(n) worst | O(1) at the ends, O(n) in the middle | O(1) for the minimum only |
| `x in c` | O(n) | O(1) average | O(n) | O(n) |
| Append or add | O(1) amortised | O(1) average | O(1) both ends | push O(log n) |
| Pop from the end you append to | O(1) | not applicable | O(1) both ends | pop minimum O(log n) |
| Insert or delete in the middle | O(n) | O(1) average delete | O(n) | not supported |
| Build from n items | O(n) | O(n) | O(n) | heapify O(n) |
| Sort | O(n log n), stable | not ordered by key | not supported | heapsort O(n log n), not stable |

The dict and set worst case is O(n) because every key can land in the same hash bucket; the average case assumes a good hash function. Two traps follow from the table: `list.pop(0)` and `list.insert(0, x)` are O(n), so a queue built on a list is quadratic overall (use `deque.popleft`), and `x in some_list` inside a loop is quadratic (convert to a set first).

```python
import timeit

n = 100_000
as_list = list(range(n))
as_set = set(as_list)
t_list = timeit.timeit(lambda: (n - 1) in as_list, number=200)   # O(n) scan
t_set = timeit.timeit(lambda: (n - 1) in as_set, number=200)     # O(1) average hash lookup
print(f"list membership took about {t_list / t_set:,.0f} times longer than set membership")
```
```text
list membership took about 8,599 times longer than set membership
```

The exact ratio depends on the machine; the point is that it grows with n.

### Counter and defaultdict

`Counter` is a dict subclass for counting. Missing keys read as 0, `most_common(n)` returns pairs in descending count (ties keep first seen order), arithmetic keeps only positive counts, and `total()` arrived in Python 3.10. `defaultdict(factory)` calls the factory on a missing key, but only for `d[key]` access: `d.get(key)` and `key in d` never insert. That difference causes a common bug where a defaultdict silently fills with empty lists because the code reads keys with brackets.

```python
from collections import Counter, defaultdict

words = "to be or not to be that is the question".split()
c = Counter(words)
print(c.most_common(2))      # ties keep first seen order
print(c["missing"])          # 0, no KeyError
print(c.total())             # sum of counts, Python 3.10+

a, b = Counter("aabbc"), Counter("abd")
print(a - b)                 # subtraction keeps only positive counts

by_len = defaultdict(list)   # no "if key not in d" dance
for w in words:
    by_len[len(w)].append(w)
print(dict(by_len))
print(by_len.get(99), 99 in by_len)   # .get never inserts
by_len[99]                            # [] access inserts an empty list
print(99 in by_len)
```
```text
[('to', 2), ('be', 2)]
0
10
Counter({'a': 1, 'b': 1, 'c': 1})
{2: ['to', 'be', 'or', 'to', 'be', 'is'], 3: ['not', 'the'], 4: ['that'], 8: ['question']}
None False
True
```

### heapq

`heapq` turns a plain list into a binary min heap: `h[0]` is always the smallest item, `heapify` is linear, push and pop are O(log n). Python 3.14 added max heap functions such as `heappush_max`; on 3.12 you negate the key. Tuples compare element by element, so `(priority, payload)` breaks when two priorities tie and the payloads cannot be compared; the documented fix is a running counter as the second element. The docs also say when to use what: `nlargest` and `nsmallest` for small k, `sorted` for large k, `min` or `max` for k equal to one.

```python
import heapq
import itertools

h = [5, 1, 9, 3, 7]
heapq.heapify(h)             # O(n), in place, min heap
print(h[0])                  # the smallest item is always at index 0
heapq.heappush(h, 0)
print([heapq.heappop(h) for _ in range(3)])

# max heap on Python 3.12: push negated keys (3.14 adds heappush_max and friends)
scores = {"doc_a": 0.91, "doc_b": 0.42, "doc_c": 0.77}
mx = [(-s, d) for d, s in scores.items()]
heapq.heapify(mx)
print(heapq.heappop(mx)[1])

# a running counter breaks ties so the payloads are never compared
counter = itertools.count()
pq = []
for prio, task in [(2, {"id": 1}), (2, {"id": 2}), (1, {"id": 3})]:
    heapq.heappush(pq, (prio, next(counter), task))   # dicts cannot be compared with <
print([heapq.heappop(pq)[2]["id"] for _ in range(3)])
```
```text
1
[0, 1, 3]
doc_a
[3, 1, 2]
```

### bisect

`bisect_left` and `bisect_right` return insertion points in a sorted list in O(log n). They answer "how many values are below x", "which bucket does x fall in" and "count in a range" without scanning. The `key` argument exists since Python 3.10. `insort` finds the position in O(log n) but the list insert is O(n), so a sorted list maintained with `insort` costs O(n) per insert.

```python
import bisect

# bucket model scores into risk bands with sorted breakpoints
breaks = [0.2, 0.5, 0.8]
bands = ["low", "medium", "high", "critical"]
print([bands[bisect.bisect_right(breaks, s)] for s in [0.1, 0.2, 0.65, 0.95]])

# count values in [lo, hi) of a sorted list with two binary searches, O(log n)
lat = sorted([12, 15, 15, 20, 31, 40, 41, 55])
print(bisect.bisect_left(lat, 41) - bisect.bisect_left(lat, 15))

# empirical percentile of a new value
print(bisect.bisect_right(lat, 33) / len(lat))

# insort keeps the list sorted, but the insert itself is O(n)
bisect.insort(lat, 33)
print(lat)
```
```text
['low', 'medium', 'high', 'critical']
5
0.625
[12, 15, 15, 20, 31, 33, 40, 41, 55]
```

> **Remember:** `bisect_right(breaks, x)` puts a value equal to a breakpoint into the upper bucket; `bisect_left` puts it into the lower one. Say which you want before you code it.

## Python idioms and gotchas

### Sorting with keys

`sorted` and `list.sort` take a `key` function called once per element, and Python's sort (Timsort) is guaranteed stable, so equal keys keep their input order. Stability gives two techniques interviewers like. A tuple key sorts by several fields at once, negating numeric fields for descending order. When a field cannot be negated (a string), sort in passes: secondary key first, then the primary key.

```python
from operator import itemgetter

runs = [("bert", 0.81, 120), ("gpt", 0.86, 300), ("t5", 0.86, 200), ("lstm", 0.74, 50)]

# best accuracy first, ties broken by lower latency: negate the numeric key
print(sorted(runs, key=lambda r: (-r[1], r[2])))

# stability lets you sort in passes: secondary key first, then primary key
by_latency = sorted(runs, key=itemgetter(2))
print(sorted(by_latency, key=itemgetter(1), reverse=True))

# a string cannot be negated, so name descending within accuracy ascending needs two passes
by_name_desc = sorted(runs, key=itemgetter(0), reverse=True)
print([r[0] for r in sorted(by_name_desc, key=itemgetter(1))])

# argsort in plain Python
vals = [0.3, 0.9, 0.1]
print(sorted(range(len(vals)), key=vals.__getitem__))
```
```text
[('t5', 0.86, 200), ('gpt', 0.86, 300), ('bert', 0.81, 120), ('lstm', 0.74, 50)]
[('t5', 0.86, 200), ('gpt', 0.86, 300), ('bert', 0.81, 120), ('lstm', 0.74, 50)]
['lstm', 'bert', 't5', 'gpt']
[2, 0, 1]
```

### Generators and comprehensions

A generator produces values lazily, so a pipeline over a 50 GB log file uses constant memory. A generator expression is a comprehension in parentheses. Two properties matter in interviews: memory (a list of a million ints is megabytes, the generator is a couple hundred bytes) and single use (a second pass over a spent generator yields nothing, which silently returns 0 from `sum`). In a nested comprehension the left `for` is the outer loop, matching the order you would write the loops.

```python
import itertools
import sys

def batches(iterable, size):
    """Yield lists of up to size items without loading the whole input in memory."""
    it = iter(iterable)
    while chunk := list(itertools.islice(it, size)):
        yield chunk

print(list(batches(range(7), 3)))

squares_list = [i * i for i in range(10**6)]       # materialised: megabytes
squares_gen = (i * i for i in range(10**6))        # lazy: a couple hundred bytes
print(sys.getsizeof(squares_list) > 1_000_000, sys.getsizeof(squares_gen) < 1_000)
print(sum(squares_gen) == sum(squares_list))
print(sum(squares_gen))                            # 0: a generator is spent after one pass

vocab = ["the", "cat", "sat"]
tok2id = {t: i for i, t in enumerate(vocab)}       # dict comprehension
lengths = {len(t) for t in vocab}                  # set comprehension
pairs = [(a, b) for a in vocab for b in vocab if a < b]   # the left loop is the outer loop
print(tok2id, lengths, pairs)
```
```text
[[0, 1, 2], [3, 4, 5], [6]]
True True
True
0
{'the': 0, 'cat': 1, 'sat': 2} {3} [('cat', 'the'), ('cat', 'sat'), ('sat', 'the')]
```

### The gotchas interviewers plant

1. **Mutable default arguments.** Defaults are evaluated once, at definition time, so a default list is shared by every call. Use `None` and create the list inside.
2. **Late binding closures.** A lambda made in a loop reads the loop variable when it is called, so every function sees the last value. Bind it with a default argument (`i=i`) or `functools.partial`.
3. **List multiplication aliasing.** `[[0] * 3] * 3` is three references to one inner list.
4. **Float equality.** `0.1 + 0.2 == 0.3` is false; use `math.isclose` or `np.isclose`.
5. **Floor division.** `-7 // 2` is -4 because `//` rounds toward negative infinity; `int(-7 / 2)` is -3.

```python
import math

# 1. a mutable default argument is created once, when the function is defined
def add_bad(x, seen=[]):
    seen.append(x)
    return seen
add_bad(1)
print(add_bad(2))

def add_good(x, seen=None):
    seen = [] if seen is None else seen
    seen.append(x)
    return seen
add_good(1)
print(add_good(2))

# 2. late binding: the closure reads i when it runs, not when it was made
fns = [lambda x: i * x for i in range(3)]
print([f(10) for f in fns])
fns = [lambda x, i=i: i * x for i in range(3)]
print([f(10) for f in fns])

# 3. [[0] * 3] * 3 repeats one inner list three times
grid = [[0] * 3] * 3
grid[0][0] = 1
print(grid)
grid = [[0] * 3 for _ in range(3)]
grid[0][0] = 1
print(grid)

# 4. compare floats with a tolerance
print(0.1 + 0.2 == 0.3, math.isclose(0.1 + 0.2, 0.3))

# 5. floor division rounds toward negative infinity, int() truncates toward zero
print(7 // 2, -7 // 2, int(-7 / 2))
```
```text
[1, 2]
[2]
[20, 20, 20]
[0, 10, 20]
[[1, 0, 0], [1, 0, 0], [1, 0, 0]]
[[1, 0, 0], [0, 0, 0], [0, 0, 0]]
False True
3 -4 -3
```

> **Interview trap:** Saying "Python passes by reference". Python passes object references by value: rebinding a parameter inside a function does not affect the caller, but mutating the object it points to does. That is the whole mechanism behind gotchas 1 and 3.

## Algorithm patterns for ML screens

Most DS and MLE algorithm questions are one of eight patterns in disguise. Recognising the signal in the problem statement is most of the work.

| Pattern | Signal in the prompt | ML flavoured example | Typical cost |
| --- | --- | --- | --- |
| Hash map, prefix sums | "count", "seen before", "pairs that sum to", "subarray sum equals" | Day ranges whose signups total k | O(n) time, O(n) space |
| Two pointers | Two sorted inputs, or pairs in one sorted array | Intersect posting lists, merge ranked shards | O(n + m) |
| Sliding window | "contiguous", "longest or shortest stretch", "at most k" | Longest run of requests with at most k failures | O(n) |
| Heap for top k | "k largest", "k closest", a stream | k most similar documents to a query | O(n log k) |
| Binary search | Sorted data, or a monotone yes or no question over an answer | Fewest pipeline stages for a cost cap | O(n log range) |
| Intervals | Start and end times | GPU busy time, peak concurrent jobs | O(n log n) |
| BFS and DFS | Graphs, grids, dependencies | Friend suggestions, topological order of a compute graph | O(V + E) |
| Dynamic programming | "minimum edits", "number of ways", overlapping subproblems | Word error rate via edit distance | O(m n) |

### Hash maps and prefix sums

A hash map turns "have I seen this before" into an O(1) check. The classic extension is prefix sums plus a hash map: the sum of a contiguous range from i to j equals prefix[j] minus prefix[i], so "how many ranges sum to k" becomes "for each prefix, how many earlier prefixes equal prefix minus k". Chip Huyen's book lists exactly this question (count continuous subarrays summing to k). It handles negative numbers, which the sliding window cannot.

```python
import random
from collections import defaultdict

def count_windows_with_total(daily, k):
    """Number of contiguous day ranges whose values sum to exactly k. O(n) time, O(n) space."""
    seen = defaultdict(int)
    seen[0] = 1                    # the empty prefix
    prefix = count = 0
    for x in daily:
        prefix += x
        count += seen[prefix - k]  # every earlier prefix p with prefix - p == k ends a valid range here
        seen[prefix] += 1
    return count

def brute(daily, k):
    n = len(daily)
    return sum(sum(daily[i:j]) == k for i in range(n) for j in range(i + 1, n + 1))

random.seed(0)
for _ in range(300):
    arr = [random.randint(-3, 3) for _ in range(random.randint(0, 12))]
    k = random.randint(-4, 4)
    assert count_windows_with_total(arr, k) == brute(arr, k)
print(count_windows_with_total([1, 2, 3, -2, 2], 3))
print("matches brute force on 300 random cases")
```
```text
4
matches brute force on 300 random cases
```

The four ranges in the example are [1, 2], [3], [2, 3, -2] and [3, -2, 2].

### Two pointers

With two sorted inputs, walk one pointer through each and always advance the one pointing at the smaller value. Intersecting posting lists in a search index and merging per shard result lists in a retrieval system are both this loop. It is O(n + m) time and O(1) extra space, against O(n m) for the nested loop.

```python
import random

def intersect_sorted(a, b):
    """Doc ids present in both sorted posting lists. O(len(a) + len(b)) time, O(1) extra space."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out

def merge_ranked(a, b, k):
    """Merge two lists of (score, doc) already sorted by descending score; return the top k."""
    i = j = 0
    out = []
    while len(out) < k and (i < len(a) or j < len(b)):
        if j == len(b) or (i < len(a) and a[i][0] >= b[j][0]):
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out

print(intersect_sorted([1, 4, 7, 9, 12], [2, 4, 9, 10, 12, 15]))
print(merge_ranked([(0.9, "d1"), (0.5, "d4")], [(0.8, "d2"), (0.7, "d3")], 3))

random.seed(1)
for _ in range(200):
    a = sorted(random.sample(range(50), random.randint(0, 20)))
    b = sorted(random.sample(range(50), random.randint(0, 20)))
    assert intersect_sorted(a, b) == sorted(set(a) & set(b))
print("intersection matches set logic")
```
```text
[4, 9, 12]
[(0.9, 'd1'), (0.8, 'd2'), (0.7, 'd3')]
intersection matches set logic
```

### Sliding window

For contiguous ranges with a constraint that only gets worse as the window grows (at most k failures, at most k distinct tokens, sum at most a budget with non negative values), grow the right edge and shrink the left edge while the window is invalid. Each index enters and leaves once, so it is O(n) even though there is a nested while loop. A fixed size window just adds the new value and subtracts the oldest.

```python
import random

def longest_run_with_at_most_k_failures(status, k):
    """Variable window: longest stretch of requests with at most k failures (1 = failed). O(n)."""
    best = left = fails = 0
    for right, s in enumerate(status):
        fails += s
        while fails > k:              # shrink from the left until the window is valid again
            fails -= status[left]
            left += 1
        best = max(best, right - left + 1)
    return best

def max_window_sum(xs, w):
    """Fixed window: the best total over any w consecutive values. O(n)."""
    cur = best = sum(xs[:w])
    for i in range(w, len(xs)):
        cur += xs[i] - xs[i - w]      # add the new value, drop the oldest
        best = max(best, cur)
    return best

print(longest_run_with_at_most_k_failures([0, 1, 0, 0, 1, 1, 0, 0, 0, 1], 1))
print(max_window_sum([5, 3, 8, 6, 2, 9, 4], 3))

random.seed(2)
for _ in range(300):
    s = [random.randint(0, 1) for _ in range(random.randint(1, 15))]
    k = random.randint(0, 3)
    want = max(j - i for i in range(len(s)) for j in range(i, len(s) + 1) if sum(s[i:j]) <= k)
    assert longest_run_with_at_most_k_failures(s, k) == want
print("variable window matches brute force")
```
```text
4
17
variable window matches brute force
```

### Heaps for top k

To keep the k best items of a stream, keep a min heap of size k: the root is the worst item you are keeping, so a new item only enters if it beats the root. That is O(n log k) time and O(k) memory, better than sorting when k is much smaller than n and the only option when the stream does not fit in memory. When everything fits in a NumPy array, `np.argpartition` finds the top k in O(n) and you sort only those k.

```python
import heapq
import numpy as np

def top_k_similar(query, doc_stream, k):
    """Stream of (doc_id, vector). Keep the k best cosine scores in a min heap of size k. O(n log k)."""
    q = query / np.linalg.norm(query)
    heap = []                                      # (score, doc_id); the worst kept score sits on top
    for doc_id, vec in doc_stream:
        s = float(vec @ q / np.linalg.norm(vec))
        if len(heap) < k:
            heapq.heappush(heap, (s, doc_id))
        elif s > heap[0][0]:
            heapq.heapreplace(heap, (s, doc_id))   # pop the worst and push the new one in one sift
    return [d for s, d in sorted(heap, reverse=True)]

rng = np.random.default_rng(0)
docs = rng.normal(size=(1000, 16))
query = rng.normal(size=16)
got = top_k_similar(query, enumerate(docs), 5)

cos = docs @ query / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query))
want = [int(i) for i in np.argsort(-cos)[:5]]
print(got, got == want)

# everything in memory: argpartition finds the top k in O(n), then sort only those k
idx = np.argpartition(-cos, 5)[:5]
print([int(i) for i in idx[np.argsort(-cos[idx])]] == want)
```
```text
[119, 810, 942, 272, 103] True
True
```

### Binary search on the answer

Binary search applies whenever a yes or no question is monotone: if capacity C works, every larger capacity also works. Search the smallest C for which a greedy feasibility check passes. Splitting a model's layers into k contiguous pipeline stages so the slowest stage is as fast as possible is this problem (it is LeetCode's "split array largest sum" in ML clothing). The test compares against a brute force over every set of cut points.

```python
import itertools
import random

def min_max_stage_cost(costs, k):
    """Split layers, in order, into at most k contiguous pipeline stages so the slowest stage is as
    fast as possible. Binary search on the answer: feasibility is monotone in the capacity.
    O(n log(sum of costs))."""
    def stages_needed(cap):
        stages, cur = 1, 0
        for c in costs:
            if cur + c > cap:          # start a new stage
                stages, cur = stages + 1, 0
            cur += c
        return stages

    lo, hi = max(costs), sum(costs)    # the answer lives in [max, sum]
    while lo < hi:
        mid = (lo + hi) // 2
        if stages_needed(mid) <= k:
            hi = mid                   # feasible: try a smaller capacity
        else:
            lo = mid + 1               # infeasible: need more capacity
    return lo

def brute(costs, k):
    n, best = len(costs), sum(costs)
    for m in range(1, min(k, n)):
        for cuts in itertools.combinations(range(1, n), m):
            bounds = (0,) + cuts + (n,)
            best = min(best, max(sum(costs[a:b]) for a, b in zip(bounds, bounds[1:])))
    return best

print(min_max_stage_cost([4, 2, 7, 3, 5, 1, 6], 3))
random.seed(3)
for _ in range(300):
    costs = [random.randint(1, 9) for _ in range(random.randint(1, 8))]
    k = random.randint(1, 4)
    assert min_max_stage_cost(costs, k) == brute(costs, k)
print("matches brute force on 300 random cases")
```
```text
12
matches brute force on 300 random cases
```

### Intervals

Sort by start, then merge each interval into the last kept one if it overlaps. For "how many at once", sweep over events: +1 at each start, -1 at each end, and order ends before starts at the same timestamp when intervals are half open.

```python
import random

def merge_intervals(iv):
    """Merge overlapping [start, end) job intervals. O(n log n) for the sort."""
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def peak_concurrency(iv):
    """Most jobs running at once. Sweep +1 at a start and -1 at an end. At equal times the
    (t, -1) event sorts before (t, +1), so a job ending at t and one starting at t do not overlap."""
    events = sorted([(s, 1) for s, _ in iv] + [(e, -1) for _, e in iv])
    cur = best = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best

jobs = [(1, 4), (2, 5), (7, 9), (8, 10), (10, 12)]
print(merge_intervals(jobs))
print(sum(e - s for s, e in merge_intervals(jobs)), "busy hours")
print(peak_concurrency(jobs))

random.seed(4)
for _ in range(300):
    iv = [(s, s + random.randint(1, 5)) for s in (random.randint(0, 20) for _ in range(random.randint(1, 8)))]
    want = max(sum(s <= t < e for s, e in iv) for t in range(30))
    assert peak_concurrency(iv) == want
print("sweep matches a brute force scan")
```
```text
[[1, 5], [7, 12]]
9 busy hours
2
sweep matches a brute force scan
```

### BFS and DFS

BFS with a queue gives shortest hop counts in an unweighted graph; mark nodes when you enqueue them so nothing is queued twice. DFS post order gives a topological order, which is exactly the order backpropagation needs in reverse: micrograd and every autograd engine build it this way. Tracking an "active" state detects cycles. Recursion in Python stops at about 1,000 frames by default, so for deep graphs write the DFS with an explicit stack.

```python
from collections import Counter, deque

def bfs_distance(graph, src):
    """Hops from src to every reachable node. O(V + E)."""
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v not in dist:          # mark when enqueued so nothing is queued twice
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def friend_suggestions(graph, user, k):
    """People at distance exactly 2, ranked by number of mutual friends."""
    friends = set(graph[user])
    mutual = Counter(fof for f in friends for fof in graph[f] if fof != user and fof not in friends)
    return sorted(mutual, key=lambda p: (-mutual[p], p))[:k]

def topo_order(inputs):
    """Order compute graph nodes so each comes after its inputs (DFS post order). Raises on a cycle."""
    order, state = [], {}
    def visit(u):
        if state.get(u) == "active":
            raise ValueError(f"cycle through {u}")
        if state.get(u) == "done":
            return
        state[u] = "active"
        for v in inputs.get(u, []):
            visit(v)
        state[u] = "done"
        order.append(u)
    for u in inputs:
        visit(u)
    return order

g = {"ana": ["bo", "cy"], "bo": ["ana", "dee", "eli"], "cy": ["ana", "dee"],
     "dee": ["bo", "cy", "fay"], "eli": ["bo"], "fay": ["dee"]}
print(bfs_distance(g, "ana"))
print(friend_suggestions(g, "ana", 2))

# loss = mean(relu(x @ W + b)): backprop visits this order reversed
graph = {"loss": ["mean"], "mean": ["relu"], "relu": ["add"], "add": ["matmul", "b"],
         "matmul": ["x", "W"], "x": [], "W": [], "b": []}
order = topo_order(graph)
print(order)
assert all(order.index(v) < order.index(u) for u in graph for v in graph[u])
try:
    topo_order({"a": ["b"], "b": ["a"]})
except ValueError as e:
    print("ValueError:", e)
```
```text
{'ana': 0, 'bo': 1, 'cy': 1, 'dee': 2, 'eli': 2, 'fay': 3}
['dee', 'eli']
['x', 'W', 'matmul', 'b', 'add', 'relu', 'mean', 'loss']
ValueError: cycle through a
```

> **Watch:** [Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4652s) shows the topological sort that drives `backward()` (chapter "implementing the backward function for a whole expression graph").

### Simple dynamic programming

Dynamic programming applies when the answer to a big problem is built from answers to overlapping smaller ones. Edit distance is the ML favourite because word error rate in speech recognition is edit distance over words divided by the reference length. Define the table entry in words first ("dp[i][j] is the cost to turn the first i reference tokens into the first j hypothesis tokens"), then the recurrence, then notice each row only needs the previous one, which cuts space to O(n).

```python
import functools
import random

def edit_distance(ref, hyp):
    """Levenshtein distance over tokens. dp[i][j] = edits to turn ref[:i] into hyp[:j].
    O(m n) time, O(n) space because each row only needs the previous row."""
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            substitute = prev[j - 1] + (ref[i - 1] != hyp[j - 1])
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, substitute)   # delete, insert, substitute
        prev = cur
    return prev[-1]

def word_error_rate(ref, hyp):
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / len(r)

print(edit_distance("kitten", "sitting"))
print(round(word_error_rate("the cat sat on the mat", "the cat sit on mat"), 4))

def slow(a, b):
    @functools.lru_cache(maxsize=None)
    def go(i, j):
        if i == 0 or j == 0:
            return i + j
        return min(go(i - 1, j) + 1, go(i, j - 1) + 1, go(i - 1, j - 1) + (a[i - 1] != b[j - 1]))
    return go(len(a), len(b))

random.seed(5)
for _ in range(300):
    a = "".join(random.choice("abc") for _ in range(random.randint(0, 7)))
    b = "".join(random.choice("abc") for _ in range(random.randint(0, 7)))
    assert edit_distance(a, b) == slow(a, b)
print("bottom up table matches the memoised recursion")
```
```text
3
0.3333
bottom up table matches the memoised recursion
```

## NumPy vectorisation and broadcasting

Vectorising means replacing a Python loop over elements with whole array operations that run in compiled code. It matters in interviews for two reasons: a loop over a million elements takes seconds where the array version takes milliseconds, and interviewers read loop free NumPy as a sign that you think in shapes. The test below measures one run; the exact factor varies by machine.

```python
import time
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=1_000_000)
y = rng.normal(size=1_000_000)

def mse_loop(a, b):
    total = 0.0
    for i in range(len(a)):          # one interpreted iteration per element
        d = a[i] - b[i]
        total += d * d
    return total / len(a)

def mse_vec(a, b):
    return np.mean((a - b) ** 2)     # one call, the loop runs in compiled code

t0 = time.perf_counter(); slow = mse_loop(x, y)
t1 = time.perf_counter(); fast = mse_vec(x, y)
t2 = time.perf_counter()
print(np.isclose(slow, fast), f"vectorised was about {(t1 - t0) / (t2 - t1):.0f}x faster here")
```
```text
True vectorised was about 92x faster here
```

### Broadcasting rules

NumPy compares shapes from the trailing dimension backwards. Two dimensions are compatible when they are equal or one of them is 1, and a missing leading dimension counts as 1. The result takes the larger size in each position. So (4, 3) with (3,) works (the (3,) row is reused for every sample), (4, 3) with (4, 1) works (one value per row), and (4, 3) with (4,) fails because the trailing sizes are 3 and 4. `keepdims=True` on a reduction keeps the reduced axis as size 1 so the result broadcasts back against the input.

The dangerous case is the one that does not fail: a (n, 1) column minus a (n,) vector is a (n, n) matrix. This happens when predictions come out of a model as a column and labels are flat. The loss is then the mean of n squared numbers, the code runs, and the model trains on nonsense. Assert shapes at the boundaries of every function.

```python
import numpy as np

X = np.arange(12, dtype=float).reshape(4, 3)   # 4 samples, 3 features
mu, sd = X.mean(axis=0), X.std(axis=0)         # shape (3,)
Z = (X - mu) / sd                              # (4, 3) op (3,): broadcast across rows
print(np.allclose(Z.mean(axis=0), 0), Z.std(axis=0))

row_sums = X.sum(axis=1, keepdims=True)        # (4, 1): keepdims leaves a broadcastable axis
print((X / row_sums).sum(axis=1))

a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([1.0, 2.0, 3.0])
print((a[:, None] + b).shape)                  # (4, 1) op (3,) -> (4, 3), an outer sum

try:
    X / X.sum(axis=1)                          # (4, 3) op (4,): trailing sizes 3 and 4 clash
except ValueError as e:
    print("ValueError:", e)

p = np.array([0.2, 0.9, 0.4]).reshape(-1, 1)   # predictions as a column (3, 1)
y = np.array([0.0, 1.0, 0.0])                  # labels as a flat vector (3,)
print((p - y).shape)                           # (3, 3): no error, silently wrong
```
```text
True [1. 1. 1.]
[1. 1. 1. 1.]
(4, 3)
ValueError: operands could not be broadcast together with shapes (4,3) (4,) 
(3, 3)
```

### Pairwise distances without loops

Pairwise squared distances power kNN, k-means and retrieval. The direct broadcast `A[:, None, :] - B[None, :, :]` is easy to read but builds an n by m by d intermediate. The expansion of the squared norm uses one matrix multiply and only n by m memory; it can produce tiny negative values from rounding, so clip at zero before a square root.

```python
import numpy as np
from scipy.spatial.distance import cdist

def pairwise_sq_dists(A, B):
    """Squared Euclidean distances between rows of A (n, d) and B (m, d), returned as (n, m).
    Uses |a - b|^2 = |a|^2 + |b|^2 - 2 a.b, so peak memory is O(n m), not O(n m d)."""
    a2 = (A ** 2).sum(axis=1)[:, None]     # (n, 1)
    b2 = (B ** 2).sum(axis=1)[None, :]     # (1, m)
    d2 = a2 + b2 - 2.0 * A @ B.T           # (n, m), one matrix multiply
    return np.maximum(d2, 0.0)             # rounding can produce tiny negatives

def pairwise_sq_dists_broadcast(A, B):
    diff = A[:, None, :] - B[None, :, :]   # (n, m, d) intermediate: clear but memory hungry
    return (diff ** 2).sum(axis=-1)

rng = np.random.default_rng(0)
A, B = rng.normal(size=(500, 64)), rng.normal(size=(300, 64))
ref = cdist(A, B, "sqeuclidean")
print(np.allclose(pairwise_sq_dists(A, B), ref), np.allclose(pairwise_sq_dists_broadcast(A, B), ref))
print(f"broadcast intermediate: {500 * 300 * 64 * 8 / 1e6:.1f} MB, result: {500 * 300 * 8 / 1e6:.1f} MB")
```
```text
True True
broadcast intermediate: 76.8 MB, result: 1.2 MB
```

### Axes, views and copies

`axis=0` reduces over rows (one value per column); `axis=1` reduces over columns (one value per row). Basic slicing returns a view that shares memory, so writing to the slice writes to the original. Integer array indexing and boolean mask reads return copies, but assigning through a mask writes in place. `reshape` returns a view when the memory layout allows, and a transpose is a view with different strides, which is why it is not C contiguous.

```python
import numpy as np

a = np.arange(10)
s = a[2:5]                    # basic slicing returns a view
s[0] = 99
print(a[2], np.shares_memory(a, s))

f = a[[2, 3, 4]]              # integer array (fancy) indexing returns a copy
f[0] = -1
print(a[2], np.shares_memory(a, f))

m = a[a > 5]                  # a boolean mask read is also a copy
a[a > 5] = 0                  # but assigning through a mask writes into a
print(a, np.shares_memory(a, m))

b = a.reshape(2, 5)           # reshape returns a view when the memory layout allows it
print(np.shares_memory(a, b), b.T.flags["C_CONTIGUOUS"])
```
```text
99 True
99 False
[0 1 0 3 4 5 0 0 0 0] False
True False
```

### Vectorising a nested loop

Chip Huyen's book includes a question in this exact form: a function builds a 3D mask with three nested loops and a per element distance check; what is wrong and how do you vectorise it? The answer is open grids. `np.ogrid` gives index arrays of shapes (X, 1, 1), (1, Y, 1) and (1, 1, Z) that broadcast into the full grid only inside the arithmetic.

```python
import numpy as np

def sphere_mask_loops(shape, center, r):
    mask = np.zeros(shape, dtype=bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                d2 = (i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2
                mask[i, j, k] = d2 <= r * r
    return mask

def sphere_mask_vec(shape, center, r):
    i, j, k = np.ogrid[:shape[0], :shape[1], :shape[2]]   # shapes (X,1,1), (1,Y,1), (1,1,Z)
    return (i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2 <= r * r

shape, c, r = (40, 50, 60), (20, 25, 30), 12
fast = sphere_mask_vec(shape, c, r)
print(np.array_equal(sphere_mask_loops(shape, c, r), fast), int(fast.sum()))
```
```text
True 7153
```

### Numerical precision when summing

The same book asks how to average a million floats with minimal precision loss. A naive running sum in float32 loses the low bits of every addition once the accumulator is large: here it ends up about 9.5 away from the true mean of 1000.5. NumPy's `np.sum` uses pairwise summation, which keeps the error growth near logarithmic in n, and accumulating in float64 (or `math.fsum`, which is exact up to the final rounding) removes the problem. Subtracting a reference value before summing also helps, and Welford's algorithm in the streaming section is the one pass version of that idea.

```python
import math
import numpy as np

rng = np.random.default_rng(0)
x = (rng.random(1_000_000) + 1000).astype(np.float32)    # a million values near 1000.5
exact = math.fsum(x.astype(np.float64)) / len(x)         # correctly rounded reference

acc = np.float32(0)
for v in x:                                              # naive running sum in float32
    acc = np.float32(acc + v)
naive = float(acc) / len(x)
pairwise = float(np.sum(x)) / len(x)                     # NumPy sums float32 pairwise
upcast = float(np.sum(x, dtype=np.float64)) / len(x)     # accumulate in float64
print(f"exact {exact:.6f}  naive {naive:.6f}  pairwise {pairwise:.6f}  float64 {upcast:.6f}")
```
```text
exact 1000.500159  naive 1009.997696  pairwise 1000.500160  float64 1000.500159
```

> **Watch:** [Karpathy, makemore part 1](https://www.youtube.com/watch?v=PaCmpygFfXo&t=2177s) works through broadcasting while normalising a count matrix row by row, including a keepdims slip that silently normalises the wrong axis.

## Pandas in interview form

Pandas screens test about six operations. Know their shapes: which return one row per group, which return one row per input row, and which can change the row count.

| Operation | Output rows | SQL analogue | Common bug |
| --- | --- | --- | --- |
| `groupby().agg()` | One per group | GROUP BY | Counting rows when you meant unique users |
| `groupby().transform()` | One per input row | Window function with PARTITION BY | Using agg then a merge back, which is slower and easy to misalign |
| `merge` | Can grow or shrink | JOIN | Duplicated keys multiplying rows |
| `shift`, `diff`, `cumcount`, `rank` per group | One per input row | LAG, ROW_NUMBER, RANK | Forgetting to sort by time first |
| `rolling`, `resample` | One per row or per period | Window frames, date bucketing | Row windows on irregular dates |
| `pivot_table`, `unstack`, `crosstab` | One per index value | Conditional aggregation | Mean of a 0 or 1 column is a rate; sum is a count |

### groupby, agg and transform

Named aggregation (`agg(users=("user", "nunique"))`) gives readable output columns in one call. `transform` broadcasts the group result back to every row, which is how you compute a share of group total or a deviation from the group mean. `filter` keeps or drops whole groups, and `idxmax` per group picks the top row.

```python
import pandas as pd

ev = pd.DataFrame({
    "user": ["a", "a", "b", "b", "b", "c"],
    "country": ["US", "US", "IN", "IN", "IN", "US"],
    "revenue": [10.0, 5.0, 0.0, 7.0, 3.0, 20.0],
})
# named aggregation: one output row per group, one column per (input column, function)
per_country = ev.groupby("country").agg(
    users=("user", "nunique"), revenue=("revenue", "sum"), orders=("revenue", "size")
).reset_index()
print(per_country)

# transform returns one value per input row: the tool for "share of group" columns
ev["share_of_user"] = ev["revenue"] / ev.groupby("user")["revenue"].transform("sum")
print(ev.round(2))

# filter keeps or drops whole groups
print(ev.groupby("user").filter(lambda g: len(g) >= 2)["user"].unique().tolist())

# the top row per group
print(ev.loc[ev.groupby("user")["revenue"].idxmax(), ["user", "revenue"]])
```
```text
  country  users  revenue  orders
0      IN      1     10.0       3
1      US      2     35.0       3
  user country  revenue  share_of_user
0    a      US     10.0           0.67
1    a      US      5.0           0.33
2    b      IN      0.0           0.00
3    b      IN      7.0           0.70
4    b      IN      3.0           0.30
5    c      US     20.0           1.00
['a', 'b']
  user  revenue
0    a     10.0
3    b      7.0
5    c     20.0
```

### merge without surprises

Before joining, say the grain of each table out loud ("users is one row per user, orders is one row per order"). Then use `validate` to make pandas check it and `indicator=True` to see which rows matched. pandas 3.0 (released January 2026) added `how="left_anti"` for "users with no orders"; the `isin` version works on every version. Report orphan keys instead of dropping them silently.

```python
import pandas as pd

users = pd.DataFrame({"user_id": [1, 2, 3, 4], "plan": ["free", "pro", "free", "pro"]})
orders = pd.DataFrame({"user_id": [1, 1, 2, 5], "amount": [9.0, 12.0, 30.0, 4.0]})

m = users.merge(orders, on="user_id", how="left", validate="one_to_many", indicator=True)
print(m)
print(len(users), "users became", len(m), "rows")        # user 1 has two orders

print(users.merge(orders, on="user_id", how="left_anti"))  # users with no orders, pandas 3.0+
print(users[~users["user_id"].isin(orders["user_id"])])    # the same on any pandas version
print(orders[~orders["user_id"].isin(users["user_id"])])   # orphan orders: report them

try:
    users.merge(orders, on="user_id", validate="one_to_one")
except pd.errors.MergeError as e:
    print("MergeError:", e)
```
```text
   user_id  plan  amount     _merge
0        1  free     9.0       both
1        1  free    12.0       both
2        2   pro    30.0       both
3        3  free     NaN  left_only
4        4   pro     NaN  left_only
4 users became 5 rows
   user_id  plan  amount
3        3  free     NaN
4        4   pro     NaN
   user_id  plan
2        3  free
3        4   pro
   user_id  amount
3        5     4.0
MergeError: Merge keys are not unique in right dataset; not a one-to-one merge
Duplicates in right:
  user_id
       1 ...
```

### Window style operations: lag, sessions, row numbers

The single most common pandas screen question is some form of sessionisation: sort by user and time, compute the gap to the previous event per user, start a new session when the gap exceeds a threshold, and number sessions with a cumulative sum of the "new session" flag. The same three tools (`diff` or `shift` per group, a boolean flag, `cumsum` per group) answer "time since previous purchase", "first event per user" and "streak length" questions.

```python
import pandas as pd

ev = pd.DataFrame({
    "user": ["a", "a", "a", "b", "b", "a"],
    "ts": pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:10", "2026-01-01 11:00",
                          "2026-01-01 09:05", "2026-01-01 09:50", "2026-01-01 11:20"]),
})
ev = ev.sort_values(["user", "ts"]).reset_index(drop=True)
ev["gap"] = ev.groupby("user")["ts"].diff()                     # like LAG() in SQL
ev["new_session"] = ev["gap"].isna() | (ev["gap"] > pd.Timedelta(minutes=30))
ev["session"] = ev.groupby("user")["new_session"].cumsum()      # running count per user
ev["event_no"] = ev.groupby("user").cumcount() + 1              # like ROW_NUMBER()
print(ev)
print(ev.groupby(["user", "session"]).size().rename("events").reset_index())
```
```text
  user                  ts             gap  new_session  session  event_no
0    a 2026-01-01 09:00:00             NaT         True        1         1
1    a 2026-01-01 09:10:00 0 days 00:10:00        False        1         2
2    a 2026-01-01 11:00:00 0 days 01:50:00         True        2         3
3    a 2026-01-01 11:20:00 0 days 00:20:00        False        2         4
4    b 2026-01-01 09:05:00             NaT         True        1         1
5    b 2026-01-01 09:50:00 0 days 00:45:00         True        2         2
  user  session  events
0    a        1       2
1    a        2       2
2    b        1       1
3    b        2       1
```

### Rolling windows and ranks

`rolling(3)` is a window of three rows; `rolling("7D")` on a DatetimeIndex is a window of seven calendar days, which is what a "7 day average" means when days can be missing. `min_periods` decides whether the first partial windows return a value. `rank(method="dense")` matches SQL DENSE_RANK.

```python
import pandas as pd

daily = pd.DataFrame({
    "day": pd.date_range("2026-03-01", periods=10, freq="D"),
    "signups": [5, 3, 8, 6, 2, 9, 4, 7, 1, 10],
}).set_index("day")
daily["avg_7d"] = daily["signups"].rolling("7D").mean()            # time based window
daily["avg_3rows"] = daily["signups"].rolling(3, min_periods=3).mean()
daily["pct_change"] = daily["signups"].pct_change()
daily["rank"] = daily["signups"].rank(ascending=False, method="dense").astype(int)
print(daily.round(2))
```
```text
            signups  avg_7d  avg_3rows  pct_change  rank
day                                                     
2026-03-01        5    5.00        NaN         NaN     6
2026-03-02        3    4.00        NaN       -0.40     8
2026-03-03        8    5.33       5.33        1.67     3
2026-03-04        6    5.50       5.67       -0.25     5
2026-03-05        2    4.80       5.33       -0.67     9
2026-03-06        9    5.50       5.67        3.50     2
2026-03-07        4    5.29       5.00       -0.56     7
2026-03-08        7    5.57       6.67        0.75     4
2026-03-09        1    5.29       4.00       -0.86    10
2026-03-10       10    5.57       6.00        9.00     1
```

### Copy on Write in pandas 3.0

pandas 3.0 made Copy on Write the default. Every indexing result behaves as a copy, chained assignment such as `df[mask]["b"] = 0` never modifies `df` (it now raises a `ChainedAssignmentError` warning), and `SettingWithCopyWarning` is gone. The correct form has always been a single `df.loc[mask, "b"] = 0`. Two other 3.0 changes show up in screens: text columns infer the new `str` dtype instead of `object`, and parsed datetimes default to microsecond resolution (`datetime64[us]`).

```python
import warnings
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    df[df["a"] > 1]["b"] = 0        # chained assignment: writes into a temporary copy
print(df["b"].tolist(), [type(w.message).__name__ for w in caught])

df.loc[df["a"] > 1, "b"] = 0        # one .loc call with the row mask and the column
print(df["b"].tolist())

sub = df[df["a"] > 1]
sub.loc[:, "b"] = -1                # a derived frame behaves as a copy under Copy-on-Write
print(df["b"].tolist(), sub["b"].tolist(), pd.__version__)
```
```text
[10, 20, 30] ['ChainedAssignmentError']
[10, 0, 0]
[10, 0, 0] [-1, -1] 3.0.6
```

### Reshaping and missing values

`pivot_table` aggregates into a wide table; `groupby` plus `unstack` does the same and is easier to reason about; `crosstab` counts. Before any modelling, count missing values per column, and fill numeric gaps with statistics computed on the training split only, never on the full dataset.

```python
import pandas as pd

df = pd.DataFrame({
    "variant": ["A", "A", "B", "B", "B", "A", "B", "A"],
    "device": ["ios", "web", "ios", "web", "web", "ios", "ios", "web"],
    "converted": [1, 0, 1, 1, 0, 0, 1, 1],
})
print(df.pivot_table(index="device", columns="variant", values="converted", aggfunc="mean"))
print(pd.crosstab(df["device"], df["variant"], margins=True))
print(df["variant"].value_counts(normalize=True))
wide = df.groupby(["device", "variant"])["converted"].mean().unstack("variant")
print(wide)
```
```text
variant    A    B
device           
ios      0.5  1.0
web      0.5  0.5
variant  A  B  All
device            
ios      2  2    4
web      2  2    4
All      4  4    8
variant
A    0.5
B    0.5
Name: proportion, dtype: float64
variant    A    B
device           
ios      0.5  1.0
web      0.5  0.5
```

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"age": [25, np.nan, 40, 31], "city": ["NY", None, "SF", "NY"],
                   "signup": ["2026-01-03", "2026-02-10", None, "2026-02-28"]})
print(df.isna().sum().to_dict())
print(df.dtypes.astype(str).to_dict())           # pandas 3.0 infers str for text, not object
df["age"] = df["age"].fillna(df["age"].median())  # in a real pipeline use the TRAIN median
df["city"] = df["city"].fillna("unknown")
df["signup"] = pd.to_datetime(df["signup"])
df["month"] = df["signup"].dt.to_period("M")
print(df)
print(df["signup"].dtype)
```
```text
{'age': 1, 'city': 1, 'signup': 1}
{'age': 'float64', 'city': 'str', 'signup': 'str'}
    age     city     signup    month
0  25.0       NY 2026-01-03  2026-01
1  31.0  unknown 2026-02-10  2026-02
2  40.0       SF        NaT      NaT
3  31.0       NY 2026-02-28  2026-02
datetime64[us]
```

> **Interview trap:** Calling `df.fillna(df.mean())` before the train test split. The test rows contributed to the mean, so the evaluation has seen test data. Fit every statistic on the training rows, then apply it to both.

## Linear and logistic regression

These are the two most requested from scratch models because they test the whole loop in a few lines: a model, a loss, a gradient, an update, and a correctness check.

### Linear regression in closed form

Add a column of ones to X to get the design matrix A, so the intercept is just another weight. The squared loss is

$$
L(w) = \lVert A w - y \rVert^2, \qquad \nabla L = 2 A^\top (A w - y) = 0 \;\Rightarrow\; A^\top A \, w = A^\top y .
$$

These are the normal equations. Solve them with `np.linalg.solve` rather than computing the inverse: solving is cheaper and more accurate. Forming $A^\top A$ squares the condition number of A, so when features are nearly collinear `np.linalg.lstsq` (based on the SVD) is the safer call, and when A is rank deficient (a duplicated column) only `lstsq` or a ridge penalty gives an answer at all. Ridge adds $\lambda \lVert w \rVert^2$, which changes the system to $(A^\top A + \lambda I') w = A^\top y$ with the intercept entry of $I'$ set to zero, and makes the matrix invertible for any $\lambda > 0$. Cost: $O(n d^2)$ to form $A^\top A$ plus $O(d^3)$ to solve.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

def fit_linear(X, y, l2=0.0):
    """Least squares with an intercept: solve (A^T A + P) w = A^T y, where A = [1, X] and
    P = l2 * I except that the intercept is not penalised. O(n d^2 + d^3)."""
    A = np.c_[np.ones(len(X)), X]
    P = l2 * np.eye(A.shape[1])
    P[0, 0] = 0.0
    w = np.linalg.solve(A.T @ A + P, A.T @ y)     # solve the system, never form the inverse
    return w[0], w[1:]

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = 4.0 + X @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.1, size=200)

b, w = fit_linear(X, y)
sk = LinearRegression().fit(X, y)
print(np.round(w, 4), round(float(b), 4))
print(np.allclose(w, sk.coef_), np.isclose(b, sk.intercept_))

b2, w2 = fit_linear(X, y, l2=10.0)
skr = Ridge(alpha=10.0).fit(X, y)
print(np.allclose(w2, skr.coef_), np.isclose(b2, skr.intercept_))

# a duplicated column makes A^T A singular; lstsq (SVD based) still returns the minimum norm fit
Xd = np.c_[X, X[:, 0]]
A = np.c_[np.ones(len(Xd)), Xd]
print(np.linalg.matrix_rank(A.T @ A), "of", A.shape[1])
w_ls, *_ = np.linalg.lstsq(A, y, rcond=None)
print(np.allclose(A @ w_ls, sk.predict(X)))
```
```text
[ 1.991  -1.0096  0.5053] 3.9964
True True
True True
4 of 5
True
```

### Linear regression by gradient descent

For mean squared error $\frac{1}{n}\lVert Xw + b - y \rVert^2$ the gradient is $\frac{2}{n} X^\top (Xw + b - y)$ for the weights and $\frac{2}{n} \sum_i (x_i^\top w + b - y_i)$ for the bias. Each epoch is $O(nd)$, so gradient descent wins when d is large (no $d^3$ solve) or when data arrives in mini batches. Convergence needs a learning rate below $2 / \lambda_{\max}$ of the Hessian $\frac{2}{n} X^\top X$. When one feature is in thousands and another in fractions, the eigenvalues differ by orders of magnitude and no single learning rate works; standardising the features fixes that. Map the weights back with $w = w_z / \sigma$ and $b = b_z - w^\top \mu$.

```python
import numpy as np

def linreg_gd(X, y, lr=0.1, epochs=500):
    """Batch gradient descent on mean squared error. Each epoch costs O(n d)."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        err = X @ w + b - y                    # (n,)
        w -= lr * (2.0 / n) * (X.T @ err)      # gradient of mean((Xw + b - y)^2)
        b -= lr * (2.0 / n) * err.sum()
    return w, b

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3)) * np.array([1.0, 10.0, 0.1]) + 5.0   # very different scales
y = 4.0 + X @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.1, size=200)

mu, sd = X.mean(axis=0), X.std(axis=0)
Z = (X - mu) / sd                       # standardise so one learning rate suits every feature
wz, bz = linreg_gd(Z, y)
w, b = wz / sd, bz - (wz / sd) @ mu     # map the weights back to the original units

A = np.c_[np.ones(len(X)), X]
w_cf = np.linalg.lstsq(A, y, rcond=None)[0]
print(np.round(w, 3), np.round(w_cf[1:], 3), np.allclose(w, w_cf[1:], atol=1e-3))

w_raw, b_raw = linreg_gd(X, y, lr=0.1, epochs=50)    # the same learning rate on raw features
mse = lambda w_, b_: float(np.mean((X @ w_ + b_ - y) ** 2))
print(f"MSE after GD: standardised {mse(w, b):.4f}, raw features {mse(w_raw, b_raw):.2e}")
```
```text
[ 1.991 -1.001  0.553] [ 1.991 -1.001  0.553] True
MSE after GD: standardised 0.0100, raw features 1.48e+144
```

The raw feature run is not a crash, it is a loss of about $10^{144}$: the largest Hessian eigenvalue is about 289, so each step multiplies the error along that direction by about 28 (the factor is the absolute value of 1 minus 0.1 times 289), and the smallest eigenvalue is about 0.0007, a condition number near 400,000.

### Logistic regression by gradient descent

The model is $p = \sigma(w^\top x + b)$ with $\sigma(z) = 1 / (1 + e^{-z})$, trained on binary cross entropy $-[y \log p + (1 - y) \log(1 - p)]$. The derivative of the loss with respect to the logit z collapses to $p - y$, so the gradient over a batch is

$$
\nabla_w = \frac{1}{n} X^\top (p - y) + \lambda w, \qquad \nabla_b = \frac{1}{n} \sum_i (p_i - y_i).
$$

The loss is convex, so gradient descent reaches the global optimum, but there is no closed form. Two stability details separate a good answer from a fragile one. A sigmoid written as `1 / (1 + np.exp(-z))` overflows for z near -1000; the branch version only calls exp on non positive numbers. The loss is best written in logits, $\log(1 + e^{z}) - yz$, using `np.logaddexp(0, z)`, which never takes the log of zero. The test matches scikit-learn exactly once you map the objectives: scikit-learn minimises $C \sum_i \ell_i + \frac{1}{2}\lVert w \rVert^2$, which equals the mean loss plus $\frac{\lambda}{2}\lVert w \rVert^2$ when $C = 1 / (n \lambda)$.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    """Stable: exp is only ever called on non positive numbers."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def bce_from_logits(z, y):
    """Mean binary cross entropy written in logits: log(1 + e^z) - y z, via logaddexp."""
    return float(np.mean(np.logaddexp(0.0, z) - y * z))

def logreg_gd(X, y, lam=0.0, lr=0.5, epochs=5000):
    """Minimise mean BCE + (lam / 2) |w|^2 by batch gradient descent. O(n d) per epoch."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        g = sigmoid(X @ w + b) - y            # dL/dz for sigmoid plus cross entropy
        w -= lr * (X.T @ g / n + lam * w)     # the bias is not regularised
        b -= lr * g.mean()
    return w, b

rng = np.random.default_rng(1)
X = rng.normal(size=(500, 4))
y = (rng.random(500) < sigmoid(X @ np.array([1.5, -2.0, 0.0, 0.7]) - 0.3)).astype(float)

lam = 0.01
w, b = logreg_gd(X, y, lam=lam)
# scikit-learn minimises C * sum(log loss) + 0.5 |w|^2, so C = 1 / (n lam) is the same objective
sk = LogisticRegression(C=1.0 / (len(X) * lam), tol=1e-10, max_iter=10_000).fit(X, y)
print(np.round(w, 4), np.round(sk.coef_.ravel(), 4))
print(np.allclose(w, sk.coef_.ravel(), atol=1e-4), np.isclose(b, sk.intercept_[0], atol=1e-4))
print(round(bce_from_logits(X @ w + b, y), 4))
print(sigmoid(np.array([-1000.0, 0.0, 1000.0])))      # no overflow, no nan
```
```text
[ 1.0236 -1.5779 -0.0344  0.5398] [ 1.0236 -1.5779 -0.0344  0.5398]
True True
0.4358
[0.  0.5 1. ]
```

> **Interview trap:** Training unregularised logistic regression on linearly separable data and expecting it to converge. The loss keeps falling as the weights grow, so the weights go to infinity. Add an L2 penalty or stop early, and say why.

Follow ups to expect: why cross entropy and not squared error (squared error on a sigmoid is not convex and its gradient vanishes when the model is confidently wrong); how to go multiclass (softmax with a weight matrix, covered below); what Newton's method would do (the Hessian is $X^\top S X$ with $S = \operatorname{diag}(p(1-p))$, which gives iteratively reweighted least squares); and how to handle class imbalance (class weights in the loss or threshold tuning on a validation set).

> **Watch:** [CS229 Lecture 3: Weighted Least Squares](https://www.youtube.com/watch?v=uJF_gL3jhxI) and [CS229 Lecture 4: Exponential Family, GLMs Classification](https://www.youtube.com/watch?v=8gVi4Rk21Eg) derive both models.

## k-means and kNN

### k-means with k-means++

k-means minimises inertia, the sum of squared distances from each point to its assigned centre. Lloyd's algorithm alternates two steps, each of which cannot increase inertia: assign every point to its nearest centre, then move each centre to the mean of its points. It therefore converges, but only to a local optimum that depends on the starting centres. k-means++ (Arthur and Vassilvitskii, SODA 2007) picks the first centre uniformly and each next one with probability proportional to $D(x)^2$, the squared distance to the nearest centre already chosen; in expectation the result is within a factor $O(\log k)$ of the optimal inertia. scikit-learn uses k-means++ by default and reruns several initialisations.

Implementation details interviewers check: distances vectorised with the expansion trick ($O(nkd)$ per iteration, $O(nk)$ memory), a convergence test on centre movement, and a rule for empty clusters (keep the old centre, or re seed it at the point farthest from its centre, which is what the code does). The test compares inertia and labels (via the adjusted Rand index, which ignores label permutation) against scikit-learn with ten restarts.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

def sq_dists(X, C):
    return np.maximum((X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T, 0.0)

def kmeans_pp_init(X, k, rng):
    """First centre uniform at random; each next centre drawn with probability proportional
    to D(x)^2, the squared distance to the nearest centre chosen so far. O(n k d)."""
    n = len(X)
    centers = [X[rng.integers(n)]]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        idx = rng.choice(n, p=d2 / d2.sum())
        centers.append(X[idx])
        d2 = np.minimum(d2, ((X - X[idx]) ** 2).sum(axis=1))
    return np.array(centers)

def kmeans(X, k, n_iter=100, tol=1e-8, seed=0, init=None):
    """Lloyd's algorithm. O(n k d) per iteration. Returns centres, labels, inertia."""
    rng = np.random.default_rng(seed)
    C = kmeans_pp_init(X, k, rng) if init is None else np.array(init, dtype=float)
    for _ in range(n_iter):
        d2 = sq_dists(X, C)
        labels = d2.argmin(axis=1)                     # assignment step
        new_C = C.copy()
        for j in range(k):                             # update step
            members = X[labels == j]
            if len(members):
                new_C[j] = members.mean(axis=0)
            else:                                      # empty cluster: re seed at the worst served point
                new_C[j] = X[d2.min(axis=1).argmax()]
        shift = np.abs(new_C - C).max()
        C = new_C
        if shift < tol:
            break
    d2 = sq_dists(X, C)
    return C, d2.argmin(axis=1), float(d2.min(axis=1).sum())

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=7)
C, labels, inertia = kmeans(X, 4)
sk = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
print(round(inertia, 3), round(float(sk.inertia_), 3), adjusted_rand_score(labels, sk.labels_))

# an initial centre far from every point gets no members and must be re seeded
bad_init = np.r_[X[:3], [[1e3, 1e3]]]
C2, labels2, inertia2 = kmeans(X, 4, init=bad_init)
print(np.bincount(labels2, minlength=4).min() > 0, round(inertia2, 3))
```
```text
1127.402 1127.402 1.0
True 1127.402
```

Follow ups: choosing k (elbow on inertia, silhouette score, or the downstream task); why scale features first (distances are dominated by the largest unit); what shapes k-means fails on (non convex or very unequal clusters, because each cluster is a Voronoi cell around a mean); and scale (k-means++ needs k sequential passes over the data, k-means|| by Bahmani and others at VLDB 2012 samples many centres per pass, and mini batch k-means updates centres from small batches).

> **Watch:** [CS229 Lecture 9: K-Means and GMM](https://www.youtube.com/watch?v=bSmIGBCoffA).

### kNN

k nearest neighbours stores the training set and, for each query, takes a majority vote among the k closest training points. Training is O(1) beyond storing $O(nd)$ data; prediction costs $O(nd)$ per query for the distances. `np.argpartition` selects the k smallest distances per row in linear time without sorting all n. Tie breaking must be stated: `argmax` over `bincount` returns the smallest label among tied classes, which is also what scikit-learn does, and the test agrees with scikit-learn on every query for k of 1, 5 and 8.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def knn_predict(X_train, y_train, X_test, k):
    """Majority vote of the k nearest training points.
    O(n_test n_train d) for distances, O(n_test n_train) for the selection, O(n_train d) memory to store."""
    d2 = (X_test ** 2).sum(1)[:, None] + (X_train ** 2).sum(1)[None, :] - 2 * X_test @ X_train.T
    nn = np.argpartition(d2, k - 1, axis=1)[:, :k]     # the k smallest per row, unordered, O(n_train)
    votes = y_train[nn]                                 # (n_test, k) neighbour labels
    counts = np.apply_along_axis(np.bincount, 1, votes, minlength=y_train.max() + 1)
    return counts.argmax(axis=1)                        # ties go to the smallest label

rng = np.random.default_rng(0)
X_train = rng.normal(size=(300, 5))
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int) + (X_train[:, 2] > 1).astype(int)
X_test = rng.normal(size=(100, 5))

for k in (1, 5, 8):
    mine = knn_predict(X_train, y_train, X_test, k)
    sk = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_test)
    print(k, (mine == sk).mean())
```
```text
1 1.0
5 1.0
8 1.0
```

Follow ups: k trades variance (k = 1 memorises noise) for bias (large k smooths over real structure), so pick it by cross validation; distance weighting; feature scaling; the curse of dimensionality (in high dimensions distances concentrate, so nearest and farthest neighbours look alike); and scale (KD trees and ball trees for low dimensions, approximate nearest neighbour indexes such as HNSW or IVF for embeddings).

## Decision tree splits

A classification tree chooses, at each node, the feature and threshold that most reduce weighted impurity. Gini impurity for class proportions $p_c$ is $G = 1 - \sum_c p_c^2$: the chance that two labels drawn at random from the node disagree. A split into left and right children with $n_L$ and $n_R$ points scores

$$
G_{\text{split}} = \frac{n_L}{n} G_L + \frac{n_R}{n} G_R, \qquad \text{gain} = G_{\text{parent}} - G_{\text{split}} .
$$

Trying every threshold naively costs $O(n)$ per threshold and $O(n^2)$ per feature. The standard trick sorts the feature once and sweeps left to right, moving one point at a time from the right counts to the left counts, which makes each candidate O(C) for C classes and the feature $O(n \log n)$ overall. Skip positions where adjacent values are equal (no threshold separates them) and put the threshold at the midpoint, as scikit-learn does. The test rounds features to create ties and matches scikit-learn's depth one stump on feature and threshold.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def gini(counts):
    p = counts / counts.sum()
    return 1.0 - float((p ** 2).sum())

def best_split_feature(x, y, n_classes):
    """Best threshold on one feature by weighted child Gini.
    Sort once, then sweep left to right updating class counts: O(n log n + n C)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = len(y)
    left = np.zeros(n_classes)
    right = np.bincount(ys, minlength=n_classes).astype(float)
    best_score, best_thr = np.inf, None
    for i in range(n - 1):
        left[ys[i]] += 1
        right[ys[i]] -= 1
        if xs[i] == xs[i + 1]:
            continue                                   # no threshold separates equal values
        n_left = i + 1
        score = (n_left * gini(left) + (n - n_left) * gini(right)) / n
        if score < best_score:
            best_score, best_thr = score, (xs[i] + xs[i + 1]) / 2.0   # midpoint, like scikit-learn
    return best_score, best_thr

def best_split(X, y):
    """Return (feature, threshold, impurity decrease) of the best single split."""
    n_classes = int(y.max()) + 1
    parent = gini(np.bincount(y, minlength=n_classes).astype(float))
    results = [best_split_feature(X[:, j], y, n_classes) + (j,) for j in range(X.shape[1])]
    score, thr, feat = min(results, key=lambda r: r[0])
    return feat, thr, parent - score

rng = np.random.default_rng(0)
for trial in range(5):
    X = rng.normal(size=(300, 4)).round(2)              # rounding creates tied values
    y = ((X[:, trial % 4] > 0.3) ^ (rng.random(300) < 0.15)).astype(int)
    feat, thr, gain = best_split(X, y)
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    same = feat == stump.tree_.feature[0] and np.isclose(thr, stump.tree_.threshold[0])
    print(trial, feat, round(thr, 3), round(gain, 4), same)
```
```text
0 0 0.31 0.169 True
1 1 0.3 0.2405 True
2 2 0.295 0.2526 True
3 3 0.295 0.1814 True
4 0 0.3 0.2175 True
```

Entropy, $-\sum_c p_c \log_2 p_c$, gives the information gain criterion; it ranks splits almost the same way as Gini and costs a log. Regression trees use variance reduction instead, with the same sorted sweep over running sums of y and y squared. Follow ups: why trees need no feature scaling (only the order of values matters); how a random forest decorrelates trees (bootstrap rows and a random subset of features at each split); how gradient boosting uses the same split finder on residuals or gradients; and how categorical features are handled (one hot, or sort categories by their mean target and treat them as ordered).

## Softmax, cross entropy and backprop

### Numerically stable softmax

Softmax maps logits to probabilities, $\operatorname{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$. It is unchanged by adding the same constant to every logit, so subtract the row maximum first: the largest exponent becomes $e^0 = 1$, nothing overflows, and at least one term in the denominator is 1, so nothing underflows to a division by zero. Without the shift, a logit of 1000 overflows to infinity and the output is `nan`.

### Cross entropy from logits

For integer labels, cross entropy is the mean of $-\log p_{y}$. Computing softmax and then `np.log` can take the log of an underflowed zero. The stable route works in log space:

$$
\log \operatorname{softmax}(z)_y = z_y - \operatorname{logsumexp}(z), \qquad \operatorname{logsumexp}(z) = m + \log \sum_j e^{z_j - m}, \; m = \max_j z_j .
$$

This is why PyTorch's `F.cross_entropy` takes logits, not probabilities. A useful sanity value: with C classes and uniform predictions the loss is $\log C$ (2.3026 for ten classes), which is what a freshly initialised classifier should report.

```python
import numpy as np
from scipy.special import logsumexp as sp_logsumexp, softmax as sp_softmax
from sklearn.metrics import log_loss

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)    # softmax(z) == softmax(z - c); now the max is 0
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)

def cross_entropy(logits, y):
    """Mean negative log likelihood from raw logits (n, C) and integer labels (n,)."""
    log_probs = logits - logsumexp(logits)[:, None]     # log softmax without ever taking log(0)
    return float(-log_probs[np.arange(len(y)), y].mean())

z = np.array([[1000.0, 1001.0, 1002.0], [-5.0, 0.0, 5.0]])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
print(naive[0])                                          # exp(1000) is inf, inf / inf is nan
print(softmax(z)[0], np.allclose(softmax(z), sp_softmax(z, axis=1)))

rng = np.random.default_rng(0)
logits = rng.normal(size=(8, 4)) * 3
y = rng.integers(0, 4, size=8)
print(np.isclose(cross_entropy(logits, y), log_loss(y, sp_softmax(logits, axis=1), labels=range(4))))
print(np.allclose(logsumexp(logits), sp_logsumexp(logits, axis=1)))
print(round(cross_entropy(np.zeros((1, 10)), np.array([3])), 4), round(float(np.log(10)), 4))
```
```text
[nan nan nan]
[0.09003057 0.24472847 0.66524096] True
True
True
2.3026 2.3026
```

### The softmax cross entropy gradient

With $p = \operatorname{softmax}(z)$ and a one hot target y, $\partial L / \partial z = p - y$. The derivation is short enough to give at a whiteboard: $L = -z_y + \log \sum_j e^{z_j}$, so $\partial L / \partial z_i = -[i = y] + e^{z_i} / \sum_j e^{z_j} = p_i - y_i$. Averaged over a batch it is $(P - Y)/n$. The sigmoid with binary cross entropy gives the same $p - y$ form, which is why the logistic regression gradient above looks the way it does.

### A two layer MLP with manual backprop

Write the backward pass as the forward pass read in reverse, one line per line, and check every gradient's shape against its parameter's shape. For $h = \tanh(XW_1 + b_1)$ and logits $= hW_2 + b_2$:

| Forward | Backward |
| --- | --- |
| logits $= hW_2 + b_2$ | $dW_2 = h^\top d\text{logits}$, $db_2 = \sum_{\text{rows}} d\text{logits}$, $dh = d\text{logits}\, W_2^\top$ |
| $h = \tanh(u)$ | $du = dh \odot (1 - h^2)$ |
| $u = XW_1 + b_1$ | $dW_1 = X^\top du$, $db_1 = \sum_{\text{rows}} du$ |

Two rules cover every linear layer: the weight gradient is (input transposed) times (upstream gradient), and the bias gradient sums the upstream gradient over the batch, because the bias was broadcast across it.

### Gradient checking

Verify analytic gradients against centred finite differences, $(f(\theta + h) - f(\theta - h)) / 2h$, whose error is $O(h^2)$ against $O(h)$ for the one sided version. CS231n recommends comparing with the relative error $|a - n| / \max(|a|, |n|)$, working in float64, using a step near $10^{-5}$, and checking a few data points. Its rough scale: above $10^{-2}$ the gradient is probably wrong, around $10^{-7}$ or below is good, and kinks such as ReLU at zero can make a correct gradient look worse. The check below is on a tanh network for that reason; its worst relative error is $6 \times 10^{-8}$. Training then reaches 99 percent accuracy on an XOR style problem that no linear model can fit.

```python
import numpy as np

def init(d_in, d_h, d_out, rng):
    return {"W1": rng.normal(size=(d_in, d_h)) / np.sqrt(d_in), "b1": np.zeros(d_h),
            "W2": rng.normal(size=(d_h, d_out)) / np.sqrt(d_h), "b2": np.zeros(d_out)}

def forward_backward(params, X, y):
    """Two layer MLP: tanh hidden layer, softmax output, mean cross entropy. Returns loss and grads."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    n = len(X)
    # forward
    h_pre = X @ W1 + b1                           # (n, H)
    h = np.tanh(h_pre)                            # (n, H)
    logits = h @ W2 + b2                          # (n, C)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    loss = float(-np.log(p[np.arange(n), y]).mean())
    # backward: one line per forward line, in reverse order
    dlogits = p.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n                                  # softmax plus cross entropy: (p - onehot) / n
    dW2 = h.T @ dlogits                           # (H, C)
    db2 = dlogits.sum(axis=0)                     # a bias gradient sums over the batch
    dh = dlogits @ W2.T                           # (n, H)
    dh_pre = dh * (1.0 - h ** 2)                  # tanh'(u) = 1 - tanh(u)^2
    dW1 = X.T @ dh_pre                            # (D, H)
    db1 = dh_pre.sum(axis=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def grad_check(params, X, y, eps=1e-5):
    """Centred differences on every parameter; returns the worst relative error."""
    _, grads = forward_backward(params, X, y)
    worst = 0.0
    for name, P in params.items():
        for i in np.ndindex(P.shape):
            old = P[i]
            P[i] = old + eps
            loss_plus, _ = forward_backward(params, X, y)
            P[i] = old - eps
            loss_minus, _ = forward_backward(params, X, y)
            P[i] = old
            num = (loss_plus - loss_minus) / (2 * eps)
            ana = grads[name][i]
            worst = max(worst, abs(num - ana) / max(abs(num), abs(ana), 1e-12))
    return worst

rng = np.random.default_rng(0)
X = rng.normal(size=(6, 4))
y = rng.integers(0, 3, size=6)
print(f"worst relative error {grad_check(init(4, 5, 3, rng), X, y):.1e}")

# the gradients are right, so plain gradient descent should learn a nonlinear boundary
X = rng.normal(size=(400, 2))
y = (X[:, 0] * X[:, 1] > 0).astype(int)          # XOR style quadrants, not linearly separable
params = init(2, 16, 2, rng)
for step in range(3001):
    loss, g = forward_backward(params, X, y)
    for name in params:
        params[name] -= 0.5 * g[name]
    if step % 1000 == 0:
        print(step, round(loss, 4))
h = np.tanh(X @ params["W1"] + params["b1"])
print("train accuracy", ((h @ params["W2"] + params["b2"]).argmax(1) == y).mean())
```
```text
worst relative error 6.3e-08
0 0.7896
1000 0.0649
2000 0.0448
3000 0.0363
train accuracy 0.9925
```

> **Watch:** [Karpathy, Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=5191s) derives the cross entropy backward pass by hand (chapter "exercise 2"), and [building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1930s) starts from manual backprop on a scalar expression.

Follow ups: why initialise weights with variance $1/\text{fan in}$ (keeps activations from saturating tanh or exploding); why not zero initialisation (every hidden unit gets the same gradient and they never differentiate); what a starting loss far above $\log C$ means (overconfident random predictions, usually a scale problem in the last layer); and the cost of backprop (about twice the forward pass in compute, plus the memory to store every activation).

## Attention from scratch

### Scaled dot product attention

For queries Q, keys K and values V,

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V .
$$

Each output row is a weighted average of value rows, with weights from how well that query matches every key. The division by $\sqrt{d_k}$ is the follow up everyone gets: if query and key entries are independent with variance 1, their dot product has variance $d_k$, so without scaling the logits grow with head size, softmax saturates toward one hot, and gradients through it shrink. Scaling keeps the logit variance near 1.

### Masks

A mask sets disallowed logits to $-\infty$ before the softmax, so they get exactly zero weight. A causal mask is lower triangular: position t may attend to positions 0 through t, which is what lets a decoder train on every position of a sequence in parallel without seeing the future. A padding mask blocks keys that are padding tokens. If a whole row could be masked (an all padding query), $-\infty$ everywhere gives `nan`; use a large negative number or mask the output instead.

### Multi head attention

Project x to Q, K and V with (D, D) matrices, reshape the last dimension into h heads of size D / h, move the head axis next to the batch axis, run attention on all heads at once, then reverse the reshape and apply an output projection. The reshape and transpose is where most bugs live, so the test compares the batched version against a slow loop over batch items and heads, checks that weight rows sum to 1 with zero weight on the future, and perturbs the last token to confirm no earlier output changes.

```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q (..., Tq, dk), K (..., Tk, dk), V (..., Tk, dv); mask is True where attending is allowed.
    Cost O(Tq Tk dk) time and O(Tq Tk) memory for the score matrix."""
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])   # (..., Tq, Tk)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)                 # masked logits get weight 0
    weights = softmax(scores, axis=-1)
    return weights @ V, weights

def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))                  # row t may see columns 0..t

def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    """x (B, T, D); each W is (D, D). Heads split D into n_heads slices of size D // n_heads."""
    B, T, D = x.shape
    hd = D // n_heads
    def split(t):                                                # (B, T, D) -> (B, h, T, hd)
        return t.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(T) if causal else None)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)             # concatenate the heads
    return out @ Wo, w

rng = np.random.default_rng(0)
B, T, D, H = 2, 5, 8, 2
x = rng.normal(size=(B, T, D))
Wq, Wk, Wv, Wo = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(4))
out, w = multi_head_attention(x, Wq, Wk, Wv, Wo, H)

# 1. every row of weights is a distribution and the future gets exactly zero weight
print(np.allclose(w.sum(-1), 1.0), np.all(w[..., np.triu_indices(T, 1)[0], np.triu_indices(T, 1)[1]] == 0))

# 2. the batched version equals a slow loop over batch items and heads
hd = D // H
slow = np.zeros((B, T, D))
for b in range(B):
    heads = []
    for h in range(H):
        sl = slice(h * hd, (h + 1) * hd)
        q, k, v = x[b] @ Wq[:, sl], x[b] @ Wk[:, sl], x[b] @ Wv[:, sl]
        s = q @ k.T / np.sqrt(hd)
        s[np.triu_indices(T, 1)] = -np.inf
        heads.append(softmax(s) @ v)
    slow[b] = np.concatenate(heads, axis=-1) @ Wo
print(np.allclose(out, slow))

# 3. causality: changing the last token cannot change any earlier output
x2 = x.copy()
x2[:, -1] += 10.0
out2, _ = multi_head_attention(x2, Wq, Wk, Wv, Wo, H)
print(np.allclose(out[:, :-1], out2[:, :-1]), np.allclose(out[:, -1], out2[:, -1]))
```
```text
True True
True
True False
```

Cost: computing the scores is $O(T^2 d)$ time and $O(T^2)$ memory per head, which is the bottleneck for long contexts and the reason for FlashAttention (tiles the computation so the full score matrix is never stored) and for sparse or sliding window variants. Multi query and grouped query attention share key and value heads across query heads to shrink the KV cache.

### KV cache

During generation each new token needs attention against all previous tokens. Under a causal mask the keys and values of past tokens never change, so cache them and compute only the new token's query, key and value: per step cost drops from recomputing $O(t^2 d)$ scores to $O(t d)$. The test shows token by token decoding with a cache reproduces full causal attention exactly. Memory is the price: $2 \times \text{layers} \times \text{kv heads} \times \text{head dim} \times \text{tokens} \times \text{bytes}$. For a configuration with 32 layers, 8 key value heads of dimension 128 and 16 bit values, an 8,192 token sequence needs 1 GiB.

```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def full_causal_attention(x, Wq, Wk, Wv):
    T, D = x.shape
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    s = q @ k.T / np.sqrt(D)
    s[np.triu_indices(T, 1)] = -np.inf
    return softmax(s) @ v                              # recomputes all T x T scores

def decode_with_kv_cache(x, Wq, Wk, Wv):
    """Process one token at a time. Keys and values of past tokens never change under a causal
    mask, so cache them and compute only the new row: O(t D) per step instead of O(t^2 D)."""
    D = x.shape[1]
    K_cache, V_cache, outs = [], [], []
    for x_t in x:
        K_cache.append(x_t @ Wk)                        # append, never recompute
        V_cache.append(x_t @ Wv)
        K, V = np.stack(K_cache), np.stack(V_cache)     # (t + 1, D)
        w = softmax((x_t @ Wq) @ K.T / np.sqrt(D))      # one query row against all cached keys
        outs.append(w @ V)
    return np.stack(outs)

rng = np.random.default_rng(0)
T, D = 6, 8
x = rng.normal(size=(T, D))
Wq, Wk, Wv = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(3))
print(np.allclose(full_causal_attention(x, Wq, Wk, Wv), decode_with_kv_cache(x, Wq, Wk, Wv)))

n_layers, n_kv_heads, head_dim, seq, bytes_per = 32, 8, 128, 8192, 2
print(f"KV cache per sequence: {2 * n_layers * n_kv_heads * head_dim * seq * bytes_per / 2**30:.2f} GiB")
```
```text
True
KV cache per sequence: 1.00 GiB
```

> **Watch:** [Karpathy, Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s) builds self attention step by step ("version 4"), with the scaling explained at [01:16:56](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4616s) and multi head attention at [01:21:59](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4919s). [CME295 Lecture 1](https://www.youtube.com/watch?v=Ub3GoFaUcds&t=4007s) covers the mechanism and [CME295 Lecture 3](https://www.youtube.com/watch?v=Q5baLehv5So&t=5100s) the KV cache.

## Tokenization and decoding

### Byte pair encoding

BPE (Sennrich and others for subword NMT, then GPT-2 at byte level) learns a vocabulary by repeatedly merging the most frequent adjacent pair of symbols. On a corpus of words with frequencies: split each word into characters, count adjacent pairs weighted by word frequency, merge the top pair everywhere, record the merge, repeat until the vocabulary is the target size. On the Hugging Face course corpus (hug 10, pug 5, pun 12, bun 4, hugs 5) the first merges are ug (count 20), then un (16), then hug (15). Encoding a new word applies the learned merges in the order they were learned, so an unseen word such as "thug" still encodes as t plus hug, and nothing is ever out of vocabulary at the byte level.

Byte level BPE, as in GPT-2 and Karpathy's minbpe, starts from the 256 byte values and numbers new tokens from 256. On minbpe's example string "aaabdaaabac" three merges give the ids [258, 100, 258, 97, 99], and decoding returns the original string. Note the tie at the second merge: pairs (256, 97) and (97, 98) both occur twice, and `max` over a Counter picks the first one seen. The merge table depends on the tie rule, so say yours.

```python
from collections import Counter

def pair_counts(words):
    """words maps a tuple of symbols to its corpus frequency; pair counts are frequency weighted."""
    counts = Counter()
    for symbols, freq in words.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += freq
    return counts

def merge_symbols(symbols, pair, new):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(new)
            i += 2                      # non overlapping, left to right
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)

def train_bpe(word_freqs, n_merges):
    """Each merge costs one pass over the corpus: O(n_merges * corpus symbols)."""
    words = {tuple(w): f for w, f in word_freqs.items()}      # start from single characters
    merges = []
    for _ in range(n_merges):
        counts = pair_counts(words)
        if not counts:
            break
        best = max(counts, key=counts.get)                      # ties: the first pair seen wins
        merges.append(best)
        words = {merge_symbols(s, best, best[0] + best[1]): f for s, f in words.items()}
    return merges, words

def encode(word, merges):
    """Apply the learned merges in the order they were learned."""
    symbols = tuple(word)
    for pair in merges:
        symbols = merge_symbols(symbols, pair, pair[0] + pair[1])
    return symbols

corpus = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}   # Hugging Face LLM course example
merges, words = train_bpe(corpus, 3)
print(merges)
print(words)
print(encode("hugs", merges), encode("bug", merges), encode("thug", merges))

# byte level, as in GPT-2 and minbpe: start from the 256 byte values, new ids from 256 upward
def train_byte_bpe(text, n_merges):
    ids = list(text.encode("utf-8"))
    merges = {}
    for i in range(n_merges):
        counts = Counter(zip(ids, ids[1:]))
        pair = max(counts, key=counts.get)
        merges[pair] = 256 + i
        ids = list(merge_symbols(tuple(ids), pair, 256 + i))
    return ids, merges

def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new in merges.items():                          # dicts keep insertion order
        vocab[new] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8")

ids, m = train_byte_bpe("aaabdaaabac", 3)
print(ids, decode(ids, m) == "aaabdaaabac")
```
```text
[('u', 'g'), ('u', 'n'), ('h', 'ug')]
{('hug',): 10, ('p', 'ug'): 5, ('p', 'un'): 12, ('b', 'un'): 4, ('hug', 's'): 5}
('hug', 's') ('b', 'ug') ('t', 'hug')
[258, 100, 258, 97, 99] True
```

Complexity: the naive loop recounts all pairs after every merge, $O(M \cdot N)$ for M merges over N symbols, which is fine for an interview; production trainers update pair counts incrementally with a priority queue. Real tokenizers also pre split text with a regex (GPT-2 separates letters, digits, punctuation and leading spaces) so merges never cross word boundaries.

> **Watch:** [Karpathy, Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s) walks through the BPE algorithm and then [counts and merges pairs in code](https://www.youtube.com/watch?v=zduSFxRajkE&t=1715s).

### Greedy and beam search

Greedy decoding takes the most likely token at every step. Beam search keeps the B best partial sequences by total log probability, expands each by every token, and keeps the best B again. It can find a sequence greedy misses: in the Hugging Face blog example, greedy goes "The nice woman" with probability $0.5 \times 0.4 = 0.2$, while "The dog has" has $0.4 \times 0.9 = 0.36$ and a beam of 2 finds it. Beam search is still not exact search, it favours short outputs unless scores are length normalised (divide the log probability by length raised to some power), and in open ended generation it tends to repeat itself, which is why it is used for input grounded tasks such as translation and speech recognition rather than chat.

```python
import heapq
import math

# toy next token table; the nice and dog branches follow the Hugging Face decoding blog example
LM = {
    ("The",): {"nice": 0.5, "dog": 0.4, "car": 0.1},
    ("The", "nice"): {"woman": 0.4, "house": 0.3, "guy": 0.3},
    ("The", "dog"): {"has": 0.9, "runs": 0.05, "and": 0.05},
    ("The", "car"): {"is": 0.3, "drives": 0.5, "turns": 0.2},
}

def next_probs(seq):
    return LM.get(tuple(seq), {"<eos>": 1.0})

def greedy(prefix, steps):
    seq, logp = list(prefix), 0.0
    for _ in range(steps):
        tok, p = max(next_probs(seq).items(), key=lambda kv: kv[1])
        seq.append(tok)
        logp += math.log(p)
    return seq, round(math.exp(logp), 4)

def beam_search(prefix, steps, beam_width):
    """Keep the beam_width best partial sequences by total log probability.
    Each step scores beam_width * V candidates."""
    beams = [(0.0, list(prefix))]
    for _ in range(steps):
        candidates = [(lp + math.log(p), seq + [tok])
                      for lp, seq in beams for tok, p in next_probs(seq).items()]
        beams = heapq.nlargest(beam_width, candidates, key=lambda c: c[0])
    lp, seq = beams[0]
    return seq, round(math.exp(lp), 4)

print(greedy(["The"], 2))
print(beam_search(["The"], 2, beam_width=2))
print(beam_search(["The"], 2, beam_width=1))     # width 1 is greedy
```
```text
(['The', 'nice', 'woman'], 0.2)
(['The', 'dog', 'has'], 0.36)
(['The', 'nice', 'woman'], 0.2)
```

### Temperature, top k and top p

Sampling draws the next token from the model's distribution. Temperature divides the logits by T before the softmax: T below 1 sharpens the distribution toward greedy, T above 1 flattens it. Top k (Fan and others, 2018) keeps only the k most likely tokens and renormalises. Top p, or nucleus sampling (Holtzman and others, ICLR 2020), keeps the smallest set of top tokens whose cumulative probability reaches p, so the candidate set shrinks when the model is confident and grows when it is not. The usual order is temperature, then top k, then top p, then renormalise and draw.

On probabilities [0.5, 0.25, 0.15, 0.1], top p of 0.7 keeps the first two tokens (cumulative 0.75) and top p of 0.8 keeps three (cumulative 0.9). Sampling with top k of 2 draws the first two tokens about two thirds and one third of the time.

```python
import numpy as np

def top_k_filter(logits, k):
    """Keep the k largest logits; everything else becomes -inf."""
    if k >= len(logits):
        return logits
    kth = np.partition(logits, -k)[-k]
    return np.where(logits >= kth, logits, -np.inf)       # ties at the kth value survive

def top_p_filter(logits, p):
    """Nucleus: keep the smallest set of top tokens whose cumulative probability reaches p."""
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    keep = order[: np.searchsorted(cum, p) + 1]           # first index where cum >= p, inclusive
    out = np.full_like(logits, -np.inf)
    out[keep] = logits[keep]
    return out

def sample(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """Temperature first, then the filters, then renormalise and draw."""
    if temperature == 0:
        return int(np.argmax(logits))                     # greedy
    z = np.asarray(logits, dtype=float) / temperature
    if top_k is not None:
        z = top_k_filter(z, top_k)
    if top_p is not None:
        z = top_p_filter(z, top_p)
    probs = np.exp(z - z.max())
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))

probs = np.array([0.5, 0.25, 0.15, 0.1])
logits = np.log(probs)
print(np.isfinite(top_k_filter(logits, 2)).nonzero()[0], np.isfinite(top_p_filter(logits, 0.7)).nonzero()[0],
      np.isfinite(top_p_filter(logits, 0.8)).nonzero()[0])

for T in (0.5, 1.0, 2.0):
    q = np.exp(logits / T) / np.exp(logits / T).sum()
    print(T, q.round(3))                                   # low T sharpens, high T flattens

rng = np.random.default_rng(0)
draws = np.bincount([sample(logits, rng, top_k=2) for _ in range(20_000)], minlength=4) / 20_000
print(draws.round(3))                                      # about 2/3 and 1/3, zero elsewhere
```
```text
[0 1] [0 1] [0 1 2]
0.5 [0.725 0.181 0.065 0.029]
1.0 [0.5  0.25 0.15 0.1 ]
2.0 [0.37  0.262 0.203 0.166]
[0.664 0.336 0.    0.   ]
```

> **Watch:** [CME295 Lecture 3](https://www.youtube.com/watch?v=Q5baLehv5So&t=2314s) covers greedy decoding and beam search, then [sampling methods](https://www.youtube.com/watch?v=Q5baLehv5So&t=2796s) and [temperature](https://www.youtube.com/watch?v=Q5baLehv5So&t=3144s).

## Metrics from scratch

### Precision and recall at a threshold

At threshold t, predict positive when the score is at least t, then count the confusion matrix: true positives, false positives, false negatives, true negatives. Precision is TP / (TP + FP), the share of flagged items that are real. Recall is TP / (TP + FN), the share of real items that were flagged. F1 is their harmonic mean, $2PR / (P + R)$, which is low when either one is low. Decide the zero division convention out loud: with no positive predictions precision is undefined, and scikit-learn's `zero_division=0` reports 0. On the ten item example, lowering the threshold from 0.5 to 0.25 raises recall from 0.75 to 1.0 and lowers precision from 0.75 to 0.667.

```python
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def confusion_at(y_true, scores, t):
    pred = scores >= t
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    return tp, fp, fn, tn

def precision_recall_f1(y_true, scores, t):
    tp, fp, fn, tn = confusion_at(y_true, scores, t)
    precision = tp / (tp + fp) if tp + fp else 0.0      # nothing predicted positive: report 0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
s = np.array([0.95, 0.80, 0.55, 0.30, 0.70, 0.45, 0.20, 0.15, 0.10, 0.05])
for t in (0.5, 0.25, 0.99):
    mine = precision_recall_f1(y, s, t)
    pred = (s >= t).astype(int)
    sk = (precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0),
          f1_score(y, pred, zero_division=0))
    print(t, confusion_at(y, s, t), np.round(mine, 3), np.allclose(mine, sk))
```
```text
0.5 (3, 1, 1, 5) [0.75 0.75 0.75] True
0.25 (4, 2, 0, 4) [0.667 1.    0.8  ] True
0.99 (0, 0, 4, 6) [0. 0. 0.] True
```

### ROC AUC from scores

ROC AUC is the probability that a randomly chosen positive gets a higher score than a randomly chosen negative, with ties counting one half. That definition gives an $O(PN)$ pairwise implementation, which is the clearest test oracle. The efficient version uses ranks: sort all scores, give tied scores the average of their ranks, and apply the Mann Whitney identity

$$
\text{AUC} = \frac{\sum_{i \in \text{pos}} r_i - P(P+1)/2}{P \cdot N},
$$

which is $O(n \log n)$. The subtracted term is the smallest possible rank sum for P positives. AUC depends only on the order of the scores, so any strictly increasing transform (scaling, exp, a sigmoid) leaves it unchanged, which is why it is a ranking metric and says nothing about calibration. It is undefined when only one class is present, which happens in small bootstrap resamples. The Gini coefficient used in credit scoring is $2 \cdot \text{AUC} - 1$.

```python
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def average_ranks(s):
    """1 based ranks; tied values share the mean of the ranks they span."""
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks

def auc_rank(y, s):
    """Mann Whitney: AUC = (sum of positive ranks - P(P + 1)/2) / (P N). O(n log n)."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined when only one class is present")
    r = average_ranks(s)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def auc_pairwise(y, s):
    """The definition: P(random positive outscores random negative), ties count one half. O(P N)."""
    diff = s[y == 1][:, None] - s[y == 0][None, :]
    return ((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size

y0 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
s0 = np.array([0.95, 0.80, 0.55, 0.30, 0.70, 0.45, 0.20, 0.15, 0.10, 0.05])
print(auc_rank(y0, s0), auc_pairwise(y0, s0), roc_auc_score(y0, s0))

rng = np.random.default_rng(0)
for _ in range(200):
    n = int(rng.integers(2, 60))
    y = rng.integers(0, 2, size=n)
    if y.min() == y.max():
        continue
    s = rng.integers(0, 5, size=n).astype(float)        # few distinct values, many ties
    assert np.isclose(auc_rank(y, s), roc_auc_score(y, s))
    assert np.isclose(auc_pairwise(y, s), roc_auc_score(y, s))
    assert np.allclose(average_ranks(s), rankdata(s))
print("rank and pairwise AUC match scikit-learn with ties")
print(roc_auc_score(y0, s0), roc_auc_score(y0, 10 * s0 - 3), roc_auc_score(y0, np.exp(s0)))   # ranks only
```
```text
0.875 0.875 0.875
rank and pairwise AUC match scikit-learn with ties
0.875 0.875 0.875
```

> **Interview trap:** Passing hard 0 or 1 predictions to `roc_auc_score`. AUC needs scores; with labels you get the area of a single threshold, which is the balanced accuracy, not the AUC.

Under heavy class imbalance, precision recall curves and average precision are more informative than ROC, because the false positive rate stays small even when false positives outnumber true positives.

### nDCG at k

Discounted cumulative gain sums the relevance of the items you ranked, discounted by position: $\text{DCG@}k = \sum_{i=1}^{k} g(\text{rel}_i) / \log_2(i + 1)$. Normalise by the DCG of the ideal ordering (the same items sorted by true relevance, also cut at k) to get nDCG in [0, 1]. Two gain conventions exist and sources disagree on which is standard: scikit-learn's `ndcg_score` uses the relevance directly (linear gain), while many ranking papers and libraries use $2^{\text{rel}} - 1$, which rewards highly relevant items more. The code reproduces the scikit-learn docs example exactly (0.6957, 0.4937, and 0.352 at k = 4) and shows the exponential gain value for comparison. scikit-learn also averages over tied scores by default; the code assumes no ties.

```python
import numpy as np
from sklearn.metrics import ndcg_score

def dcg_at_k(rels, k, gain="linear"):
    rels = np.asarray(rels, dtype=float)[:k]
    g = rels if gain == "linear" else 2.0 ** rels - 1.0
    return float((g / np.log2(np.arange(2, len(rels) + 2))).sum())   # rank i is discounted by log2(i + 1)

def ndcg_at_k(true_rel, scores, k, gain="linear"):
    ranked = np.asarray(true_rel)[np.argsort(-np.asarray(scores), kind="stable")]
    ideal = np.sort(true_rel)[::-1]
    idcg = dcg_at_k(ideal, k, gain)
    return dcg_at_k(ranked, k, gain) / idcg if idcg > 0 else 0.0

true = [10, 0, 0, 1, 5]                                  # the example in the scikit-learn docs
cases = [([.1, .2, .3, 4, 70], 5), ([.05, 1.1, 1., .5, 0.], 5), ([.05, 1.1, 1., .5, 0.], 4)]
for scores, k in cases:
    print(k, round(ndcg_at_k(true, scores, k), 4), round(float(ndcg_score([true], [scores], k=k)), 4))
print("exponential gain:", round(ndcg_at_k(true, [.1, .2, .3, 4, 70], 5, gain="exp"), 4))
print("perfect ranking:", ndcg_at_k(true, true, 3))
```
```text
5 0.6957 0.6957
5 0.4937 0.4937
4 0.352 0.352
exponential gain: 0.4097
perfect ranking: 1.0
```

Follow ups: recall at k (did the relevant item appear in the top k at all, the standard first stage retrieval metric), MRR (the mean of one over the rank of the first relevant item), and why nDCG suits graded relevance while MRR suits a single right answer.

## Splitting and resampling

### Train test split and k fold

A split is a permutation followed by a cut. k fold cross validation partitions the indices into k folds; each fold is the test set once and the other k minus 1 folds train the model. scikit-learn's `KFold` without shuffling uses contiguous folds where the first n mod k folds get one extra item, and the code reproduces it exactly. Shuffle first unless the order carries meaning (time).

### Stratification

Stratifying keeps the class mix of every fold equal to the full data. With 10 percent positives in 100 rows, plain shuffled k fold put 3, 2, 1, 3 and 1 positives in its five test folds; stratified folds get exactly 2 each. Stratify whenever a class is rare, or a fold's metric is mostly noise. The implementation shuffles each class and deals its members round robin across folds, continuing the deal across classes so fold sizes stay balanced too.

```python
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def train_test_split_idx(n, test_frac, rng):
    perm = rng.permutation(n)                     # shuffle once, then cut
    n_test = int(round(n * test_frac))
    return perm[n_test:], perm[:n_test]

def kfold_indices(n, k):
    """Contiguous folds; the first n % k folds get one extra item, like scikit-learn KFold."""
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1
    idx, start = np.arange(n), 0
    for size in sizes:
        test = idx[start:start + size]
        yield np.r_[idx[:start], idx[start + size:]], test
        start += size

def stratified_kfold_indices(y, k, rng):
    """Shuffle each class, then deal its members round robin across folds, continuing the
    deal across classes so fold sizes stay balanced too."""
    folds, pos = [[] for _ in range(k)], 0
    for c in np.unique(y):
        for m in rng.permutation(np.flatnonzero(y == c)):
            folds[pos % k].append(m)
            pos += 1
    everything = np.arange(len(y))
    for f in folds:
        test = np.sort(np.array(f, dtype=int))
        yield np.setdiff1d(everything, test), test

rng = np.random.default_rng(0)
train, test = train_test_split_idx(23, 0.2, rng)
print(len(train), len(test), len(np.intersect1d(train, test)))

mine = [t.tolist() for _, t in kfold_indices(23, 5)]
sk = [t.tolist() for _, t in KFold(5).split(np.zeros(23))]
print(mine == sk, [len(t) for t in mine])

y = np.array([0] * 90 + [1] * 10)                 # 10 percent positives
for name, folds in [("mine", stratified_kfold_indices(y, 5, rng)),
                    ("sklearn", StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(100), y)),
                    ("plain", KFold(5, shuffle=True, random_state=0).split(np.zeros(100)))]:
    print(name, [int(y[t].sum()) for _, t in folds])   # positives per test fold
```
```text
18 5 0
True [5, 5, 5, 4, 4]
mine [2, 2, 2, 2, 2]
sklearn [2, 2, 2, 2, 2]
plain [3, 2, 1, 3, 1]
```

### Leakage

Every statistic used by the model (scaling means, imputation values, selected features, target encodings, vocabulary) must be fit on training rows only. The demonstration mirrors the scikit-learn common pitfalls page: with 10,000 pure noise features and random labels, choosing the 25 features most correlated with the label using all rows gives a mean test accuracy of 0.834 over ten seeds, while choosing them on the training rows alone gives 0.496, the honest chance level. scikit-learn's own version of the example reports 0.76 against 0.5. In production code a `Pipeline` enforces the rule inside cross validation. Two more splits prevent leakage by construction: a group split (every row of a user, a patient or a document on one side) and a time split (train on the past, test on the future, never shuffle).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def top_features(X, y, k=25):
    """The k features most correlated with the label."""
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    corr = np.abs(Xc.T @ yc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc))
    return np.argsort(-corr)[:k]

def one_run(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 10_000))            # pure noise features
    y = rng.integers(0, 2, size=200)              # random labels: honest accuracy is about 0.5
    train, test = np.arange(150), np.arange(150, 200)
    def accuracy(cols):
        model = LogisticRegression(max_iter=1000).fit(X[np.ix_(train, cols)], y[train])
        return model.score(X[np.ix_(test, cols)], y[test])
    leaky = accuracy(top_features(X, y))                  # selected on all rows, test included
    honest = accuracy(top_features(X[train], y[train]))   # selected on training rows only
    return leaky, honest

runs = np.array([one_run(seed) for seed in range(10)])
print("mean test accuracy over 10 seeds: leaky", runs[:, 0].mean().round(3), "honest", runs[:, 1].mean().round(3))

# group split: every row of a user lands on one side
rng = np.random.default_rng(0)
users = rng.integers(0, 40, size=200)
is_test = np.isin(users, rng.permutation(np.unique(users))[:8])
print("users on both sides:", len(np.intersect1d(users[is_test], users[~is_test])))
```
```text
mean test accuracy over 10 seeds: leaky 0.834 honest 0.496
users on both sides: 0
```

### Stratified and weighted sampling

Sampling the same fraction from every stratum keeps rare groups at their population share; a simple random sample of 200 from data with 2 percent spam gave 3, 5 and 5 spam rows on three seeds, while the stratified sample gave 4 every time. For weighted sampling without replacement, `rng.choice(..., replace=False, p=w / w.sum())` works in memory; the weighted reservoir in the next section works on a stream.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = pd.DataFrame({"label": rng.choice(["spam", "ham"], size=10_000, p=[0.02, 0.98]),
                   "x": rng.normal(size=10_000)})

def stratified_sample(df, by, frac, seed=0):
    """Sample the same fraction from every stratum so rare groups keep their share."""
    return df.groupby(by, group_keys=False).sample(frac=frac, random_state=seed)

print(df["label"].value_counts().to_dict())
for seed in range(3):
    simple = df.sample(n=200, random_state=seed)
    strat = stratified_sample(df, "label", 0.02, seed)
    print(seed, simple["label"].value_counts().to_dict(), strat["label"].value_counts().to_dict())

# weighted sampling without replacement: probability proportional to a weight column
w = np.where(df["label"] == "spam", 10.0, 1.0)
picked = rng.choice(len(df), size=500, replace=False, p=w / w.sum())
print((df["label"].to_numpy()[picked] == "spam").mean().round(3))
```
```text
{'ham': 9810, 'spam': 190}
0 {'ham': 197, 'spam': 3} {'ham': 196, 'spam': 4}
1 {'ham': 195, 'spam': 5} {'ham': 196, 'spam': 4}
2 {'ham': 195, 'spam': 5} {'ham': 196, 'spam': 4}
0.15
```

### Bootstrap confidence intervals

The percentile bootstrap resamples the n rows with replacement B times, recomputes the statistic on each resample, and reports the 2.5th and 97.5th percentiles. It needs no formula for the standard error, so it works for medians, AUCs and ratios. For the mean of 200 exponential draws it agrees with the t interval to two decimals and with SciPy's `stats.bootstrap`. To compare two models, resample the same rows for both (a paired bootstrap) and look at the interval of the difference: here the AUC difference of -0.020 has an interval from -0.086 to 0.048, which includes zero, so this test set cannot separate the models. Skip resamples that contain one class only, because AUC is undefined there. The bootstrap assumes independent rows; with users contributing many rows, resample users, not rows.

```python
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

def bootstrap_ci(x, stat=np.mean, n_boot=10_000, alpha=0.05, seed=0):
    """Percentile bootstrap: resample rows with replacement, recompute, take quantiles.
    O(n_boot * n) time; the (n_boot, n) index matrix is built in one call."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    return np.quantile(boots, [alpha / 2, 1 - alpha / 2])

rng = np.random.default_rng(42)
x = rng.exponential(scale=2.0, size=200)
lo, hi = bootstrap_ci(x)
t_lo, t_hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
sp = stats.bootstrap((x,), np.mean, confidence_level=0.95, method="percentile", random_state=0)
print(f"mean {x.mean():.3f}  mine [{lo:.3f}, {hi:.3f}]  t [{t_lo:.3f}, {t_hi:.3f}]  "
      f"scipy [{sp.confidence_interval.low:.3f}, {sp.confidence_interval.high:.3f}]")
print("median CI", np.round(bootstrap_ci(x, stat=np.median), 3))

# paired bootstrap for the AUC difference of two models scored on the same test rows
y = rng.integers(0, 2, size=400)
s_a = y + rng.normal(scale=1.0, size=400)
s_b = y + rng.normal(scale=1.2, size=400)
diffs = []
for _ in range(2000):
    i = rng.integers(0, 400, size=400)               # the same rows for both models
    if y[i].min() == y[i].max():
        continue
    diffs.append(roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]))
print(f"AUC A minus B: {roc_auc_score(y, s_a) - roc_auc_score(y, s_b):.3f}, 95% CI {np.round(np.quantile(diffs, [0.025, 0.975]), 3)}")
```
```text
mean 1.928  mine [1.682, 2.187]  t [1.673, 2.183]  scipy [1.685, 2.184]
median CI [1.059 1.84 ]
AUC A minus B: -0.020, 95% CI [-0.086  0.048]
```

> **Watch:** [CS229 Lecture 6: Dataset Split, ML Advice](https://www.youtube.com/watch?v=llnEgyyuYkQ) and [Karpathy, makemore part 2](https://www.youtube.com/watch?v=TCH_1BHY58I&t=3200s) on why train, validation and test splits exist.

## Streaming statistics

### Reservoir sampling

Algorithm R keeps a uniform random sample of k items from a stream whose length is unknown, in one pass and O(k) memory. Keep the first k items; for item number i (1 based) with i greater than k, draw j uniformly from 1 to i and, if j is at most k, replace slot j. The proof is an induction: item i enters with probability k / i, and an item already in the reservoir survives step i + 1 with probability $1 - \frac{k}{i+1} \cdot \frac{1}{k} = \frac{i}{i+1}$, so its inclusion probability stays $\frac{k}{i} \cdot \frac{i}{i+1} = \frac{k}{i+1}$. The empirical test over 20,000 runs gives every item a frequency within 0.003 of 0.3. Algorithm L (Li, 1994) skips ahead with geometric jumps and runs in $O(k(1 + \log(n/k)))$ expected time. For weighted sampling, A-Res (Efraimidis and Spirakis, 2006) gives each item the key $u^{1/w}$ for a uniform u and keeps the k largest keys in a min heap; with k = 1 it picks each item with probability proportional to its weight.

```python
import heapq
import random
from collections import Counter
from scipy.stats import chisquare

def reservoir_sample(stream, k, rng=random):
    """Algorithm R: a uniform sample of k items from a stream of unknown length.
    O(n) time, O(k) memory, one pass."""
    res = []
    for i, x in enumerate(stream):              # i is 0 based, so x is item number i + 1
        if i < k:
            res.append(x)
        else:
            j = rng.randint(0, i)               # inclusive: uniform over i + 1 positions
            if j < k:
                res[j] = x                      # kept with probability k / (i + 1)
    return res

rng = random.Random(0)
counts, trials, n, k = Counter(), 20_000, 10, 3
for _ in range(trials):
    counts.update(reservoir_sample(range(n), k, rng))
print({x: round(counts[x] / trials, 3) for x in range(n)})       # each close to k / n = 0.3
print(chisquare([counts[x] for x in range(n)]).pvalue > 0.01)

def weighted_reservoir(stream, k, rng=random):
    """A-Res (Efraimidis and Spirakis): give each item the key u ** (1 / w), keep the k largest."""
    heap = []
    for x, w in stream:
        key = rng.random() ** (1.0 / w)
        if len(heap) < k:
            heapq.heappush(heap, (key, x))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, x))
    return [x for _, x in heap]

wc = Counter(weighted_reservoir([("a", 1.0), ("b", 2.0), ("c", 7.0)], 1, rng)[0] for _ in range(20_000))
print({x: round(c / 20_000, 3) for x, c in sorted(wc.items())})  # about 0.1, 0.2, 0.7 when k = 1
```
```text
{0: 0.303, 1: 0.299, 2: 0.298, 3: 0.3, 4: 0.299, 5: 0.301, 6: 0.3, 7: 0.303, 8: 0.3, 9: 0.297}
True
{'a': 0.099, 'b': 0.196, 'c': 0.706}
```

### Online mean and variance with Welford

The textbook one pass variance, (sum of squares minus the square of the sum over n) divided by n minus 1, subtracts two huge nearly equal numbers. On the Wikipedia example (4, 7, 13, 16 shifted by $10^9$, true sample variance 30) my run of the naive formula returns 0.0; the Wikipedia article reports -170.67 for the same data, and the exact garbage depends on the order of operations. Welford's update keeps the running mean and $M_2$, the sum of squared deviations from the current mean:

$$
\delta = x - \bar{x}_{n-1}, \quad \bar{x}_n = \bar{x}_{n-1} + \frac{\delta}{n}, \quad M_{2,n} = M_{2,n-1} + \delta\,(x - \bar{x}_n),
$$

with sample variance $M_2 / (n - 1)$. Two summaries combine exactly (Chan and others): with $\delta = \bar{x}_B - \bar{x}_A$, the merged $M_2$ is $M_{2,A} + M_{2,B} + \delta^2 n_A n_B / n$. This is how you compute variance over sharded data or in a distributed job, and it is the same algebra behind batch norm's running statistics and Adam's moment estimates.

```python
import numpy as np

class RunningStats:
    """Welford: one pass, O(1) memory, no catastrophic cancellation."""
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)     # the old delta times the distance to the NEW mean

    @property
    def var(self):                              # sample variance, ddof = 1
        return self.m2 / (self.n - 1) if self.n > 1 else float("nan")

    def merge(self, other):
        """Chan and others: combine two summaries, for sharded or parallel data."""
        out = RunningStats()
        out.n = self.n + other.n
        delta = other.mean - self.mean
        out.mean = self.mean + delta * other.n / out.n
        out.m2 = self.m2 + other.m2 + delta ** 2 * self.n * other.n / out.n
        return out

def naive_var(xs):
    n, s, sq = len(xs), sum(xs), sum(x * x for x in xs)
    return (sq - s * s / n) / (n - 1)

data = [1e9 + 4, 1e9 + 7, 1e9 + 13, 1e9 + 16]   # Wikipedia's example: the true sample variance is 30
rs = RunningStats()
for v in data:
    rs.update(v)
print(naive_var(data), rs.var)

rng = np.random.default_rng(0)
x = rng.normal(loc=50, scale=3, size=10_001)
a, b = RunningStats(), RunningStats()
for v in x[:4000]:
    a.update(v)
for v in x[4000:]:
    b.update(v)
both = a.merge(b)
print(np.isclose(both.mean, x.mean()), np.isclose(both.var, x.var(ddof=1)))
```
```text
0.0 30.0
True True
```

### Moving averages over a stream

A windowed moving average keeps the last w values in a deque and a running total: add the new value, subtract the one that falls out, divide. That is O(1) per update and O(w) memory, against O(w) per update for re summing the window. Over very long streams the running total accumulates rounding error, so recompute it from the buffer every so often. An exponential moving average, $m_t = \alpha x_t + (1 - \alpha) m_{t-1}$, needs O(1) memory, weights the past geometrically, and has an effective window of about $2/\alpha - 1$ points. The tests match `np.convolve`, pandas `rolling` and pandas `ewm(adjust=False)`.

```python
from collections import deque
import numpy as np
import pandas as pd

class MovingAverage:
    """Mean of the last w values of a stream: O(1) per update, O(w) memory."""
    def __init__(self, w):
        self.w, self.buf, self.total = w, deque(), 0.0

    def update(self, x):
        self.buf.append(x)
        self.total += x
        if len(self.buf) > self.w:
            self.total -= self.buf.popleft()      # drop the value leaving the window
        return self.total / len(self.buf)

def ema(xs, alpha):
    """Exponential moving average: O(1) memory, weight alpha on the newest value."""
    out, m = [], None
    for x in xs:
        m = x if m is None else alpha * x + (1 - alpha) * m
        out.append(m)
    return out

rng = np.random.default_rng(0)
xs = rng.normal(size=1000).cumsum()
ma = MovingAverage(7)
mine = np.array([ma.update(v) for v in xs])
print(np.allclose(mine[6:], np.convolve(xs, np.ones(7) / 7, mode="valid")))
print(np.allclose(mine, pd.Series(xs).rolling(7, min_periods=1).mean()))
print(np.allclose(ema(xs, 0.2), pd.Series(xs).ewm(alpha=0.2, adjust=False).mean()))
```
```text
True
True
True
```

## Debugging and AI assisted rounds

### What an ML debugging round looks like

You get code that runs but trains badly, or a model that gives wrong outputs, and you have about an hour. The aced.io guide to the OpenAI research engineer loop describes a transformer of roughly 300 lines with four planted bugs, after which you implement a KV cache. The more general form is a short NumPy or PyTorch training loop with something broken. PracHub's version of a broken transformer question lists where bugs hide: attention internals, positional encodings, label alignment, tensor shapes, gradient flow, optimiser interactions, and train versus eval behaviour. None of these sources are official company documentation, so treat them as candidate reports.

### A method that finds bugs

1. **Reproduce and measure.** Run it, read the loss curve, check the starting loss against $\log C$.
2. **Overfit one tiny batch.** A correct model and training loop drive the loss on eight examples toward zero. If it cannot, the bug is in the model, loss or update, not in the data volume.
3. **Check shapes at every boundary.** Print or assert them; broadcasting bugs never crash.
4. **Check data and labels together.** Look at a few decoded inputs with their labels after every transform.
5. **Bisect the pipeline.** Replace pieces with known good versions (a reference loss, a fixed batch, no augmentation) until the symptom disappears.
6. **Verify the fix against the original symptom**, and say what test would have caught it.

The demonstration plants four classic bugs in a logistic regression loop. Each one runs without an error.

```python
import numpy as np

def make_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 3))
    y = (X @ np.array([2.0, -1.0, 0.5]) + 0.3 * rng.normal(size=400) > 0).astype(float)
    return X, y

def fit(X, y, bug=None, lr=0.5, steps=300):
    """Logistic regression by gradient descent with four classic planted bugs."""
    n, d = X.shape
    if bug == "shuffled_x_only":
        X = X[np.random.default_rng(1).permutation(n)]   # rows of X no longer line up with y
    w = np.zeros((d, 1)) if bug == "column_weights" else np.zeros(d)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y)          # column_weights: p is (n, 1), so p - y is (n, n)
        if bug != "no_mean":
            grad = grad / n
        w = w + lr * grad if bug == "sign_flip" else w - lr * grad
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
    eps = 1e-12
    loss = float(np.mean(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
    acc = float(np.mean((p > 0.5) == y))
    return w.shape, round(loss, 3), round(acc, 3)

X, y = make_data()
for bug in [None, "sign_flip", "no_mean", "shuffled_x_only", "column_weights"]:
    print(f"{str(bug):16s}", fit(X, y, bug))

# sanity check every training loop should pass first: overfit a tiny batch
Xs, ys = X[:8], y[:8]
print("tiny batch loss", fit(Xs, ys, steps=5000)[1])
```
```text
None             ((3,), 0.132, 0.955)
sign_flip        ((3,), 24.566, 0.048)
no_mean          ((3,), 0.179, 0.953)
shuffled_x_only  ((3,), 0.677, 0.573)
column_weights   ((3, 400), 0.681, 0.555)
tiny batch loss 0.004
```

| Bug | Symptom in the run | Why |
| --- | --- | --- |
| Gradient sign flipped | Loss 24.6, accuracy 0.048 | The model climbs the loss, so it learns the inverted rule |
| Gradient not divided by n | Loss 0.179 against 0.132, accuracy almost unchanged | The effective learning rate is 400 times larger; on a convex problem it still trains, so only a baseline comparison reveals it |
| X shuffled without y | Accuracy 0.573, near chance | Input and label pairs are broken, so there is nothing to learn |
| Weights initialised as a column | Weights end with shape (3, 400), accuracy 0.555 | (n, 1) minus (n,) broadcasts to (n, n) and every later step stays wrong |

Transformer specific bugs worth rehearsing: a mask that lets positions see the future (validation loss suspiciously low, generation poor), targets not shifted by one position, softmax over the wrong axis, missing $\sqrt{d_k}$ scaling, heads merged with a reshape that skips the transpose, positional encodings added to the wrong tensor or not at all, dropout left on at evaluation, and a KV cache that recomputes positions or forgets to offset them.

### AI assisted coding rounds

Canva's engineering blog (June 2025) says backend, ML and frontend candidates are expected to use AI tools such as Copilot, Cursor or Claude in technical interviews. The round replaced the old computer science fundamentals screen, and its problems are deliberately too ambiguous for one prompt (their example is a control system for takeoffs and landings at a busy airport). The post says interviewers look for clarifying questions, using AI for well defined subtasks, catching and fixing poor AI output, and full ownership of the final code.

Meta's version started in October 2025, replacing one of the two onsite coding rounds, according to two prep sites. They agree on the outline (a 60 minute CoderPad session with an AI assistant and a multi file codebase you did not write) and disagree on details. Hello Interview describes three phases: fix a bug, implement a core feature of 120 or more lines, then optimise against tiered tests; problems are algorithmic, such as a maze solver. interviewing.io describes more operational tasks such as log parsing, shell scripts and API endpoints, and lists the models offered as including GPT-5, Claude Sonnet 4.5 and Gemini 2.5 Pro. I trust the shared outline more than either set of details; confirm the format with your recruiter.

What both sources and Canva agree on:

- Read the code and state your plan before prompting. The AI is a tool for steps you have already designed.
- Delegate well specified pieces (boilerplate, a known data structure, test scaffolding) with specific prompts that include the schema and the expected behaviour.
- Review every output, run it, and explain it. Code you cannot explain counts against you.
- Keep narrating: what you asked for, what you accepted, what you changed and why.
- Test more, not less. The AI makes writing tests cheap, so there is no excuse for skipping edge cases.

> **Interview trap:** Pasting the whole problem into the assistant. Every source above names this as the most common way to fail: the interviewer is grading your judgment, and a full solution prompt shows none.

## What interviewers probe

These are the follow up chains that come after a working solution. Have an answer ready for each step.

- **Any implementation:** What is the time and space complexity? What happens on empty input, one element, all ties? How would you test it? What changes when the data does not fit in memory?
- **Hash maps:** Why is lookup O(1) on average and when is it O(n)? What can be a dict key, and why must keys be hashable?
- **Heaps:** Why a min heap of size k for the k largest? When is sorting better? How does `heapify` run in linear time?
- **NumPy:** What are the broadcasting rules? Why did this (n, 1) and (n,) operation give a matrix? Is this slice a view or a copy? How much memory does this intermediate take?
- **pandas:** What is the grain of each table? Why did the row count change after the merge? What is the denominator? What changed with Copy on Write?
- **Linear regression:** Derive the normal equations. Why `solve` and not `inv`? What if $X^\top X$ is singular? When is gradient descent better? Why standardise?
- **Logistic regression:** Derive $p - y$. Why not squared error? What happens on separable data? Map your regularisation to scikit-learn's C.
- **k-means:** Is it guaranteed to converge? To the global optimum? What does k-means++ guarantee? What about empty clusters, choosing k, and scaling?
- **kNN:** Cost at training and at prediction time? Effect of k? Why do high dimensions hurt? How do you scale to a billion vectors?
- **Trees:** Gini or entropy, and does it matter? Why sort before sweeping? How do random forests and boosting reuse this?
- **Softmax and cross entropy:** Why subtract the max? Why take logits? What should the initial loss be?
- **Backprop:** Write the backward pass for each layer. How do you know it is right? What relative error is acceptable and why use float64?
- **Attention:** Why divide by $\sqrt{d_k}$? What does the mask do and where does it go? What are the shapes after the head split? What is the complexity in sequence length, and what do FlashAttention, grouped query attention and the KV cache change?
- **BPE:** How are ties broken? How do you encode a new word? Why byte level? What is the training cost?
- **Decoding:** Why does beam search repeat itself? Why length normalise? What does temperature do to the distribution? Top k versus top p?
- **Metrics:** Why is AUC invariant to monotone transforms? What do you do with ties? When is PR AUC better than ROC AUC? Linear or exponential gain in nDCG?
- **Splits:** Why stratify? Where does leakage come from? When must you split by group or by time?
- **Bootstrap:** What does the interval assume? How do you compare two models? What do you resample when rows are not independent?
- **Streaming:** Prove reservoir sampling is uniform. Why is the naive variance unstable? How do you merge two summaries?
- **Debugging:** What do you check first? How do you know the fix worked? What test would have caught it?
- **AI assisted:** What did you delegate and why? How did you verify the generated code? What would you have written by hand?

## Sources

- [Machine Learning Interviews: ML coding chapter (alirezadir)](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLC/ml-coding.md)
- [Machine Learning Interviews: DSA coding chapter (alirezadir)](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/lc-coding.md)
- [Chip Huyen, ML Interviews Book: common interview formats](https://huyenchip.com/ml-interviews-book/contents/2.2.1-common-interview-formats.html)
- [Chip Huyen, ML Interviews Book: algorithms questions](https://huyenchip.com/ml-interviews-book/contents/6.1-algorithms.html)
- [Chip Huyen, ML Interviews Book: complexity and numerical analysis](https://huyenchip.com/ml-interviews-book/contents/6.2-complexity-and-numerical-analysis.html)
- [Hello Interview: Meta's AI enabled coding interview](https://www.hellointerview.com/blog/meta-ai-enabled-coding)
- [interviewing.io: how to use AI in Meta's AI assisted coding interview](https://interviewing.io/blog/how-to-use-ai-in-meta-s-ai-assisted-coding-interview-with-real-prompts-and-examples)
- [Canva engineering: Yes, you can use AI in our interviews](https://canva.dev/blog/engineering/yes-you-can-use-ai-in-our-interviews)
- [aced.io: OpenAI research engineer interview guide](https://www.aced.io/guides/openai-research-engineer-interview-guide)
- [PracHub: debug a broken transformer](https://prachub.com/interview-questions/debug-a-broken-transformer)
- [PracHub: how take home data challenges are graded](https://prachub.com/learning/data-science-projects/how-take-home-data-challenges-are-graded)
- [Tech Interview Handbook: coding interview techniques](https://www.techinterviewhandbook.org/coding-interview-techniques/)
- [Python wiki: TimeComplexity](https://wiki.python.org/moin/TimeComplexity)
- [Python docs: heapq](https://docs.python.org/3/library/heapq.html)
- [Python docs: collections](https://docs.python.org/3/library/collections.html)
- [NumPy docs: broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [pandas 3.0.0 release notes](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html)
- [Stanford CS231n: neural networks part 3, gradient checks](https://cs231n.github.io/neural-networks-3/)
- [scikit-learn docs: ndcg_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- [scikit-learn docs: common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [Wikipedia: algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
- [Wikipedia: reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling)
- [Karpathy, minbpe](https://github.com/karpathy/minbpe)
- [Hugging Face LLM course: byte pair encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
- [Hugging Face blog: how to generate text](https://huggingface.co/blog/how-to-generate)
