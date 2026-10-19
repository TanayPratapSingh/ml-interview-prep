# Interview loops for ML, AI and data roles in 2026

An interview loop is the full run of interviews a company puts you through: a recruiter call, one or two screens, then a set of back to back rounds called the onsite (often held on video now). This guide says what each round tests, how loops differ by role and by company size, which rounds are new since 2025, and how to prepare for each one. Each source is named next to what it says. Companies rarely publish their loops, so treat each description as a pattern to confirm with your recruiter, not a promise.

## How far to trust these descriptions

Three kinds of source sit behind this guide, and they deserve different amounts of trust.

- **Company posts.** The [Canva engineering blog](https://canva.dev/blog/engineering/yes-you-can-use-ai-in-our-interviews) and [Amazon's interview guide](https://www.aboutamazon.com/news/workplace/amazon-interview-guide) are official, but each speaks only for that company.
- **Prep sites and candidate write ups.** [Hello Interview](https://www.hellointerview.com/blog/meta-ai-enabled-coding), [interviewing.io](https://interviewing.io/blog/how-to-use-ai-in-meta-s-ai-assisted-coding-interview-with-real-prompts-and-examples), [aced.io](https://www.aced.io/guides/openai-research-engineer-interview-guide) and [PracHub](https://prachub.com/learning/data-science-projects/how-take-home-data-challenges-are-graded) repeat what candidates reported. They sometimes disagree, and the text says so when they do.
- **Books and repositories.** [Chip Huyen's ML interviews book](https://huyenchip.com/ml-interviews-book/) and the [alirezadir Machine Learning Interviews repository](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLC/ml-coding.md) describe patterns seen across many companies. Most of that material predates 2025.

Where I combine sources into a table or a comparison, the text says it is my synthesis. For the details of each round, the matching module on this site goes deeper: [ML coding](#/module/ml-coding?d=deep), [ML system design](#/module/ml-system-design?d=deep), [Product analytics](#/module/product-analytics?d=deep), [SQL](#/module/sql-analytics?d=deep) and [Narration](#/module/narration?d=deep).

## Terms used in this guide

Short definitions, so nothing below needs a second search.

- **Rubric.** A scoring guide an interviewer fills in after a round.
- **Take home.** An assignment you do on your own time and send back, instead of a live round.
- **Grain.** What one row of a table stands for, such as one order, or one user on one day.
- **Window function.** A calculation across related rows that keeps every row, such as a running total or a rank within a group.
- **CTE.** A common table expression: a named temporary query that you write with `WITH`.
- **Cohort retention.** The share of a group of users, such as everyone who signed up in January, who come back in later months.
- **Gaps and islands.** A family of SQL problems about finding unbroken runs of consecutive days or numbers.
- **MVCC.** Multi version concurrency control: how a database lets readers and writers work at the same time by keeping old versions of rows.
- **A/B test.** An experiment that randomly splits users between two versions and compares a metric.
- **p value.** The chance of seeing data at least this extreme if nothing real is going on.
- **Confidence interval.** A range of values that is likely to contain the true number.
- **Transformer and attention.** The network design behind modern language models. Attention lets each word look at the other words in the text and weigh them.
- **Token.** A piece of text, often a word or part of a word, that a language model reads as one unit.
- **Byte pair encoding.** A way to split text into tokens by repeatedly merging the most common pair of symbols.
- **KV cache.** Saved attention keys and values for the words already processed, so the model does not recompute them for each new word.
- **LoRA.** A cheap way to fine tune a large model by training small extra matrices instead of all its weights.
- **DPO.** Direct preference optimization: tuning a model from pairs of a better and a worse answer.
- **RAG.** Retrieval augmented generation: the system fetches relevant documents and hands them to the model before it answers.
- **Evals.** Tests that measure how good a model or system is.
- **ROC AUC.** A score between 0 and 1 for how well a classifier ranks true positives above negatives.
- **PCA.** Principal component analysis: compressing many features into a few that keep most of the variation.
- **Reservoir sampling.** A way to pick a fair random sample from a stream of unknown length.
- **Idempotent.** Safe to repeat: doing it twice has the same effect as doing it once.
- **STAR.** Situation, task, action, result: a four part shape for telling a story.

## The stages of a loop

Most loops follow the same order. This is a common pattern, not a source backed rule.

1. **Recruiter call.** A short chat about your background, the role, your timing and your other processes. Chip Huyen's book says every company asks whether you have an offer or a deadline that they should know about.
2. **Screen.** One or two shorter technical rounds, often a coding or SQL exercise on video. The book describes a coding challenge as one to three questions, taking 30 minutes to 3 hours, mostly for new graduates and interns.
3. **Onsite loop.** Several rounds in a row, each with a different interviewer and a different skill to test. The word "onsite" is still used when the rounds are on video.
4. **Team match and offer.** Some companies hire into a pool and then match you to a team.

Ask the recruiter for the round list, the length of each round, the languages and libraries allowed, and whether your code will be run or only read. That one email removes most of the surprises.

## The rounds at a glance

This table is my synthesis of the sources named in the sections below. The roles column shows who meets the round most often, and titles vary by company.

| Round | Who meets it | What happens | What is graded |
| --- | --- | --- | --- |
| Background and project talk | Everyone | You tell your story and defend your resume | Clarity, honesty, ownership |
| SQL | Data scientist, analyst, some ML engineer | Queries on a small schema | Correct logic, grain, window functions |
| Python and pandas | Data scientist, analyst | Clean and summarize a small table | Correct denominators, checking your output |
| Statistics and probability | Data scientist, research | Puzzles, p values, intervals, Bayes | Correct reasoning said out loud |
| Experiment design | Data scientist, product analyst | Design or critique an A/B test | Sound design, awareness of pitfalls |
| Product sense and metric cases | Product data scientist, analyst | Define a metric or diagnose a drop | Structure, judgment, a recommendation |
| ML concepts | Every ML role | Explain how methods work and fail | Depth under follow up questions |
| ML coding from scratch | ML engineer, research engineer | Implement a model or metric in NumPy | Correct math, tests, numerical care |
| Algorithms | ML engineer, AI engineer at larger companies | LeetCode style problems | Optimal solution, complexity |
| ML debugging | Research engineer, ML engineer at labs | Fix code that runs but trains badly | Method, transformer knowledge |
| ML system design | ML engineer, AI engineer, senior data scientist | Open ended design conversation | Framing, data, metrics, operations |
| LLM application design | AI engineer | Design a RAG or agent system | Retrieval, evals, safety, cost |
| Take home | Data scientist, analyst, some ML engineer | A dataset and a few days | Right answer, readable write up |
| AI assisted coding | Software and ML engineer at some companies | Code with an assistant in your editor | Judgment about what to delegate |
| Behavioral | Everyone | Stories about past work | Communication, scope, ownership |

## Background and project talk

This round opens most loops and often returns later with a different interviewer. Chip Huyen's book lists what it covers: "Tell me about yourself" (answer in under a minute, personal and professional), your level of involvement with areas such as computer vision or language, your career history with the decisions you made at each job, and why you are looking now.

The part that decides the round is the project deep dive. The book says an interviewer may take one project from your resume and make you explain every choice in it, which is how they tell people who talk about work from people who did it. A tool listed on your resume is fair game too: the book's example is TensorFlow, where you should be ready to discuss eager execution, estimators and distributed training.

> **Interview trap:** Claiming more experience than you have. The book is blunt that interviewers find out, and one overclaim makes them doubt everything else you said.

**How to prepare.** Write a one minute self introduction, then explain each project you list at three lengths: 15 seconds, 60 seconds and five minutes. Know the numbers, the baseline, what broke and what you would change. The [Narration module](#/module/narration?d=deep) and the spoken Drill on this site exist for this.

## SQL

SQL is the round with the clearest public question lists. A [techinterview.org guide from April 2026](https://www.techinterview.org/post/3233460377/sql-interview-questions-2026-window-functions-optimization-and-advanced-queries/) sorts questions by seniority:

- **Junior, must know cold:** the four join types, `WHERE` versus `HAVING`, finding duplicates, the second highest salary.
- **Senior, expected:** how an index works, when a window function beats a subquery, reading a slow query plan with `EXPLAIN ANALYZE`, a cohort retention query, `UNION` versus `UNION ALL`.
- **Staff, impressive:** how PostgreSQL keeps old row versions for concurrent readers (called MVCC, multi version concurrency control), why a query ignores an index that exists, a schema for a multi tenant product, and gaps and islands streak queries.

StrataScratch's [Uber data scientist page](https://www.stratascratch.com/blog/uber-data-scientist-interview-questions) lists the concepts its practice questions test: aggregation and sorting, subqueries, `LAG` and `LEAD`, window functions and CTEs. A **window function** runs a calculation across related rows while keeping every row. A **CTE** (common table expression) is a named temporary query you write with `WITH`. DataLemur tags real questions by company, such as a click through rate query from Facebook and a repeated payments query from Stripe, and notes which problems match the SQL, statistics and machine learning chapters of the book Ace the Data Science Interview, for example the [App Click through Rate question](https://datalemur.com/questions/click-through-rate).

**How to prepare.** Write each query by hand, say the grain of every table (what one row means) before you join, and check the row count after each join. Work through the [SQL module](#/module/sql-analytics?d=deep), then filter the Questions page by kind "sql" and time yourself at 15 to 20 minutes per question.

## Python and pandas

The data science screen is the one closest to daily work: a small table, a business question, and 20 to 40 minutes, according to this site's [ML coding guide](#/module/ml-coding?d=deep). Typical asks are a group by with several aggregates, a merge followed by a count, "the previous event per user" and a rolling average. Interviewers care less about exact function names than about whether you checked the grain before joining, noticed that a left join multiplied rows, and can state the denominator of a rate.

**How to prepare.** Do ten small pandas problems on your own data, each ending with a sanity check such as "the counts add up to the total".

## Statistics and probability

[StrataScratch's list of 30 questions](https://www.stratascratch.com/blog/30-probability-and-statistics-interview-questions-for-data-scientists) shows the range, and it names the companies it ties to each question, including Facebook, Google, Amazon, Microsoft and Jane Street. Probability questions use cards, dice, coins and expected value. Statistics questions cover mean versus median, the definition of variance, how to explain a p value to an engineer, confidence intervals and sample sizes, checking that A/B assignment was random, Bayes' theorem and the frequentist versus Bayesian split. DataLemur adds puzzles such as the expected number of rolls until two 5s in a row.

> **Say it like this:** "A p value is the chance of seeing data at least this extreme if the null hypothesis were true. It is not the chance that the null is true."

**How to prepare.** Compute every puzzle in code after you solve it on paper, so you catch arithmetic slips. Then practice saying the answer in two sentences. The [Probability and statistics module](#/module/probability-statistics?d=deep) covers the standard set.

## Experiment design

Product companies test whether you can run an A/B test (a randomized comparison of two versions) without fooling yourself. [Amplitude's list of 22 A/B testing questions](https://www.amplitude.com/blog/a-b-testing-interview-questions.md), published in June 2022, groups them as: design and setup (ideal conditions, what to test, how long to run), tools, fixing problems (small samples, interference between groups), common scenarios, analysis and decisions (an inconclusive result, network effects, two metrics that disagree), and team workflows. Several questions are about the interviewer's own product, so using it before the interview helps.

**How to prepare.** Be able to do a sample size calculation by hand, name three ways a test goes wrong (peeking, interference, a bad assignment) and say what you would do about each. See the [Experiments and causal inference module](#/module/experimentation-causal?d=deep).

## Product sense and metric cases

[The Product Data Science Interview book, chapter 6](https://bookdown.org/madinterviewllc/Product-Data-Science-Interview-The-Bar-to-Get-Hired/metrics.html) lists four question types: measure something for the first time, find the root cause of a recent metric change, estimate a counterfactual impact (what would have happened without the change), and weigh a trade off. Its worked cases include a sudden engagement drop on a Q and A forum and measuring the impact of a filter that removes stale models in an ads system. In the engagement case the interviewer answers questions about how big the drop is, whether it is the same across all users, and which part of engagement fell.

**How to prepare.** Use the same five moves on every open question: restate and clarify, announce a structure out loud, work the buckets in priority order with rough numbers, give the answer first with evidence, and finish with a recommendation and what would change your mind. This is the method in the [Product analytics module](#/module/product-analytics?d=deep), which is a synthesis and not an official rubric.

## ML concepts

This is the theory round for every ML role. Alexey Grigorev's [theory question list](https://github.com/alexeygrigorev/data-science-interviews/blob/master/theory.md) runs through linear regression, validation, classification, regularization, trees, random forests, boosting, neural networks, optimization, vision, text, clustering, dimensionality reduction, ranking, recommenders and time series. Chip Huyen's book and Sebastian Raschka's [Machine Learning Q and AI](https://sebastianraschka.com/books/ml-q-and-ai-chapters/ch05/) cover the same ground with more depth on training and deployment.

The pattern is a definition first, then follow ups that push on why it works, when it fails and what you would do instead. For example: "What is L1 regularization?" then "Why does it give sparse weights?" then "When would you not use it?"

**How to prepare.** Work the [Classical ML](#/module/classical-ml?d=deep) and [Deep learning](#/module/deep-learning?d=deep) cards in order, and answer each one out loud before you flip it.

## ML coding from scratch

Here you implement a model, a layer or a metric, usually in NumPy and often without scikit-learn. The [alirezadir ML coding chapter](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLC/ml-coding.md) groups its practice problems into four families:

- **Classic ML:** a stable softmax and cross entropy, linear and logistic regression with gradient descent, k-nearest neighbors, k-means, a decision tree split, PCA, a 2D convolution, ROC AUC and reservoir sampling.
- **Language models:** TF IDF, scaled dot product attention, byte pair encoding, a causal mask, positional encoding, top-k and top-p sampling and a KV cache.
- **Generative AI:** a LoRA update, a contrastive loss, cosine top-k retrieval for RAG, the DPO loss and classifier free guidance.
- **Agentic coding:** validating a tool call's JSON, a permission aware tool registry, retries that are safe to repeat (idempotent), conversation memory and a bounded tool loop.

The chapter suggests about 15 minutes for an easy problem, 20 to 35 for a medium one and 40 to 60 or more for a hard one.

**How to prepare.** Implement the classic set from an empty file, then test each against scikit-learn on random data. Say what you are doing as you type. The [ML coding module](#/module/ml-coding?d=deep) has working versions and the follow up questions that come after.

## Algorithm rounds

MLE and AI engineer loops at larger companies still include one or two general algorithm rounds, per this site's ML coding guide. The [alirezadir data structures and algorithms chapter](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/lc-coding.md) lists fourteen pattern families, such as hashing, two pointers, sliding windows, heaps and graphs. For an ML candidate the highest value subset is hash maps, two pointers, sliding windows, heaps for top k, binary search, intervals, graph search and simple dynamic programming.

**How to prepare.** Practice patterns, not individual problems. After each solution, state the time and space cost and two edge cases.

## ML debugging

You receive code that runs but trains badly, and you have about an hour. The [aced.io guide to the OpenAI research engineer loop](https://www.aced.io/guides/openai-research-engineer-interview-guide) describes a transformer of about 300 lines with four planted bugs, followed by implementing a KV cache. A [PracHub question on a broken transformer](https://prachub.com/interview-questions/debug-a-broken-transformer) lists where bugs tend to hide: attention internals, positional encodings, label alignment, tensor shapes, gradient flow and the difference between training and evaluation mode. Both are candidate reports, not company documents.

**How to prepare.** Learn a fixed method: check the data, overfit one tiny batch first (a correct model drives the loss toward zero on eight examples), then check shapes, masks, learning rate and mode switches. The ML coding module walks through four planted bugs.

## ML system design

The prompt is one line, such as "design a feed ranking system", and you spend 45 to 60 minutes turning it into a plan while the interviewer answers your questions. Chip Huyen reports that almost every company she spoke with asks at least one such question. The [alirezadir system design chapter](https://raw.githubusercontent.com/alirezadir/Machine-Learning-Interviews/main/src/MLSD/ml-system-design.md) sorts its sample prompts into groups: LLM systems (marked 2026), recommendation, search, ranking, language, vision, autonomous driving and others such as fraud and harmful content detection.

Interviewers listen for problem framing, data and labels, metrics, a justified model choice, serving and scale, operations (experiments, rollout, monitoring, retraining) and communication. That list is a synthesis of several write ups, not a published rubric. Missing a whole area costs more than being shallow in one.

> **Interview trap:** Drawing a neural network in the first minute. Start with the goal, the scale, the latency limit and the data, then propose a simple baseline.

**How to prepare.** Practice the eleven step framework in the [ML system design module](#/module/ml-system-design?d=deep) with a timer, and do five designs aloud, covering recommendation, search, ads click prediction, fraud and an LLM assistant.

## LLM application design

AI engineer loops add questions about building with language models. The [Let's Data Science list of 50 questions for 2026](https://letsdatascience.com/blog/50-llm-and-ai-engineer-interview-questions-for-2026), published in March 2026, sorts them by seniority. Architecture and training questions suit all levels. Retrieval (RAG, retrieval augmented generation), fine tuning and production questions are aimed at mid level and above, and agent questions at senior level and above. Safety questions suit all levels at AI labs and senior candidates at startups.

Design prompts follow the same pattern: a RAG question answering system, a support chatbot with guardrails and a human fallback, an agent that plans and uses tools, search with cited LLM answers, and a coding agent that verifies its work, all from the alirezadir list.

**How to prepare.** Be ready to say how you would measure a RAG system, handle a tool call that fails, defend against prompt injection and cut cost. The [RAG, agents and evals module](#/module/rag-agents-evals?d=deep) covers each, and CME295 lectures 7 and 8 teach the same topics on video.

## Behavioral

A behavioral round asks for stories about past work. Amazon says it judges every hiring decision against its 16 Leadership Principles and asks for answers in the STAR shape (situation, task, action, result) in the first person, usually 2 to 3 minutes each. [Exponent's Amazon guide](https://blog.tryexponent.com/how-to-nail-amazons-behavioral-interview-questions/), built from candidate reports, describes a 60 minute round that covers two principles, about 25 minutes each. That leaves most of the time for follow ups, not the first telling.

The [Tech Interview Handbook rubric](https://www.techinterviewhandbook.org/behavioral-interview-rubrics/) names eight focus areas: motivation, proactivity, working in an unstructured environment, perseverance, conflict resolution, empathy, growth and communication. Interviewers also judge scope, meaning whether your story moved only you, your team or an organization. An [interviewing.io essay](https://interviewing.io/blog/stop-memorizing-star-for-behavioral-interviews-start-selecting-better-stories) argues for choosing stories well over memorizing the format.

**How to prepare.** Write six stories with real numbers: a disagreement, a failure, a deadline, a time you led without authority, a time you changed your mind and a project you are proud of. Rehearse each in 2 minutes and again in 30 seconds.

## Take home assignments

A take home gives you a dataset and a question, and usually a few days. Chip Huyen's book treats a coding challenge as a shorter version aimed at new graduates. A [PracHub grading guide](https://prachub.com/learning/data-science-projects/how-take-home-data-challenges-are-graded) reports that reviewers read the write up before the notebook, look for a number and a decision in the first paragraph, treat analysis errors as a gate (a mistake there fails you regardless of polish), and use craft as a tiebreaker. Craft means a notebook that runs from top to bottom and assumptions that are written down.

> **Remember:** Put the answer in the first paragraph and make the notebook run from a fresh kernel.

**How to prepare.** Keep the scope small, spend your time on checking the data and the result, and say what you did not do and why. Ask in advance how much time they expect you to spend and stay near it.

## Rounds that allow an AI assistant

Two public examples changed the picture in 2025.

- **Canva.** The company's June 2025 post says backend, ML and frontend candidates are expected to use AI tools such as Copilot, Cursor or Claude in technical interviews. The round replaced its computer science fundamentals screen, and the problems are deliberately too vague to solve with one prompt. Interviewers look for clarifying questions, AI used for well defined subtasks, poor AI output being caught and fixed, and full ownership of the final code.
- **Meta.** According to two prep sites, an AI enabled coding round began in October 2025 and replaces one of the two onsite coding rounds. Both describe a 60 minute CoderPad session with an assistant and a multi file codebase you did not write. [Hello Interview](https://www.hellointerview.com/blog/meta-ai-enabled-coding) describes three phases: fix a bug, build a feature of 120 or more lines, then optimize against tiered tests. [interviewing.io](https://interviewing.io/blog/how-to-use-ai-in-meta-s-ai-assisted-coding-interview-with-real-prompts-and-examples) describes more operational tasks, such as log parsing, shell scripts and API endpoints, and lists GPT-5, Claude Sonnet 4.5 and Gemini 2.5 Pro among the models offered. I trust the shared outline more than either set of details, so confirm the format with the recruiter.

I found no source describing an AI assisted round for data science or analyst roles. Ask whether yours allows one.

> **Interview trap:** Pasting the whole problem into the assistant. A full solution prompt shows no judgment, and judgment is what the round grades.

**How to prepare.**

- Read the code and say your plan before you prompt.
- Delegate small, well specified jobs, and include the schema and the expected behavior in the prompt.
- Run and read everything the assistant writes, and be ready to explain every line.
- Keep narrating what you asked for, what you kept and what you changed.
- Write more tests than usual, because the assistant makes tests cheap.

## Loops by role

This section is my synthesis. Each role lists its likely rounds, drawn from the sources above.

### Machine learning engineer

Expect ML concepts, ML coding from scratch, one or two algorithm rounds, ML system design and a project talk, with behavioral questions near the end. Larger companies standardize this mix. Labs may swap an algorithm round for ML debugging.

### AI engineer

Expect LLM application design, RAG and agent questions, an ML or software coding round, a system design round that centers on an LLM product, and a project talk. Evals (how you measure quality) come up in nearly every round, so prepare a clear answer for how you would know your system works.

### Research engineer

Expect ML coding and debugging rounds, deep questions on transformers and training, and a paper or project discussion. The only loop description I found is one prep guide for one lab ([aced.io](https://www.aced.io/guides/openai-research-engineer-interview-guide)), so treat the shape as uncertain and ask the recruiter for specifics.

### Data scientist, product

Expect SQL, statistics and probability, experiment design, a metric or product case, and behavioral questions. Some loops add a Python or pandas screen or a take home. The product data science book lists the four case types above.

### Data scientist, machine learning

Expect a mix of the product loop and the ML engineer loop: SQL or pandas, statistics, ML concepts, a modeling case, and often a lighter system design round.

### Data analyst and product analyst

Expect SQL as the main round, a metric or product case, a dashboard or business intelligence question, and a stakeholder style behavioral round. Analysts are tested on explaining a number to a non technical listener as much as on computing it. The SQL and product analytics modules on this site map to this loop.

## Startups versus large companies

No source I could fetch compares the two, so this section is my reading of the patterns above, not data.

| Aspect | Large company | Startup |
| --- | --- | --- |
| Loop shape | Standard rounds, written rubrics by level | Fewer rounds, shape depends on who is hiring |
| Question source | Large public pools, often tagged by company | Often tied to the company's own product and data |
| Take homes | Less common for engineers, common for some analyst roles | More common, often a small real problem |
| Behavioral weight | Heavy at some companies (Amazon) | Lighter, mixed in with a project talk |
| Breadth | One skill per round | One person may test several skills |

One public example of product flavored questions: Amplitude's list includes "from your experience of using our product, what would you improve?" and a question on learning its own experimentation tool.

**How to prepare.** For large companies, drill the standard rounds and the story bank. For startups, read the product, use it, and bring one idea you would test or build.

## Questions to ask your recruiter

- Which rounds will I have, and how long is each?
- Can I use my own editor, libraries or an AI assistant in the coding rounds?
- Will code be run, or only read?
- Is there a take home, and how many hours do you expect it to take?
- Which round covers machine learning concepts, and which covers system design?
- Will the loop test SQL, and which database dialect?
