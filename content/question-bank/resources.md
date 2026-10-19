# Best resources for each interview goal

This page picks the three to six best resources for each goal: fundamentals fast, LLMs in depth, SQL practice, product cases, system design, mock interviews and staying current. Every web link was fetched during research for this site, with the limits noted in each row (a few pages loaded only in part), and each row says what the resource is best for and how to use it. The video links come from this site's own playlist data. The four lecture playlists come last, with an order to watch them in.

## Fundamentals fast

You need the standard machine learning toolkit in a few weeks, in a form you can say out loud. Start with the questions, then fill the gaps they expose.

| Resource | Best for | How to use it |
| --- | --- | --- |
| [Chip Huyen, Machine Learning Interviews book](https://huyenchip.com/ml-interviews-book/) | Real questions grouped by topic, plus how interviews are run | Answer each question aloud before you read anything else, then turn every miss into a card |
| [Alexey Grigorev, theory questions](https://github.com/alexeygrigorev/data-science-interviews/blob/master/theory.md) | A long checklist of classic ML questions, each marked by difficulty | Use it as a self test: any question you cannot answer in two sentences goes on your review list |
| [Sebastian Raschka, Machine Learning Q and AI](https://sebastianraschka.com/books/ml-q-and-ai-chapters/ch01/) | 30 short chapters that each answer one practical question, such as how to reduce overfitting or what a distribution shift is | Read one chapter a day on the topics you missed above |
| [Stanford CS229 main notes, Spring 2026](https://cs229.stanford.edu/main_notes.pdf) | The derivations: normal equations, logistic loss, generalized linear models, naive Bayes, the SVM, bias and variance, EM (expectation maximization) and PCA (principal component analysis) | Read the chapter after the matching lecture and redo one derivation on paper per week |
| [scikit-learn, common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | The canonical examples of data leakage (information from the test set sneaking into training) and how a Pipeline prevents it | Read it once, then rebuild one of your own projects with a Pipeline and compare the scores |

> **Remember:** Questions first, theory second. Reading a chapter feels productive, but only answering aloud shows what you cannot yet explain.

## LLMs in depth

You want to explain how a language model works from tokens to deployment, and to build small versions yourself.

| Resource | Best for | How to use it |
| --- | --- | --- |
| [CME295 cheatsheet by Afshine and Shervine Amidi](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models) | The official course cheatsheet: tokens, attention, mixture of experts (many specialist sub networks, a few used per token), decoding, the KV cache (stored attention keys and values), prefill and decode (reading the prompt, then writing the answer), speculative decoding (a small model drafts, a large one checks) and FlashAttention (a faster, memory saving way to compute attention) | Skim it after each CME295 lecture, and use it as your last read before a screen |
| [Vaswani et al., Attention Is All You Need](https://arxiv.org/html/1706.03762v7) | The original transformer: scaled dot product attention, sinusoidal positions, the model settings | Read the model architecture section after lecture 1, then rebuild one attention block in NumPy |
| [Karpathy, nanoGPT model.py](https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py) | A complete GPT in readable code: the block layout, layer norm, the MLP and weight tying | Read it next to the Zero to Hero video that builds GPT, then close it and write the model from memory |
| [Chip Huyen, RLHF](https://huyenchip.com/2023/05/02/rlhf.html) | A practitioner walk through of the three phases of RLHF (reinforcement learning from human feedback), the reward model loss and why it might work | Read it before CME295 lecture 5 so preference tuning is not new |
| [Anthropic, Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | The difference between workflows and agents, five workflow patterns and tool design advice | Use the five patterns as your vocabulary when an interviewer asks how you would build an agent |
| [Hamel Husain, Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) | The three levels of evaluation and why generic metrics mislead | Read it before any RAG (retrieval augmented generation) or agent design round, and prepare your own answer to "how do you know it works?" |

## SQL practice

Interviewers test SQL more than any other single skill for data roles, and the best practice is writing queries against real questions with a timer.

| Resource | Best for | How to use it |
| --- | --- | --- |
| [DataLemur questions](https://datalemur.com/questions/click-through-rate) | Real SQL, statistics and ML questions tagged by company, with solutions on most pages | Solve before you read the solution, and time yourself at 15 to 20 minutes |
| [StrataScratch, Uber data scientist questions](https://www.stratascratch.com/blog/uber-data-scientist-interview-questions) | Step by step walkthroughs: assumptions, logic, then code, using aggregation, window functions and CTEs (named temporary queries) | Copy its habit of stating assumptions about the data before writing any SQL |
| [techinterview.org, SQL questions for 2026](https://www.techinterview.org/post/3233460377/sql-interview-questions-2026-window-functions-optimization-and-advanced-queries/) | The question list by seniority, from joins to MVCC (how a database keeps old row versions for concurrent readers) | Use it as a checklist and tick off each question you can answer without notes |
| [techinterview.org, the SQL patterns behind most questions](https://www.techinterview.org/post/3233476262/sql-interview-question-patterns/) | Eight recurring patterns: top N, Nth highest, dedup, running totals, LAG, gaps and islands, pivots, self joins | Write each pattern once on a small table you invent, until you can do it from a blank editor |
| [SQLite, window functions](https://www.sqlite.org/windowfunctions.html) | The exact rules for frames, which are the sets of rows a window covers: ROWS, RANGE and GROUPS | Run the examples in Python's built in sqlite3 module so you can test queries with nothing installed |
| [Use The Index, Luke, date conditions](https://use-the-index-luke.com/sql/where-clause/obfuscation/dates) | Why a function on an indexed column stops the index being used | Read it before any question about slow queries |

## Product cases

Product rounds ask you to define metrics and find causes. You learn them by working cases aloud, not by reading frameworks.

| Resource | Best for | How to use it |
| --- | --- | --- |
| [Product Data Science Interview book, chapter 6](https://bookdown.org/madinterviewllc/Product-Data-Science-Interview-The-Bar-to-Get-Hired/metrics.html) | The four question types and worked cases, including an engagement drop | Pause after each interviewer question in the worked case and answer it yourself first |
| [Mandy Liu, how would you investigate a 20 percent drop](https://mandyliu.substack.com/p/how-would-you-investigate-a-20-drop) | A short framing for metric drops: understand the change, build a funnel, slice the data, recommend. Only the free part of the post loaded when I checked it | Use the four steps as your opening structure on any "metric fell" question |
| [a16z, 16 startup metrics](https://a16z.com/16-startup-metrics/) | Definitions of customer acquisition cost (CAC), lifetime value (LTV), churn variants, cohort retention, gross merchandise value (GMV) versus revenue, and annual recurring revenue (ARR) | Learn each definition with a number, such as a 4 percent monthly churn |
| [Google Research, HEART framework](https://research.google/pubs/measuring-the-user-experience-on-a-large-scale-user-centered-metrics-for-web-applications/) | A way to map a product goal to metrics. Only the abstract loaded when I checked it | Apply it to a product you use: list a goal, a signal and a metric for each part |
| [Amplitude, North Star Playbook](https://amplitude.com/en-us/books/north-star/about-north-star-framework) | The north star metric idea, which links the customer, the product and the business. Only the introduction was read | Pick a north star for three products and defend each choice in 30 seconds |
| [Amplitude, 22 A/B testing interview questions](https://www.amplitude.com/blog/a-b-testing-interview-questions.md) | A question list that spans experiment setup, analysis and team workflows | Answer five a day aloud, and give each answer a number |

## System design

The design round rewards a repeatable framework and a few real systems you can describe in detail. The papers below are the real systems.

| Resource | Best for | How to use it |
| --- | --- | --- |
| [alirezadir, ML system design chapter](https://raw.githubusercontent.com/alirezadir/Machine-Learning-Interviews/main/src/MLSD/ml-system-design.md) | A step by step framework and a large list of sample prompts, including LLM systems marked 2026 | Pick two prompts a week and design each aloud in 45 minutes |
| [Google, Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) | Practical rules: launch a simple baseline first, watch for training and serving skew, log features at serving time | Quote a rule by number when you justify a baseline |
| [Sculley et al., Hidden Technical Debt in ML Systems](https://papers.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) | Why ML systems rot: entangled changes, undeclared consumers, feedback loops | Use it for the "what can go wrong after launch" part of every design |
| [Covington et al., Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) | The two stage recommender: candidate generation, then ranking | Retell it in five minutes as your recommendation example |
| [He et al., Practical Lessons from Predicting Clicks on Ads at Facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf) | An ads click model: feature design, calibration, data freshness and online learning | Retell it as your ads and ranking example |
| [Hello Interview, system design delivery framework](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery) | A phase by phase plan and timings for a 45 minute design round | Use its timings to practice pacing, because running out of time is the most common failure |

## Mock interviews

A mock interview trains delivery, not knowledge. The cheapest mock is the spoken Drill on this site: it gives you a question and a timer, you answer aloud, and then you check your answer against the must mention list. Use a friend or a recording for the rest.

| Resource | Best for | How to use it |
| --- | --- | --- |
| [Tech Interview Handbook, coding interview rubrics](https://www.techinterviewhandbook.org/coding-interview-rubrics/) | The grading dimensions and the communication bands from strong hire to strong no hire | Hand it to a friend and ask them to score you |
| [Tech Interview Handbook, behavioral interview rubrics](https://www.techinterviewhandbook.org/behavioral-interview-rubrics/) | The eight focus areas and how the same story reads as junior, senior or staff | Score your own recorded answers on each area |
| [interviewing.io, stop memorizing STAR](https://interviewing.io/blog/stop-memorizing-star-for-behavioral-interviews-start-selecting-better-stories) | How to choose stories, with a brainstorm and filter process | Build your story list before you rehearse any delivery |
| [interviewing.io, system design interview guide](https://interviewing.io/guides/system-design-interview) | Why a design round is a conversation, and how to justify trade offs | Run a 45 minute mock with a friend as the interviewer, and let them interrupt |
| [Amazon interview guide](https://www.aboutamazon.com/news/workplace/amazon-interview-guide) | Amazon's own advice: STAR, "I" statements, 2 to 3 minute answers, recording yourself | Record yourself and check the length of each answer |
| [Tech Interview Handbook, final questions to ask](https://www.techinterviewhandbook.org/final-questions/) | A bank of closing questions grouped by interviewer type | Pick three that you honestly want answered |

StrataScratch's site menu lists a Mock Interview section, and interviewing.io and Exponent publish the articles above. I read only the articles, so check what live mock services they offer and what they cost.

## Staying current

Model names and best practices move fast, so spend about 15 minutes a day on new material. The Daily digest on this site is built for that: each morning it collects the top Hugging Face daily papers, new posts from 24 blogs, AI stories from Hacker News and Reddit, new videos in the four playlists below, and open roles with direct apply links.

| Resource | Best for | How to use it |
| --- | --- | --- |
| [Anthropic, How we built our multi agent research system (2025)](https://www.anthropic.com/engineering/multi-agent-research-system) | A real multi agent design with measured gains, token costs and evaluation lessons | Read it for numbers you can quote when asked about agent cost |
| [Eugene Yan, Evaluating the Effectiveness of LLM Evaluators (2024)](https://eugeneyan.com/writing/llm-evaluators/) | A survey of papers on LLM judges (models that grade another model's answers), pairwise versus direct scoring, and kappa versus correlation as agreement measures | Use it to answer "can you trust an LLM judge?" |
| [Hamel Husain, Using LLM as a judge](https://hamel.dev/blog/posts/llm-judge/) | How to build a pass or fail judge with a domain expert, including data splits and sample sizes for validation | Follow it on a small dataset of your own |
| [OWASP, Top 10 for LLM Applications (2025)](https://genai.owasp.org/llm-top-10/) | The 2025 list of LLM application risks, from prompt injection to unbounded consumption | Memorize the top five and one defense each |
| [Let's Data Science, 50 LLM and AI engineer questions for 2026](https://letsdatascience.com/blog/50-llm-and-ai-engineer-interview-questions-for-2026) | What current AI engineer interviews ask, sorted by seniority (published March 2026) | Check it each quarter and add any topic you cannot answer |
| [Hello Interview, Meta's AI enabled coding interview](https://www.hellointerview.com/blog/meta-ai-enabled-coding) | How interview formats are changing, with phases and common mistakes | Re-read it before any coding round that allows an assistant |

## The four lecture playlists

This site has timestamped notes and flashcards for each of the four playlists (CS329A is still being added part by part). Every key idea in a note links to the minute of the video where it is taught, and the cards turn those ideas into questions. The Lectures page lists them all, and each module links the lecture that teaches each of its sections. Lectures are numbered here in playlist order, so a few YouTube titles differ (the last three CS229 videos are titled Lecture 16, 18 and 20 on YouTube).

| Playlist | Lectures | Total time | Notes and cards on this site | Best for |
| --- | --- | --- | --- | --- |
| Stanford CME295, Transformers and LLMs, Autumn 2025 | 9 | 16.2 hours | All 9 lectures, 192 cards | How language models work, from the transformer to agents and evaluation |
| Stanford CS229, Machine Learning, Spring 2026 | 17 | 20.6 hours | All 17 lectures, 249 cards | The derivations behind classical machine learning, then neural networks, diffusion, LLMs and RL |
| Stanford CS329A, Self Improving AI Agents | 9 | 10.7 hours | Added part by part, so check the Lectures page for how far it has got | Test time compute, verification and learning from feedback |
| Karpathy, Neural Networks: Zero to Hero | 10 | 19.4 hours | All 10 videos, 226 cards | Building backprop, a GPT and a tokenizer from an empty file |

### Recommended viewing order

Watch in this order. Each step builds on the one before it, and the module column says where to find the matching cards.

| Step | Watch | Hours | Why here | Modules |
| --- | --- | --- | --- | --- |
| 1 | [CS229 Lecture 1](https://www.youtube.com/watch?v=DATnpGoGhM8), [CS229 Lecture 2](https://www.youtube.com/watch?v=cmNIMjPYdgM), [CS229 Lecture 3](https://www.youtube.com/watch?v=uJF_gL3jhxI), [CS229 Lecture 4](https://www.youtube.com/watch?v=8gVi4Rk21Eg), [CS229 Lecture 5](https://www.youtube.com/watch?v=zRdE8A4UZes), [CS229 Lecture 6](https://www.youtube.com/watch?v=llnEgyyuYkQ), [CS229 Lecture 7](https://www.youtube.com/watch?v=fRM41w9jzQo), [CS229 Lecture 8](https://www.youtube.com/watch?v=ne2ngVAoMG8) | 9.2 | The supervised learning foundations: regression, logistic regression, GLMs, bias and variance, neural networks and backpropagation | Classical ML, Deep learning |
| 2 | [Zero to Hero Video 1](https://www.youtube.com/watch?v=VMj-3S1tku0), [Zero to Hero Video 7](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 4.4 | Micrograd turns backpropagation into code you wrote, and "build GPT" shows a whole transformer in one sitting | Deep learning, Transformers and LLMs, ML coding |
| 3 | [CME295 Lecture 1](https://www.youtube.com/watch?v=Ub3GoFaUcds), [CME295 Lecture 2](https://www.youtube.com/watch?v=yT84Y5zCnaA), [CME295 Lecture 3](https://www.youtube.com/watch?v=Q5baLehv5So), [CME295 Lecture 4](https://www.youtube.com/watch?v=VlA_jt_3Qc4), [CME295 Lecture 5](https://www.youtube.com/watch?v=PmW_TMQ3l0I), [CME295 Lecture 6](https://www.youtube.com/watch?v=k5Fh-UgTuCo), [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg), [CME295 Lecture 8](https://www.youtube.com/watch?v=8fNP4N46RRo), [CME295 Lecture 9](https://www.youtube.com/watch?v=Q86qzJ1K1Ss) | 16.2 | The core LLM course, in order: transformer, tricks, decoding, training, tuning, reasoning, agents, evaluation | Transformers and LLMs, Post training and reasoning, RAG, agents and evals |
| 4 | [Zero to Hero Video 8](https://www.youtube.com/watch?v=bZQun8Y4L2A), [Zero to Hero Video 9](https://www.youtube.com/watch?v=zduSFxRajkE) | 2.9 | How assistants are trained, and tokenization, which interviewers love | Transformers and LLMs |
| 5 | [CS229 Lecture 9](https://www.youtube.com/watch?v=bSmIGBCoffA), [CS229 Lecture 10](https://www.youtube.com/watch?v=sUS-eTa0l6s), [CS229 Lecture 11](https://www.youtube.com/watch?v=dqUMCzWjZSI), [CS229 Lecture 12](https://www.youtube.com/watch?v=_kREM2UAiJ8), [CS229 Lecture 13](https://www.youtube.com/watch?v=lNTajqxxOn4), [CS229 Lecture 14](https://www.youtube.com/watch?v=pwQ0l4hFCVI), [CS229 Lecture 15](https://www.youtube.com/watch?v=hHC-SF3utxg), [CS229 Lecture 16](https://www.youtube.com/watch?v=xveNBYVTrqw), [CS229 Lecture 17](https://www.youtube.com/watch?v=J7CossjMvEg) | 11.4 | Unsupervised learning, diffusion, language models and reinforcement learning, with the derivations | Classical ML, Post training and reasoning |
| 6 | [CS329A Part 1](https://www.youtube.com/watch?v=6YnLB0XbTnI), [CS329A Part 2](https://www.youtube.com/watch?v=-Ggc37xLj_Y), [CS329A Part 3](https://www.youtube.com/watch?v=p7TdPUcPoik), [CS329A Part 4](https://www.youtube.com/watch?v=Lxh9RF5S-K0), then [CS329A Part 5](https://www.youtube.com/watch?v=Ml_fp9XkB8Y), [CS329A Part 6](https://www.youtube.com/watch?v=yVnmHSAy3ck), [CS329A Part 7](https://www.youtube.com/watch?v=Uni9dqyuuDM), [CS329A Part 8](https://www.youtube.com/watch?v=8JAqLnTaZu4), [CS329A Part 9](https://www.youtube.com/watch?v=AyO6wyu4DEg) | 4.6 then 6.1 | Newer material on agents, test time compute and verification. Notes and cards are still being added for the later parts | Post training and reasoning, RAG, agents and evals |
| 7 | [Zero to Hero Video 2](https://www.youtube.com/watch?v=PaCmpygFfXo), [Zero to Hero Video 3](https://www.youtube.com/watch?v=TCH_1BHY58I), [Zero to Hero Video 4](https://www.youtube.com/watch?v=P6sfmUTpUmc), [Zero to Hero Video 5](https://www.youtube.com/watch?v=q8SA3rM6ckI), [Zero to Hero Video 6](https://www.youtube.com/watch?v=t3YJ5hKiMQ0), then [Zero to Hero Video 10](https://www.youtube.com/watch?v=l8pRSuU81PU) | 8.0 then 4.0 | The slow, deep version of the language model material, for when you have time | Deep learning, ML coding |

If you have only about 19 hours, watch [CS229 Lecture 2](https://www.youtube.com/watch?v=cmNIMjPYdgM), [CS229 Lecture 6](https://www.youtube.com/watch?v=llnEgyyuYkQ), [CS229 Lecture 8](https://www.youtube.com/watch?v=ne2ngVAoMG8), [Zero to Hero Video 1](https://www.youtube.com/watch?v=VMj-3S1tku0), [Zero to Hero Video 7](https://www.youtube.com/watch?v=kCc8FmEb1nY), and [CME295 Lecture 1](https://www.youtube.com/watch?v=Ub3GoFaUcds), [CME295 Lecture 3](https://www.youtube.com/watch?v=Q5baLehv5So), [CME295 Lecture 5](https://www.youtube.com/watch?v=PmW_TMQ3l0I), [CME295 Lecture 6](https://www.youtube.com/watch?v=k5Fh-UgTuCo), [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg), [CME295 Lecture 8](https://www.youtube.com/watch?v=8fNP4N46RRo). That set covers the questions asked most often.

> **Watch:** [CME295 Lecture 1](https://www.youtube.com/watch?v=Ub3GoFaUcds) is the best single place to start for LLM interviews. It builds attention and the transformer from the ground up.

> **Remember:** Do not watch passively. Pause after each section, say what you just learned out loud, and then open the matching cards on the Learn page.
