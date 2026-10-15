# RAG, Agents and Evals: the plain version

This is the easy version of three ideas that sit behind most AI products. Retrieval augmented generation (RAG) lets a language model look things up before it answers. Agents let a model take actions with tools, step by step, until a task is done. Evals are the tests that tell you whether either one works. Each section gives the intuition, an everyday analogy, one example with real numbers, and where the idea breaks.

## RAG versus fine tuning versus long context

A language model is like a well read person answering from memory. RAG hands them the relevant pages before they answer. Fine tuning is sending them on a training course that changes their habits. Long context is giving them the whole binder and hoping they find the right page.

Use RAG for facts, especially facts that change or that you must cite. Use fine tuning for behaviour: a format, a tone, a skill. Use long context when everything fits and you can afford to send it every time.

> **Worked example:** the original RAG paper (2020) built one index from Wikipedia as of 2016 and another from 2018. Asked about world leaders, the model got about 70 percent right when the index matched the year and only 4 to 12 percent when it did not. Swapping the index updated the model with no retraining.

Where it breaks: models read long inputs unevenly. A 2023 study found they miss facts placed in the middle of a long prompt more than facts at the start or end, so "just send everything" is not free.

## The RAG pipeline end to end

Think of a library. Before anyone asks a question, librarians collect books, clean them up, cut them into index cards, label each card, and file the cards two ways: by meaning and by keywords. When a question arrives, a clerk rephrases it, pulls a stack of maybe 100 cards, sorts the best 10 to the top, and hands them to a writer who answers and points to the cards used. If the cards do not contain the answer, the writer should say so.

Every step can fail on its own: a table turned into gibberish when the PDF was read, a card cut in the wrong place, the right card filed but never pulled. That is why you test each step separately.

> **Remember:** the writer can only use the cards it is given. If retrieval misses the answer, nothing downstream can fix it.

## Chunking strategies and trade offs

A chunk is one index card. Small cards describe one idea sharply, so a question about that idea finds them, but a card that says "revenue grew 3 percent" without naming the company is useless on its own. Big cards keep the context but blur several topics together, so they match every question weakly and cost more to read.

Common ways to cut: every few hundred words, by sentence or paragraph, by the document's own headings, or by detecting where the topic shifts. A neat trick from Anthropic (2024) adds a short note to each card saying where it came from, like "From ACME's Q2 2023 report, revenue section". That cut their retrieval misses from 5.7 percent to 3.7 percent, and to 1.9 percent with the other steps below.

> **Worked example:** a 1,000 word document cut into 256 word cards with 64 words of overlap gives 5 cards; with no overlap it gives 4. Overlap protects answers that sit on a boundary but makes the index bigger.

## Embeddings and bi encoders

An embedding turns text into a list of numbers, a point on a map where texts with similar meaning land close together. A bi encoder makes one point for the question and one for each document separately, so all the document points can be made in advance. Searching is then just finding the nearest points.

The more careful alternative, a cross encoder, reads the question and a document together, like a person comparing them side by side. It judges better but must read every pair fresh, so it is too slow for millions of documents.

Where it breaks: a map of meaning is fuzzy about exact things such as product codes, numbers, names and "not". A model trained on one kind of text can also do badly on another: a 2021 benchmark called BEIR found a well known dense retriever scored about 48 percent below plain keyword search on average across 18 unfamiliar datasets.

## Training retrievers contrastively

You teach a retriever like a matching game. Show it a question, the right passage, and some wrong ones, and reward it for placing the right one closest. A cheap trick: in a batch of 128 question and answer pairs, every other question's answer is a free wrong option, so each question gets 127 of them at no extra cost.

The wrong options that teach the most are near misses, passages that share the question's words but do not answer it. In the DPR paper (2020), adding one such near miss per question raised the share of questions with the answer in the top 5 results from 55.8 to 65.8 percent.

Where it breaks: sometimes a "wrong" option is actually right but nobody labelled it, and the model learns to push good answers away.

## Vector indexes: flat, IVF, HNSW, PQ

Finding the nearest point among millions by checking every one (flat search) is exact but slow. The shortcuts:

- **IVF** sorts points into neighbourhoods and only searches the few neighbourhoods closest to the question.
- **HNSW** builds a layered road network: highways on top for long jumps, local streets at the bottom, and the search drives down from highway to street.
- **PQ** compresses each point into a short code, like storing a postcode instead of a full address.

> **Worked example:** 10 million vectors of 768 numbers take 30.7 GB stored exactly. Compressed to 96 byte codes they take about 1 GB. In a small test, searching 1 neighbourhood out of 64 found only 30 percent of the true nearest neighbours; searching 16 found 87 percent while still skipping three quarters of the data.

Where it breaks: speed costs some accuracy, the right settings depend on your data, and some indexes (HNSW in the FAISS library) cannot delete items without a rebuild.

## Sparse retrieval and BM25

BM25 is classic keyword search with three sensible rules. Rare words count more than common ones. Repeating a word helps, but less each time. Long documents are discounted because they contain more words by chance.

> **Worked example:** with typical settings, a word appearing once in an average length document scores 1.0 (times its rarity weight), twice scores 1.375, ten times 1.96, and it can never pass 2.2. A 13 word document mentioning "refund policy" scored 0.78 against 1.19 for a 5 word one with the same words.

BM25 needs no training and is excellent at exact names, codes and error messages. Its blind spot is different words for the same thing: "cancel my subscription" does not match "terminate plan".

## Hybrid search and reciprocal rank fusion

Keyword search and meaning search make different mistakes, so good systems run both and merge the lists. Their scores are on different scales, so the usual merge ignores scores and uses positions: each document earns 1 divided by (60 plus its rank) from every list it appears in, and the totals decide the order. This is reciprocal rank fusion, from a 2009 paper that picked 60 in a pilot study.

> **Worked example:** document A is first in the keyword list and missing from the other; document B is third and second. A earns 1/61 = 0.0164. B earns 1/63 + 1/62 = 0.0320 and wins. Agreement between methods beats one method's enthusiasm.

Where it breaks: if one of the lists is poor, it adds noise rather than help.

## Cross encoder reranking

Reranking is a second, slower look at a short list. Pull 100 to 150 candidates cheaply, let a cross encoder read each one with the question and rescore it, and keep the best 5 to 20. In the BEIR benchmark this was the most accurate approach on average, about 11 percent better than keyword search alone, but it took about 450 milliseconds per question against 20 for keyword search.

Where it breaks: a reranker can only reorder what it was given. If the first search missed the right passage, reranking cannot find it.

## Query rewriting, HyDE and multi hop retrieval

People ask questions badly for search. "What about the annual one?" means nothing without the chat before it, so systems rewrite it into a full query, split compound questions apart, or try several phrasings and merge the results.

HyDE is a clever variant: ask a model to write a fake answer first, then search for real documents that look like that fake answer. Answers resemble documents more than questions do.

Multi hop questions need a chain: "Where was the author of Things Fall Apart born?" needs the author first, then the birthplace. The system searches, reads, then searches again with what it learned. Every extra hop is another chance to go wrong.

## Metadata filters, freshness and deletion

Labels on each card let you filter: only this customer's documents, only HR policies, only things after 2024. Permission filters must happen in the search itself. Telling the model "ignore documents this user cannot see" after showing them to it is a leak.

> **Worked example:** if a filter matches 1 percent of documents and you filter after fetching the top 100, you expect about one result left. Filter first, or fetch far more.

Freshness means reindexing documents when they change. Deletion means removing a document everywhere it was copied: its cards, its vectors, keyword entries, cached answers and saved memories. A retracted paper should be removed or hard blocked at once, never just ranked lower.

## Retrieval metrics

These score a ranked list against known right answers.

- **Recall at k:** of all the right documents, what share appeared in the top k.
- **Precision at k:** of the top k, what share are right.
- **MRR:** 1 divided by the position of the first right answer, averaged over questions.
- **nDCG:** gives more credit for right answers near the top and for very relevant ones, scaled so a perfect ranking scores 1.

> **Worked example:** ten results where positions 2, 4 and 6 are relevant (graded 3, 2 and 1), and one more relevant document (grade 2) was never found. Precision at 5 is 2 out of 5 = 0.40. Recall at 10 is 3 out of 4 = 0.75. The first hit is at position 2, so reciprocal rank is 0.5. nDCG at 10 works out to 0.55, checked in Python against scikit-learn.

Where it breaks: the "right answers" lists are often incomplete, so a new system that finds good documents nobody labelled gets marked down.

## Grounded generation, citations and refusal

A grounded answer only says what the retrieved text supports. A correct answer matches the truth. They are different: an answer copied faithfully from an outdated page is grounded and wrong.

Citations help only if you check them. Models happily attach a real source to a sentence that source does not say, so systems check each cited passage against its sentence afterward.

The system also needs a way to say "the documents do not answer this". Test it with questions that have no answer in the corpus and count two mistakes: answering when it should refuse, and refusing when the answer was there. Lowering one raises the other, so you pick the balance based on which mistake costs more.

## RAG evaluation frameworks and their limits

Frameworks such as RAGAS (2023) use a language model as the grader. Faithfulness: split the answer into claims and count how many the context supports. Answer relevance: generate questions from the answer and see how similar they are to the real question. Context relevance: count how much of the retrieved text was needed.

On a test of 50 Wikipedia pages, RAGAS agreed with human preferences 95 percent of the time on faithfulness but only 70 percent on context relevance.

Where it breaks: the scores depend on which model grades and how it is prompted, and "consistent with the context" is not the same as "true". Check the grader against your own human labels before trusting it.

## Hallucination mitigation

A hallucination is a confident statement with no support. No single fix removes them, so you stack defences: retrieve better, send fewer and cleaner passages, tell the model to answer only from sources and allow "I don't know", require a citation per claim, check claims against their citations afterward, sample several answers and distrust ones that disagree, and send high stakes answers to a person.

Where it breaks: the model can still misread a passage it was given, so the checking step is never optional for important answers.

## Agents and tool calling

An agent is a model in a loop. It reads the task, decides to use a tool (search, a database lookup, a calculator), your code runs the tool and hands back the result, and the model decides the next step, until it answers or hits a limit.

Tool calling works like a form: you describe each tool's name, purpose and inputs, the model fills in the form instead of writing prose, and your program does the actual work.

Anthropic's advice (2024) is to start simple. One good call with retrieval often beats an agent, because agents cost more, take longer, and their mistakes pile up. Tools deserve as much care as prompts: clear descriptions, obvious inputs, helpful error messages.

## Workflow patterns

Before full agents come fixed recipes:

- **Chaining:** step one's output feeds step two, with checks between.
- **Routing:** sort the request first, then send it to a specialist, for example easy questions to a cheap model.
- **Parallel:** split a task into independent parts, or ask several times and vote.
- **Orchestrator and workers:** a lead model decides the subtasks as it goes and hands them out.
- **Evaluator and optimiser:** one model drafts, another critiques, repeat until good enough.

Pick the simplest recipe that fits. A full agent is for open ended tasks where you cannot predict the steps.

## ReAct, planning and multi step reasoning

ReAct (2022) has the model alternate between thinking out loud and acting: "I need the author first" then search, read, "now the birthplace", search again, answer. Mixing the two let it fetch facts instead of inventing them, and on two interactive benchmarks it beat earlier methods by 34 and 10 percentage points.

Planning variants: write a full plan first and revise it when a step fails; reflect on a failed attempt and retry; or explore several possible paths and pick the best. Exploring only works where actions can be undone. You cannot try sending an email and take it back.

## Memory: short term and long term

Short term memory is what fits in the model's current input, like a desk. It fills up, so agents summarise old parts and keep the important notes. Long term memory is a filing cabinet outside the model: saved facts about the user, notes from earlier tasks. Using it is just retrieval again.

Where it breaks: old memories go stale, a malicious instruction saved in memory comes back every session, and private memories need the same deletion rules as documents.

## Multi agent systems

Several model instances work on one job, usually a lead that plans and helpers that work in parallel. Anthropic's research tool (2025) did 90 percent better than a single agent on their internal test, but used about 15 times the tokens of a normal chat.

That cost is the catch. Much of the gain comes from spending more effort in parallel. It suits wide research across many sources. It suits poorly tasks where everyone needs the same shared picture, like most coding. Before building one, try a single agent with the same budget.

## MCP, computer use and coding agents

The Model Context Protocol (MCP) is like a universal plug. Instead of writing a custom connector between every AI app and every tool, each tool becomes an MCP server and each app speaks MCP once. Servers offer tools (actions), resources (data to read) and prompts (templates). The protocol was revised in 2026, so details depend on the version.

Computer use agents look at screenshots and click and type like a person. They work where no API exists but are slow and easy to trick with text on a page.

Coding agents are the strongest case because tests tell them right away whether a change worked. People still need to review the code, and the agent must not be allowed to "fix" a failing test by editing the test.

## Agent failure modes

Agents fail in recognisable ways: repeating the same action in a loop, calling the wrong tool or inventing arguments, building on an early mistake, giving up too soon, forgetting the goal in a long session, breaking a business rule, and running up a huge bill.

> **Worked example:** if each step works 99 percent of the time, a 50 step task works only about 60 percent of the time, because 0.99 multiplied by itself 50 times is 0.605. At 95 percent per step, a 20 step task succeeds 36 percent of the time.

Defences: cap steps and spending, detect repeated calls, check results along the way, and hand off to a person when stuck.

## Guardrails and prompt injection

Prompt injection is text that hijacks the model. Direct injection is a user typing "ignore your instructions". Indirect injection is sneakier: the instruction hides in a web page, email or document the system reads, so the attacker never talks to the system at all.

No filter catches everything, because to the model instructions and data are the same stuff: text. So you limit the damage instead: give the agent only the permissions it needs, ask a person before anything irreversible, keep agents that read untrusted pages away from powerful tools, and never let output carry data out through links or images.

## Human in the loop design

Put people where their judgment changes outcomes: approving risky actions like refunds or deletions, taking over when the system is stuck or unsure, reviewing a sample of answers, and labelling the test cases that everything else depends on.

Gate only risky actions. If people approve hundreds of harmless ones a day, they stop reading. Show the evidence and the proposed action together, make approve and reject one click, and track how often people change what the system proposed.

## Evals and golden sets

Evals are the tests that tell you whether a change made things better. Without them, every fix is a guess and old bugs quietly come back.

Start by reading real conversations and writing down what went wrong. Group the problems, count them, then write tests for the common and costly ones. A golden set is a saved list of test cases, each with the input, the expected behaviour (answer, answer with caution, refuse, or block), the documents that should be used, and a note on why the case exists.

> **Worked example:** with 100 test cases and a 70 percent pass rate, the honest range is roughly 60 to 78 percent. A 5 point improvement is invisible at that size, so important comparisons need hundreds of cases.

## Offline and online evaluation

Offline evaluation runs the saved test set before release, every time something changes. Online evaluation watches real users after release: thumbs up and down, people rephrasing and retrying, people giving up, people asking for a human, plus sampled reviews and A/B tests.

The two drift apart because real users keep doing new things. The fix is a loop: every confirmed real world failure becomes a new test case.

## LLM as a judge and its biases

When answers have no single right form, a strong model can grade them. It can compare two answers, score one answer, or compare against a reference answer.

Judges have measurable habits. They favour whichever answer comes first, longer answers, and sometimes their own writing. In a 2023 study, one judge gave the same verdict after the two answers were swapped only 24 percent of the time; the best judge, GPT-4, managed 65 percent. Fixes: grade both orders, compare answers of similar length, give the judge a reference answer, and use a judge from a different model family.

The same study found GPT-4 agreed with human experts 85 percent of the time, slightly more than humans agreed with each other, but mostly on easy comparisons.

## Calibrating a judge against human labels

Treat the judge like a new employee: check its work against an expert's before trusting it. Have the expert label examples pass or fail with a reason, tune the judge's instructions on part of them, and test on the rest.

Plain agreement can fool you. If 80 percent of answers are good, a judge that passes everything agrees 80 percent of the time while catching nothing. Cohen's kappa fixes this by subtracting the agreement you would get by chance.

> **Worked example:** 100 labelled answers, 20 bad. The judge catches 14 of the 20 bad ones and wrongly fails 6 of the 80 good ones. Agreement is 88 percent, but chance alone gives 68 percent, so kappa is 0.625. The pass everything judge scores kappa 0.

## Regression gates in CI

A regression gate is an automatic check before release: if quality drops, the change is blocked. Good gates have layers: basic checks that must always pass, critical test cases with zero tolerance, overall scores compared with the last release, and limits on speed and cost.

> **Worked example:** on 200 test cases a new prompt broke 18 that used to pass and fixed 8, a 5 point drop. A statistics test says this could be chance (p = 0.076), so the gate also blocks on any failed critical case, and someone reads the 18 broken cases.

Never loosen a threshold just to get a change through.

## Benchmarks and contamination

Public benchmarks are shared exams for models: knowledge, maths, coding, safety, agent tasks. Contamination is when the exam questions leaked into the training data, so a high score shows memory, not skill.

Signs: the model recites reference answers word for word, or scores drop sharply on reworded or newer questions. Defences: test on questions written after the model was trained, keep test sets private, and reword questions. Your own test set can leak too, if you tune prompts on it or copy cases into examples.

## Agent benchmarks and long horizon tasks

SWE-bench gives a model a real GitHub issue and the code repository and checks whether its fix passes the project's tests. It has 2,294 tasks; in 2023 the best model solved under 2 percent.

tau-bench puts an agent in a customer service chat with a simulated customer, a database and company rules, then checks the final database. It also asks whether the agent succeeds every time: for a task solved 6 times out of 8, any 4 of those tries include at least one success, but all 4 succeed only about 21 percent of the time.

Long horizon tests ask how long a task an agent can finish. Work covered in Stanford's CS329A found that length roughly doubling every seven months.

## Observability, latency and cost

Observability means recording every step of every request: what was searched, what came back, what the model said, how long each step took and what it cost. When something goes wrong you replay the trace instead of guessing.

Speed and cost come from a few habits: cache repeated work (Anthropic reported caching long prompts cut cost by up to 90 percent), send easy requests to a smaller model, run independent steps at the same time, and cap how much each request may spend.

> **Worked example:** if 70 percent of traffic goes to a model one tenth the price, total cost falls to 37 percent of the original, a 63 percent saving, as long as you check the cheap model handles that traffic well.
