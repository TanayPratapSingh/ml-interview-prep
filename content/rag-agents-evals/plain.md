# RAG, Agents and Evals: the plain version

This guide explains three ideas that sit behind most artificial intelligence (AI) products, in everyday words. The first is retrieval augmented generation (RAG). Retrieval means finding the right documents for a question. RAG lets a large language model (LLM) look things up before it answers. An LLM is a program trained on huge amounts of text to predict what text comes next. The second idea is agents. An agent is a model that takes actions with tools, one step at a time, until a task is done. The third idea is evals, short for evaluations. Evals are the tests that tell you whether a RAG system or an agent works. Each section gives the intuition, an everyday analogy, one example with real numbers, and where the idea breaks.

## RAG versus fine tuning versus long context

There are three ways to give a model knowledge or skills, and each fits a different need. Picture a well read person answering a question from memory. RAG hands that person the relevant pages just before they answer. Fine tuning sends them on a training course that changes their habits. Long context gives them the whole binder and hopes they find the right page.

Fine tuning means training an existing model further on your own examples. This changes the numbers inside the model, which are called its weights. Long context means pasting a lot of text into the model's input, which is called the prompt. A model can read only a limited amount at once, and that limit is its context window.

Use RAG for facts, especially facts that change or that you must cite. Use fine tuning for behaviour, such as a format, a tone or a skill. Use long context when all the text fits and you can afford to send it every time.

> **Worked example:** the original RAG paper (2020) built one index (a searchable catalogue of documents) from Wikipedia as of 2016 and another from 2018. Asked about world leaders, the model got about 70 percent right when the index matched the year. It got only 4 to 12 percent right when the index did not match. Swapping the index updated the model with no retraining (no further training).

Where it breaks: models read long inputs unevenly. A 2023 study found that models miss facts placed in the middle of a long prompt more often than facts at the start or end. So "just send everything" is not free.

## The RAG pipeline end to end

A RAG system has two phases. One is preparation, which happens before anyone asks. The other is answering, which happens when a question arrives. A pipeline is the chain of steps that data passes through. Think of a library.

Before anyone asks a question, librarians collect books and clean them up. They cut the books into index cards and label each card. They then file the cards two ways: by meaning and by keywords.

When a question arrives, a clerk rephrases it. The clerk pulls a stack of maybe 100 cards. The clerk then sorts the best 10 to the top. A writer reads those 10 and answers, pointing to the cards used. If the cards do not contain the answer, the writer should say so.

Every step can fail on its own. A table can turn into gibberish when the PDF (portable document format, a common file type for documents) is read. A card can be cut in the wrong place. The right card can be filed but never pulled. That is why you test each step separately.

> **Remember:** the writer can only use the cards it is given. If retrieval misses the answer, no later step in the pipeline can fix it.

## Chunking strategies and trade offs

A chunk is one piece of a document that gets stored and searched as a unit, like one index card. Small chunks describe one idea sharply, so a question about that idea finds them. But a card that says "revenue grew 3 percent" without naming the company is useless on its own. Big chunks keep the context. They blur several topics together, though. So they match every question weakly and cost more to read.

There are four common ways to cut text into chunks. You can cut every few hundred words, or by sentence or paragraph. You can cut by the document's own headings. You can also detect where the topic shifts and cut there.

A neat trick from Anthropic (2024) adds a short note to each chunk saying where it came from. An example note is "From ACME's Q2 2023 report, revenue section". Q2 means the second quarter of the year. This cut their retrieval misses (questions where the right chunk was not found) from 5.7 percent to 3.7 percent. With the other steps described below, the misses fell to 1.9 percent.

> **Worked example:** a 1,000 word document cut into 256 word chunks with 64 words of overlap gives 5 chunks. With no overlap it gives 4. Overlap means neighbouring chunks share some words. It protects answers that sit on a boundary, but it makes the index bigger.

## Embeddings and bi encoders

An embedding turns text into a list of numbers. Think of it as a point on a map, where texts with similar meaning land close together. A bi encoder is a model that makes one point for the question and one point for each document, separately. The document points do not depend on the question, so you can make them all in advance. Searching is then just finding the nearest points.

The more careful alternative is a cross encoder. A cross encoder reads the question and a document together, like a person comparing them side by side. It judges better, but it must read every pair fresh. That makes it too slow to run over millions of documents.

Where it breaks: a map of meaning is fuzzy about exact things such as product codes, numbers, names and the word "not". A model trained on one kind of text can also do badly on another kind. A benchmark (a standard test used to compare systems) called BEIR (Benchmarking Information Retrieval) came out in 2021. It found that a well known dense retriever scored about 48 percent below plain keyword search on average across 18 unfamiliar datasets. A dense retriever finds documents by comparing embeddings.

## Training retrievers contrastively

Contrastive training teaches a retriever like a matching game. A retriever is the part of the system that finds documents for a question. Contrastive means learning by contrast: matching pairs are pulled together and non matching pairs are pushed apart. You show the retriever a question, the right passage, and some wrong passages. You reward it for placing the right passage closest to the question.

A cheap trick uses the batch, which is a group of training examples processed together. In a batch of 128 question and answer pairs, every other question's answer is a free wrong option. So each question gets 127 wrong options at no extra cost.

The wrong options that teach the most are near misses. These are passages that share the question's words but do not answer it. The DPR (Dense Passage Retrieval) paper (2020) tested this. Adding one such near miss per question raised the share of questions with the answer in the top 5 results. That share rose from 55.8 to 65.8 percent.

Where it breaks: sometimes a "wrong" option is actually right, but nobody labelled it. Then the model learns to push good answers away.

## Vector indexes: flat, IVF, HNSW, PQ

A vector is a list of numbers, such as an embedding. A vector index is a structure that finds the stored vectors nearest to a question vector quickly. Checking every stored vector one by one finds the exact nearest points, but it is slow when you have millions. That method is called flat search. The shortcuts are:

- **IVF** (inverted file index) sorts points into neighbourhoods. It only searches the few neighbourhoods closest to the question.
- **HNSW** (hierarchical navigable small world) builds a layered road network: highways on top for long jumps, and local streets at the bottom. The search drives down from highway to street.
- **PQ** (product quantization) compresses each point into a short code, like storing a postcode instead of a full address.

> **Worked example:** 10 million vectors of 768 numbers each take 30.7 GB (gigabytes) stored exactly. Compressed to 96 byte codes (96 bytes per vector), they take about 1 GB. In a small test, searching 1 neighbourhood out of 64 found only 30 percent of the true nearest neighbours. These are the stored points truly closest to the question. Searching 16 found 87 percent while still skipping three quarters of the data.

Where it breaks: speed costs some accuracy, and the right settings depend on your data. Some indexes cannot delete items without a rebuild. HNSW in the FAISS (Facebook AI Similarity Search) library is one of them.

## Sparse retrieval and BM25

Sparse retrieval is classic keyword search. It matches the exact words in a question against the words in documents. It is called sparse because any one document contains only a few of all possible words, so its word count list is mostly zeros. BM25 (Best Matching 25) is the standard scoring formula for keyword search. It follows three sensible rules. Rare words count more than common ones. Repeating a word helps, but it helps less each time. Long documents are discounted because they contain more words by chance.

> **Worked example:** with typical settings, a word appearing once in an average length document scores 1.0, times its rarity weight (the bonus that rare words get). Twice scores 1.375. Ten times scores 1.96. It can never pass 2.2. A 13 word document mentioning "refund policy" scored 0.78 against 1.19 for a 5 word one with the same words.

BM25 needs no training and is excellent at exact names, codes and error messages. Its blind spot is different words for the same thing. For example, "cancel my subscription" does not match "terminate plan".

## Hybrid search and reciprocal rank fusion

Hybrid search runs keyword search and meaning search together, then merges the two result lists. Good systems do this because the two methods make different mistakes. Their scores sit on different scales, so the usual merge ignores scores and uses positions, also called ranks. Each document earns 1 divided by (60 plus its rank) from every list it appears in. The totals decide the final order. This is called reciprocal rank fusion. It comes from a 2009 paper that picked the number 60 in a pilot study (a small early test).

> **Worked example:** document A is first in the keyword list and missing from the other list. Document B is third in one list and second in the other. A earns 1/61 = 0.0164. B earns 1/63 + 1/62 = 0.0320 and wins. Agreement between methods beats one method's enthusiasm.

Where it breaks: if one of the lists is poor, it adds noise rather than help.

## Cross encoder reranking

Reranking is a second, slower look at a short list. First, pull 100 to 150 candidates cheaply. Then let a cross encoder read each one together with the question and rescore it. Keep the best 5 to 20. In the BEIR benchmark this was the most accurate approach on average. It was about 11 percent better than keyword search alone. But it took about 450 milliseconds (thousandths of a second) per question, against 20 for keyword search.

Where it breaks: a reranker can only reorder what it was given. If the first search missed the right passage, reranking cannot find it.

## Query rewriting, HyDE and multi hop retrieval

People often ask questions in a way that is bad for search. "What about the annual one?" means nothing without the chat before it. So systems rewrite it into a full query. They also split compound questions apart. Or they try several phrasings and merge the results.

HyDE (hypothetical document embeddings) is a clever variant. First, ask a model to write a fake answer. Then search for real documents that look like that fake answer. This works because answers resemble documents more than questions do.

Multi hop questions need a chain of lookups. "Where was the author of Things Fall Apart born?" needs the author first, then the birthplace. The system searches and reads. Then it searches again with what it learned. Every extra hop is another chance to go wrong.

## Metadata filters, freshness and deletion

Metadata is information about a document, such as its owner, type or date. Labels on each chunk let you filter on metadata. Examples are only this customer's documents, only HR (human resources) policies, or only things after 2024. Permission filters must happen inside the search itself. Showing the model a document and then telling it "ignore documents this user cannot see" is a leak.

> **Worked example:** if a filter matches 1 percent of documents and you filter after fetching the top 100, you expect about one result left. Filter first, or fetch far more.

Freshness means reindexing documents when they change. Reindexing means rebuilding the index entries for them. Deletion means removing a document everywhere it was copied. That includes its chunks, its vectors, its keyword entries, cached answers (saved replies) and saved memories. A retracted paper (a paper the journal has withdrawn because of errors) should be removed or hard blocked at once. Never just rank it lower.

## Retrieval metrics

Retrieval metrics score a ranked list of results against known right answers. A ranked list is the ordered list of documents the system returns. Four metrics are common.

- **Recall at k:** of all the right documents, the share that appeared in the top k results.
- **Precision at k:** of the top k results, the share that are right.
- **MRR** (mean reciprocal rank): 1 divided by the position of the first right answer, averaged over questions.
- **nDCG** (normalised discounted cumulative gain): gives more credit for right answers near the top and for very relevant ones. It is scaled so a perfect ranking scores 1.

> **Worked example:** ten results where positions 2, 4 and 6 are relevant (graded 3, 2 and 1). One more relevant document (grade 2) was never found. Precision at 5 is 2 out of 5 = 0.40. Recall at 10 is 3 out of 4 = 0.75. The first hit is at position 2, so reciprocal rank is 0.5. nDCG at 10 works out to 0.55. This was checked in Python (a programming language) against scikit-learn (a standard machine learning library).

Where it breaks: the "right answers" lists are often incomplete. A new system that finds good documents nobody labelled gets marked down.

## Grounded generation, citations and refusal

A grounded answer only says what the retrieved text supports. A correct answer matches the truth. These are different. An answer copied faithfully from an outdated page is grounded and wrong.

Citations (pointers to the source of each claim) help only if you check them. Models happily attach a real source to a sentence that the source does not say. So systems check each cited passage against its sentence afterward.

The system also needs a way to say "the documents do not answer this". That is called refusal. Test it with questions that have no answer in the corpus (the full collection of documents). Then count two mistakes: answering when it should refuse, and refusing when the answer was there. Lowering one raises the other. So you pick the balance based on which mistake costs more.

## RAG evaluation frameworks and their limits

Frameworks such as RAGAS (Retrieval Augmented Generation Assessment, 2023) use a language model as the grader. They score three things.

- **Faithfulness:** split the answer into claims (single statements) and count how many the context supports. The context is the retrieved text.
- **Answer relevance:** generate questions from the answer and see how similar they are to the real question.
- **Context relevance:** count how much of the retrieved text was needed.

On a test of 50 Wikipedia pages, RAGAS agreed with human preferences 95 percent of the time on faithfulness. It agreed only 70 percent of the time on context relevance.

Where it breaks: the scores depend on which model does the grading and on how you instruct it. "Consistent with the context" is also not the same as "true". Check the grader against your own human labels before trusting it.

## Hallucination mitigation

A hallucination is a confident statement with no support. No single fix removes them, so you stack defences.

- Retrieve better.
- Send fewer and cleaner passages.
- Tell the model to answer only from sources, and allow "I don't know".
- Require a citation for each claim.
- Check claims against their citations afterward.
- Sample several answers, which means asking the model the same question several times. Distrust answers that disagree.
- Send high stakes answers to a person.

Where it breaks: the model can still misread a passage it was given. So the checking step is never optional for important answers.

## Agents and tool calling

An agent is a model in a loop. It reads the task and decides to use a tool. A tool can be a search, a database lookup (a query to a store of structured records) or a calculator. Your code runs the tool and hands the result back. The model then decides the next step. This repeats until the model answers or hits a limit.

Tool calling works like a form. You describe each tool's name, purpose and inputs. The model fills in the form instead of writing prose. Your program does the actual work.

Anthropic's advice (2024) is to start simple. One good call with retrieval often beats an agent. Agents cost more and take longer, and their mistakes pile up. Tools deserve as much care as prompts. Give them clear descriptions, obvious inputs and helpful error messages.

## Workflow patterns

Workflow patterns are fixed recipes that come before full agents. In a workflow, your code decides the steps. In an agent, the model decides. Five recipes are common.

- **Chaining:** step one's output feeds step two, with checks between.
- **Routing:** sort the request first, then send it to a specialist. For example, send easy questions to a cheap model.
- **Parallel:** split a task into independent parts, or ask several times and vote.
- **Orchestrator and workers:** a lead model decides the subtasks as it goes and hands them out.
- **Evaluator and optimiser:** one model drafts, another critiques, and they repeat until the result is good enough.

Pick the simplest recipe that fits. A full agent is for open ended tasks where you cannot predict the steps.

## ReAct, planning and multi step reasoning

ReAct (short for reasoning and acting, 2022) has the model alternate between thinking out loud and acting. For example, the model says "I need the author first", then searches and reads. Next it says "now the birthplace", searches again and answers. Mixing the two let the model fetch facts instead of inventing them. On two interactive benchmarks it beat earlier methods by 34 and 10 percentage points. A percentage point is the plain gap between two percentages. Going from 20 percent to 30 percent is a gain of 10 points.

There are three planning variants. One is to write a full plan first and revise it when a step fails. Another is to reflect on a failed attempt and retry. A third is to explore several possible paths and pick the best. Exploring only works where actions can be undone. You cannot try sending an email and take it back.

## Memory: short term and long term

Short term memory is what fits in the model's current input, like a desk. It fills up. So agents summarise old parts and keep the important notes. Long term memory is a filing cabinet outside the model. It holds saved facts about the user and notes from earlier tasks. Using it is just retrieval again.

Where it breaks: old memories go stale. A malicious instruction saved in memory comes back every session. Private memories need the same deletion rules as documents.

## Multi agent systems

A multi agent system has several model instances working on one job. Usually a lead agent plans, and helper agents work in parallel, which means at the same time. Anthropic's research tool (2025) did 90 percent better than a single agent on their internal test. But it used about 15 times the tokens of a normal chat. A token is a small chunk of text, roughly a word or part of a word. You pay for models by the token.

That cost is the catch. Much of the gain comes from spending more effort in parallel. The design suits wide research across many sources. It suits poorly tasks where everyone needs the same shared picture, like most coding. Before building one, try a single agent with the same budget.

## MCP, computer use and coding agents

The Model Context Protocol (MCP) is like a universal plug. Without it, you write a custom connector between every AI app and every tool. With it, each tool becomes an MCP server, which is a small program that offers its abilities to AI apps. Each app speaks MCP once. Servers offer three kinds of thing: tools (actions), resources (data to read) and prompts (templates). The protocol was revised in 2026, so details depend on the version.

Computer use agents look at screenshots and click and type like a person. They work where no API exists. An API (application programming interface) is a way for programs to talk to each other. Computer use agents are slow and easy to trick with text on a page.

Coding agents are the strongest case because tests tell them right away whether a change worked. People still need to review the code. The agent must not be allowed to "fix" a failing test by editing the test.

## Agent failure modes

Agents fail in recognisable ways.

- They repeat the same action in a loop.
- They call the wrong tool or invent arguments, which are the inputs a tool needs.
- They build on an early mistake.
- They give up too soon.
- They forget the goal in a long session.
- They break a business rule.
- They run up a huge bill.

> **Worked example:** if each step works 99 percent of the time, a 50 step task works only about 60 percent of the time. The reason is that 0.99 multiplied by itself 50 times is 0.605. At 95 percent per step, a 20 step task succeeds 36 percent of the time.

Defences: cap steps and spending, detect repeated calls, check results along the way, and hand off to a person when stuck.

## Guardrails and prompt injection

Guardrails are the safety checks and limits placed around a model. The biggest threat they face is prompt injection. Prompt injection is text that hijacks the model. Direct injection is a user typing "ignore your instructions". Indirect injection is sneakier. The instruction hides in a web page, email or document that the system reads. So the attacker never talks to the system at all.

No filter catches everything, because to the model instructions and data are the same stuff: text. So you limit the damage instead.

- Give the agent only the permissions it needs.
- Ask a person before anything irreversible.
- Keep agents that read untrusted pages away from powerful tools.
- Never let output carry data out through links or images.

## Human in the loop design

Human in the loop means people take part in the system's decisions. Put people where their judgment changes outcomes. There are four such places.

- Approving risky actions, such as refunds or deletions.
- Taking over when the system is stuck or unsure.
- Reviewing a sample of answers.
- Labelling the test cases that everything else depends on.

Gate only risky actions. A gate is a point where the system waits for a person. If people approve hundreds of harmless actions a day, they stop reading. Show the evidence and the proposed action together. Make approve and reject one click. Track how often people change what the system proposed.

## Evals and golden sets

Evals are the tests that tell you whether a change made things better. Without them, every fix is a guess and old bugs quietly come back.

Start by reading real conversations and writing down what went wrong. Group the problems and count them. Then write tests for the common and costly ones. A golden set is a saved list of test cases. Each case has the input, the expected behaviour, the documents that should be used, and a note on why the case exists. The expected behaviour is one of four: answer, answer with caution, refuse, or block.

> **Worked example:** with 100 test cases and a 70 percent pass rate, the honest range is roughly 60 to 78 percent. The honest range is the span of true pass rates that could plausibly have produced your result. A 5 point improvement is invisible at that size, so important comparisons need hundreds of cases.

## Offline and online evaluation

Offline evaluation runs the saved test set before release, every time something changes. Online evaluation watches real users after release. It looks at thumbs up and down, people rephrasing and retrying, people giving up, and people asking for a human. It also uses sampled reviews and A/B tests. An A/B test shows two versions to two random groups of users so you can compare them.

The two drift apart because real users keep doing new things. The fix is a loop. Every confirmed real world failure becomes a new test case.

## LLM as a judge and its biases

An LLM judge is a strong model that grades other answers. It helps when answers have no single right form. It can compare two answers, score one answer, or compare against a reference answer, which is a known good answer.

Judges have measurable habits. They favour whichever answer comes first. They favour longer answers. Sometimes they favour their own writing. In a 2023 study, one judge gave the same verdict after the two answers were swapped only 24 percent of the time. The best judge, GPT-4 (generative pre trained transformer 4, a large model from OpenAI), managed 65 percent. There are four fixes. Grade both orders. Compare answers of similar length. Give the judge a reference answer. Use a judge from a different model family.

The same study found GPT-4 agreed with human experts 85 percent of the time. That is slightly more than humans agreed with each other. But the agreement was mostly on easy comparisons.

## Calibrating a judge against human labels

Calibrating a judge means checking its grades against an expert's. Treat the judge like a new employee: check its work against an expert's before trusting it. Have the expert label examples pass or fail, with a reason. Tune the judge's instructions on part of the examples, and test on the rest.

Plain agreement can fool you. If 80 percent of answers are good, a judge that passes everything agrees 80 percent of the time while catching nothing. Cohen's kappa fixes this by subtracting the agreement you would get by chance. In words, kappa is the gain over chance divided by the most gain that was possible.

> **Worked example:** 100 labelled answers, 20 bad. The judge catches 14 of the 20 bad ones and wrongly fails 6 of the 80 good ones. Agreement is 88 percent, but chance alone gives 68 percent, so kappa is 0.625. That is 0.88 minus 0.68, divided by 1 minus 0.68. The pass everything judge scores kappa 0.

## Regression gates in CI

A regression is a change that makes things worse than before. A regression gate is an automatic check before release. If quality drops, the change is blocked. The check runs in CI (continuous integration), which means automatic checks that run whenever someone changes the code. Good gates have layers. There are basic checks that must always pass. There are critical test cases with zero tolerance. There are overall scores compared with the last release. There are also limits on speed and cost.

> **Worked example:** on 200 test cases a new prompt broke 18 that used to pass and fixed 8, a 5 point drop. A statistics test says this could be chance (p = 0.076). That p value means a gap this big would show up about 7.6 percent of the time if the prompt were really no worse. So the gate also blocks on any failed critical case, and someone reads the 18 broken cases.

Never loosen a threshold just to get a change through.

## Benchmarks and contamination

Public benchmarks are shared exams for models. They cover knowledge, maths, coding, safety and agent tasks. Contamination is when the exam questions leaked into the training data, which is the text the model learned from. A high score then shows memory, not skill.

There are signs to watch for. The model recites reference answers word for word. Or its scores drop sharply on reworded or newer questions. Defences are to test on questions written after the model was trained, keep test sets private, and reword questions. Your own test set can leak too, if you tune prompts on it or copy cases into examples.

## Agent benchmarks and long horizon tasks

SWE-bench (software engineering benchmark) gives a model a real GitHub issue, which is a bug report or feature request. It also gives the model the code repository, which is the project's folder of code. The benchmark checks whether the model's fix passes the project's tests. It has 2,294 tasks. In 2023 the best model solved under 2 percent.

tau-bench puts an agent in a customer service chat with a simulated customer, a database and company rules. It then checks the final database. It also asks whether the agent succeeds every time. Suppose an agent solves a task in 6 of 8 tries. Any 4 of those tries include at least one success. But all 4 succeed only about 21 percent of the time.

Long horizon tests ask how long a task an agent can finish. Work covered in Stanford's CS329A, a university course, found that this length roughly doubles every seven months.

## Observability, latency and cost

Observability means recording every step of every request. That includes what was searched, what came back, what the model said, how long each step took and what it cost. When something goes wrong, you replay the trace (the saved record of one request) instead of guessing.

Speed and cost come from a few habits.

- Cache repeated work, which means saving and reusing results. Anthropic reported that caching long prompts cut cost by up to 90 percent.
- Send easy requests to a smaller model.
- Run independent steps at the same time.
- Cap how much each request may spend.

Latency, the waiting time before an answer, drops with these habits too.

> **Worked example:** if 70 percent of traffic goes to a model one tenth the price, total cost falls to 37 percent of the original. That is a 63 percent saving, as long as you check that the cheap model handles that traffic well.
