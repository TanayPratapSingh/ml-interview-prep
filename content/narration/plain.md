# Narration: explaining, structuring and telling your project stories

This is the easy version of the narration module: how to explain things out loud so an interviewer can follow you, how to structure answers when you are nervous, and how to tell your own project stories in 15 seconds, 60 seconds or 3 minutes. Each section matches a section of the deep version, which carries the technical detail and the full scripts.

## How communication gets scored

An interviewer is taking notes while you talk. "Good communication" means they can write down your reasoning without asking you to repeat it. That is how many rubrics score it: the Tech Interview Handbook's coding rubric separates "the interviewer had no trouble following you" (strong hire) from "they had to ask follow up questions to understand your approach" (leaning hire) from "you jumped into code without explaining" (leaning no hire).

Think of yourself as a tour guide. A good guide says where the group is going, points at things as they pass, and checks nobody got lost. A bad guide walks fast and talks to the wall.

Different rounds weight this differently. In system design, talking is the whole round. In behavioural rounds, especially at Amazon, a single story can get 25 minutes of follow up questions, so the story is only the start.

> **Remember:** Your goal in every answer is fewer "wait, what do you mean?" questions from the interviewer.

## The layered answer

Answer in layers, like a news article: headline first, then detail for whoever wants it.

1. **One sentence** that answers the question.
2. **About 30 seconds**: one intuition and one example.
3. **About 2 minutes**: how it works, the main trade off, when it fails.
4. **The deep dive**, only if they ask.

After the 30 second layer, pause or offer: "I can go into the maths or into when it breaks. Which is more useful?" That one question tells you what the interviewer wants and saves you from lecturing on the wrong thing.

> **Worked example:** "What is dropout?" One sentence: "During training it randomly switches off some neurons so the network can't lean on any single one." Thirty seconds: "It's like training lots of slightly different networks that share weights, then averaging them at test time." Stop there and offer more.

The most common mistake is starting with the deep dive. The interviewer wanted to tick a box and move on, and you spent four minutes of their plan.

## The concept answer template

When asked to explain a concept, use the same five steps every time:

1. **What it is**, in one sentence.
2. **The intuition**, an analogy or the mechanism in plain words.
3. **A concrete example**, with numbers if you can.
4. **The trade off**: what it costs or when it breaks.
5. **When you would use it**, and what you would use instead.

This comes from the Feynman technique: if you cannot explain an idea in words a 12 year old would follow, you do not fully understand it yet, and the place where your simple explanation breaks is exactly where the interviewer's follow up will land. Practise by explaining each concept on your resume to a friend with no ML background and noting every word they stop you on.

> **Worked example:** Reciprocal rank fusion merges several ranked lists by giving each document points for how high it ranks in each list. It works because documents that several systems like tend to be good, and it never compares raw scores, which are on different scales. The cost is that a system that is very confident gets no extra say.

## Thinking aloud while coding

Thinking aloud means the interviewer always knows which step you are on. A simple order:

1. Repeat the problem in your own words and ask two or three questions (can the input be empty? is it sorted?).
2. Describe one slow approach and one faster one before choosing.
3. Ask "does this approach sound good?" before writing code.
4. While coding, say what each block is for, not what you are typing.
5. When done, do not say "done". Walk through an example by hand, out loud, then try the awkward cases: empty input, one item, duplicates.
6. State how fast it is and what you would improve.

If you get stuck, keep talking about what you know and what you are trying. "I'm stuck on avoiding the second pass. I know the output follows sorted order, so maybe..." gives the interviewer a chance to nudge you. A short announced silence ("give me thirty seconds") is fine. Long unexplained silence is the worst outcome on the rubric.

## Narrating math on a whiteboard

A derivation is a short story: here is where I'm going, here is what each letter means, here is each step and why it is allowed, here is a check that it makes sense.

For example, deriving the gradient of logistic loss: say the goal, define the input, the label and the predicted probability, take the steps, then say the result in words ("the update is the prediction error times the input") and check a simple case ("if the prediction is already right, the update is zero, which makes sense").

Board habits: write the goal at the top, keep definitions in a corner you never erase, number your lines, leave space between them. If you make a mistake, say so calmly and fix it in place. Read formulas aloud in words ("the square root of n"), never as code.

## Signposting in system design

Signposting is telling the listener the route before you drive it. In the first minute of a design round, say your plan: "I'll start with requirements, then the main data objects, then the API, then a simple end to end design, and then spend the last ten minutes on the weakest parts."

Then say where you are at each step ("that's requirements done; moving to the API"), and check in at each boundary ("anything you'd add or cut?"). This keeps you on time and lets the interviewer redirect you at a natural point instead of interrupting.

Give every choice a reason and a condition: "I'd use a queue here because the scorer can fall behind. If the load were tiny, I'd call it directly." Build something simple that works end to end before adding complexity; the most common way to fail is never finishing.

## Analytics and product cases

For "a metric dropped", say a structure out loud before diving in:

1. What exactly is the metric, over what time window, compared to what?
2. Is it a real change or a measurement problem (a logging bug, a new release, a timezone change)?
3. Was the drop sudden (often a release or outage) or gradual (often product or outside causes)?
4. Which segment is it concentrated in: platform, country, new versus returning users?
5. Rank your guesses and say what data would confirm each one.
6. Recommend an action.

For "how would you measure success of this feature", say what the feature is for, one main metric, two or three things that must not get worse, and how you would test it.

Writing the steps on the board and pointing at them as you go is most of the narration skill here.

## Behavioural answers with STAR and CAR

STAR stands for Situation, Task, Action, Result. CAR is the short version: Context, Action, Result. Both say the same thing: set the scene briefly, spend most of your time on what you did and why, and finish with what happened, with a number if you have one.

A good 2 minute STAR answer spends about 20 seconds on the situation and task, about 70 on your actions, about 20 on the result, and about 10 on what you learned. Use "I" for your own actions; credit the team once. Start with a title that hints at the challenge, such as "the time a default setting silently cut every document to 100 tokens", so the interviewer knows which skill you are about to show.

Do not memorise word for word. Write bullet points and practise saying them so the answer sounds natural.

## Your story bank

Prepare 8 to 10 true stories ahead of time and know which questions each can answer. You have strong material:

| If asked about | Tell |
| --- | --- |
| Conflict or disagreement | JMA Wireless: different teams defined the same KPIs differently |
| Influencing without authority | JMA Wireless: showing the constraint was material flow, not the machine |
| A failure | EvidenceOps: the retraction registry caught 0 of 3 retractions on real data |
| Ambiguity | Gravity Spy: five undocumented metadata schemas |
| A decision with incomplete data | Dr. GRPO: reporting one seed honestly as "not reproduced here" |
| A mistake you caught | BEIR: a hidden 100 token limit that broke one result |
| A result that surprised you | Distributed ViT: faster GPUs made scaling worse |
| Teaching or explaining | Cybersecurity TA: 15 plus hands on labs |

Practise each story twice: a 30 second version for phone screens and a 2 minute version for full interviews.

## Saying I don't know

When you do not know, say so plainly, then say what you do know nearby, then say how you would find out. "I haven't used that directly. I know it works like X, so I'd guess Y, but I'd check by Z." That turns a gap into a demonstration of how you think.

Never invent a number. If you do not remember a figure, give the direction and where it came from. For your own projects, know every number, which is what Part 2 is for.

## Handling pushback

When an interviewer disagrees, it is usually a test, not a verdict. Do three things:

1. Repeat their concern in your own words.
2. Work out whether they have new information, a different assumption, or are testing your conviction. Ask if unsure.
3. If they are right, say so and show what changes. If you still disagree, explain why and say what would change your mind.

Changing your mind because of a good argument is a good sign. Changing it because someone frowned is not. Stay friendly either way.

## Clarifying questions

Interview questions are vague on purpose, to see whether you check before building. Ask two or three questions that would change your answer, then say your assumptions for the rest out loud.

For coding: how big is the input, can it be empty, is it sorted? For ML design: what is the goal, where do labels come from, how fast must it be? For a metrics case: exactly which metric, over what period? Repeating the problem back in your own words catches misunderstandings in seconds.

## Questions to ask the interviewer

"Do you have any questions for me?" is still part of the interview. Having none can look like low interest. Prepare two good questions per interviewer type:

- Engineer: "What early technical decision does the team still live with?"
- ML lead: "How do you decide a model change is better before shipping it?"
- Manager: "What would you want me to have solved in my first few months?"
- Founder: "What has to be true next year for the company to work?"

The best question picks up something the interviewer said earlier. Leave pay and logistics for the recruiter.

## Pacing, pauses and filler words

Speak a little slower than normal and pause after each important claim, so the interviewer can write. Pauses feel long to you and normal to them.

Filler words ("um", "like", "basically") mostly appear while you plan the next sentence. Three fixes: pause silently instead, use short sentences, and signpost ("there are two parts to this") to give yourself a breath. Record yourself and count fillers per minute; most people speak fastest in the first five minutes.

Save hedges like "sort of" for things you are actually unsure about. At the end of a long answer, repeat the headline in one sentence.

## Tell me about yourself

This is a 60 to 90 second pitch, not your life story: who you are now, what you are good at, two short proofs, and why this role.

> **Say it like this:** "I'm finishing an MS in Applied Data Science at Syracuse. I build retrieval, evaluation and agent systems, and the part I'm best at is measuring where they break. For example, I replicated the BEIR retrieval benchmark on a laptop and traced the one result that didn't reproduce to a hidden 100 token limit. I also work on Gravity Spy, an NSF funded LIGO citizen science project. I'm applying here because..."

Swap the two proofs to match the role: replications for research roles, distributed ViT for infrastructure, JMA Wireless and ProProfs for analyst roles, handoff for agent roles. Keep visa, pay and contact details out of it.

## Daily practice routine

Thirty minutes a day:

1. **Warm up (5 minutes):** three concept cards, each said in 10 seconds and then 30 seconds.
2. **Card drill (15 minutes):** five cards, answered out loud and recorded on your phone. Then check the card's "must mention" list and give yourself a point for each item you said.
3. **Project story (5 minutes):** one project a day, the 60 second version from memory, then two follow ups.
4. **Review (5 minutes):** play one recording back. Did the first sentence answer the question? How many fillers? Where did you lose the thread? Write one fix for tomorrow.

Once a week, do a mock interview with a friend or an AI told to keep asking follow ups, and explain one project to someone non technical.

## How to tell a project story

Every project gets five layers: a 15 second version, a 60 second version, a 3 minute deep dive (problem, why it matters, what I built, the hardest decision, the result with caveats, what's next), five likely follow ups, and the weak spots to own.

Three rules keep you credible. Every number comes with what it is out of ("0 of 3", not "low"). Every caveat goes in the same sentence as its number. And lead with the surprise, the result that did not go as expected, because that is what makes a project memorable.

Owning a weak spot means saying it before they do, saying why, and saying what would fix it.

## EvidenceOps

**In plain words.** A question answering system over medical research papers that refuses to cite papers that have been retracted. A retracted paper is often the most relevant one, so a normal search system will happily quote it, and the reader cannot tell.

**15 seconds.** "EvidenceOps answers questions over PubMed Central papers and treats retraction as a hard rule: if the best evidence was withdrawn, it refuses and says why."

**The caveat you must say.** Its evaluation numbers come from a four article synthetic test corpus and a stand in "stub" model, not a real model. What is solidly real: 107 tests and test coverage that rose from 75 to 84 percent, and a test on 5,000 real articles (no model involved) that showed the first design caught 0 of 3 retractions, which led to a fix that caught 3 of 3.

**Weak spot to own.** No real language model has been run behind it yet. Say so first.

## BEIR replication

**In plain words.** BEIR is a famous benchmark that showed simple keyword search (BM25) beats fancier neural search on unfamiliar topics. I rebuilt part of it on a laptop to see if that still holds.

**15 seconds.** "It held for the paper's models, it failed against a newer 2023 model, and the one number that didn't match came from a hidden setting that cut every abstract to 100 tokens."

**The story.** DPR scored 0.260 against a published 0.318. Searching with titles only scored the same as titles plus text, which meant the text was being thrown away. Raising the limit to 512 tokens matched the paper exactly.

**Weak spot to own.** Only two datasets, and my keyword search is a Python rewrite, not the paper's Java tool, which explains two small differences.

## Dr. GRPO replication

**In plain words.** GRPO is a way of training language models with rewards for correct maths answers. A 2025 paper claimed one detail of GRPO makes wrong answers get longer, and proposed a fix called Dr. GRPO. I tested it.

**15 seconds.** "On one training run, accuracy matched (51.6 against 51.2 percent on MATH500), but both of the paper's claims about length came out the other way round."

**Why it is a good story.** I said "not reproduced here" instead of "the paper is wrong", because one run on small hardware cannot settle it. That is a decision made with incomplete data, reported honestly.

**Weak spot to own.** One seed, 100 training steps and a short 1024 token limit.

## handoff

**In plain words.** An incident response system where an AI agent must ask a human before doing something risky, and, crucially, knows what to do if nobody answers.

**15 seconds.** "Escalations have an owner and a deadline. If nobody answers in time, the system runs a safer substitute, like putting a database into read only mode instead of failing it over."

**Weak spot to own.** The cluster is simulated and the agents are simple rules, not language models. The point is the escalation design.

## Distributed ViT on CIFAR-10

**In plain words.** I trained an image model on two GPUs and measured what the second GPU actually bought.

**15 seconds.** "Two GPUs gave a 98 percent efficient speedup, but when I made each GPU faster with lower precision, efficiency fell to 57 percent, because the computer's CPUs could no longer prepare images fast enough."

**Why it is a good story.** It is the "result that surprised me" answer: speeding up one part moved the bottleneck somewhere else. I proved it by feeding fake data that needs no preparation, which brought efficiency back to 96 percent.

**Weak spot to own.** A small dataset and one machine with two GPUs.

## Tool dependency graph

**In plain words.** AI agents call tools, and many tools need an ID that only another tool can give them. This project works out the order automatically from the tools' descriptions.

**15 seconds.** "From 1,391 tool descriptions, it works out which tools must run first and which inputs must come from the user."

**Weak spot to own.** It covers 65 percent of required inputs with typed IDs; the 99.9 percent figure is coverage of those, not of everything. Say that distinction yourself.

## ZTF transient pipeline

**In plain words.** A telescope survey sends out alerts whenever something in the sky changes. This pipeline scores each alert with a model and stores the results in a tested data warehouse.

**15 seconds.** "Real astronomical alerts over Kafka, a model that reads the images and the measurements, and a dbt warehouse with tests on top."

**Weak spot to own.** The classifier is not benchmarked; the labels are coarse. It is a pipeline project.

## ShopFloor AI

**In plain words.** A factory system that predicts machine failures from sensor data, shows live production numbers, and answers questions about maintenance manuals.

**15 seconds.** "A failure classifier at 0.85 F1 on a public synthetic dataset, with streaming KPIs, daily retraining behind a quality gate, and an assistant over six manuals."

**Weak spot to own.** The dataset is synthetic, so the accuracy says more about the pipeline than about real machines. Be ready for questions about leakage.

## media-pulse and comms-agent

**In plain words.** A system that reads news mentions of a brand, labels them, and sends the uncertain ones to a person to check before the numbers go into a report; plus an agent that runs that whole loop on a schedule.

**15 seconds.** "Low confidence predictions go to a human reviewer; on synthetic data that lifted topic accuracy from about 0.88 to 0.96."

**Weak spot to own.** The data and the reviewer are both simulated, so the uplift is partly built in.

## The live data dashboards

Four web pages that compute statistics live from public data feeds, each with a logged test run:

- **Wikipedia edits:** compact counting tricks checked against exact counts; one estimate was off by only 0.4 percent.
- **Bitcoin order book:** a live copy of an exchange's order book, verified against the exchange's checksum on every one of tens of thousands of updates.
- **Earthquakes:** classic seismology statistics plus a log of every correction the catalog makes after the fact.
- **Space weather:** solar wind measured by a spacecraft, timed to when it reaches Earth.

**Weak spot to own.** Each ran for minutes, not months. Present them as careful measurement, not as long term studies.

## Work and research experience

- **Gravity Spy (research assistant, February 2026 to now):** built the pipeline for 276,000 plus volunteer classifications of LIGO detector glitches and a model that reaches 0.89 validation AUC. Do not claim the model finds the causes of glitches; it predicts volunteer labels.
- **JMA Wireless (Lean Six Sigma project lead, January to May 2026):** led a team of four. Making two factory lines comparable showed the automated cell was starved of materials, not broken. The roughly USD 355,000 a year is a projection, not a saving yet.
- **ProProfs (data and product analyst intern, July to August 2023):** customer and feature analysis that fed product priorities, and dashboards on under used features. Know how the 5 percent churn figure was measured.
- **Cybersecurity TA (summer 2025):** built and taught 15 plus hands on labs. This is your teaching story.
