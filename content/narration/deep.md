# Narration: explaining, structuring and telling your project stories

This module trains the part of the interview that most candidates never practise: saying things out loud. Part 1 covers general narration, from the layered answer and the concept template to thinking aloud while coding, narrating a derivation, signposting a system design, structuring analytics cases and behavioural stories, and handling "I don't know", pushback and the closing questions. Part 2 turns each of your projects and roles into stories at three lengths (15 seconds, 60 seconds, 3 minutes), with the follow ups an interviewer at a research lab will ask and the weak spots they will probe. Every project number below is copied from the project's own README or results file, with its caveat in the same sentence.

## How communication gets scored

Interviewers do not grade "communication" as charm. They grade whether they could follow your reasoning without having to pull it out of you. The Tech Interview Handbook's coding rubric, which its author describes as roughly shared across large tech companies, splits a coding round into four dimensions: communication, problem solving, technical competency and testing. Its communication bands are worth memorising because they describe behaviour you can rehearse:

| Band | What the interviewer experienced |
| --- | --- |
| Strong hire | Followed your thought process with no effort: how you understood the question, the approach, the trade offs |
| Leaning hire | Clear and organised, but they had to ask follow ups to understand parts of your approach |
| Leaning no hire | You jumped into code without explaining, or the explanation was disorganised |
| Strong no hire | Silence, even when addressed |

The gap between "leaning hire" and "strong hire" is the number of follow up questions the interviewer had to ask just to understand you. Every technique in this module reduces that number.

Behavioural rounds score communication too. The Handbook's behavioural rubric (contributed by a former Meta engineering manager) lists eight focus areas: motivation, proactivity, working in an unstructured environment, perseverance, conflict resolution, empathy, growth and communication. Interviewers also judge scope: the same "proactive change" story reads as junior if only you worked on it, senior if it moved your whole team, staff if it moved an organisation. You cannot fake scope, but you can fail to show the scope you have by telling the story badly.

System design is where narration is the whole round. The interviewing.io system design guide puts it bluntly: without communication nothing happens, because the format is a conversation. Its advice is to play tech lead and treat the interviewer as a junior engineer who will implement your design and is welcome to ask questions. Hello Interview names the most common mid level failure as not delivering a working system, which usually shows up in feedback as "time management". That is a narration failure: no plan announced, no plan followed.

Amazon is the extreme case for behavioural weight. Its own interview guide says every hiring decision is made against its 16 Leadership Principles, that answers should use the STAR structure with "I" statements, and that each STAR answer should typically run 2 to 3 minutes. Exponent's Amazon guide, built from candidate reports, describes a loop round as 60 minutes covering two principles, roughly 25 minutes per question. Twenty five minutes on one story means the first two minutes are the trailer and the rest is follow ups. Prepare for the follow ups, not just the story.

> **Remember:** The test in every round is the same: could a tired interviewer write accurate notes about your reasoning from what you said, without asking you to repeat it?

> **Watch:** [Patrick Winston, How to Speak (MIT)](https://www.youtube.com/watch?v=Unzc731iCUY&t=255s) starts at "How to Start". His chapter on job talks and oral exams starts at [41:30](https://www.youtube.com/watch?v=Unzc731iCUY&t=2490s) and the one on how to stop at [53:06](https://www.youtube.com/watch?v=Unzc731iCUY&t=3186s) (timestamps from the MIT OpenCourseWare outline).

## The layered answer

A layered answer gives the conclusion first and then lets the interviewer choose how deep to go. It has four layers, and you stop after each one unless invited to continue:

| Layer | Length | Job |
| --- | --- | --- |
| Headline | One sentence | The answer itself, in words a non specialist could repeat |
| Short | About 30 seconds | One mechanism or intuition, one concrete example |
| Medium | About 2 minutes | How it works, the main trade off, when it fails |
| Deep dive | On request | Derivation, edge cases, complexity, numbers, citations |

Why it works: interviewers have a list of signals to collect, and they cannot collect the next one while you are still delivering a lecture on the first. The headline lets them tick "knows the answer" in ten seconds. The pause after the short layer is a handover: they either say "go on", which tells you depth is wanted, or they move to the question they actually care about. Both outcomes save you time. Military writing calls the same idea BLUF, bottom line up front, and Patrick Winston's version is to open with a promise of what the listener will know by the end.

The failure modes are symmetric. Starting at the deep dive buries the answer and spends minutes the interviewer had planned for other questions. Stopping at the headline when depth was wanted looks shallow. The fix for both is the same habit: say the headline, give the short layer, then offer the next layer explicitly. "I can go into how the gradient flows through that, or into when it breaks. Which is more useful?" is a complete sentence that buys you their preference.

> **Worked example:** "What is dropout?"
> Headline: "Dropout randomly zeroes a fraction of activations during training so the network cannot rely on any single unit."
> Short: "It acts like training a large ensemble of thinned networks that share weights. At test time you use all units, scaled so the expected activation matches training."
> Medium: inverted dropout scales by one over the keep probability during training so inference needs no change; it regularises by injecting multiplicative noise; it interacts badly with batch norm statistics if placed carelessly; large pretrained transformers often use little or none.
> Deep: the ensemble view, the Gaussian dropout equivalence, Monte Carlo dropout for uncertainty, and why the scaling preserves the expectation.

How to practise it: take any card in this site, set a timer, and record three takes of the same answer at 10 seconds, 30 seconds and 2 minutes. If the 10 second take does not contain the answer, the other two are built on sand.

## The concept answer template

For "explain X" questions, use five moves in a fixed order: what it is, the intuition, a concrete example, the trade off, and when you would use it. The order matters. Definition first answers the question. Intuition next gives the interviewer a hook to remember it. The example proves you can apply it. The trade off shows judgement. "When I would use it" connects it to engineering, which is the job.

| Move | What you say | Common miss |
| --- | --- | --- |
| What it is | One sentence definition with the category: "a regularisation method", "a ranking metric" | Starting with history or the paper |
| Intuition | An analogy or mechanism in plain words | An analogy that is wrong at the edges and you do not say so |
| Example | Numbers, a tiny dataset, a case from your own work | Staying abstract |
| Trade off | What it costs, what it assumes, what breaks it | Presenting it as free |
| When to use | A decision rule against the alternative | Never naming the alternative |

The Feynman technique is the study method behind this template. Farnam Street describes it as four steps: map what you know about a concept, explain it in words a 12 year old would follow, go back to the source wherever the simple version breaks, then test it by teaching someone without notes. The second step is the one that matters for interviews. If your plain explanation needs a term you cannot define, that term is where the follow up will land.

> **Worked example:** "Explain reciprocal rank fusion."
> What: "RRF merges ranked lists by summing, for each document, one over a constant plus its rank in each list."
> Intuition: "It rewards documents that several rankers place near the top, and it ignores raw scores, so you never have to calibrate BM25 against cosine similarity."
> Example: with the usual constant of 60, a document ranked 1st by BM25 and 3rd by a dense model scores 1/61 + 1/63, about 0.0323; one ranked 2nd and 1st scores 1/62 + 1/61, about 0.0325, so it wins narrowly.
> Trade off: it throws away score magnitude, so a dense retriever that is very confident gets no extra say.
> When: as the default fusion when the score scales are incomparable; in my BEIR replication, RRF over BM25 and TAS-B scored 0.687 nDCG@10 on SciFact against 0.677 for BM25 and 0.643 for TAS-B, and it is labelled as not in the paper.

Interviewers at research labs probe the trade off and the edges. After any concept answer, expect "when does that fail?" and "what would you use instead?". Pre load both for every concept you list on your resume.

## Thinking aloud while coding

Thinking aloud means the interviewer can always see which step you are on and why. The Tech Interview Handbook's checklist gives the sequence, and each step has a sentence you can say:

1. **Restate and clarify.** "Let me restate it: given a list of intervals, return the merged list. Can intervals be unsorted? Can they touch at an endpoint, and does touching count as overlapping?" The Handbook recommends at least two or three clarifying questions and a tiny example checked with the interviewer before any code.
2. **Offer approaches before choosing.** "The brute force compares every pair, which is quadratic. If I sort by start, one pass is enough, so it is n log n for the sort and linear after. I'll go with sorting unless you want me to avoid modifying the input."
3. **Get the green light.** "Does that approach sound good before I write it?" The Handbook lists starting to code before agreement as a mistake.
4. **Narrate intent, not keystrokes.** Say what the next block is for ("this loop extends the current interval while the next start is inside it"), not what you are typing ("for i in range"). Name variables so the code narrates itself.
5. **Flag shortcuts.** "In production I'd validate the input here; I'm skipping that for time."
6. **Do not announce you are done.** Trace one example by hand, out loud, updating variables as you go, then try edge cases: empty input, one element, duplicates, everything overlapping.
7. **Restate complexity and what you would improve.**

When you are stuck, keep narrating the search, not the panic. "I'm stuck on how to avoid the second pass. What I know is the output order matches the sorted order, so maybe I can build it in place. Let me check that on the example." Saying a half formed idea out loud lets the interviewer hint, and interviewers prefer giving a small hint to watching silence.

Silence has a place: a short, announced one. "Give me thirty seconds to think about the invariant" is fine. Two minutes of unannounced silence is the "strong no hire" row of the rubric.

> **Interview trap:** Narrating every keystroke. It slows you down and hides the reasoning. Narrate decisions and invariants; let the code speak for syntax.

> **Watch:** [Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) is two hours of thinking aloud while coding. Pause before each cell runs and say what you expect it to print.

## Narrating math on a whiteboard

A derivation is a story with a fixed plot: state the goal, define every symbol, take one step at a time and say why it is legal, check the result. Interviewers asking you to derive something are testing whether you can be checked, so make each step checkable.

The script:

1. **Goal.** "I want the gradient of the logistic loss with respect to the weights."
2. **Symbols, out loud and written.** "x is one input vector, y is its label in zero or one, sigma is the sigmoid, p is sigma of w dot x."
3. **Steps with reasons.** "The loss for one example is minus y log p minus one minus y log one minus p. I'll use the fact that the derivative of sigma is sigma times one minus sigma. Chain rule gives..."
4. **Result in words.** "So the gradient is p minus y, times x: the prediction error times the input."
5. **Sanity check.** Units, a limiting case, or a number. "If p equals y the gradient is zero, which is right: no error, no update."

$$
\ell(w) = -y\log p - (1-y)\log(1-p), \quad p = \sigma(w^\top x), \quad \frac{\partial \ell}{\partial w} = (p - y)\,x
$$

Board habits that help: write the goal at the top and box it; keep a definitions column on the left that you never erase; number the lines so you can say "from line 3"; leave space between lines in case you need to insert a step (the Handbook's whiteboard space advice). When you make an algebra slip, say so and fix it in place: "that sign is wrong, the derivative of log one minus p carries a minus." Interviewers read calm self correction as a positive signal.

Say formulas in words the listener can write down: "the square root of n", "the sum over i of", "one over the length of the response". Never read LaTeX aloud.

> **Say it like this:** "Before I expand this, here is where I'm going: I want to show the per token gradient depends on the response length under GRPO and not under Dr. GRPO. Then I'll check it with two response lengths."

> **Watch:** [CS229 Lecture 4: Exponential family, GLMs](https://www.youtube.com/watch?v=8gVi4Rk21Eg) is a long run of board derivations; pause before each step and narrate it yourself.

## Signposting in system design

Signposting means announcing the structure before you fill it in, then saying where you are in it. Listeners drift, and a signpost lets them rejoin. In a 45 to 60 minute design round it also protects your time, because the interviewer can redirect you at a signpost instead of interrupting mid thought.

A delivery plan you can say in the first minute, adapted from Hello Interview's framework:

| Phase | Rough share of a 45 minute round | What you say |
| --- | --- | --- |
| Requirements | About 5 minutes | "Users should be able to..." for the top three features; then quantified qualities: "search under 500 ms" beats "low latency" |
| Core entities | A minute or two | A short list of nouns: for a RAG service, Document, Chunk, Query, Answer, Citation |
| API or interface | About 5 minutes | One or two endpoints, data flow as a list |
| High level design | The middle block | Boxes and arrows, built endpoint by endpoint, saying what state changes on each request |
| Deep dives | The last 10 or so minutes | The two or three places the design is weakest against the non functional requirements |

Hello Interview advises skipping upfront back of the envelope maths unless a number changes the design, and saying so: "I'll skip estimates for now and do the maths where it decides something." For ML system design, the number that usually decides something is latency per request or index size. EvidenceOps has a real example: its results file computes that 10 million chunks at 384 dimensions need 15.50 GB in float32 against 4.77 GB in int8, a calculation that came from the project's own capacity planner on the synthetic fixture setup, and it is what justified quantising.

Check ins are short questions at the boundaries between phases:

- "Those are the three features I'd prioritise. Anything you want added or cut?"
- "I've got a working path end to end. I see two weak spots, the feed fan out and the cache invalidation. Which would you like me to go deeper on?"
- "I'm choosing a queue here instead of a synchronous call because the scorer can lag. Does that match how you'd want it to behave under load?"

The interviewing.io guide describes a quieter skill: leave breadcrumbs that steer the interviewer toward the areas where you are strongest, so their follow ups land on your preparation. Mentioning "I'd resolve trust at query time instead of baking it into the index, and I can explain why" is a breadcrumb.

Every decision gets a reason and a condition under which you would decide differently. "I'd use Postgres here because the access pattern is relational and the volume is modest. If writes went past what one primary handles, I'd revisit." Interviewers are not grading the choice so much as whether you know what it depends on.

> **Interview trap:** Layering on complexity before you have a working end to end path. Hello Interview names this as the common way candidates never arrive at a complete design.

> **Watch:** [CME295 Lecture 7: Agentic LLMs](https://www.youtube.com/watch?v=h-7S6HNq0Vg) and [CME295 Lecture 8: LLM Evaluation](https://www.youtube.com/watch?v=8fNP4N46RRo) give you the vocabulary for ML and agent design rounds.

## Analytics and product cases

Analytics and product cases reward a visible structure more than a clever answer. Two shapes cover most prompts.

**Metric moved ("daily active users dropped 10 percent yesterday").** Say the structure first, then walk it:

1. Clarify the metric and the change: exact definition, time window, compared against what (day over day, week over week, same day last year), size of the drop.
2. Rule out measurement: logging or pipeline changes, a deploy, a timezone or definition change. A drop that is a tracking bug is the most common real answer and the most commonly skipped.
3. Shape: sudden step at one time points to a release or outage; gradual slide points to product or external causes.
4. Segment: platform, app version, region, new versus returning users, acquisition channel. Look for the segment where the drop concentrates; an aggregate can hide segments moving in opposite directions.
5. Internal versus external hypotheses, ranked by likelihood and how cheap they are to check, each paired with the cut that would confirm or kill it.
6. Recommend: what you would do now, and what number would change your mind.

**Measure or launch ("how would you measure success of a new feature?").** Goal of the feature in one sentence, the user action that shows the goal was met, one primary metric, two or three guardrail metrics (what must not get worse), how you would test it (an experiment if possible, with the unit of randomisation), and what result would make you not launch.

Narration tips specific to cases:

- Write the framework on the board as five words before filling it in, and point to it as you move.
- Number your hypotheses. "Hypothesis three" is easy for the interviewer to note and to come back to.
- State assumptions as assumptions: "I'll assume the product is mobile first; tell me if not."
- End with a decision, not a list. Listing a dozen causes in no order reads as no depth.

Your JMA Wireless engagement is a strong worked example of the measurement shape done in real life: definitions first, then the comparison that located the constraint (see the experience section).

> **Remember:** The product analytics and experimentation modules on this site carry the statistics. This section is about saying the structure so the interviewer can follow it.

## Behavioural answers with STAR and CAR

STAR is Situation, Task, Action, Result. CAR folds the first two into Context: Context, Action, Result. Amazon's own guide asks for STAR and for "I" statements that separate your contribution from the team's. Exponent's guides say the Action part should get most of the time and the Result should carry a number where you can. The Tech Interview Handbook adds a second R, Reflection, as a seniority signal: what you learned and changed afterwards. The interviewing.io article on story selection calls the reflective variant CARL, with L for learning.

| Format | Use it when | Time split for a 2 minute answer |
| --- | --- | --- |
| STAR | The task was distinct from the situation, for example you were assigned a goal inside a messy context | Situation and task about 20 seconds, action about 70, result about 20, reflection about 10 |
| CAR or CARL | Phone screens, or when the context is simple | Context about 15 seconds, action about 75, result and learning about 30 |

Rules that hold for either:

- **Open with a title.** The interviewing.io article recommends a one line headline that teases the challenge and the evidence: "the time a default setting silently cut every document to 100 tokens". It tells the interviewer which signal you are about to give them.
- **Actions are decisions.** For each action say what you did, why, and what the alternative was. The Handbook frames actions as the repeatable behaviours the interview is named after.
- **I, not we.** Credit the team in one sentence, then say what you did.
- **Results with numbers and caveats.** A number with its denominator and its limits is more credible than a big adjective. "Projected, not realised, pending finance review" is a strength when it is true.
- **Stop.** End on the result or the learning. Do not trail into a second story.

> **Interview trap:** Rushing to the result. The actions are where the skill is visible, which is why Exponent calls the Action part the substance of the answer. The other trap is memorising word for word; the Handbook recommends bullet points and practice aloud so the answer stays natural.

## Your story bank

A story bank is 8 to 10 stories you have written as bullets, each mapped to the prompts it can answer. The interviewing.io article suggests brainstorming 20 to 30 candidates, writing down for each what made it hard and what evidence proves it, deleting any story without both, and keeping 6 to 10. The Handbook suggests anchoring on 3 to 5 large projects, since one project can answer several prompts with different evidence.

Your bank, built only from facts on your public portfolio and READMEs:

| Prompt | Story | The evidence you will cite |
| --- | --- | --- |
| Conflict or competing views | JMA Wireless: functions defined Uptime and Schedule Attainment differently | Definition interviews across manufacturing, quality and supervision; a sponsor approved definitions document for five KPIs |
| Influence without authority | JMA Wireless: the constraint was upstream material flow, not the equipment people suspected | 97.3 percent uptime against 46.3 percent of weekly target; recommendation picked with a Pugh matrix from six candidates |
| Failure | EvidenceOps: the notice based retraction registry caught 0 of 3 retractions on a real slice | Measured on 5,000 real articles (parsing only, no model); fixed by adding the publisher flag as a second source, 3 of 3 |
| Ambiguity | JMA Wireless Define phase, or Gravity Spy's five undocumented metadata schemas | Schemas reconciled across 25,104 glitch subjects |
| Decision with incomplete data | Dr. GRPO: reporting one seed as "not reproduced here" instead of "refuted" | Paired bootstrap intervals; one seed against the paper's three |
| A mistake you caught | BEIR: DPR scored 0.260 against a published 0.318 because of a 100 token default | Sweep to 512 tokens matched 0.318 exactly |
| A result that surprised you | Distributed ViT: fp16 made two GPU scaling fall from 98 to 57 percent | Synthetic input isolation restored 96 percent |
| Raising the bar on your own work | handoff: three bugs kept in history; or distributed ViT's `make verify` that found five unsupported claims | 200 tests; claims traced to artifacts |
| Teaching or explaining | Cybersecurity TA: 15 plus hands on labs, grading the reasoning not only the answer | Labs across crypto, network security, pentest and forensics |
| Customer or user focus | ProProfs: customer analysis that fed feature prioritisation | 2,000 client records; dashboards on under used features |

Practise each story at two lengths: a 30 second CAR for screens and a 2 minute STAR for loops. Then write the three follow ups you most fear for each and answer them aloud.

> **Interview trap:** Telling a team success with no conflict or difficulty. The interviewing.io article's filter deletes stories where everything went smoothly, because they give the interviewer nothing to score.

## Saying I don't know

Saying "I don't know" well has three parts: say it plainly, say what you do know that is adjacent, and say how you would find out or reason toward it. Bluffing is the expensive option because interviewers at research labs ask the next question precisely where a bluff would break.

A four step script:

1. **Boundary.** "I haven't worked with ColBERT's late interaction scoring directly."
2. **Adjacent knowledge.** "I know it keeps one vector per token and scores a query against a document with a sum of maximum similarities, which is why its index is much larger than a single vector model's."
3. **Reasoning attempt, labelled.** "If I had to guess why it does well on SciFact, I'd say token level matching helps with rare scientific terms, but that's a hypothesis."
4. **How you would check.** "I'd read the scoring function in the paper and run it on one dataset against the single vector baseline."

The interviewing.io system design guide notes that interviewers will sometimes ask you to design internals you have never studied, and they are judging the approach from first principles, not recall. Treat "I don't know" as the start of that approach.

Calibrate your language to your confidence, and say which mode you are in: "I'm sure", "I believe, but I'd verify", "I'm guessing". A candidate whose "I'm sure" is always right earns trust for the rest of the loop.

> **Interview trap:** Saying "I don't know" and stopping. It ends the signal. The adjacent knowledge and the plan are what get scored.

> **Interview trap:** Inventing a number. If you do not remember a benchmark figure, say the direction and the source: "TAS-B was close to BM25 on SciFact in the BEIR paper; I don't remember the exact figure." For your own projects you should know every number cold, which is what Part 2 is for.

## Handling pushback

Pushback is usually a probe, not a verdict. The interviewer wants to see whether you can hold a reasoned position, update on a good argument, and stay pleasant either way. A three move response works in almost every case:

1. **Restate their concern in your words.** "So the worry is that query time trust lookups add latency to every request."
2. **Decide which kind of pushback it is.** New information, a different assumption, or a test of conviction. Ask if unclear: "Is that a concern about the p99 budget specifically?"
3. **Respond on the merits.** If they are right, say so and show what changes: "That's fair. If the registry lived in a remote service I'd cache it locally with a short time to live, accepting a few minutes of staleness." If you still disagree, give the reason and the condition that would change your mind: "The registry is thousands of rows, so the lookup is in memory. If it grew past what fits in memory I'd revisit."

Amazon's "Have Backbone; Disagree and Commit" principle names both halves: challenge respectfully, then commit once the decision is made. Interviewers sometimes push back on correct answers to see if you fold. The Tech Interview Handbook advises against arguing with a coding interviewer, since they know the question well; the useful middle ground is to test the claim aloud: "Let me check that on the example... with that input my loop does handle the empty case, here's the trace."

Phrases worth having ready:

- "You're right, I missed that. Here's the fix."
- "I see it differently, and here's the assumption I'm making."
- "That's a real trade off. I'd pick this side because of X, and I'd switch if Y."

> **Remember:** Changing your mind for a good reason is a positive signal. Changing it because someone frowned is not.

## Clarifying questions

Clarifying questions are part of the answer, not a delay before it. The Tech Interview Handbook says coding prompts are underspecified on purpose, to test attention to detail, and the interviewing.io design guide warns against assuming you are rebuilding Instagram when the prompt said "a photo sharing service".

What to ask, by round:

| Round | Questions that change the answer |
| --- | --- |
| Coding | Input size and range; sorted or not; duplicates, negatives, empty input; may I modify the input; what to return on invalid input |
| ML system design | What is the product goal; what is the label and how is it collected; latency and cost budget; online or batch; how is success measured offline and online; what are the failure costs |
| Analytics case | Exact metric definition; time window and comparison; which platform or market; any known launches or incidents |
| Behavioural | Rarely needed, but "would you like a technical or a people example?" is fine |
| Research discussion | "Are you asking about the training objective or the evaluation?" when a question could mean either |

Ask two or three, not ten. Then state the assumptions you are making for everything else, out loud, so the interviewer can correct them. Paraphrase the problem back before starting; the Handbook recommends it because it catches misunderstanding in five seconds instead of fifteen minutes.

> **Say it like this:** "Before I design this, three questions: is this ranking for a feed or for search, what latency budget per request, and do we have click labels or only explicit ratings? I'll assume search, 200 milliseconds and clicks unless you say otherwise."

## Questions to ask the interviewer

The closing "any questions for me?" is still being scored. The Tech Interview Handbook says candidates with no questions can seem less interested, and that the questions you ask show what you care about. Prepare two for each interviewer type, and make at least one specific to what you heard in the interview.

| Interviewer | Questions that get a real answer |
| --- | --- |
| Engineer | "What is the most costly technical decision made early that the team lives with now?" "How do you evaluate a model change before it ships?" |
| ML or research lead | "What does your eval set look like, and who decides when it is wrong?" "What is a result the team believed that turned out not to hold?" |
| Hiring manager | "What would you want me to have solved in my first few months?" "How do you know an engineer on this team is doing well?" |
| Founder at a startup | "What has to be true in the next year for the company to work?" "What would you build first if you had one more engineer?" |

Questions specific to your projects make good closers because they invite a conversation you are prepared for: "You mentioned retrieval quality. How do you handle documents that become invalid after they are indexed?" is a breadcrumb back to EvidenceOps.

Avoid questions answered on the company's front page, and leave compensation and logistics for the recruiter.

## Pacing, pauses and filler words

Pacing is about giving the listener time to write. A useful target for explanations is a little slower than conversation, with a full pause after each claim the interviewer might note. Pauses feel long to you and normal to them.

Filler words ("um", "like", "so basically", "kind of") are not a moral failing; they mostly signal that you are planning the next sentence while still talking. Three fixes work:

1. **Replace filler with a pause.** Silence of one or two seconds reads as thought. Practise by recording and counting fillers per minute, then repeating the answer with the instruction "pause instead".
2. **Signpost to buy planning time.** "There are two parts to that" gives you a structure and a breath.
3. **Shorter sentences.** Fillers cluster at clause boundaries in long sentences. One claim per sentence leaves fewer boundaries.

Hedges are a separate problem from fillers. "Sort of", "I think maybe" and "probably kind of" on a claim you are sure of undercuts it. Save hedges for real uncertainty and say the level: "I'm fairly confident" or "I'd need to check".

Winston's point about attention applies here: at any moment some share of the room has drifted, so he repeats key ideas several times. In an interview, repeat the headline at the end of a long answer: "So, in short, the gap came from the dataloader, not the all reduce."

> **Interview trap:** Speeding up when nervous. Record a mock round; most people speak noticeably faster in the first five minutes.

## Tell me about yourself

"Tell me about yourself" is a 60 to 90 second pitch tailored to the role, not a history. The Tech Interview Handbook frames it as an elevator pitch: who you are, two or three highlights with a number or a challenge, and why you fit this role, without a year by year history. Keep project detail at the headline level and let them ask.

A template for you:

1. **Now:** "I'm finishing an MS in Applied Data Science at Syracuse, graduating in December."
2. **Through line:** "I build retrieval, evaluation and agent systems, and I care most about measuring where they break."
3. **Two proofs**, chosen per role:
   - Research engineer or retrieval: the BEIR replication (the DPR truncation bug) and EvidenceOps (retraction as a correctness constraint).
   - Post training or RL: the Dr. GRPO replication (accuracy held, length claims reversed, one seed).
   - ML infrastructure: distributed ViT (98 percent to 57 percent scaling, cause isolated).
   - Data or product analyst: JMA Wireless (definitions located the real constraint) and ProProfs.
   - Agents: handoff and the tool dependency graph.
4. **Why here:** one sentence connecting the through line to their product.

> **Say it like this:** "I'm finishing an MS in Applied Data Science at Syracuse. I build retrieval, evaluation and agent systems, and the part I'm best at is measuring where they break. For example, I replicated the BEIR benchmark on a laptop and traced the one number that didn't reproduce to a 100 token default that silently truncated every abstract. I also research with Gravity Spy, the NSF funded LIGO citizen science project, where I built the ingestion and a paired spectrogram model. I'm applying here because your team ships retrieval to users who act on the answers, which is exactly the failure surface I like working on."

Keep the public version free of anything you would not want on a slide. Work authorisation, salary and personal contact details belong with the recruiter, not in this answer.

## Daily practice routine

Narration improves with reps and feedback, and the flashcards are built for that. A 30 minute daily routine:

1. **Warm up, 5 minutes.** Pick three concept cards. Say each at 10 seconds, then 30 seconds. Check the 10 second version contains the answer.
2. **Card drill, 15 minutes.** Pick five cards across levels. Read the question, answer aloud without looking, recording on your phone. Then open the card's must mention checklist and score yourself: one point per item you said, minus one per trap you fell into.
3. **Project story, 5 minutes.** One project per day, rotating. Say the 60 second version from memory, then answer two of its follow ups.
4. **Review, 5 minutes.** Play back one recording at normal speed. Note filler count, the first sentence (did it answer?), and where you lost the thread. Write one fix for tomorrow.

Weekly: one full mock with a friend or an AI interviewer asked to push follow ups, which the Tech Interview Handbook also suggests; one session explaining a project to a non technical friend, Feynman style, noting every term they stop you on.

Amazon's guide suggests recording yourself or practising with a friend so STAR answers stay concise. The recording is the part people skip, and it is the part that shows you what the interviewer hears.

> **Remember:** Score against the checklist, not against how it felt. Answers that felt fluent often skipped the trade off.

## How to tell a project story

Every project gets the same five layers, so you can scale any of them to the time you are given:

| Layer | Length | Contents |
| --- | --- | --- |
| 15 seconds | One or two sentences | What it is and the one finding that makes it interesting |
| 60 seconds | About 120 words | Problem, what you built, the result with its caveat, one hook for a follow up |
| 3 minute deep dive | Six beats | Problem; why it matters; what I built; the hardest decision and its trade off; the result with honest caveats; what I would do next |
| Follow ups | Five prepared | The questions the interviewer is most likely to ask, with answers |
| Weak spots | Two to five | What an informed interviewer will probe, and how to own it before they do |

Three rules from your own READMEs make these stories credible, and interviewers at research labs check all three:

1. **Every number has a denominator and a provenance.** "0 of 3 retractions caught, on 5,000 real articles" beats "poor recall".
2. **Caveats travel with the number in the same sentence.** "On a four article synthetic corpus with a deterministic stub adapter, the pass rate was 0.857" is honest; the same number with the caveat two minutes later is not.
3. **Lead with the surprise.** Your best projects are about the cell that disagreed (BEIR's DPR), the claim that reversed (Dr. GRPO) or the speedup that vanished (distributed ViT). The surprise is the story. The plumbing is the follow up.

Owning a weak spot means naming it before the interviewer does, saying why it exists, and saying what would fix it. "It's one seed. The intervals cover variation across problems and steps, not across seeds, so I read it as not reproduced here. Seeds 1 and 2 are about 4.4 GPU hours each and are the first thing I'd run." That answer turns a limitation into evidence of judgement.

> **Remember:** If a number is not in the project's README or results file, do not say it. If you are unsure of a number, say the direction and offer to share the repository.

## EvidenceOps

**15 seconds.** EvidenceOps is a retrieval and answer system over the PubMed Central open access papers that treats retraction status as a correctness constraint: if the best evidence has been withdrawn, it refuses and says so instead of citing it fluently.

**60 seconds.** "EvidenceOps answers questions over PubMed Central open access papers, built on one observation: an answer that cites a retracted paper is worse than a refusal, because it is fluent, correctly sourced and wrong. So retraction is a correctness constraint with its own registry, looked up at query time instead of written into the index, so a retraction published tonight applies to the next query. Retrieval is BM25 plus dense with rank fusion, and nine deterministic guardrails run before any model call. Its evaluation numbers come from a four article synthetic corpus and a deterministic stub adapter, so they show plumbing, not model quality; the reproducible facts are 107 tests and statement coverage that rose from 75 to 84 percent. On 5,000 real articles, parsed with no model, the notice based registry caught 0 of 3 retractions, which changed the design."

**3 minute deep dive.**

*Problem.* Retrieval quality work optimises relevance and assumes every indexed document is a valid thing to cite. In scientific literature that is false: papers are retracted, often years later, and the retracted paper is frequently the most relevant document for the question. No reranker fixes that, because the relevance is real.

*Why it matters.* A reader cannot detect a fluent, correctly cited answer built on withdrawn evidence. The system is also explicitly not a clinical decision tool, and its refusal of individualised medical advice is enforced on both the request and the response and scored in the evaluation suite.

*What I built.* Six services: ingest (public S3 access, JATS parsing, chunking, embedding, nightly trust sweeps), retrieval (dense plus BM25, fused with reciprocal rank fusion, reranked), guard (nine deterministic detectors, citation verification, policy decisions), gateway (routing, semantic cache, budget), evalsvc (golden set, release gates, judge calibration) and agent (permissioned tools, dry runs, approvals, a state machine). Four storage planes split by access pattern: DuckDB with dbt for metadata, a vector store, a BM25 index on the same chunk ids for exact strings embeddings blur (HbA1c, IL-6, trial numbers), and a small JSONL trust registry read on every query. The domain core depends only on the standard library, numpy and pyyaml, with zero cost defaults behind protocols (a hashing embedder, an in memory vector store, a deterministic extractive adapter), so CI runs the real pipeline with no network or GPU.

*Hardest decision and the trade off.* Trust resolved at query time instead of index time (ADR 0003). Writing trust into the vector payload is one lookup and fails silently: an article indexed in good standing and retracted eighteen months later still says ok until someone reindexes. Resolving live costs a registry lookup on the serving path and wastes storage on chunks that can never be cited. The second half of the decision was to keep retracted chunks retrievable, so the guard can say "the best evidence was withdrawn" instead of quietly serving the second best chunk with the same confidence.

*Result, with caveats.* On the four article synthetic fixture with the stub adapter, the demo question about an XG-1 biomarker returns a block with confidence 0.0573 even though retrieval scored 0.5933 and citation support 1.0; with the registry removed, the same stub pipeline produces a fluent, cited answer from the retracted paper. On the same fixture and stub, the 14 case golden set passes at 0.857 with a retracted evidence leak of 0.000, and the results file itself says the zero was measured under conditions that guaranteed it, because the fixture placed the notice next to its target. First contact with 5,000 real articles (parsing only, no model) found six bugs the fixtures hid. The worst: the parser read the PMC id from an attribute value real files do not use, so every live article parsed to an empty id and every trust lookup would have failed open while the suite stayed green. The structural finding: all 11 notices in the slice pointed at articles outside it, so the notice derived registry caught 0 of the 3 articles PMC flags as retracted; adding PMC's own flag as a second source caught 3 of 3.

*What next.* Put a real model behind the gateway (the Ollama path has not been run), grow the golden set from 14 to about 120 cases on the real corpus, hand label 50 calibration pairs instead of 8, and run the retrieval ablation that matters most: hybrid plus rerank against dense only on the same questions.

**Five follow ups.**

1. *"Why not just filter retracted papers out of the index?"* Because a filter answers a different question badly. If the best evidence is withdrawn and I silently return the second best chunk, the answer looks equally confident and the user never learns the field changed. Keeping it retrievable lets the guard distinguish "no good evidence exists" from "the good evidence was withdrawn". The cost is storage on chunks that can never be cited, and that is the trade I chose.
2. *"How do you know the registry is complete?"* It is not, and I measured how incomplete. On a 5,000 article real slice the notice based registry caught 0 of 3 flagged retractions, because any filtered corpus splits notices from their targets. Merging PMC's own flag took it to 3 of 3 on that slice. It still only sees PMC; Retraction Watch is not ingested, and coverage has to be audited against a hand checked sample.
3. *"Your metrics come from a stub. What do they prove?"* Plumbing: that retrieval reaches the right source, citations bind to the right chunks, retracted evidence is blocked and gates fire. Not model quality. Every report records the adapter and embedder names so a stub run cannot be mistaken for a model run. Swapping in bge-small even showed the harness's blind spot: pass rate rose from 0.857 to 0.929 on the same synthetic fixture and stub, yet the lexical reranker produced byte identical scores and one case passed with an answer that did not address the question.
4. *"How do you verify citations without a judge call on every one?"* Lexical overlap bands: below 0.25 the citation is rejected free, above 0.80 accepted free, and only the middle reaches a judge. On the fixture with the stub, 1 of 6 citations needed the judge, which says little because the stub is extractive. The real number has to be measured with a real model.
5. *"Why not LangGraph or Pydantic everywhere?"* ADR 0002 and 0004: the core runs with no framework so CI exercises the real pipeline in under a minute, and the agent state machine maps one to one onto LangGraph node names. The cost is hand written validation (including the Python case where a boolean satisfies an integer parameter, since bool subclasses int) and no built in streaming, checkpointing or retries.

**Weak spots and how to own them.**

- *No real model yet.* Say it in the first minute, as in the 60 second version. Then point to what is real: the 5,000 article parse, the six bug fix list, the registry recall finding.
- *The calibration set.* Eight hand labelled pairs give agreement 0.625 and Cohen's kappa 0.25 for the stub judge, with 3 false positives and 0 false negatives. Be ready for an interviewer who computes kappa: if the labels were binary with 5 supported, a judge that says "supported" every time would score agreement 0.625 and kappa 0, and 3 false positives with 0 false negatives describes exactly that judge, so check how the labels are coded before you quote 0.25.
- *Small n on link rate.* All 11 notices resolved through the structured attribute, but 10 of them were corrections against one retraction, so 100 percent means "no counterexample yet".
- *Test count.* The README says 107 tests (86 on a bare install, 21 more with extras); an earlier section of the results file lists 89 with extras. Run the suite and quote the number you just saw.
- *Lexical reranker, no figures.* Every number came from the lexical reranker, and 6 real articles of 5,000 (0.12 percent) hold their body only in figure captions, which nothing extracts yet.

> **Watch:** CME295 Lecture 8 on [inter rater agreement metrics](https://www.youtube.com/watch?v=8fNP4N46RRo&t=428s) and [LLM as a judge](https://www.youtube.com/watch?v=8fNP4N46RRo&t=1680s), the vocabulary for the kappa and judge questions.

## BEIR replication

**15 seconds.** I reproduced BEIR's claim that BM25 beats dense retrievers out of domain, on SciFact and NFCorpus on one laptop. It held for the paper's models, did not survive a 2023 embedder, and the one cell that missed traced to a 100 token default.

**60 seconds.** "I rebuilt the central claim of BEIR, that dense retrievers trained on MS MARCO lose to BM25 on new domains, on its two scientific datasets that fit a laptop, SciFact and NFCorpus. It held for the models the paper tested: BM25 still beat TAS-B and DPR on both. It did not survive bge-small, a 2023 model that scored 0.713 nDCG@10 on SciFact against 0.677 for my BM25, which I label as an extension. The best part was the cell that missed: DPR on SciFact came in at 0.260 against a published 0.318. The sentence-transformers port of DPR's context encoder ships a 100 token limit that silently truncates abstracts; at 512 it matched 0.318 exactly. NFCorpus hid the bug, with 38.2 judged documents per query against SciFact's 1.1."

**3 minute deep dive.**

*Problem.* BEIR (Thakur and colleagues, NeurIPS 2021 Datasets and Benchmarks) reported that dense retrievers trained on MS MARCO look excellent in domain and then lose to a bag of words baseline on new domains. On SciFact and NFCorpus, the paper's BM25 scored 0.665 and 0.325 nDCG@10.

*Why it matters.* That result shaped the default for a generation of RAG systems: start with BM25 or hybrid. If it no longer holds, the default should change, and the only way to know is to rebuild it.

*What I built.* The official BEIR release (corpus, queries, test qrels), scoring only judged queries as BEIR does, with document and query counts matching the paper (SciFact 5,183 and 300, NFCorpus 3,633 and 323). BM25 reimplemented in Python with Anserini's k1 of 0.9 and b of 0.4, Porter stemming and the Lucene English stopword list. Dense retrieval through sentence-transformers with each checkpoint's own similarity: dot product for TAS-B and DPR, cosine plus the query instruction for bge. Metrics from `pytrec_eval`, the same trec_eval bindings BEIR uses, so the metric code cannot be the difference. Every number in the README comes from a run directory with `report.json`.

*Hardest decision and the trade off.* Reimplementing BM25 instead of running Anserini, which needs a Java runtime the machine did not have. My version concatenates title and body into one field and uses exact document lengths, where Lucene indexes two fields and quantises lengths to one byte. That is the likely source of the two inexact BM25 cells, plus 0.012 on SciFact and minus 0.005 on NFCorpus. I stated the deviation up front instead of tuning until the numbers matched.

*Result, with caveats.* Reproduced nDCG@10: BM25 0.677 and 0.320, TAS-B 0.643 and 0.319 (both exact), DPR 0.318 and 0.188 after the fix. bge-small-en-v1.5, 33M parameters and not in the paper, scored 0.713 and 0.343, beating BM25 on both. Hybrid RRF over BM25 and TAS-B, also not in the paper, scored 0.687 and 0.341 and took the best SciFact Recall@100 at 0.948. The DPR story: the first run gave 0.260. The tell was that encoding only the title scored 0.261, the same as title plus text, which only makes sense if the body is being discarded. Sweeping `max_seq_length` with `title [SEP] text` encoding gave 0.276 at 100, 0.298 at 256, 0.315 at 350 and 0.318 at 512, with Recall@100 rising from 0.601 to 0.727, matching both published numbers.

*What next.* Run Anserini to close the BM25 gap, report capped Recall@100 the way the paper's Table 12 does, add a cross encoder reranking row, and extend to more BEIR datasets with few judged documents per query, since those are the ones that expose ranking damage.

**Five follow ups.**

1. *"Why did NFCorpus hide the bug?"* NFCorpus has 38.2 judged documents per query and SciFact 1.1. With dozens of relevant documents, a degraded ranking still puts some of them in the top 10, so nDCG@10 barely moves. With about one relevant document per query, losing that one document from the top 10 drops the query's score to zero. Datasets with sparse judgements are the sensitive instrument.
2. *"Why does the similarity function matter?"* Each checkpoint was trained with one scoring function. TAS-B and DPR were trained for dot product, so cosine changes the ranking by normalising away the norm the model learned to use; bge expects cosine and a query instruction. The README calls getting this wrong the easiest way to manufacture a fake negative result about dense retrieval.
3. *"Why did RRF beat both of its inputs?"* BM25 and TAS-B make different mistakes, lexical misses versus semantic drift, and RRF rewards documents that both rank highly using only ranks, so it needs no score calibration. With the constant 60, a document ranked 1st and 3rd outscores one ranked 1st by one system and absent from the other.
4. *"What is the difference between capped and plain Recall@100?"* Capped recall divides by the smaller of 100 and the number of relevant documents, so a query with 150 relevant documents can still reach 1. Plain recall divides by all relevant documents. They agree whenever a query has fewer than 100 relevant documents, which is every SciFact query but not every NFCorpus query.
5. *"What would you change?"* Run the real Anserini BM25 so the baseline is identical, and add more datasets from the benchmark. I would also add an automated check that compares each model's configured maximum sequence length against the corpus length distribution, since the DPR failure mode exists in any RAG system whose chunks are longer than its embedder's limit.

**Weak spots and how to own them.**

- *Two datasets.* They are the two smallest scientific ones, chosen to fit a laptop. Say so, and say the claim is tested on scientific text only.
- *BM25 is not Anserini.* Own the plus 0.012 and minus 0.005 and their likely cause.
- *Extensions are not reproductions.* bge-small and hybrid RRF are labelled as not in the paper everywhere. Keep that label when you speak.
- *A sentence in your own README.* It says BM25 beats every dense model evaluated on these two datasets, but its own table lists ColBERT at 0.671 on SciFact against BM25's 0.665. Fix the sentence before an interviewer reads it aloud to you.
- *Two different 100 token numbers.* The first run gave 0.260; the sweep at 100 tokens gave 0.276 because it used `title [SEP] text` encoding. If asked, explain the encoding difference rather than looking inconsistent.

> **Watch:** CME295 Lecture 7 on [BM25](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=2065s) and [retrieval evaluation with NDCG and MRR](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=2869s), and CME295 Lecture 2's [BERT deep dive](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=4298s) for the encoder family behind DPR and TAS-B.

## Dr. GRPO replication

**15 seconds.** I replicated Dr. GRPO on Qwen2.5-1.5B with verifiable math rewards. On one seed, the accuracy claim held (51.6 against 51.2 percent on MATH500) and both length claims came out reversed.

**60 seconds.** "I replicated Dr. GRPO, a 2025 paper arguing that GRPO's per response length normalisation lets wrong answers grow long. I ran R1-Zero style RL on Qwen2.5-1.5B with the authors' grader and data, with both arms trained on the same 16 questions per step and evaluated on the same 500 MATH500 problems. On seed 0 the accuracy claim held: 51.6 against 51.2 percent, a paired difference of plus 0.4 points with an interval from minus 3.4 to plus 4.0. Both length claims reversed: Dr. GRPO ran longer, including on wrong answers, 680 against 545 tokens. I read it as not reproduced here, not refuted, because it is one seed, 100 LoRA steps and a 1024 token budget. Before trusting the loss I tested its per token gradient against the closed form."

**3 minute deep dive.**

*Problem.* Liu and colleagues (2025, "Understanding R1-Zero-Like Training: A Critical Perspective") argue that GRPO, the objective behind DeepSeek-R1-Zero, has two biases. It averages each response's token losses over that response's own length, and it divides each question's advantages by the standard deviation of the group's rewards. Dr. GRPO removes both. The paper reports that length keeps growing under GRPO and not Dr. GRPO, that Dr. GRPO's wrong answers are substantially shorter, and that accuracy is maintained.

*Why it matters.* Length inflation is costly at inference time and can masquerade as "more reasoning". If the cause is an objective term, it is cheap to fix.

*What I built.* Qwen2.5-1.5B base, the paper's prompt template, reward, training data and eval set, with the authors' grader vendored unmodified. Each step samples 8 responses for each of 16 questions, 128 in all. The two arms differ in exactly two flags. LoRA rank 64 on all seven projections, a vLLM sampler and a PyTorch learner on Kaggle's two T4s in fp16.

The objective, per response $o$ with advantage $A$:

$$
\nabla_{\text{GRPO}} \propto -\frac{A}{|o|}\ \text{per token}, \qquad \nabla_{\text{Dr. GRPO}} \propto -\frac{A}{B}\ \text{per token, with } B \text{ the generation budget}
$$

Summed over a response's tokens, GRPO gives every response the same total weight $A$ whatever its length, while Dr. GRPO's total grows with length as $A|o|/B$. With $A = -1$ and a budget of 1024, a 100 token wrong answer gets $-0.01$ per token under GRPO and a 400 token one $-0.0025$, so the long wrong answer is penalised less per token; under Dr. GRPO both get about $-0.00098$ per token and the longer one carries four times the total penalty. The standard deviation term: rewards of 1, 1, 1, 0 give advantages of about 0.58 and minus 1.73, while 1, 0, 1, 0 give plus and minus 1, so near solved or near hopeless questions are upweighted.

*Hardest decision and the trade off.* Stopping at seed 0. The paper used three seeds; each extra seed costs about 4.4 GPU hours. I chose to publish one seed with paired bootstrap intervals and to word the verdict as "not reproduced here". The trade off is that the intervals capture variation across problems and steps, not across seeds, and two identical runs drift apart after the first update because fp16 backward passes on GPUs are not bit reproducible.

*Result, with caveats.* Dr. GRPO was 99 tokens longer per step on average [77, 123] and 239 longer over the last third of training [217, 262]; the length trend was minus 0.2 tokens per step for GRPO and plus 3.0 for Dr. GRPO. Wrong answers ran 680 against 545 tokens, and on the 199 problems both arms got wrong Dr. GRPO ran 120 tokens longer [80, 160]. Accuracy: 51.6 against 51.2 percent, paired difference plus 0.4 points [minus 3.4, plus 4.0]. Half the mechanism did appear: GRPO's correct answers were shorter, 262 tokens against 399. Truncation does not explain the gap: 55 of 500 final responses hit the 1024 token cap in each arm. For the first 50 steps the arms were indistinguishable.

*What next.* Seeds 1 and 2, then a longer run with a larger generation budget, since the effect may need room to grow past 1024 tokens.

**Five follow ups.**

1. *"How do you know the trainer is correct?"* I tested the objective before trusting it: the per token gradient equals minus A over the response length under GRPO and minus A over the budget under Dr. GRPO; PyTorch and MLX losses agree to 1e-6; batched padded log probabilities match an fp64 per sequence forward to 1e-5; merged LoRA weights reproduce the adapter model's outputs; and during training the sampler's and learner's log probabilities for the same tokens differed by only 0.002 to 0.004 nats, so the sampled policy was the trained policy.
2. *"Why LoRA and a learning rate of 5e-5 when the paper used 1e-6?"* Free GPUs have 15 GB, so full fine tuning was out, and LoRA conventionally runs 10 to 100 times higher learning rates. Every deviation applies to both arms, and neither term under test depends on them, but they bound how far the result generalises.
3. *"Why 1.5B and not 0.5B?"* Qwen2.5-0.5B scored 0 of 128 under the strict answer format, so GRPO would get no reward signal and no gradient. 1.5B is also the model in the paper's figure.
4. *"Could truncation explain the reversal?"* 55 of 500 final responses hit the cap in each arm, and in training the arms truncated 8.2 and 9.2 of 128 responses per step, so truncation is similar across arms.
5. *"What engineering problems did you hit?"* vLLM on a T4 ran at 350 tokens per second; the cause was per request sampling seeds, not LoRA, and removing them tripled throughput. PyTorch reports bf16 as supported on a T4 by emulating it, so precision is keyed on compute capability. Building all merged weights at once ran the sampler out of memory, so they stream one tensor at a time.

**Weak spots and how to own them.**

- *One seed.* Lead with it. "Not reproduced here" is the claim; "refuted" would be wrong.
- *Scale.* 100 steps against several hundred, 1024 tokens against 3000, one on policy update per 128 responses against eight sequential updates per 1024.
- *Unexplained structure.* GRPO's length collapse between steps 55 and 80 is unexplained. Say so; do not invent a mechanism.

> **Watch:** CME295 Lecture 6 on [GRPO](https://www.youtube.com/watch?v=k5Fh-UgTuCo&t=3464s), [length bias](https://www.youtube.com/watch?v=k5Fh-UgTuCo&t=4574s) and [DAPO and Dr. GRPO](https://www.youtube.com/watch?v=k5Fh-UgTuCo&t=5100s); [CS329A Part 6: Train Time Scaling and Scaling RL](https://www.youtube.com/watch?v=yVnmHSAy3ck) for the wider setting.

## handoff

**15 seconds.** handoff is a supervisor orchestrated agent system for incident response where every escalation has an owner, a deadline and a declared behaviour for silence, so an unanswered page runs a safer substitute instead of hanging.

**60 seconds.** "handoff is a supervisor agent system for incident response where escalations have deadlines. Most human in the loop is a confirm dialog that blocks forever, and what actually happens at 3am is that nobody answers. Here each tool declares its risk, whether it is reversible, its blast radius and a strictly safer substitute, and policy is a YAML table: a high risk action pages the primary with 180 seconds to answer, and on expiry runs the safer fallback. So an unanswered database failover becomes read only mode, which loses no in flight transactions and a human can undo. No expiry rule can approve an action nobody looked at. The cluster is simulated and the workers are rule based, which I say up front; it has 200 tests, 11 scenarios and no dependencies."

**3 minute deep dive.**

*Problem.* An agent parked on an unanswered approval prompt is worse than one that never asked: the incident keeps burning and nothing is watching it.

*What I built.* A supervisor decomposes an incident, fans out to five workers (four report, one proposes) and is the only component that can call a mutating tool, because workers receive a registry filtered to read only tools. Risk lives on the tool, and `build_registry` refuses a substitute that is missing, self referential or not actually lower risk, so a bad declaration fails at import. Disagreement is escalated, not averaged: a hypothesis with confident support and contradiction is contested, and any action on it has its risk raised one level. Deadlines fire in a caller driven `tick`, with nothing sleeping on a background thread, so each deadline is tested at its exact boundary. The policy table runs from read only (no approval) to critical (600 seconds, hold, quorum of 2 across three tiers), and four misconfigurations are refused at load time.

*Hardest decision and the trade off.* No ALLOW on expiry (ADR 0004). The four expiry behaviours are escalate, safe fallback, drop and hold. Forbidding "approve after timeout" means some urgent actions wait for a human, which is slower in the cases where the action was right; the trade buys the guarantee that nothing irreversible ever runs unseen.

*Result, with caveats.* 200 tests and 11 scenarios passing, suite wall time 0.08 to 0.10 seconds, 3,666 source lines across 35 modules, 15 tools across five risk levels, no third party dependencies, all on a simulated cluster with rule based workers and no claim about production.

*What next.* LLM backed workers behind the existing `Worker` interface, to test whether the escalation layer holds up when workers are non deterministic, a real scheduler for `tick`, and a pager attached to `Journal.subscribe`.

**Five follow ups.**

1. *"Why is escalation a policy and not a callback?"* As data, it can be reviewed in a pull request by someone who does not read Python, and misconfigurations are caught at load time (ADR 0003).
2. *"Why raise risk on disagreement?"* Uncertainty about the diagnosis and danger in the fix are different problems with the same remedy: involve a person. A low risk restart that would run unattended stops and asks when the diagnosis is contested.
3. *"Tell me about a bug."* The contested risk bump was unreachable: the planner only received confirmed hypotheses, so a contested diagnosis produced no proposal. It read like a guarantee and was decoration. A second one: the engine escalated a contested action at medium, then looked up the rule by the tool's declared low risk, whose empty ladder dropped the action, which reads in the journal like a policy decision.
4. *"How do you test a 180 second deadline without waiting 180 seconds?"* Time is an input. `tick` is driven by the caller, nothing sleeps, so a test advances the clock to exactly the boundary and asserts which side of it the decision lands on. That is also why the whole suite runs in about a tenth of a second. The cost is that a deployment needs a scheduler to call `tick`, which is deliberately missing.
5. *"Why no agent framework?"* Explicit state machines (ADR 0001) make every transition testable, and the zero dependency core (ADR 0002) keeps the suite at about a tenth of a second.

**Weak spots.** Simulated cluster; rule based workers with no LLM anywhere; no scheduler; no notification channel; a static roster of seven people with no rotation or timezones. All five are in the README's limitations, so name them before you are asked. The public portfolio quotes a different test count from the README; quote the README or run the suite first.

> **Watch:** [CME295 Lecture 7: Agentic LLMs](https://www.youtube.com/watch?v=h-7S6HNq0Vg) for the agent vocabulary interviewers will use.

## Distributed ViT on CIFAR-10

**15 seconds.** I trained a vision transformer on CIFAR-10 across two T4 GPUs, proved the two GPU run computes the same gradient as one, and then found why mixed precision made the speedup vanish: the host could not decode images fast enough.

**60 seconds.** "I trained a ViT on CIFAR-10 across two T4 GPUs and built it so every claim was checkable. First a test that a two rank DDP run computes the same gradient as one process with the doubled batch, shipped with a deliberately broken twin to show the test can fail. Then the surprise: in fp32 two GPUs scaled at 98 percent efficiency, but in fp16 the same setup fell to 57 percent. Each GPU got twice as fast, and Kaggle's 4 vCPUs could not decode images fast enough to feed the second one. Feeding synthetic tensors with no dataloader restored 96 percent, so the all reduce was never the bottleneck. I also made the code refuse impossible topologies, like distributed training on Apple's MPS, instead of silently copying to CPU."

**3 minute deep dive.**

*Problem.* Most distributed training results cannot be checked: you cannot tell whether the two rank job trained the same model as the baseline, or whether the baseline was tuned as carefully.

*What I built.* A ViT written out by hand, 1,806,538 parameters, patch size 4 on 32 by 32 images, so each all reduce moves about 7 MB of fp32 gradients. DDP and FSDP, automatic mixed precision, gradient accumulation inside `no_sync()`, a sweep runner and a dashboard generated only from run artifacts.

The correctness test rests on one identity. DDP averages gradients, so with $R$ ranks each holding $B$ examples and a mean reduced loss,

$$
\frac{1}{R}\sum_{r=1}^{R} \frac{1}{B}\sum_{i \in \text{rank } r} \nabla \ell_i \;=\; \frac{1}{RB}\sum_{i=1}^{RB} \nabla \ell_i
$$

which holds only when every rank has the same $B$. The test spawns a real gloo process group and checks it to `rtol=1e-4`, and its twin computes the wrong reduction to prove the check would fail.

*Hardest decision and the trade off.* How to find the cause of the 57 percent. Raising dataloader workers and watching throughput would give a correlation. Instead, a synthetic input mode serves pre normalised tensors from memory, removing decode and augmentation entirely, so one rank against two measures compute plus communication and nothing else. It restored 1.92 times at 96 percent. The trade off is a second code path to maintain, which pays for itself the first time it isolates a cause.

*Result, with caveats.* On 2 T4s with nccl over 30 epochs: 1,853 images per second on one GPU in fp32 and 3,637 on two (98 percent), 3,810 on one in fp16 and 4,376 on two (57 percent). FSDP ran at 0.66 times DDP's throughput, since a 1.8M parameter model has almost nothing to shard. On the M5 laptop, fp16 autocast alone was slower than fp32, 726 against 1,061 images per second, because eager mode casts around every operator, and the same config under `torch.compile` reached 2,361, 2.23 times the baseline. Moving from 1 to 4 gloo ranks on the 10 core M5 raised throughput from 552 to 879 images per second, the opposite of the slowdown the suite was built to show, so the prediction was deleted.

*What next.* Faster decode (GPU side augmentation or pre decoded tensors) to recover fp16 scaling on real data, and more ranks on a host with more cores.

**Five follow ups.**

1. *"Why does accumulation need `no_sync()`?"* Without it DDP all reduces after every microbatch backward, so accumulation pays one full communication per microbatch and gets slower instead of cheaper. Wrapping the non final microbatches skips the reduction until the last one.
2. *"Why did fp16 hurt scaling if communication did not change?"* Efficiency is speedup over ideal speedup. Halving compute time per step leaves the fixed host side cost (decoding images on 4 vCPUs) as a larger share, so the second GPU waits for data. Scaling efficiency is a property of the whole pipeline, not of the interconnect.
3. *"Why refuse distributed MPS?"* gloo cannot all reduce an MPS tensor and nccl is CUDA only. Copying to CPU and calling it distributed MPS would produce numbers nobody could trust. `resolve_backend` raises `UnsupportedTopology` with the three paths that do work.
4. *"How do you stop yourself from quoting wrong numbers?"* `make verify` extracts every number with a unit from the README and results, and fails unless it rounds from an artifact value or a ratio of two artifact values on the same hardware. It found five unsupported claims the first time it ran.
5. *"Why fp16 and not bf16 on the T4?"* The T4 is compute capability 7.5, which has no native bf16, so `pick_amp_dtype` reads the card and uses fp16 with a gradient scaler.

**Weak spots.** CIFAR-10 is small, so this is a systems measurement, not an accuracy result; two GPUs on one host say nothing about multi node scaling; and the Kaggle host's 4 vCPUs are an unusual constraint. Own all three: the point is the method for locating a bottleneck.

> **Watch:** Karpathy's GPT-2 reproduction on [fp16 and gradient scalers](https://www.youtube.com/watch?v=l8pRSuU81PU&t=5978s), [torch.compile](https://www.youtube.com/watch?v=l8pRSuU81PU&t=6495s), [gradient accumulation](https://www.youtube.com/watch?v=l8pRSuU81PU&t=9249s) and [DDP](https://www.youtube.com/watch?v=l8pRSuU81PU&t=10012s).

## Tool dependency graph

**15 seconds.** I built a graph from 1,391 tool schemas that tells an agent which tools must run before which, and which arguments must be asked of the user, because agents fail on handles like a thread id they cannot invent.

**60 seconds.** "Agents fail tool calls for a boring reason: the call needs an argument they cannot produce, like a thread id that only a listing call returns. I built a dependency graph from 1,391 tool schemas across two large toolkits that says which tools must run first and which inputs must come from the user. Matching input names to output names fails for three reasons: generic names like id collide across services, a bare id's meaning lives in its parent path, and responses embed almost every object. So handles resolve to namespaced entities, paths resolve in three tiers, and producers are ranked by the head noun of the tool's name. Tools and entities are both nodes, which keeps edges to producers plus consumers. Of 2,728 required inputs, 65 percent resolve to an entity, and 99.9 percent of those have a producer."

**3 minute deep dive.**

*Problem.* `REPLY_TO_THREAD` needs a `thread_id`, an opaque handle that only exists because a prior call returned it. An agent without that ordering either hallucinates the handle or asks the user for something the user cannot know.

*What I built.* JSON Schema flattening that knows nothing about any service, an entity registry with service inference and producer ranking, a bipartite graph builder and a static HTML viewer. Resolution runs leaf name first, then the nearest meaningful ancestor (so `threads[].id` is a thread), then the producing tool's own object. Each input lands in one of five buckets: tool derived, user supplied, control, enum or constant, and unresolved.

*Hardest decision and the trade off.* Primary versus incidental producers. A workflow run response contains a repository, so taken literally a cancel tool "produces" a repository name, true and useless. A producer is primary when yielding the entity is what the tool is for, judged by the head noun of its slug's object phrase (English compounds are head final, so `CREATE_A_REPOSITORY_VARIABLE` yields a variable, not a repository) and by how shallow the field sits. The trade off is a heuristic that gets common cases right and will get exotic ones wrong.

*Result, with caveats.* 1,391 tools, 46 canonical entities, 3,278 edges, 7,101 input slots of which 2,728 are required. 1,770 of 2,728 required slots (65 percent) resolve to an entity, 1,768 of those 1,770 (99.9 percent) have at least one producer, 1,238 of 1,391 tools (89 percent) have a real output schema, and the slug inference fallback fired zero times, so every edge traces to a declared output field. 568 handle shaped slots still resolve to nothing.

*What next.* Transitive planning (chaining "what comes immediately before" into a full plan) and growing the registry beyond 46 entities, using the unresolved list as the to do list.

**Five follow ups.**

1. *"Why bipartite?"* One entity, the repository owner, is required by hundreds of tools and produced by dozens, so tool to tool edges would number producers times consumers. With entities as nodes it is producers plus consumers, the hubs become visible, and the two hop path is exactly "run A before B".
2. *"Isn't 99.9 percent suspiciously high?"* It is coverage of slots that resolved to an entity, not of all required inputs. The honest headline is that 65 percent of required inputs are typed handles and almost all of those can be satisfied by another tool.
3. *"How did you validate it?"* Eight hand written assertions in `check.py`, each of which started out failing; four are the bugs found along the way, including the bare id under `threads[]`, cross service producers leaking into a plan, requiredness not compounding through an optional parent, and the head noun error.
4. *"Where does it get things wrong?"* Schemas thinner than reality: one contacts search tool declares no email field, so it gets no credit for supplying an address. The graph reflects the catalog, not the API behind it.
5. *"How would an agent use it?"* Before a call, look up each required slot: if tool derived, run a producer first; if user supplied, ask; if unresolved, flag it rather than guess.

**Weak spots.** The registry is hand built; producer ranking is a heuristic; no transitive planning; and the raw catalog is not redistributed, only the derived graph and an eight tool fixture, so an interviewer cannot rerun the full build.

## ZTF transient pipeline

**15 seconds.** A streaming pipeline that classifies real astronomical alerts from the Zwicky Transient Facility, paired with a dbt warehouse layer that turns the alerts into tested, governed tables.

**60 seconds.** "This is a pipeline over the Zwicky Transient Facility's public alert stream, which delivers optical transient alerts as Avro over Kafka, each with three image cutouts: science, template and difference. A multimodal model scores each alert from the cutouts plus tabular features, predictions land in a warehouse, and dbt builds staging models, object lightcurve marts and an incremental nightly summary, with tests and a PSI drift monitor on predictions. It defaults to DuckDB so it builds with no cloud account, and an Airflow DAG runs download, load, score, dbt build and drift check in order. I replay a real archived night through my own Kafka broker, because the live broker needs IP whitelisting. The honest limit is the classifier: the labels are coarse, so it demonstrates the pipeline, not a publishable classifier."

**3 minute deep dive.**

*Problem.* Time domain surveys produce far more alerts than anyone can look at, most of them artifacts. Difference imaging, subtracting a template sky from a new image, leaves what changed.

*What I built.* Two planes that meet in the warehouse: a streaming ML plane (Avro deserialisation, a stream scorer in batch and Kafka modes) writing `raw_predictions`, and a warehouse plane modelling raw alerts through dbt. Staging models are typed and deduped with `observed_at` derived from Julian date; an intermediate lightcurve model uses window functions for detection sequence, magnitude delta, cadence and running peak; marts include the classified transients product table, an incremental daily summary and model monitoring. Tests cover not null, unique, accepted values, relationships, dbt_utils range checks, source freshness, and a singular test against double scoring.

*Hardest decision and the trade off.* Flattening prior detections in Python at load time so the SQL is portable across DuckDB, BigQuery and Snowflake. The cost is losing the native array; the README documents the `unnest` pattern for BigQuery and `lateral flatten` for Snowflake if you want it back.

*Result, with caveats.* CI builds the whole dbt project on DuckDB from generated seeds on every push. The sample fixture is a CI fixture, not the project's data; the classifier's labels are coarse; and there is no validation against community broker labels yet.

*What next.* Benchmark the classifier against community broker labels (Lasair, ALeRCE and Fink expose them), and move from replay to the live broker.

**Five follow ups.**

1. *"Why replay instead of live?"* The live UW broker needs your IP whitelisted and each night's topic expires after about two weeks; replaying the archive puts real payloads through a real broker without that dependency.
2. *"How do you prevent double scoring?"* A singular dbt test guards against it, and staging dedupes alerts.
3. *"What does the drift monitor measure?"* The population stability index of the prediction distribution over time, which flags a change in what the model sees before labels arrive.
4. *"Why DuckDB?"* Anyone can build the analytics layer with no account; the same models port to BigQuery or Snowflake with a profile change.
5. *"How good is the classifier?"* Say plainly that it is not benchmarked; the labels are coarse and this is a systems project first.

**Weak spots.** No classifier metric in the README, so do not quote one. The portfolio describes a real night run with counts that the README does not contain; either add them to the README with the log, or leave them out of the interview.

## ShopFloor AI

**15 seconds.** A predictive maintenance platform on the AI4I 2020 dataset: a gradient boosting failure classifier tracked in MLflow, Kafka replaying sensor data for live KPIs, a gated daily retrain in Airflow, and a RAG assistant over six maintenance SOPs.

**60 seconds.** "ShopFloor AI is a predictive maintenance platform on the AI4I 2020 dataset, 10,000 synthetic machine records with a 3.39 percent failure rate. I engineered 16 features from five sensor readings, such as power from torque and speed and the gap between process and air temperature, trained logistic regression, random forest and gradient boosting, and tracked all three in MLflow. Gradient boosting was selected at 0.85 F1 and 0.96 AUC. Around it, Kafka replays the sensor data to compute rolling KPIs, a daily Airflow DAG retrains behind a data quality gate and only registers a model above 0.80 F1, and a RAG assistant answers questions over six SOPs. It runs as seven Docker services. The caveat is that AI4I is synthetic, so the F1 says more about the pipeline than about real machines."

**3 minute deep dive.**

*Problem.* Predict machine failure before it happens, and give operators the procedure when it does.

*What I built.* Feature engineering (`temp_diff`, `power_W` as torque times angular speed, `wear_torque_product`, power flags at 3,500 W and 9,000 W, a heat dissipation risk flag), three models in MLflow with the best registered, a FastAPI backend with `/predict` and `/ask`, LangChain plus ChromaDB plus the OpenAI API for the SOP assistant, a Kafka producer and consumer for rolling KPIs and alerts, a Streamlit dashboard with four pages, and 15 unit tests.

*Hardest decision and the trade off.* Which model to promote. Gradient boosting had the best F1 (0.8548) and random forest the best AUC (0.9823 against 0.9591). The README records the selection on F1; be ready to defend that F1 at a chosen threshold matches a decision made at a threshold, and to say that at a 3.39 percent base rate a precision recall curve would be more informative than ROC AUC.

*Result, with caveats.* Gradient boosting 0.8548 F1 and 0.9591 AUC; random forest 0.8387 and 0.9823; logistic regression 0.3054 and 0.9455. The data are synthetic; streaming replays history; the assistant depends on six bundled SOPs.

*What next.* SHAP attributions in the prediction API, multi model comparison in the DAG, and real MES or SCADA data.

**Five follow ups.**

1. *"Why is logistic regression's AUC 0.95 but its F1 only 0.31?"* AUC measures ranking across all thresholds; F1 is measured at one threshold. With 3.39 percent positives, a model can rank well and still have poor precision and recall at the default threshold. Tuning the threshold or reweighting classes would move its F1.
2. *"Did you use the failure mode flags as features?"* The dataset ships five failure mode indicators alongside the target. If any were features, that is leakage. Know your answer from the code before the interview.
3. *"Are your engineered features just the simulator's rules?"* AI4I is synthetic, and several features (power flags at 3,500 and 9,000 W, tool wear over 240) mirror the operating envelope written into the bundled SOPs. Read the dataset's documentation, and if its failure modes are rule generated, say first that a high F1 partly measures how well the features recover those rules.
4. *"Why gate promotion at 0.80 F1?"* A floor stops a bad batch from reaching the registry; a relative check against the current model would be the next improvement.
5. *"How does the RAG part avoid hallucinating procedures?"* Six SOPs only; answers depend on retrieval over those documents. Without an evaluation set for the assistant, say that its quality is unmeasured.

**Weak spots.** Synthetic data, possible leakage questions, no evaluation of the RAG assistant, and the README's claim of F1 "across 5 failure modes" for what is a binary target. Answer each from the code, and lead with the synthetic caveat.

## media-pulse and comms-agent

These two are one story: media-pulse is the measurement system and comms-agent is the automation that runs it.

**15 seconds.** media-pulse classifies brand mentions for sentiment and topic, sends the low confidence tail to a human review queue with an audit log, and reports comms metrics plus a review of the model's own calibration; comms-agent wraps that loop in an explicit state graph with a conditional review gate.

**60 seconds (media-pulse).** "media-pulse is a communications measurement system. It ingests brand mentions, classifies each for sentiment and topic with a confidence attached, routes predictions below a confidence threshold to a human review queue, writes every correction to an immutable audit log, and reports share of voice, message pull through and sentiment over time, alongside a review of the classifier itself: held out accuracy, expected calibration error and accuracy by outlet type. On the bundled synthetic corpus of 120 mentions, review lifted held out topic accuracy from about 0.88 to 0.96 and sentiment accuracy from about 0.50 to 0.85. About 30 percent of the articles genuinely span two topics, which is what gives review something real to catch."

**60 seconds (comms-agent).** "comms-agent is the automation layer on top of media-pulse. It is a small state graph engine with named nodes, fixed and conditional edges, per node retries, a max step guard and a full trace, and the agent runs plan, ingest, classify, a review gate, aggregate and report. The gate is a conditional edge: if any mention came back below the confidence threshold it routes to human review, otherwise it skips to aggregation. On 24 sample mentions it flagged 9 and the reviewer corrected 7 before anything was reported. The default LLM is a deterministic mock and the reviewer is an oracle in the demo, so this demonstrates orchestration and the gate, not model quality."

**3 minute deep dive.**

*Problem.* Comms teams act on numbers produced by classifiers that are wrong on the hard tail, and they need to know which numbers a human checked.

*What I built.* Pluggable ingestion on one schema (a sample source and an RSS adapter), VADER and TF-IDF with logistic regression when installed and numpy fallbacks when not, a review queue that overwrites model output and logs every action, metrics including reach weighted share of voice, and a four tab Streamlit dashboard. comms-agent adds the graph engine with no framework, so porting to LangGraph is mechanical.

*Hardest decision and the trade off.* Where to put the confidence threshold. Lower sends fewer items to people and leaves more errors in the report; higher costs reviewer time. The calibration review exists so the threshold can be set on confidence scores that mean something.

*Result, with caveats.* The accuracy uplift is measured on synthetic data with an oracle style correction, so it measures how many errors fell below the threshold, which is a property of the confidence scores, more than it measures human review in practice.

*What next.* Real mentions, a real reviewer UI with inter reviewer agreement, and an LLM classifier behind the existing `predict(text)` interface.

**Five follow ups.**

1. *"What does the 0.50 to 0.85 sentiment uplift tell me?"* That the confidence score separated many wrong predictions from right ones on this synthetic corpus. A baseline of 0.50 also says the default sentiment model is weak here; I would report that first.
2. *"Why calibration error and not just accuracy?"* The routing rule uses confidence, so if confidences are miscalibrated the gate routes the wrong items.
3. *"Why accuracy by outlet type?"* Averages hide where the model is weakest, and the review policy may need a lower threshold for that subgroup.
4. *"Why write your own graph engine?"* To make the control flow explicit and auditable with zero dependencies; the same nodes map onto LangGraph.
5. *"What would break with a real reviewer?"* Reviewers disagree and get tired; I would measure inter reviewer agreement and review time per item before trusting the uplift.

**Weak spots.** Synthetic, deterministic data about a fictional brand; mock LLM; oracle reviewer; the uplift is partly by construction. Lead with these.

## The live data dashboards

Four browser dashboards on public live feeds, each with the statistics computed in a pure module with no DOM and tested under `node:test`, and each with a logged run on October 8, 2026 behind every number. Tell them as a set when asked about data engineering or statistics, or pick the one that fits the role.

### Wikipedia edit stream

**15 seconds.** Every Wikipedia edit counted twice, exactly and with streaming sketches, so you can watch how far each sketch drifts from the truth.

**60 seconds.** "This dashboard runs three streaming sketches beside exact counters on the live Wikipedia edit stream. HyperLogLog with 4,096 one byte registers estimates distinct editors, Count-Min with 4 rows of 1,024 counts edits per wiki, and Space-Saving with 200 counters tracks the busiest pages with an error on every count. In a 297 second run with 9,909 events, HyperLogLog estimated 986.9 distinct editors against an exact 983, plus 0.4 percent, Space-Saving had zero bound violations, and Count-Min's top 12 were all within their error bound. I also found that binning by arrival time flagged delivery clumps as bursts, so the chart bins by each edit's own timestamp."

*Follow ups to prepare.* The HyperLogLog standard error is about 1.04 over the square root of the register count, $1.04/\sqrt{4096} \approx 1.6$ percent, so plus 0.4 percent is well inside one sigma. Count-Min with width 1,024 and depth 4 has $\varepsilon = e/1024 \approx 0.27$ percent of all events and failure probability $e^{-4} \approx 1.8$ percent. Space-Saving guarantees count minus error is at most the true count, which is at most the count. *Weak spot:* about 100 wikis in 1,024 columns rarely collide, so the Count-Min bound is shown but hardly stressed, and the README says so.

### Order book microstructure

**15 seconds.** A live copy of Kraken's BTC/USD book rebuilt in the browser and checked against the exchange's own checksum after every update, before any statistic is computed from it.

**60 seconds.** "Every trading system starts from a local order book built from a snapshot plus deltas, and the classic failure is silent drift. Kraken publishes a CRC32 of the top ten levels with every update, so this page verifies every message and resyncs on a mismatch. In two 330 second BTC/USD runs at depth 100, every checksum verified, 46,892 of 46,892 in the second run, at about 142 updates a second, with zero resyncs. Only then does it measure order flow imbalance, after Cont, Kukanov and Stoikov 2014, regressed on mid price changes: a slope of 1.10 dollars per bitcoin of net flow with an R squared of 0.54."

*Follow ups to prepare.* Why the checksum is fragile: it is computed over formatted strings, so the instrument's own precision matters. Why realized volatility was higher at 10 seconds (31.9 percent) than at 1 second (27.0 percent), against the usual microstructure noise story: the 10 second figure rests on only 30 returns, so the page compares the gap with its sampling error. *Weak spot:* a natural mismatch never happened, so the resync path is exercised only by forced corruption and tests; latency of 89 ms at the median includes clock offset.

### Live earthquake statistics

**15 seconds.** The past week of USGS earthquakes, with the Gutenberg-Richter b-value, completeness magnitude and aftershock decay computed in the browser, plus a log of every revision the catalog makes while you watch.

**60 seconds.** "Earthquake catalogs change after the fact: magnitudes move, events merge under new ids, a few are deleted. This page computes classic seismology statistics from the week feed and logs every revision. b is estimated by maximum likelihood with Utsu's binning correction, with Shi and Bolt uncertainty and a 200 resample bootstrap that re-estimates completeness each time. For California, b was 0.727, bootstrap interval 0.661 to 0.851. The pooled b of 0.343 is not physical, because the feed mixes magnitude scales across networks, and the page says so. My favourite part is a test that failed for the right reason: across 300 synthetic catalogs the 95 percent bootstrap interval covered the true b 93.7 percent of the time."

*Follow ups to prepare.* The estimator: $\hat b = \log_{10} e \,/\, (\bar M - (M_c - \Delta M/2))$. A quick simulation for this module (300 synthetic catalogs with a true b of 1) gave a mean estimate of 0.994, so the estimator is close to unbiased at that size; that is a check, not a project number. Why one failing seed is not a bug: a 95 percent interval should miss about one time in twenty, so the test now requires at least 16 of 20 catalogs to cover. *Weak spot:* no sequence reached 30 aftershocks that week, so there is no Omori fit on live data, and deletions have only been exercised on synthetic feeds.

### Space weather nowcast

**15 seconds.** The input side of a Kp forecasting model: nine NOAA feeds joined on time, solar wind propagated from L1 to Earth, and the quantities that drive geomagnetic activity, with the freshness of every input shown.

**60 seconds.** "My earlier Kp forecasting project trained an LSTM on clean hourly history. In production it would run on live feeds from several spacecraft with gaps, revisions and instrument switches, so this page builds that input side. It joins nine NOAA feeds, propagates each solar wind sample to Earth using the active spacecraft's real distance and the sample's own speed, and computes dynamic pressure, the Shue magnetopause standoff and the Newell coupling function. In one logged pass the wind was 360 kilometres a second at 5.7 protons per cubic centimetre, giving 1.24 nanopascals, a standoff of 10.9 Earth radii and arrival 73.5 minutes after measurement. The feeds also taught me my own spec was wrong about how NOAA maps Kp to the G scale."

*Follow ups to prepare.* Dynamic pressure is proton mass times density times speed squared; with the logged values it is about 1.24 nPa, which I checked. The G scale lesson: NOAA rounds Kp to the nearest whole number before mapping, so 4.67 is G1 and 5.67 is G2, and both rules are implemented and tested. *Weak spot:* ballistic propagation ignores tilted fronts, so arrival can be off by several minutes, and with no archive of past forecasts there is no skill score.

> **Remember:** Test counts on the portfolio and in these READMEs differ by two for each dashboard (the READMEs count two shared formatting tests). Quote the README.

## Work and research experience

Each role is a story bank entry. Numbers below are as they appear on your public portfolio.

### Gravity Spy research assistant

**15 seconds.** Since February 2026 I have been a research assistant on Gravity Spy 2.0, the NSF funded citizen science project supporting LIGO, building the data pipeline and a model over volunteer classifications of detector glitches.

**60 seconds.** "Gravity Spy is NSF funded citizen science that supports LIGO; the science goal is finding which auxiliary detector subsystems cause the glitches that contaminate gravitational wave data. I built the ingestion pipeline on the Zooniverse Panoptes API: more than 276,000 volunteer classifications over 23 months, which meant reconciling five undocumented metadata schemas across 25,104 glitch subjects and 8,293 auxiliary channels, and filtering about 38 science team accounts out of the volunteers. I pulled 49,000 spectrograms and converted each mosaic into paired channel heatmaps, then trained a two input CNN with a shared MobileNetV2 backbone to predict volunteer similarity labels. It reaches 0.89 validation AUC on a held out 2,000 subject set, and calibration error fell from 0.10 to 0.05 with label smoothing, AdamW and AUC based early stopping."

*Follow ups to prepare.* Why calibration mattered more than AUC (the portfolio says it did; have your reason ready, for example whether the scores are read as probabilities downstream). How the held out set was split (by subject, and whether paired images from the same time could leak across the split; know the answer). What volunteer labels are (noisy consensus, so the model learns volunteer judgement, not ground truth; 52 percent of multiply classified subjects reach perfect consensus). *Weak spot:* the model predicts similarity labels; it does not by itself establish which subsystem causes a glitch, so do not claim causal findings.

### JMA Wireless Lean Six Sigma project lead

**15 seconds.** From January to May 2026 I led a four person team through a DMAIC project at JMA Wireless, standardising five factory KPIs and showing that the automated cell's real constraint was upstream material flow.

**60 seconds.** "At JMA Wireless, a US maker of 5G infrastructure, I led a four person Lean Six Sigma team for 15 weeks on site, sponsored by the SVP of Global Operations. The first problem was that nobody could compare the manual line with the automated cell, because Uptime and Schedule Attainment meant different things to different functions, so I ran definition interviews and wrote a sponsor approved document fixing one formula and target for five KPIs. I also flattened the cell's nested JSON, over 31 GB when loaded raw, to under 5 MB. Measured the same way, the cell showed 97.3 percent uptime against 46.3 percent of its weekly target, which cannot both be an equipment problem: it was starved by kitting upstream. A two week standup pilot cut the median time from a red KPI to an owner acting from about 4 hours to about 25 minutes."

*Follow ups to prepare.* Why the two numbers locate the constraint (a machine that runs nearly all the time and still misses plan by half is input limited). How the three fixes were chosen (a Pugh matrix over six candidates: a daily standup at a dashboard wall, a two bin kitting feed, and resetting a commissioning era target to demonstrated capability). The value estimate: roughly USD 355,000 a year projected, from about 27,000 more cables and about 48 analyst hours a week returned, resting on written rate assumptions that JMA Finance still has to confirm. *Weak spot:* the pilot was two weeks, the familiarity score (2.4 to 4.1 on a five point scale) is self reported, and the value is a sized opportunity, not a realised saving. Say all three.

### ProProfs data and product analyst intern

**15 seconds.** In July and August 2023 I did product and customer analytics at ProProfs, a SaaS company, feeding feature prioritisation and dashboards on feature usage.

**60 seconds.** "At ProProfs I did product and customer analytics for SaaS products serving more than 5,000 users. I benchmarked 100 to 150 features against Zoho and Salesforce, ran SWOT and ideal customer profile analyses, and analysed customer data for 2,000 clients in Python and Excel to find behaviour patterns, pain points and what drove conversion. Those findings helped the product team prioritise two features for the next release. I also built Tableau dashboards on SQL and Excel sources that identified under used features, and churn fell by 5 percent."

*Follow ups to prepare.* How the churn change was measured and attributed, and whether 5 percent is relative or in points; in a two month internship, be precise about what you did versus what the team concluded. The portfolio also shows a rating moving from 4.2 to 4.6; know exactly what that rating was before you mention it.

### Cybersecurity teaching assistant

**15 seconds.** In summer 2025 I was the graduate TA for the Syracuse iSchool cybersecurity course, building and teaching 15 plus hands on labs.

**60 seconds.** "I was the graduate TA for the iSchool cybersecurity course. I built and taught more than 15 hands on labs across cryptography, network security, penetration testing and digital forensics, each walking students through the attacker workflow from reconnaissance and scanning to exploitation, then flipping to the defender's reading of the same evidence. I maintained the multi VM lab environments, Kali against Windows, CentOS and Metasploitable targets, so a whole section could run the same exercise without environment drift. In office hours I debugged students' own attack chains, and I graded on methodology as well as the final answer, because in security the chain of evidence matters as much as the result."

*Follow ups to prepare.* How you explained a concept to a struggling student (this is your teaching story for "explain something complex to a non expert"). How you kept environments reproducible. *Weak spot:* none of substance; keep scope and authorisation in the story, as the portfolio does.

## What interviewers probe

Follow up chains to expect, in the order they usually come:

- **Layered answers.** "Go deeper." "What's the trade off?" "When would you not use it?" "What would you use instead?" Pre load the last two for every concept on your resume.
- **Coding narration.** "What's the complexity?" "Can you do better?" "What about the empty input?" "Walk me through it with this example." If you have not traced your code aloud, expect the last one.
- **Derivations.** "Why is that step legal?" "What assumption did you just make?" "Check it with a number." Say assumptions before you use them.
- **System design.** "Why that store?" "What happens at ten times the load?" "What fails first?" "How would you know it is failing?" Have a monitoring answer for every component you draw.
- **Analytics cases.** "How would you know it's not a logging bug?" "Which segment would you check first and why?" "What would you recommend tomorrow?"
- **Behavioural.** "What did you do, specifically?" "What would you do differently?" "What did the other person think?" "How did you measure that?" Twenty five minute rounds are mostly these.
- **Your projects at a research lab.** "What is the denominator?" "Is that one seed?" "What did the baseline get on the same data?" "What would falsify your conclusion?" "Which number are you least sure of?" For EvidenceOps, expect "is any of this measured with a real model?"; for Dr. GRPO, "how confident are you with one seed?"; for BEIR, "why does your BM25 differ from Anserini?"; for distributed ViT, "how do you know the two GPU run trains the same model?"; for ShopFloor, "is there leakage?".
- **Honesty checks.** Interviewers sometimes ask a number back to you later in the conversation. If it changes, credibility goes. Know each number, its denominator and its caveat.

## Sources

Interview format and rubrics:

- [Tech Interview Handbook: coding interview cheatsheet](https://www.techinterviewhandbook.org/coding-interview-cheatsheet/)
- [Tech Interview Handbook: coding interview rubrics](https://www.techinterviewhandbook.org/coding-interview-rubrics/)
- [Tech Interview Handbook: behavioral interviews](https://www.techinterviewhandbook.org/behavioral-interview/)
- [Tech Interview Handbook: behavioral interview rubrics](https://www.techinterviewhandbook.org/behavioral-interview-rubrics/)
- [Tech Interview Handbook: final questions to ask](https://www.techinterviewhandbook.org/final-questions/)
- [Tech Interview Handbook: self introduction](https://www.techinterviewhandbook.org/self-introduction/)
- [Amazon Leadership Principles](https://www.amazon.jobs/content/en/our-workplace/leadership-principles)
- [Amazon interview guide](https://www.aboutamazon.com/news/workplace/amazon-interview-guide)
- [interviewing.io: stop memorizing STAR, start selecting better stories](https://interviewing.io/blog/stop-memorizing-star-for-behavioral-interviews-start-selecting-better-stories)
- [interviewing.io: a senior engineer's guide to the system design interview](https://interviewing.io/guides/system-design-interview)
- [Exponent: what is the STAR method](https://blog.tryexponent.com/what-is-the-star-method-and-how-to-use-it-during-interviews)
- [Exponent: Amazon behavioral interview questions](https://blog.tryexponent.com/how-to-nail-amazons-behavioral-interview-questions/)
- [Hello Interview: system design delivery framework](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery)

Explaining and speaking:

- [Farnam Street: the Feynman technique](https://fs.blog/feynman-technique/)
- [MIT OpenCourseWare: Patrick Winston, How to Speak](https://ocw.mit.edu/courses/res-tll-005-how-to-speak-january-iap-2018/pages/how-to-speak/)
- [Upworthy: five takeaways from Winston's How to Speak](https://www.upworthy.com/mit-professor-gave-the-same-how-to-speak-lecture-for-40-years-here-are-his-5-pearls-of-wisdom/)

Project and experience facts:

- [Portfolio](https://tanaypratapsingh.github.io/)
- [beir-replication](https://github.com/TanayPratapSingh/beir-replication)
- [drgrpo-replication](https://github.com/TanayPratapSingh/drgrpo-replication)
- [handoff](https://github.com/TanayPratapSingh/handoff)
- [distributed-vit-cifar10](https://github.com/TanayPratapSingh/distributed-vit-cifar10)
- [tool-dependency-graph](https://github.com/TanayPratapSingh/tool-dependency-graph)
- [ztf-transient-pipeline](https://github.com/TanayPratapSingh/ztf-transient-pipeline)
- [shopfloor-ai](https://github.com/TanayPratapSingh/shopfloor-ai)
- [media-pulse](https://github.com/TanayPratapSingh/media-pulse)
- [comms-agent](https://github.com/TanayPratapSingh/comms-agent)
- [wikipedia-edit-stream](https://github.com/TanayPratapSingh/wikipedia-edit-stream)
- [orderbook-microstructure](https://github.com/TanayPratapSingh/orderbook-microstructure)
- [live-earthquake-statistics](https://github.com/TanayPratapSingh/live-earthquake-statistics)
- [space-weather-nowcast](https://github.com/TanayPratapSingh/space-weather-nowcast)
- EvidenceOps has no public repository; its facts come from the project's own README, results file, architecture note and four ADRs.
