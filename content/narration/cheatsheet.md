# Narration cheatsheet

Ten minutes before the interview: the templates, the phrases, and the numbers you must not get wrong. Every project number here is from that project's README or results file.

## Answer shapes

| Question type | Shape | First sentence |
| --- | --- | --- |
| Any "what is X" | Layered: 1 sentence, 30 s, 2 min, deep on request | The definition with its category |
| "Explain X" | What it is, intuition, example, trade off, when to use | "X is a ... that ..." |
| Coding | Restate, 2 to 3 clarifications, 2 approaches, green light, code with intent, trace, edge cases, complexity | "Let me restate it..." |
| Derivation | Goal, symbols, steps with reasons, result in words, sanity check | "I want to show..." |
| System design | Requirements, entities, API, simple end to end design, deep dives | "Here's my plan for the next 45 minutes..." |
| Metric moved | Clarify, rule out measurement, shape, segment, ranked hypotheses, recommend | "First I'd confirm exactly what dropped..." |
| Measure a feature | Goal, primary metric, 2 to 3 guardrails, test design, kill criterion | "The feature exists to..." |
| Behavioural | Title, STAR or CAR, actions are decisions, result with number and caveat, learning | "Let me tell you about the time..." |
| Your project | 15 s, 60 s, 3 min (problem, why, built, hardest decision, result with caveat, next) | What it is plus the surprising finding |

## Phrases that buy time and structure

- "There are two parts to that. First..."
- "Let me restate the problem to make sure I have it."
- "Give me thirty seconds to think about the invariant."
- "I can go deeper on the maths or on when it breaks. Which is more useful?"
- "I'll skip estimates for now and do the maths where it decides something."
- "That's requirements. Moving to the API."
- "I see two weak spots. Which would you like me to take first?"
- "I'm sure" / "I believe, but I'd verify" / "I'm guessing": say which.

## If asked X, say Y

| If asked | Say |
| --- | --- |
| Something you don't know | Boundary, adjacent knowledge, labelled guess, how you'd check |
| "I think that's wrong" | Restate their concern, identify the assumption, update or defend with a condition |
| "Any questions for me?" | One technical, one about how success is measured, one picking up something they said |
| "Tell me about yourself" | Now, through line, two proofs for this role, why here; 60 to 90 seconds |
| "What would you do differently?" | One concrete change and why, not a list |
| "How confident are you?" | The denominator, the number of seeds or runs, what would falsify it |

## STAR timing for a 2 minute answer

| Part | Seconds |
| --- | --- |
| Situation and task | 20 |
| Action (decisions, why, alternatives) | 70 |
| Result (number, denominator, caveat) | 20 |
| Learning | 10 |

## Story bank map

| Prompt | Story |
| --- | --- |
| Conflict | JMA: functions defined Uptime and Schedule Attainment differently; definitions document for 5 KPIs |
| Influence without authority | JMA: 97.3 percent uptime against 46.3 percent of target means input limited, not equipment |
| Failure | EvidenceOps: notice registry caught 0 of 3 on 5,000 real articles (parsing only); 3 of 3 with PMC flag |
| Ambiguity | Gravity Spy: five undocumented metadata schemas, 25,104 subjects |
| Incomplete data | Dr. GRPO: one seed, reported as "not reproduced here" |
| Mistake caught | BEIR: DPR 0.260 against 0.318; 100 token default; 512 matched 0.318 |
| Surprising result | Distributed ViT: 98 percent fp32 scaling, 57 percent fp16; synthetic input restored 96 percent |
| Teaching | Cybersecurity TA: 15 plus labs, graded methodology |

## Numbers to know cold

**EvidenceOps** (caveat in the same sentence, always: four article synthetic corpus, deterministic stub adapter, not a model)

- Real and reproducible: 107 tests per the README (86 bare, 21 with extras); statement coverage over `services/` 75 to 84 percent.
- Stub on the synthetic fixture: golden set pass rate 0.857 over 14 cases, retracted evidence leak 0.000 (the fixture guaranteed it), judge kappa 0.25 on 8 labels.
- Real slice, parsing only, no model: 5,000 articles, 6 parser bugs, 11 notices all pointing outside the slice, registry recall 0 of 3, then 3 of 3 with PMC's flag.
- Capacity plan (computed by the project's estimator, not measured): 10M chunks at 384 dimensions, 4.77 GB int8 against 15.50 GB float32.

**BEIR** (nDCG@10, SciFact and NFCorpus)

- BM25 published 0.665 and 0.325, mine 0.677 and 0.320. TAS-B exact: 0.643 and 0.319.
- DPR SciFact: 0.260 first run, 0.318 at 512 tokens (published 0.318). Recall@100 0.601 at 100 tokens to 0.727 at 512.
- bge-small (not in paper) 0.713 and 0.343. Hybrid RRF (not in paper) 0.687 and 0.341, Recall@100 0.948 on SciFact.
- Judged documents per query: NFCorpus 38.2, SciFact 1.1.

**Dr. GRPO** (seed 0 only, Qwen2.5-1.5B, LoRA rank 64, 100 steps, 1024 tokens, 2 T4s)

- Accuracy held: 51.6 against 51.2 percent on MATH500, paired +0.4 points [minus 3.4, plus 4.0].
- Both length claims reversed: Dr. GRPO 99 tokens longer per step [77, 123]; wrong answers 680 against 545 tokens.
- GRPO correct answers shorter: 262 against 399. Cap hit: 55 of 500 in each arm.
- Per token gradient: minus A over response length (GRPO), minus A over budget (Dr. GRPO).

**Distributed ViT** (2 T4, nccl, 30 epochs)

- fp32: 1,853 to 3,637 img/s, 98 percent. fp16: 3,810 to 4,376 img/s, 57 percent. Synthetic input: 1.92 times, 96 percent.
- 1,806,538 parameters, about 7 MB of gradients. FSDP 0.66 times DDP.

**Tool dependency graph:** 1,391 tools, 46 entities, 3,278 edges; 1,770 of 2,728 required slots (65 percent) resolve; 1,768 of 1,770 (99.9 percent) have a producer; 568 unresolved.

**handoff:** 200 tests, 11 scenarios, 0 dependencies, 15 tools across 5 risk levels; high risk gets 180 seconds then safe fallback.

**ShopFloor AI** (AI4I 2020, synthetic, 10,000 rows, 3.39 percent positive): GB 0.8548 F1 and 0.9591 AUC; RF 0.8387 and 0.9823; LR 0.3054 and 0.9455; promotion floor 0.80 F1.

**media-pulse / comms-agent** (synthetic): topic 0.88 to 0.96, sentiment 0.50 to 0.85 with review; 9 of 24 flagged, 7 corrected.

**Dashboards (October 8, 2026 runs):** HyperLogLog 986.9 against 983 (+0.4 percent); 46,892 of 46,892 checksums; OFI slope 1.10 USD per BTC, R squared 0.54; California b 0.727 [0.661, 0.851]; bootstrap coverage 93.7 percent over 300 catalogs; solar wind 360 km/s, 1.24 nPa, 10.9 Earth radii, 73.5 minutes.

**Experience:** Gravity Spy 276,000 plus classifications, 0.89 validation AUC on 2,000 held out subjects, ECE 0.10 to 0.05. JMA: 4 person team, 31 GB to under 5 MB, 4 hours to 25 minutes in a 2 week pilot, about USD 355,000 a year projected (not realised). ProProfs: 2,000 client records, churn down 5 percent. TA: 15 plus labs.

## Traps

- Starting at the deep dive; stopping at the headline.
- Unannounced silence; narrating keystrokes.
- "We" for everything; no number in the result; a number without its denominator.
- Quoting an EvidenceOps metric without "synthetic corpus, stub adapter" in the same sentence.
- Saying "refuted" about Dr. GRPO; saying bge-small "reproduced" (it is an extension).
- Inventing a figure you half remember. Give the direction and offer the repo.
- Mentioning work authorisation, salary or contact details in answers meant for the panel.
