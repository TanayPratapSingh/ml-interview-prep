# RAG, Agents and Evals: cheatsheet

One page for the ten minutes before the interview: the formulas, the decision tables, the numbers worth quoting with their sources, and the one line answers to the questions that come up in almost every applied AI loop.

## Decide first

| If the need is | Use | Because |
| --- | --- | --- |
| Facts that change, citations, per user permissions, deletion | RAG | Index swaps update knowledge with no training (RAG paper: 70 and 68 percent with matched indexes, 12 and 4 percent mismatched) |
| Format, tone, a narrow skill, a rule the model forgets | Fine tuning | Behaviour lives in weights; facts in weights cannot be cited, scoped or deleted |
| Corpus under about 200k tokens, low traffic | Long context plus prompt caching | Anthropic (2024): caching cuts latency over 2x and cost up to 90 percent |
| Steps are fixed | Workflow (chain, route, parallel) | Predictable and cheap |
| Steps cannot be predicted | Agent with caps and checkpoints | Flexibility costs tokens and compounding errors |
| Breadth first research worth about 15x chat tokens | Multi agent | Anthropic (2025): plus 90.2 percent over single agent on their eval |

## Pipeline in one breath

Parse, clean, dedupe, chunk, attach metadata (id, version, date, ACL, trust status), embed and index for both dense and BM25. At query time rewrite, extract filters, hybrid retrieve 100 to 150, fuse with RRF, rerank to 5 to 20, generate with chunk ids, verify citations, answer or refuse, log the trace.

## Formulas

- **BM25:** $\sum_t \mathrm{IDF}(t)\,\frac{f(k_1+1)}{f + k_1(1 - b + b\,|D|/\mathrm{avgdl})}$, with $\mathrm{IDF} = \ln\left(\frac{N - n + 0.5}{n + 0.5} + 1\right)$, $k_1 \in [1.2, 2.0]$, $b = 0.75$. TF saturates at $k_1 + 1$.
- **RRF:** $\sum_r \frac{1}{60 + \mathrm{rank}_r(d)}$. Ranks only, no calibration; $k = 60$ was near optimal but not critical in Cormack et al. 2009.
- **InfoNCE / DPR loss:** $-\log \frac{e^{s(q,p^+)/\tau}}{\sum_j e^{s(q,p_j)/\tau}}$. In batch negatives: $S = QP^\top$, labels on the diagonal, $B-1$ free negatives per query.
- **ColBERT MaxSim:** $S_{q,d} = \sum_i \max_j E_{q_i} \cdot E_{d_j}$.
- **HNSW level:** $l = \lfloor -\ln U \cdot m_L \rfloor$, $m_L = 1/\ln M$, layer 0 has $2M$ links, search scales as $O(\log N)$.
- **Precision at k** = hits in top k over k. **Recall at k** = hits in top k over all relevant.
- **MRR** = mean of 1 over rank of first hit. **AP** = mean of precision at each hit, divided by all relevant.
- **nDCG at k** = $\frac{\sum_i g_i/\log_2(i+1)}{\mathrm{IDCG}}$, gain $g = rel$ or $2^{rel} - 1$, IDCG from all judged relevant items.
- **RAGAS:** faithfulness = supported statements over statements; answer relevance = mean cosine between the question and questions generated from the answer; context relevance = needed sentences over all sentences.
- **Cohen's kappa:** $\kappa = \frac{p_o - p_e}{1 - p_e}$, $p_e = \sum_c p_{human}(c)\,p_{judge}(c)$.
- **pass@k** = $1 - \binom{n-c}{k}/\binom{n}{k}$ (any of k succeeds). **pass hat k** = $\binom{c}{k}/\binom{n}{k}$ (all k succeed).
- **Compounding:** $p^n$. 0.99 over 50 steps = 0.605; 0.95 over 20 = 0.358.
- **Judge correction (Rogan-Gladen):** true rate = (observed + specificity minus 1) over (sensitivity + specificity minus 1).

## Worked numbers I checked in Python

| Case | Result |
| --- | --- |
| Grades 0,3,0,2,0,1,0,0,0,0 with relevant set 3,2,2,1 | P@5 0.40, R@10 0.75, RR 0.50, AP 0.375, nDCG@10 0.546 (0.560 exponential) |
| RRF: A ranked 1 in one list; B ranked 3 and 2 | A 0.0164, B 0.0320, consensus wins |
| BM25 TF at average length, k1 = 1.2 | 1, 2, 10 occurrences give 1.0, 1.375, 1.964 |
| 10M by 768 float32 | 30.7 GB flat; about 1.04 GB as IVF PQ with 96 byte codes |
| HNSW graph at M = 16 | about 151 bytes per vector (paper: 60 to 450 for M from 6 to 48) |
| Judge: 14 of 20 fails caught, 6 of 80 passes wrongly failed | agreement 0.88, chance 0.68, kappa 0.625; pass everything judge kappa 0 |
| Same judge, failure recall 14 of 20 | 95 percent Wilson interval 0.48 to 0.85 |
| 200 cases, 18 regressed and 8 fixed | 5 point drop, exact McNemar p = 0.076 |
| Task solved 6 of 8 times | pass hat 4 = 0.214, pass@4 = 1.0 |

## Index choice

| Index | Pick when | Knob |
| --- | --- | --- |
| Flat | Under about a million vectors, or as ground truth | none |
| HNSW | Vectors fit in RAM, need high recall at low latency | M, efSearch; no deletes in FAISS |
| IVF | Large data, can afford a training pass | nlist about $C\sqrt{N}$, nprobe |
| IVF PQ | Memory bound at 100M plus | M subvectors, nprobe; rerank shortlist with full vectors |

Tune at matched recall against flat search, then compare latency and memory.

## Numbers worth quoting (with source)

- BEIR (2021), nDCG@10 relative to BM25 averaged over 18 datasets: BM25 plus cross encoder plus 11 percent, ColBERT plus 2.5, TAS-B minus 2.8, DPR minus 47.7.
- BEIR latency on 1M docs: BM25 20 ms CPU; dense about 14 to 20 ms GPU; BM25 plus cross encoder rerank 450 ms GPU.
- DPR (2020): one BM25 hard negative lifted top 5 accuracy 55.8 to 65.8; beat BM25 by 9 to 19 points top 20.
- Contextual retrieval (2024): failure at 20 from 5.7 to 3.7 (embeddings), 2.9 (plus BM25), 1.9 percent (plus rerank, 150 to 20).
- Lost in the Middle (2023): GPT-3.5-Turbo with the answer mid context fell below its 56.1 percent closed book score.
- MT-Bench (2023): GPT-4 and humans 85 percent agreement excluding ties, humans with each other 81. Swap consistency GPT-4 65 percent, Claude-v1 23.8. Verbosity attack fooled GPT-3.5 91.3 percent, GPT-4 8.7. Math grading failures 14, 6, 3 of 20 for default, chain of thought, reference guided.
- RAGAS (2023) on 50 WikiEval pages: agreement with humans 0.95 faithfulness, 0.78 answer relevance, 0.70 context relevance.
- ReAct (2023): plus 34 points ALFWorld, plus 10 WebShop, with 1 or 2 examples.
- SWE-bench (2024): 2,294 tasks, 12 repos; Claude 2 solved 1.96 percent in the paper.
- tau-bench (2024): gpt-4o pass hat 1 about 61 percent retail, 35 airline; pass hat 8 under 25 percent retail.
- Anthropic multi agent (2025): agents about 4x chat tokens, multi agent about 15x; tokens explain 80 percent of BrowseComp variance.
- Hamel Husain: about 100 labels per failure mode to validate a judge; below 60 the intervals are too wide.

## If asked X, say Y

- **"Why not fine tune on the docs?"** No citations, no deletion, no per user access, a retrain per change, and confident errors near the learned facts.
- **"Chunk size?"** Tuned on a golden set by recall at the k I feed the model; structure aware splits, tables whole, contextual headers if chunks lose their subject.
- **"Dense or sparse?"** Both: BM25 for exact tokens and zero shot reliability, dense for paraphrase, fused with RRF, then a cross encoder on the top 100.
- **"Raise top k?"** Buys recall, costs tokens, latency and distractors; rerank instead and measure.
- **"How do you know retrieval works?"** Recall at k and nDCG at 10 on labelled queries, with first stage and reranked numbers separate.
- **"How do you know the judge works?"** Expert labels, dev and test split, failure recall, true negative rate and kappa, about 100 labels per failure mode, both orders for pairwise, recalibrate on judge upgrades.
- **"Agent loops?"** Trace it, hash repeated calls, cap steps and spend, verify intermediate results, escalate with the trace.
- **"Prompt injection?"** Cannot be filtered out completely because data and instructions share tokens; least privilege, separated trust levels, confirmation for side effects, safe output handling, red team evals.
- **"Pass rate fell 5 points?"** Paired test on the same cases, must pass cases block, read every regression, never lower the threshold.
- **"pass@k or pass hat k?"** pass@k when a verifier picks the winner; pass hat k when a customer sees every attempt.

## Traps

- Reporting ANN recall as if it were relevance.
- Ideal DCG built from the retrieved list only.
- Raw judge agreement on imbalanced labels.
- Citations counted as grounding without checking the cited span.
- Permissions enforced in the prompt instead of the retrieval filter.
- A retracted or superseded document demoted instead of removed.
- Golden set with no refusal or injection cases.
- Comparing indexes or judges at default settings instead of matched operating points.
