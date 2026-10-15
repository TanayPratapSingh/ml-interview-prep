# RAG, Agents and Evals: the exhaustive version

This is the research lab depth version of retrieval augmented generation, agents and evaluation. It covers when to retrieve instead of fine tuning or stuffing the context, every stage of a retrieval pipeline with the formulas interviewers ask you to reproduce (BM25, reciprocal rank fusion, InfoNCE, nDCG, MaxSim, HNSW level sampling), the agent patterns from Anthropic's "Building effective agents" through ReAct, MCP and multi agent research systems, and the evaluation stack that makes any of it trustworthy: golden sets, LLM judges with measured biases, judge calibration with Cohen's kappa, regression gates, contamination and agent benchmarks. Every worked number was recomputed in Python, and every number from a paper names the paper.

## RAG versus fine tuning versus long context

Retrieve when the model needs facts it does not have, facts that change, or facts you must cite and permission per user. Fine tune when you need a behaviour: a format, a tone, a narrow skill, a policy the model keeps forgetting. Put everything in the context window when the whole corpus is small enough to fit and you can pay for those tokens on every request.

The cleanest framing comes from the original RAG paper (Lewis et al., NeurIPS 2020). A language model stores knowledge in its weights, which the paper calls **parametric memory**. A retrieval index stores knowledge outside the weights, the **non parametric memory**. Their model paired a DPR retriever (two BERT base encoders) over a dense index of 21M Wikipedia passages of 100 words each with a 400M parameter BART generator, and trained the query encoder and generator end to end while keeping the document encoder and index frozen, because re encoding 21M passages during training is expensive and they found it unnecessary. RAG-Sequence set the state of the art on open Natural Questions with 44.5 exact match. The experiment interviewers like most is index hot swapping: with a December 2016 index the model answered 70 percent of questions about 2016 world leaders, with a December 2018 index it answered 68 percent about 2018 leaders, and with mismatched indexes accuracy fell to 12 and 4 percent. Swapping the index updated the model's knowledge with no training at all. That is the argument for retrieval in one experiment.

| Need | RAG | Fine tuning | Long context |
| --- | --- | --- | --- |
| Facts that change weekly | Reindex the changed docs | Retrain, and old facts linger in weights | Reload every request |
| Citations and provenance | Native: you know which chunk was used | None | Possible if you tag passages |
| Per user permissions | Filter at retrieval time | Impossible to enforce in weights | Must assemble per user |
| Deleting a fact (legal request) | Delete the chunk and its derivatives | Cannot reliably unlearn | Drop it from the prompt |
| New output format or style | Weak | Strong | Weak, few shot examples only |
| Cost per query | Retrieval plus a short prompt | Cheapest at inference | Pays for every token every time |
| Latency | Retrieval and reranking add tens to hundreds of ms | Lowest | Grows with prompt length, cut by prompt caching |

Fine tuning to inject facts is the classic wrong answer. It is slow to update, it gives no provenance, it cannot be scoped per user, and the model still hallucinates neighbours of the facts it learned. Hamel Husain puts the division well in "Your AI Product Needs Evals" (2024): fine tuning is for syntax, style and rules, retrieval is for context and current facts. The two combine: you can fine tune a generator to use retrieved context better, which is what the RAG paper's end to end training did.

Long context is a real competitor now. Anthropic's contextual retrieval post (September 2024) says that if a knowledge base is under about 200,000 tokens (roughly 500 pages) you can often skip RAG and put the whole thing in the prompt, with prompt caching cutting latency by more than 2x and cost by up to 90 percent. The catch is that models do not use long contexts uniformly. "Lost in the Middle" (Liu et al., 2023) found a U shaped curve: accuracy is highest when the answer passage is first or last and drops when it sits in the middle. In their 20 and 30 document settings GPT-3.5-Turbo with the answer in the middle did worse than its closed book score of 56.1 percent, and going from 20 to 50 retrieved documents improved open domain QA by only about 1.5 points for GPT-3.5-Turbo and about 1 point for Claude 1.3. Those are 2023 models and newer long context models do better on needle tests, so treat the numbers as a warning rather than a constant: measure position sensitivity on your own data before you decide that context length replaced retrieval.

> **Interview trap:** "We will fine tune the model on our documentation so it knows it." Say what breaks: no citations, no deletion, no per user access control, a retrain for every doc change, and confident answers about facts adjacent to the ones it saw.

> **Say it like this:** "I default to retrieval for knowledge and fine tuning for behaviour. If the corpus fits in the window and traffic is low, long context with caching is simpler, but I would still test where in the context the model actually finds things."

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=398s) opens with the RAG overview and motivation.

## The RAG pipeline end to end

A production RAG system is two pipelines that share an index. The offline pipeline turns documents into searchable units. The online pipeline turns a question into a grounded answer or a refusal. Each stage has its own failure mode and its own metric, which is why you evaluate stages separately before you evaluate the whole.

```text
OFFLINE  source systems -> fetch -> parse (PDF, HTML, XML, tables) -> clean and dedupe
         -> chunk -> enrich (title, section path, dates, ACLs, status flags, contextual header)
         -> embed (dense) + tokenize (sparse) -> write to vector index and inverted index

ONLINE   question -> rewrite / route / extract filters -> hybrid retrieve top 100 to 150
         -> fuse (RRF) -> rerank (cross encoder) to top 5 to 20 -> assemble prompt with chunk ids
         -> generate with citations -> verify claims and citations -> answer, hedge or refuse
         -> log the full trace (query, ids, scores, prompt version, model, tokens, latency)
```

Ingestion and parsing are where most real systems lose quality, and they are the least glamorous part. PDFs mix two column layouts, headers and footers that repeat on every page, tables that become word soup when flattened, figures with captions, and scanned pages that need OCR. HTML carries navigation and cookie banners. Structured formats such as XML scientific articles have their own traps: articles with no body, tables nested inside paragraphs, deep section nesting, missing licence fields. Every document needs a stable id, a version, a source timestamp, an access control list and any trust status (retracted, superseded, draft) attached at ingestion, because every later stage depends on them. Deduplicate near copies before indexing, or the top 10 fills with five versions of the same paragraph.

What each stage should log and be measured on:

| Stage | Typical failure | Metric |
| --- | --- | --- |
| Parsing | Lost tables, merged columns, boilerplate in chunks | Parse error rate, manual spot checks per source |
| Chunking | Answer split across chunks, context lost | Recall at k on a golden set per chunking config |
| First stage retrieval | Relevant chunk not in top 100 | Recall at 100 |
| Reranking | Right chunk retrieved but ranked 30th | nDCG at 10, MRR |
| Generation | Unsupported claims, ignored context | Faithfulness, citation precision |
| Refusal | Answers when evidence is missing | False answer rate on unanswerable cases |

> **Remember:** the generator can only use what retrieval hands it. Recall at the context budget caps the whole system, so measure it first.

## Chunking strategies and trade offs

A chunk is the unit you embed, retrieve and cite. Its size sets a trade off between precision of the embedding and completeness of the context. A small chunk embeds one idea cleanly, so a query about that idea finds it, but the chunk may lack the subject it refers to ("revenue grew 3 percent" without the company or quarter). A large chunk keeps context but its single vector averages several topics, so it matches each topic weakly, and it costs more tokens per retrieved hit.

| Strategy | How it splits | Strength | Weakness |
| --- | --- | --- | --- |
| Fixed token window with overlap | Every N tokens, stride N minus overlap | Simple, predictable sizes | Cuts sentences, tables and code mid unit |
| Sentence or paragraph | Natural boundaries | Units are coherent | Very uneven sizes |
| Structure aware | Headings, sections, list items, functions in code | Keeps the author's units, gives a section path for metadata | Needs a parser per format |
| Semantic | Split where similarity between adjacent sentences drops | Topic coherent chunks | Extra embedding pass, unstable boundaries |
| Parent child (small to big) | Retrieve small child chunks, return the parent section | Precise matching, full context for the generator | Two levels to store and keep in sync |
| Late chunking | Embed the whole document, then pool token embeddings per chunk span | Each chunk vector sees document context | Needs a long context embedding model |
| Contextual headers | Prepend an LLM written summary of where the chunk sits | Restores the missing subject | One LLM call per chunk at index time |

Anthropic's contextual retrieval (September 2024) is the best measured version of the last row. Claude 3 Haiku read each whole document plus one chunk and wrote 50 to 100 tokens situating the chunk, which was prepended before both embedding and BM25 indexing. Measured as 1 minus recall at 20, the failure rate fell from 5.7 percent to 3.7 percent with contextual embeddings (35 percent fewer failures), to 2.9 percent with contextual embeddings plus contextual BM25 (49 percent), and to 1.9 percent after reranking (67 percent). With prompt caching the one time cost was about USD 1.02 per million document tokens. Generic document summaries prepended to chunks did not help much in their tests, which tells you the context has to be chunk specific.

The original DPR and RAG work used disjoint 100 word passages, and DPR reported that overlapping passages were not better than non overlapping ones in their setting. Overlap is still common in practice because it protects answers that straddle a boundary, but it costs index size. With overlap share $o$ the number of chunks grows by a factor $1/(1-o)$:

```python
import math

def n_chunks(doc_tokens, size, overlap):
    stride = size - overlap
    return 1 + math.ceil(max(0, doc_tokens - size) / stride)

print(n_chunks(1000, 256, 64), n_chunks(1000, 256, 0))
for overlap_share in (0.1, 0.2, 0.5):
    print(overlap_share, "index grows by a factor of", round(1 / (1 - overlap_share), 2))
```

```text
5 4
0.1 index grows by a factor of 1.11
0.2 index grows by a factor of 1.25
0.5 index grows by a factor of 2.0
```

Two practical traps. Embedding models silently truncate input beyond their maximum sequence length, so a 2,000 token chunk sent to a 512 token encoder is embedded from its first quarter only. And chunk level relevance is not document level relevance: if three chunks from one document fill your top 3, deduplicate or diversify before you spend the context budget.

How to choose: build a golden set of questions with the chunks or passages that answer them, then sweep chunk size, overlap and strategy and compare recall at the k you will actually feed the generator. Keep tables and code blocks whole. Treat chunking as a hyperparameter you tune with data.

> **Interview trap:** defaulting to fixed 512 token windows without saying why. The follow up is always "what breaks at the edges", and the answer is sentences, tables and references to a subject defined earlier.

## Embeddings and bi encoders

A **bi encoder** maps the query and each document to vectors independently and scores relevance by a dot product or cosine. Because documents never see the query, every document vector is computed once at index time and a query costs one encoder pass plus a nearest neighbour search. DPR is the canonical example: two BERT base encoders, 768 dimensional outputs, score $s(q,p) = E_Q(q)^\top E_P(p)$.

Three architectures sit on a cost and quality line:

| Architecture | Scoring | Precompute documents? | Cost per query | Quality |
| --- | --- | --- | --- | --- |
| Bi encoder | One vector each, dot product | Yes | One encoder pass plus ANN search | Good in domain, weaker out of domain |
| Late interaction (ColBERT) | One vector per token, sum of MaxSim | Yes, but many vectors per doc | Encoder pass plus token level search | Close to cross encoders |
| Cross encoder | Query and doc concatenated through full attention, one score | No | One full forward pass per candidate | Best, used for reranking only |

ColBERT (Khattab and Zaharia, SIGIR 2020) keeps a bag of contextual token embeddings for each side, projected to 128 dimensions and L2 normalised, and scores

$$
S_{q,d} = \sum_{i \in |E_q|} \max_{j \in |E_d|} E_{q_i} \cdot E_{d_j}^\top
$$

Each query token finds its best matching document token and the matches are summed. Queries are padded to 32 tokens with BERT mask tokens, which the paper calls query augmentation and found essential. As a reranker ColBERT ran over 170 times faster than BERT based rerankers with about 14,000 times fewer FLOPs. The price is storage: BEIR (Thakur et al., 2021) reports that a ColBERT index for BioASQ (about 15M documents) needs about 900 GB against 18 GB for BM25.

Details interviewers check:

- **Normalisation.** For unit vectors, $\lVert q-d \rVert^2 = 2 - 2\,q \cdot d$, so cosine, dot product and L2 give the same ranking. Without normalisation, dot product favours long vectors, which some models learn to use as a popularity signal. Use the metric the model was trained with.
- **Pooling.** Sentence vectors come from the CLS token or the mean of token vectors. Mean pooling over a long chunk averages topics, one reason large chunks retrieve worse.
- **Asymmetry.** Questions are short and documents long. Some embedding models expect a prefix such as "query:" or "passage:" on inputs. Check the model card, because a missing prefix silently lowers recall.
- **What embeddings miss.** Negation, exact numbers, rare names, product codes and identifiers are poorly preserved by a dense bottleneck. These are exactly the cases where lexical search wins, which motivates hybrid retrieval.
- **Domain shift.** BEIR evaluated retrievers zero shot on 18 datasets. Averaged over datasets and measured relative to BM25 on nDCG at 10, DPR (trained on Natural Questions) scored 47.7 percent worse, ANCE 7.4 percent worse, TAS-B 2.8 percent worse, while BM25 plus a cross encoder reranker was 11 percent better and ColBERT 2.5 percent better. In domain wins do not transfer automatically, and BEIR also found that in domain performance correlates poorly with out of domain generalisation.

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=1659s) covers similarity search with SBERT and bi encoders.

## Training retrievers contrastively

A dense retriever is trained so that a query is closer to its relevant passage than to irrelevant ones. DPR (Karpukhin et al., 2020) writes the loss for one query $q_i$ with positive $p_i^+$ and negatives $p_{i,1}^-, \dots, p_{i,n}^-$ as the negative log likelihood of the positive:

$$
L = -\log \frac{e^{\mathrm{sim}(q_i, p_i^+)}}{e^{\mathrm{sim}(q_i, p_i^+)} + \sum_{j=1}^{n} e^{\mathrm{sim}(q_i, p_{i,j}^-)}}
$$

This is a softmax cross entropy where the correct class is the positive passage, the same family as InfoNCE. With a temperature $\tau$ the similarities are divided by $\tau$ before the softmax; a small temperature sharpens the distribution and puts more weight on the hardest negatives.

**In batch negatives** make this cheap. Take a batch of $B$ query passage pairs, embed them into matrices $Q$ and $P$ of shape $B \times d$, and form $S = QP^\top$. Row $i$ has its positive on the diagonal and $B-1$ negatives for free, so one batch trains on $B^2$ pairs. Larger batches mean more negatives, which is why retriever training papers chase batch size.

```python
import numpy as np

def info_nce(q, p, temperature=0.05):
    """In batch negatives: row i's positive is p[i], every other p[j] is a negative."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    logits = q @ p.T / temperature                      # B x B similarity matrix
    logits -= logits.max(axis=1, keepdims=True)         # numerical stability
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return -np.mean(np.diag(log_probs))                 # cross entropy, labels on the diagonal

q = np.array([[1.0, 0.1], [0.0, 1.0], [0.7, 0.7]])
p_good = np.array([[0.9, 0.2], [0.1, 0.9], [0.6, 0.8]])
p_shuffled = p_good[[1, 2, 0]]                          # positives no longer aligned
print(round(info_nce(q, p_good), 4), round(info_nce(q, p_shuffled), 4))
print(round(info_nce(q, p_good, temperature=1.0), 4))
```

```text
0.0296 7.5251
0.8638
```

Aligned pairs give a loss near zero, misaligned pairs a large loss, and the same aligned pairs at temperature 1.0 give 0.86, close to the chance value $\ln 3 \approx 1.10$, because cosine similarities between 0 and 1 barely separate before scaling.

**Hard negatives** matter more than any other choice. DPR's ablation on Natural Questions (dev set, top k retrieval accuracy) shows the pattern: in batch training with 127 negatives reached 55.8 top 5 accuracy, and adding one BM25 hard negative per question (a passage that matches the question's words but does not contain the answer) lifted it to 65.8. Adding a second BM25 negative did not help further. Growing the batch helped steadily: top 20 accuracy went from 69.1 to 70.8 to 73.0 as in batch negatives grew from 7 to 31 to 127. The full DPR retriever beat Lucene BM25 by 9 to 19 points of top 20 accuracy across open domain QA datasets.

What goes wrong:

- **False negatives.** Mined hard negatives are sometimes relevant but unlabelled, and in batch "negatives" are sometimes duplicates of the positive. Training then pushes away correct answers. Filter mined negatives with a cross encoder score threshold and deduplicate batches.
- **Easy negatives.** Random negatives are trivially separable, so the gradient vanishes. Mix in hard ones.
- **Domain gap.** BEIR's GenQ result is the standard fix: fine tuning TAS-B on synthetic queries generated from the target corpus helped on specialised domains (scientific papers, finance, StackExchange) but hurt slightly on broad Wikipedia style domains.
- **Distillation.** A common recipe trains the bi encoder to match a cross encoder teacher's scores, transferring some of the teacher's quality into the cheap architecture.

> **Interview trap:** saying in batch negatives are "random negatives". They are other queries' positives, which are on topic for some query and so more informative than random passages, and they are free.

## Vector indexes: flat, IVF, HNSW, PQ

An approximate nearest neighbour (ANN) index trades a little recall for a lot of speed and memory. Be precise about the word recall here: ANN recall at k is the share of the exact nearest neighbours (from brute force search) that the index returns. It says nothing about relevance. A perfect ANN index over bad embeddings still retrieves the wrong passages.

| Index | Idea | Bytes per vector (FAISS wiki) | Exhaustive | Knobs |
| --- | --- | --- | --- | --- |
| Flat | Compare the query with every vector | $4d$ | Yes, exact | none |
| IVF Flat | k-means into nlist cells, scan only nprobe cells | $4d + 8$ | No | nlist, nprobe |
| HNSW | Layered proximity graph, greedy descent | $4d$ plus graph links | No | M, efConstruction, efSearch |
| PQ | Split into M subvectors, store one centroid id each | $\lceil M \cdot nbits / 8 \rceil$ | Yes over codes | M, nbits |
| IVF PQ | IVF cells holding PQ codes | $\lceil M \cdot nbits / 8 \rceil + 8$ | No | nlist, nprobe, M, nbits |

**Flat** is $O(Nd)$ per query and exact. It is the right answer for a few hundred thousand vectors and the ground truth for measuring every other index.

**IVF** runs k-means on the vectors to get nlist centroids, stores each vector in the inverted list of its nearest centroid, and at query time scans only the nprobe lists whose centroids are closest. Work per query is roughly $nlist \cdot d$ to pick cells plus $(nprobe / nlist) \cdot N \cdot d$ to scan them. Recall drops when a true neighbour lives in a cell you did not probe, which happens most for queries near cell boundaries. The FAISS wiki's rule of thumb is $nlist \approx C\sqrt{N}$. IVF needs a training pass, and if the data distribution drifts after training the cells become unbalanced.

**PQ** compresses each vector by cutting it into $M$ subvectors and replacing each with the id of its nearest centroid from a 256 entry codebook (8 bits). A 768 dimensional float32 vector (3,072 bytes) becomes 96 bytes at $M = 96$, a 32x saving. Distances are computed from lookup tables of query to centroid distances, so scanning codes is fast. It is lossy, so systems often rerank the PQ shortlist with full precision vectors.

**HNSW** (Malkov and Yashunin) builds a hierarchy of proximity graphs, a skip list where linked lists are replaced by graphs. Each inserted element draws a top level $l = \lfloor -\ln(U) \cdot m_L \rfloor$ with $U$ uniform on (0, 1), so level membership decays exponentially. The paper's recommended $m_L = 1/\ln M$ matches a skip list with $p = 1/M$. Search starts at the entry point on the top layer, greedily moves to the neighbour closest to the query until no neighbour is closer, drops a layer and repeats, and on layer 0 keeps a candidate list of size ef. Insertion runs the same search with efConstruction and links the new node to M neighbours chosen by a heuristic that only adds a candidate if it is closer to the new node than to any already chosen neighbour, which keeps edges between clusters and so keeps the graph connected on clustered data. Layer 0 allows $M_{max0} = 2M$ links. The paper derives $O(\log N)$ search complexity scaling, recommends M between about 5 and 48 (larger M for high recall and high dimensional data), and gives graph memory per element as $(M_{max0} + m_L \cdot M_{max}) \cdot 4$ bytes, about 60 to 450 bytes for typical M. FAISS's HNSW index does not support removing vectors because removal would break the graph.

```python
import math

n, d = 10_000_000, 768
print("flat float32 GB:", n * d * 4 / 1e9)
print("flat float16 GB:", n * d * 2 / 1e9)
print("IVF-PQ, 96 one byte codes + 8 byte id, GB:", n * (96 + 8) / 1e9)
print("nlist rule of thumb 4 * sqrt(n):", round(4 * math.sqrt(n)))
for M in (6, 16, 48):                       # HNSW graph links: Mmax0 = 2M at layer 0, mL = 1 / ln M
    links = 2 * M + M / math.log(M)
    print(f"HNSW M={M}: {links:.1f} links, {links * 4:.0f} bytes per vector for the graph")
```

```text
flat float32 GB: 30.72
flat float16 GB: 15.36
IVF-PQ, 96 one byte codes + 8 byte id, GB: 1.04
nlist rule of thumb 4 * sqrt(n): 12649
HNSW M=6: 15.3 links, 61 bytes per vector for the graph
HNSW M=16: 37.8 links, 151 bytes per vector for the graph
HNSW M=48: 108.4 links, 434 bytes per vector for the graph
```

The computed graph sizes land inside the paper's 60 to 450 byte range. HNSW stores the raw vectors as well, so 10M 768 dimensional vectors cost about 30.7 GB plus 1.5 GB of links at M = 16, while IVF PQ fits in about 1 GB. That is the usual decision: HNSW when the vectors fit in RAM and you want the best recall at low latency, IVF PQ when memory is the constraint.

The recall versus work trade off is easy to see on a toy IVF. This builds 64 cells over 20,000 vectors with overlapping clusters and measures recall at 10 against exact search:

```python
import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng(0)
centers = rng.normal(size=(50, 32)) * 0.5
X = (centers[rng.integers(0, 50, 20000)] + rng.normal(size=(20000, 32))).astype("float32")
Q = (centers[rng.integers(0, 50, 200)] + rng.normal(size=(200, 32))).astype("float32")

def topk(D, k=10):
    return np.argsort(D, axis=1)[:, :k]

d2 = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)      # exact (flat) search
truth = topk(d2)

nlist = 64
km = KMeans(n_clusters=nlist, n_init=1, random_state=0).fit(X)
assign = km.labels_
cd = ((Q[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
for nprobe in (1, 2, 4, 8, 16):
    probe = np.argsort(cd, axis=1)[:, :nprobe]
    recalls, scanned = [], []
    for i in range(len(Q)):
        cand = np.where(np.isin(assign, probe[i]))[0]
        found = cand[np.argsort(d2[i, cand])[:10]]
        recalls.append(len(set(found) & set(truth[i])) / 10)
        scanned.append(len(cand) / len(X))
    print(f"nprobe={nprobe:2d}  recall@10={np.mean(recalls):.3f}  share of vectors scanned={np.mean(scanned):.3f}")
```

```text
nprobe= 1  recall@10=0.297  share of vectors scanned=0.016
nprobe= 2  recall@10=0.409  share of vectors scanned=0.031
nprobe= 4  recall@10=0.550  share of vectors scanned=0.062
nprobe= 8  recall@10=0.709  share of vectors scanned=0.121
nprobe=16  recall@10=0.865  share of vectors scanned=0.243
```

When I reran the same script with well separated clusters (centers scaled by 3 instead of 0.5), nprobe = 2 already gave 0.994 recall. The lesson for interviews: ANN parameters depend on how clustered your embeddings are, so you tune them on your own vectors. Fix a recall target against flat search (say 0.95 at k = 10), raise nprobe or efSearch until you hit it, then report p50 and p99 latency and memory at that setting.

> **Interview trap:** comparing two indexes at their default settings. Defaults sit at different points on the recall and latency curve, so the comparison only means something at matched recall.

## Sparse retrieval and BM25

BM25 scores a document by summing, over query terms, an inverse document frequency weight times a saturating, length normalised term frequency. It comes from the probabilistic relevance framework (Robertson and Zaragoza, 2009):

$$
\mathrm{BM25}(D, Q) = \sum_{t \in Q} \mathrm{IDF}(t) \cdot \frac{f(t, D)\,(k_1 + 1)}{f(t, D) + k_1 \left(1 - b + b \frac{|D|}{\mathrm{avgdl}}\right)}
$$

$$
\mathrm{IDF}(t) = \ln\!\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)
$$

Here $f(t, D)$ is the count of term $t$ in document $D$, $|D|$ the document length, avgdl the mean length, $N$ the number of documents and $n(t)$ the number containing $t$. The "+1" inside the log is the Lucene variant, which keeps IDF positive even for terms in more than half the documents; the original Robertson and Sparck Jones form can go negative. Typical settings are $k_1 \in [1.2, 2.0]$ and $b = 0.75$.

What each part does:

- $k_1$ controls saturation. At average length with $k_1 = 1.2$, one occurrence contributes 1.0 times IDF, two contribute 1.375, ten contribute 1.964, and the ceiling is $k_1 + 1 = 2.2$. Repeating a word stops helping, unlike raw TF IDF.
- $b$ controls length normalisation. $b = 1$ fully normalises by length, $b = 0$ ignores length. Long documents contain more words by chance, so they are penalised.
- IDF makes rare terms count. A query term in every document contributes almost nothing.

```python
import math

docs = [
    "refund policy for annual plans",
    "our refund policy explained in full detail with examples for every plan type",
    "refund requests take five days",
    "how to reset your password",
]
toks = [d.split() for d in docs]
N = len(toks)
avgdl = sum(len(d) for d in toks) / N

def idf(term):
    n = sum(term in d for d in toks)                    # document frequency
    return math.log((N - n + 0.5) / (n + 0.5) + 1)      # Lucene style, never negative

def bm25(query, doc, k1=1.2, b=0.75):
    score = 0.0
    for t in query:
        f = doc.count(t)
        norm = k1 * (1 - b + b * len(doc) / avgdl)
        score += idf(t) * f * (k1 + 1) / (f + norm)
    return score

q = "refund policy".split()
print("avgdl =", avgdl, {t: round(idf(t), 3) for t in q})
for d, t in zip(docs, toks):
    print(f"{bm25(q, t):.3f}  {d}")

# term frequency saturation at average length (k1 = 1.2, b irrelevant when len = avgdl)
for f in (1, 2, 5, 10, 100):
    print(f, round(f * 2.2 / (f + 1.2), 3))
```

```text
avgdl = 7.0 {'refund': 0.357, 'policy': 0.693}
1.189  refund policy for annual plans
0.777  our refund policy explained in full detail with examples for every plan type
0.404  refund requests take five days
0.000  how to reset your password
1 1.0
2 1.375
5 1.774
10 1.964
100 2.174
```

Documents one and two both contain each query term once; the 13 word document scores 0.777 against 1.189 for the 5 word one purely through length normalisation. "policy" appears in fewer documents than "refund", so it carries more weight.

Why BM25 is still in every serious stack: it is exact on identifiers, error codes, names and rare terms; it needs no training; its inverted index is small and fast (BEIR measured 20 ms per query on CPU over 1M DBPedia documents with a 0.4 GB index); and it generalises. BEIR's headline was that BM25 is a strong zero shot baseline that most dense and learned sparse models of 2021 failed to beat on average (DeepCT was 27.9 percent and SPARTA 20.3 percent worse, docT5query, which expands documents with generated queries before BM25 indexing, 1.6 percent better). Its weakness is vocabulary mismatch: "cancel subscription" does not match "terminate plan". Tokenisation, stemming and stop words change results, so they are part of the configuration you evaluate.

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=2065s) covers heuristic search with BM25.

## Hybrid search and reciprocal rank fusion

Hybrid search runs a lexical and a dense retriever and merges their lists, because their errors are complementary: BM25 misses paraphrases, dense retrieval misses exact tokens. The merge is the hard part, since BM25 scores are unbounded and cosine scores live in $[-1, 1]$ with model specific spreads. You can normalise scores (min max or z score per query) and take a weighted sum, which needs labelled data to tune the weight, or you can fuse ranks.

**Reciprocal rank fusion** (Cormack, Clarke and Buttcher, SIGIR 2009) scores each document by

$$
\mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}
$$

where $r(d)$ is the document's rank in list $r$ (documents absent from a list contribute nothing) and $k = 60$. The paper fixed $k = 60$ in a pilot and never changed it; its pilot table shows MAP of 0.2072 at $k = 0$, 0.2145 at 60 and 0.2098 at 500, so the choice was near optimal but not critical. Across TREC runs RRF beat Condorcet fusion, CombMNZ and the best single system by 4 to 5 percent on average, and fusing the LETOR 3 baselines gave a better ranker than any of them. The constant damps the influence of any single system's top ranks, so one outlier list cannot dominate, and documents that several lists rank moderately well rise above documents one list loves.

```python
def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])

bm25_list = ["A", "C", "B", "D"]
dense_list = ["E", "B", "F", "C"]
for k in (60, 1):
    print(k, [(d, round(s, 4)) for d, s in rrf([bm25_list, dense_list], k=k)])
```

```text
60 [('B', 0.032), ('C', 0.0318), ('A', 0.0164), ('E', 0.0164), ('F', 0.0159), ('D', 0.0156)]
1 [('B', 0.5833), ('C', 0.5333), ('A', 0.5), ('E', 0.5), ('F', 0.25), ('D', 0.2)]
```

B and C appear in both lists and win at $k = 60$, beating A and E, which each topped one list. At $k = 1$ the top ranks dominate and A almost catches C. Properties worth saying out loud: RRF uses ranks only, so it needs no score calibration; it can be computed one list at a time; weighted RRF multiplies each list's terms by a weight when one retriever is known to be better; and fusion helps only if both lists are decent, since a very weak list mostly adds noise. Fuse on chunk ids after deduplication, and fuse a deep enough list (say 50 to 100 from each retriever) that the reranker has something to work with.

## Cross encoder reranking

A reranker rescores a shortlist with a model that reads the query and each candidate together. The standard two stage design retrieves 100 to 150 candidates cheaply with hybrid search, reranks them with a cross encoder, and passes the top 5 to 20 to the generator. Anthropic's contextual retrieval pipeline reranked the top 150 to a top 20 and cut its retrieval failure rate from 2.9 to 1.9 percent. In BEIR, reranking BM25's top 100 with a cross encoder was the best method on average, 11 percent above BM25 on nDCG at 10.

The cost is what the interviewer wants you to quantify. A cross encoder needs one forward pass per candidate over a sequence of length $|q| + |d|$, with attention quadratic in that length, and nothing can be precomputed. BEIR measured about 450 ms on a V100 GPU (6.1 s on CPU) to rerank 100 BM25 candidates for one query, against 20 ms for BM25 alone and under 20 ms for dense retrieval on GPU. That is why the reranker never touches the full corpus.

Design points:

- The reranker cannot recover what the first stage missed. Report first stage recall at 100 and post rerank nDCG at 10 separately, so you know which stage to fix.
- Rerank depth is a latency knob. Score fewer candidates, use a smaller distilled reranker, batch candidates, or cache scores for repeated queries.
- LLM rerankers (asking a model to order passages) can be stronger on hard queries but cost far more per query; use them on a small shortlist or offline to create training labels.
- Calibrated reranker scores make a useful evidence gate: if the best candidate scores below a threshold tuned on labelled data, the system should hedge or refuse rather than answer.

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=2724s) covers reranking with cross encoders.

## Query rewriting, HyDE and multi hop retrieval

The user's words are often a poor search query. Rewriting fixes that before retrieval:

- **Conversational rewriting** resolves references: "what about annual plans?" becomes "refund policy for annual plans" using the chat history.
- **Expansion and multi query** generate several paraphrases, retrieve for each and fuse the lists with RRF, raising recall at the cost of more retrieval calls.
- **Decomposition** splits a compound question into sub questions answered separately.
- **Filter extraction** pulls structured constraints ("since 2024", "only HR policies") out of the text into metadata filters.
- **Routing** sends the question to the right index or tool, for example SQL for "how many orders last week" and the document index for "what is the return window".

**HyDE** (Gao, Ma, Lin and Callan, 2022, [paper](https://arxiv.org/pdf/2212.10496)) asks an instruction tuned LLM to write a hypothetical document that answers the query, embeds that document with an unsupervised contrastive encoder (Contriever), and retrieves real documents near it. The fake document may contain false details, but the encoder's dense bottleneck keeps mostly the relevance pattern, and the search grounds it in real text. No model is trained. It beat Contriever alone across web search, QA, fact verification and non English tasks. It works because answers look more like documents than questions do; it costs an LLM call before retrieval, and for niche facts the hypothetical document can point to the wrong neighbourhood.

**Multi hop** questions need facts from several documents chained together ("who directed the film that won best picture the year X was born"). Single shot retrieval fails because the second hop's query depends on the first hop's answer. Options: iterative retrieval in a ReAct style loop where the model issues the next search after reading results; decompose then retrieve per sub question; or let a reasoning model decide when to search. CS329A Part 7 covers Search-o1, which triggers a search when the reasoning model expresses uncertainty and reasons over the retrieved documents, and contrasts it with Search-R1, which learns when to search with reinforcement learning. Multi hop pipelines compound errors, so each hop needs a stop condition and the final answer needs citations from every hop.

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=2274s) covers HyDE and contextual retrieval, and [CS329A Part 7](https://www.youtube.com/watch?v=Uni9dqyuuDM) covers Search-o1 and deep research agents.

## Metadata filters, freshness and deletion

Metadata turns a similarity search into a correct search. Tenant, access control list, document type, language, date and trust status are filters, and permission filters must be enforced inside retrieval, never by asking the model to ignore documents it was shown. OWASP's 2025 list names sensitive information disclosure (LLM02) and vector and embedding weaknesses (LLM08) for this reason.

Filtering interacts with ANN search in three ways:

| Approach | How | Problem |
| --- | --- | --- |
| Filter after search | Retrieve top k, drop non matching | With a filter that passes 1 percent of documents, top 100 yields about 1 survivor |
| Filter before search | Restrict candidates, then search | Fine for small subsets with brute force; graph indexes lose connectivity when most nodes are excluded |
| Filter aware ANN | The index checks the predicate during traversal | Needs engine support; very selective filters still degrade |

The practical rule: oversample when filtering after search, switch to exact search on the filtered subset when it is small, and partition the index by tenant when isolation matters.

Freshness is an ingestion problem. Use change data capture or scheduled diffs to re embed only changed documents, store a version and timestamp per chunk, boost or filter by recency when the question is time sensitive, and put a time to live on any cache that stores answers. A stale index produces the worst kind of error, a confident and correctly cited wrong answer.

Deletion has to reach everything derived from a document: the chunks, their vectors, the BM25 postings, generated contextual headers or summaries, semantic cache entries, conversation memories, and golden set references. FAISS's HNSW index cannot remove vectors, so engines use tombstones that are filtered at query time plus periodic rebuilds. Status changes such as a paper being retracted or a policy superseded should flip a hard filter immediately, because a trust constraint is not a ranking feature: a retracted source ranked lower is still a retracted source in the context.

## Retrieval metrics

Retrieval metrics compare a ranked list against relevance judgments (qrels). Let $rel_i$ be the relevance of the item at rank $i$ and $R$ the number of relevant items for the query.

- **Precision at k:** relevant items in the top k divided by k.
- **Recall at k:** relevant items in the top k divided by $R$. For RAG this is the first metric to report at your context budget.
- **Hit rate (success) at k:** 1 if any relevant item is in the top k.
- **Reciprocal rank:** $1/\mathrm{rank}$ of the first relevant item, 0 if none. **MRR** averages it over queries. Right when one good answer is enough.
- **Average precision:** $\mathrm{AP} = \frac{1}{R}\sum_{i:\,rel_i > 0} P@i$, the mean of precision at each relevant hit, dividing by all relevant items so missed ones count as zero. **MAP** averages AP over queries.
- **DCG at k:** $\sum_{i=1}^{k} \frac{g(rel_i)}{\log_2(i+1)}$ with gain $g(r) = r$ (linear) or $2^r - 1$ (exponential, which rewards highly relevant items more).
- **nDCG at k:** DCG divided by the ideal DCG, computed from all judged relevant items sorted by grade, including ones the system never retrieved.

BEIR chose nDCG at 10 as its single metric because precision and recall ignore rank, and MRR and MAP assume binary relevance, while nDCG handles both binary and graded judgments.

> **Worked example:** ten retrieved items with grades 0, 3, 0, 2, 0, 1, 0, 0, 0, 0, and the query has four relevant items in total with grades 3, 2, 2, 1 (one grade 2 item was never retrieved). Precision at 5 is 2/5 = 0.40, recall at 5 is 2/4 = 0.50, recall at 10 is 3/4 = 0.75. The first hit is at rank 2, so reciprocal rank is 0.50. Precision at the three hits is 1/2, 2/4 and 3/6, so AP = (0.5 + 0.5 + 0.5)/4 = 0.375. Linear DCG = 3/log2(3) + 2/log2(5) + 1/log2(7) = 3.110, ideal DCG over 3, 2, 2, 1 = 5.693, so nDCG at 10 = 0.546. With exponential gain it is 0.560.

```python
import math
import numpy as np
from sklearn.metrics import ndcg_score

retrieved = [0, 3, 0, 2, 0, 1, 0, 0, 0, 0]   # graded relevance at ranks 1..10
judged = [3, 2, 2, 1]                        # all relevant docs for this query, one grade 2 doc was missed
R = len(judged)

def precision_at(k): return sum(r > 0 for r in retrieved[:k]) / k
def recall_at(k): return sum(r > 0 for r in retrieved[:k]) / R
rr = next(1 / (i + 1) for i, r in enumerate(retrieved) if r > 0)
hits = [i for i, r in enumerate(retrieved) if r > 0]
ap = sum(sum(r > 0 for r in retrieved[:i + 1]) / (i + 1) for i in hits) / R

def dcg(gains): return sum(g / math.log2(i + 2) for i, g in enumerate(gains))
ndcg_lin = dcg(retrieved) / dcg(sorted(judged, reverse=True))
ndcg_exp = dcg([2 ** g - 1 for g in retrieved]) / dcg([2 ** g - 1 for g in sorted(judged, reverse=True)])

print(f"P@5={precision_at(5):.2f} R@5={recall_at(5):.2f} P@10={precision_at(10):.2f} R@10={recall_at(10):.2f}")
print(f"RR={rr:.2f} AP={ap:.3f} nDCG@10 linear={ndcg_lin:.4f} exponential={ndcg_exp:.4f}")

# sklearn check: append the missed relevant doc with the lowest score so the ideal ranking sees it
y_true = np.array([retrieved + [2]])
y_score = np.array([list(range(11, 1, -1)) + [0]])
print(f"sklearn nDCG@10={ndcg_score(y_true, y_score, k=10):.4f}")
```

```text
P@5=0.40 R@5=0.50 P@10=0.30 R@10=0.75
RR=0.50 AP=0.375 nDCG@10 linear=0.5464 exponential=0.5603
sklearn nDCG@10=0.5464
```

Pitfalls interviewers probe:

- **Ideal DCG from retrieved items only.** If you build the ideal ranking from the retrieved list instead of all judged items, a system that misses relevant documents is not penalised. scikit-learn's `ndcg_score` builds the ideal from the `y_true` you pass, which is why the check above appends the missed item.
- **AP denominators.** Some code divides by relevant items retrieved (here 0.5) instead of all relevant items (0.375). State which.
- **Incomplete judgments.** Pooled qrels mark unjudged documents as irrelevant. BEIR found many datasets carry lexical bias because BM25 style systems built the candidate pools, which penalises dense retrievers; manually judging the holes in TREC-COVID raised non lexical systems' scores. Report the share of unjudged items in the top 10 (BEIR calls it Hole at 10) when you compare a new system against old judgments.
- **Chunks versus documents.** Decide whether relevance is per chunk or per document and deduplicate accordingly.
- **Averaging.** Metrics are averaged per query (macro). Compare systems with paired tests over the same queries, since per query variance is large.

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=2869s) covers retrieval evaluation with NDCG and MRR.

## Grounded generation, citations and refusal

Generation quality in RAG splits into properties you measure separately:

- **Faithfulness (groundedness):** every claim in the answer is supported by the retrieved context. RAGAS defines it as supported statements over total statements.
- **Answer relevance:** the answer addresses the question asked, completely and without padding.
- **Correctness:** the answer is true against a reference. A faithful answer built on a wrong or stale passage is incorrect, and a correct answer from parametric memory is unfaithful. You need both properties, and they fail independently.
- **Citation precision and recall:** of the citations given, how many actually support the sentence they are attached to; of the claims made, how many carry a supporting citation.

Mechanics that work: give each chunk an id in the prompt, ask for a citation after every sentence, and verify citations after generation by checking that the cited span entails the claim, with an NLI model or a judge prompt. Eugene Yan's "Task Specific LLM Evals that Do and Don't Work" (2024, [post](https://eugeneyan.com/writing/evals/)) reports that fine tuning an NLI model for factual consistency raised ROC AUC from 0.56 to 0.85, which makes NLI a cheap, fast verifier compared with an LLM judge per claim.

Refusal is a feature you design and test. A system that cannot say "the sources do not answer this" will answer from parametric memory and look grounded while doing it. Build an evidence sufficiency gate from signals you can calibrate: the best reranker score, agreement between retrievers, and a judge asking whether the context contains the answer. Then measure it as selective prediction: on a golden set that includes unanswerable questions, report the false answer rate (answered when it should have refused) and the false refusal rate (refused when the evidence was there). Moving the threshold trades one for the other, so pick the operating point from the cost of each error. In medicine or law a wrong cited answer is worse than a refusal.

Prompt assembly details: deduplicate near identical chunks, order by relevance with the strongest evidence first (given the lost in the middle curve), include titles, dates and section paths so the model can resolve conflicts, and tell the model what to do when sources disagree (cite both and say so).

> **Interview trap:** treating "the answer cites a source" as "the answer is grounded". Models attach plausible citations to unsupported sentences. Only a check of the cited span against the claim measures grounding.

## RAG evaluation frameworks and their limits

RAGAS (Es et al., 2023) is the reference free framework most interviewers mean. It prompts an LLM to compute three scores:

$$
F = \frac{|V|}{|S|}, \qquad AR = \frac{1}{n}\sum_{i=1}^{n} \cos(q, q_i), \qquad CR = \frac{\text{extracted sentences}}{\text{sentences in } c(q)}
$$

For faithfulness $F$, an LLM splits the answer into statements $S$ and checks which ones $V$ the context supports. For answer relevance $AR$, the LLM generates $n$ questions from the answer and the score is the mean cosine similarity between their embeddings and the original question. For context relevance $CR$, the LLM extracts the sentences of the context needed to answer and divides by the total. The validation set, WikiEval, had 50 Wikipedia pages about events after the model's training cutoff. In pairwise comparisons with human preferences, using gpt-3.5-turbo-16k as the LLM, RAGAS agreed with annotators 95 percent of the time on faithfulness, 78 percent on answer relevance and 70 percent on context relevance, against 72, 52 and 63 percent for asking ChatGPT for a 0 to 10 score directly.

Limits to state in an interview:

- The scores are judge outputs. Change the judge model or prompt and the numbers move, so scores are not comparable across setups or over time unless the judge is pinned.
- Validation was 50 pages, pairwise, on one judge from 2023. Your domain needs its own check against human labels before you trust a threshold.
- Reference free metrics measure consistency, not truth. A faithful answer to a stale passage scores perfectly.
- Answer relevance by embedding similarity rewards answers that sound on topic, including wrong ones.
- Context relevance was the hardest dimension even in the paper; the model struggled to pick the crucial sentences from long contexts.
- One average score hides failure modes. Error analysis by category tells you what to fix.

The working rule from Hamel Husain and Shreya Shankar's work: generic metrics are a starting point, never the gate. Read traces, find your failure modes, write evaluators for them, and validate each evaluator against human labels.

## Hallucination mitigation

A hallucination is output not supported by the input or by fact. In RAG, separate intrinsic errors (contradicting the context), extrinsic errors (adding unsupported content), fabricated citations, and stale facts. No single method removes them, so mitigation is layered across the pipeline:

| Stage | Mitigation |
| --- | --- |
| Retrieval | Raise recall at the context budget, hybrid search, reranking, freshness |
| Context | Fewer and better chunks, titles and dates, conflicting sources flagged |
| Prompt | Answer only from sources, cite every claim, explicit permission to say "not in the sources" |
| Decoding | Structured outputs with required citation fields, low temperature for factual tasks |
| Verification | Decompose the answer into claims, check each against cited spans with NLI or a judge, drop or flag unsupported claims |
| Consistency | Sample several answers; disagreement signals likely hallucination (the idea behind SelfCheckGPT) |
| Abstention | Evidence gate tuned on unanswerable cases |
| Humans | Review queues for high stakes outputs, sampled audits elsewhere |

Measure the hallucination rate on a golden set with known answers and known unanswerables, and track it per category. The RAG paper is a useful reminder that retrieval itself reduces hallucination (its generations were judged more factual than BART alone), but it moves the failure to retrieval and reading errors rather than ending it.

## Agents and tool calling

Anthropic's "Building effective agents" (December 2024) gives the definitions most interviewers use. The building block is the **augmented LLM**: a model with retrieval, tools and memory. **Workflows** orchestrate LLM calls and tools through predefined code paths. **Agents** let the model direct its own process and tool use, typically an LLM using tools in a loop on environment feedback, with stopping conditions such as a maximum number of iterations. The advice is to start with the simplest thing that works, often a single well prompted call with retrieval, and add agency only when it measurably helps, because agents trade latency and cost for task performance and their errors compound.

**Function calling** mechanics: the application sends tool definitions (name, description, JSON Schema for arguments) with the conversation; the model returns a structured call instead of text; the application validates and executes it and appends the result as a tool message; the loop continues until the model answers or a limit triggers. Models can emit several independent calls in one turn, which the application can run in parallel.

```python
import json

TOOLS = {"lookup_order": lambda order_id: {"order_id": order_id, "status": "delivered"}}

def fake_model(messages):
    """Stands in for an LLM: asks for a tool once, then answers."""
    if not any(m["role"] == "tool" for m in messages):
        return {"tool": "lookup_order", "args": {"order_id": "W123"}}
    return {"answer": "Order W123 was delivered."}

def run_agent(user_msg, model=fake_model, max_steps=5):
    messages = [{"role": "user", "content": user_msg}]
    seen_calls = set()
    for step in range(max_steps):
        out = model(messages)
        if "answer" in out:
            return out["answer"], step + 1
        call = json.dumps([out["tool"], out["args"]], sort_keys=True)
        if call in seen_calls:                           # same call twice means a loop
            return "Stopped: repeated tool call, escalating to a human.", step + 1
        seen_calls.add(call)
        if out["tool"] not in TOOLS:
            result = {"error": f"unknown tool {out['tool']}"}  # errors go back to the model
        else:
            result = TOOLS[out["tool"]](**out["args"])
        messages.append({"role": "tool", "content": json.dumps(result)})
    return "Stopped: step budget exhausted.", max_steps

print(run_agent("Where is my order W123?"))
looping = lambda messages: {"tool": "lookup_order", "args": {"order_id": "W123"}}
print(run_agent("Where is my order W123?", model=looping))
```

```text
('Order W123 was delivered.', 2)
('Stopped: repeated tool call, escalating to a human.', 2)
```

**Toolformer** (Schick et al., 2023) showed a model can teach itself where tool calls help. A GPT-J model with 6.7B parameters sampled candidate API calls into plain text, executed them, and kept a call only if prefixing the call and its result reduced the loss on the following tokens by at least a threshold $\tau_f$, compared with no call or a call without its result. Fine tuning on the filtered text taught it to use a calculator, a QA system, two search engines, a translator and a calendar, and it beat a much larger GPT-3 on several zero shot tasks.

**Tool design is prompt design.** Anthropic calls it the agent computer interface: write tool descriptions like docstrings for a new colleague, include examples and edge cases, keep tools distinct, choose formats close to natural text (full file rewrites are easier for a model than diffs that require counting lines), return errors as informative messages, and mistake proof arguments. Their SWE-bench agent kept making relative path errors until the tool required absolute paths. With hundreds of tools, retrieve the relevant tool definitions per request instead of sending all of them.

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=3568s) covers tool calling, and [tool selection](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=5182s) at scale.

## Workflow patterns

Anthropic's five workflow patterns, each with the condition that justifies it:

| Pattern | What it does | Use when | Example | Watch for |
| --- | --- | --- | --- | --- |
| Prompt chaining | Fixed sequence of calls, each consuming the last, with programmatic gates between | The task splits cleanly into fixed steps | Outline, check outline against criteria, write document | Latency adds up; a bad early step poisons the rest |
| Routing | Classify the input, send it to a specialised prompt or model | Categories are distinct and classification is accurate | Refund vs technical support; easy questions to a small model | Misroutes are silent unless measured |
| Parallelisation | Run calls at once and aggregate: sectioning splits the task, voting repeats it | Subtasks are independent, or several views raise confidence | Guardrail call in parallel with the answer call; several reviewers voting on vulnerabilities | Aggregation logic and cost |
| Orchestrator workers | A central LLM decides subtasks at run time, delegates and synthesises | Subtasks cannot be predicted in advance | Coding change touching an unknown number of files; multi source research | Coordination overhead |
| Evaluator optimiser | One call generates, another critiques, loop until criteria pass | Clear criteria exist and feedback measurably improves output | Literary translation; search that decides whether to search again | Loops that never converge; the evaluator's own errors |

**Autonomous agents** come after these: open ended problems where the number of steps cannot be hard coded, run in trusted or sandboxed environments, with ground truth from the environment at each step (test results, tool outputs), checkpoints for human input and iteration caps.

Two connections to research that a lab interviewer may draw. Voting is self consistency, and repeated sampling with a verifier is the Large Language Monkeys result covered in CS329A Parts 1 and 2, where solve rate rises with the number of samples if you can pick the right one. Evaluator optimiser lives or dies on the verifier, which is the subject of CS329A Part 3: outcome versus process reward models, the generation verification gap, and Weaver's ensembles of weak verifiers.

On frameworks, Anthropic's advice is to start with direct API calls, since most patterns are a few lines of code, and if you use a framework, understand what it sends, because hidden prompts and parsing are a common source of bugs.

> **Watch:** [CS329A Part 1](https://www.youtube.com/watch?v=6YnLB0XbTnI) traces the shift from chatbots to these workflow patterns, and [CS329A Part 3](https://www.youtube.com/watch?v=p7TdPUcPoik) covers verification.

## ReAct, planning and multi step reasoning

**ReAct** (Yao et al., ICLR 2023) interleaves reasoning traces and actions: the model writes a Thought, takes an Action (a tool call such as a Wikipedia search or lookup), reads the Observation, and repeats until it can answer. Reasoning lets it plan, track progress and handle exceptions; actions let it fetch facts instead of hallucinating them. On HotpotQA and FEVER it reduced the hallucination and error propagation of chain of thought alone, and on the interactive benchmarks ALFWorld and WebShop it beat imitation and reinforcement learning baselines by 34 and 10 points of absolute success rate with only one or two in context examples.

```text
Question: Which country is the birthplace of the author of "Things Fall Apart"?
Thought: I need the author, then the author's birthplace.
Action: search["Things Fall Apart"]
Observation: Things Fall Apart is a 1958 novel by Chinua Achebe...
Thought: Now find where Chinua Achebe was born.
Action: search["Chinua Achebe"]
Observation: Chinua Achebe was born in Ogidi, Nigeria...
Thought: The answer is Nigeria.
Action: finish["Nigeria"]
```

ReAct's failure modes are the agent failure modes in miniature: repeating the same search, issuing a poor query and never reformulating, and stopping early with a guess.

Planning variants, from cheapest to most expensive:

- **Plan then execute.** Write a plan up front, execute steps, replan when a step fails. Cheaper than deciding every step from scratch and easier to inspect.
- **Reflection.** After a failure, the model writes a critique of its attempt and retries with it in context.
- **Search over trajectories.** LATS (Language Agent Tree Search, CS329A Part 5) runs Monte Carlo tree search over actions, scoring nodes with an LLM value estimate and self consistency. It needs an environment you can reset, because real actions (sending an email, issuing a refund) cannot be undone to explore a sibling branch.
- **Parallel plans and trained planners.** SPRINT fine tunes reasoning models to produce independent subplans that run in parallel, and SWiRL trains multi step tool use on offline synthetic trajectories scored by a judge (both CS329A Part 5).

Since reasoning models arrived (2024 to 2025), much of the planning happens inside the model's thinking, but the loop, the tools, the budgets and the stop conditions remain your code, and that is where reliability is won.

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=5516s) covers agents with ReAct, [CS329A Part 4](https://www.youtube.com/watch?v=Lxh9RF5S-K0) covers ReAct and learning from tool feedback, and [CS329A Part 5](https://www.youtube.com/watch?v=Ml_fp9XkB8Y) covers LATS, SPRINT and SWiRL.

## Memory: short term and long term

**Short term memory** is the context window: the conversation, the scratchpad, tool results. It is bounded and every token costs money and attention, so agents manage it: truncate old tool outputs, summarise or compact the history when it nears the limit, keep a stable prefix (system prompt, tool definitions) so prompt caching applies, and write important state somewhere durable. Anthropic's research lead agent saves its plan to memory because its context can be truncated past 200,000 tokens.

**Long term memory** lives outside the model: a vector store of past interactions or notes, a key value profile of user facts and preferences, episodic logs of previous tasks. Writing is a decision (what deserves remembering, usually through an explicit memory tool), and reading is retrieval, so everything in the RAG sections applies, with recency and importance as extra ranking signals.

What goes wrong: stale memories contradict new facts unless they carry timestamps and get overwritten; injected instructions stored in memory persist across sessions, which turns a one time prompt injection into a permanent one; private data in memory needs the same access control and deletion path as documents. Evaluate memory directly: plant facts in early sessions and test recall and correct use later, including cases where the fact was updated.

> **Remember:** the KV cache is an inference optimisation inside one request; memory is application state across requests. Interviewers sometimes check that you do not mix them up.

## Multi agent systems

A multi agent system splits work across several LLM instances with separate contexts, usually an orchestrator and workers. Anthropic's research system (June 2025) is the best documented case: a lead agent on Claude Opus 4 plans, spawns 3 to 5 subagents on Claude Sonnet 4 that search in parallel and return condensed findings, and a citation agent attaches sources at the end. On their internal research eval it beat a single Opus 4 agent by 90.2 percent.

The cost is the part to lead with. Anthropic reports agents use about 4 times the tokens of a chat interaction and multi agent systems about 15 times. On BrowseComp, token usage alone explained 80 percent of the variance in performance, and tokens, tool call count and model choice together explained about 95 percent. Much of the multi agent gain is spending more tokens in parallel, in contexts that do not crowd each other.

When it is worth it:

- Breadth first tasks that split into independent threads (research across many sources).
- Information larger than one context window.
- Tasks valuable enough to pay roughly an order of magnitude more tokens.

When it is not: tasks with tight dependencies and shared state, which Anthropic says includes most coding, and anything where agents must coordinate in real time. Lessons they report: give each subagent an objective, an output format, tool guidance and boundaries (vague delegation produced duplicated work); scale the number of agents to query complexity; the lead waits on each batch of subagents, a synchronous bottleneck; small prompt changes to the lead shift subagent behaviour in ways only end to end evals catch.

> **Say it like this:** "Before I build a multi agent version I run a single agent with the same token budget. If the single agent closes the gap, the extra coordination is not paying for itself."

## MCP, computer use and coding agents

The **Model Context Protocol** standardises how an AI application connects to tools and data, turning N applications times M integrations into N plus M. Participants: a **host** (the AI application, such as an IDE or Claude Code) creates one **client** per **server**, and each server exposes context. The data layer is JSON-RPC 2.0. Transports are stdio for local servers and Streamable HTTP for remote ones (HTTP POST plus optional server sent events, with OAuth recommended for tokens). Servers expose three primitives: **tools** (actions, via `tools/list` and `tools/call`), **resources** (readable context such as files or schemas, via `resources/list` and `resources/read`) and **prompts** (reusable templates). Clients can expose **elicitation**, letting a server ask the user for input or confirmation.

The protocol changed in 2026, so date your answer. In the 2025-06-18 specification a session began with an `initialize` request and an `initialized` notification that negotiated version and capabilities ([lifecycle spec](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle)). As of protocol version 2026-07-28 the docs describe MCP as stateless: every request carries the protocol version and client capabilities in its `_meta` field, servers implement a `server/discover` request, and the sampling and logging client primitives are deprecated.

Security is the interview follow up. Tool descriptions and tool outputs enter the model's context, so a malicious or compromised server can inject instructions. Run least privilege servers, pin and review the servers you install, and require confirmation for side effects.

**Computer use** agents see screenshots and emit mouse and keyboard actions. They reach software with no API, but they are slow, brittle to layout changes, and every page they read is a prompt injection surface. Prefer an API when one exists.

**Coding agents** are the strongest agent use case because tests give ground truth feedback: the agent navigates a repository, edits files, runs tests, reads failures and iterates. Anthropic notes they spent more time on tools than on the overall prompt for their SWE-bench agent, and that human review remains necessary because passing tests do not prove a change fits the wider system. A known failure is gaming the check, such as editing or deleting the test instead of fixing the code, so evaluation must run held out tests the agent cannot touch.

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=5357s) covers the Model Context Protocol.

## Agent failure modes

The failures interviewers ask about, with the defence for each:

| Failure | What it looks like | Defence |
| --- | --- | --- |
| Looping | Same tool call or same plan step repeated | Hash recent calls, stop or escalate on repeats, cap steps |
| Tool misuse | Wrong tool, wrong or invented arguments | Schema validation, distinct tool descriptions, informative errors |
| Error compounding | One early mistake carried through every later step | Verify intermediate results, checkpoints, resume from failure |
| Premature termination | Gives up or answers before the task is done | Explicit completion criteria checked by code or a verifier |
| Poor planning | Steps out of order, missing prerequisites | Plan review step, replanning on failure |
| Context drift | Loses the goal after many turns | Restate goal and plan, compact history, persistent notes |
| Policy violation | Takes an action the rules forbid | Hard checks in the tool layer, not only in the prompt |
| Cost blowup | Thousands of calls on a simple task | Token and spend budgets per task, effort scaled to complexity |

CS329A Part 8 lists poor planning, wrong tool selection, premature abandonment and repetitive loops as the recurring long horizon failures, and tau-bench's analysis of failed gpt-4o retail trajectories found wrong arguments, wrong information, wrong decisions and partially resolved compound requests.

Error compounding is arithmetic. If each step succeeds independently with probability $p$, a task of $n$ steps succeeds with probability $p^n$:

```python
from math import comb

def pass_hat_k(n, c, k):        # all k trials succeed (tau-bench reliability metric)
    return comb(c, k) / comb(n, k)

def pass_at_k(n, c, k):         # at least one of k trials succeeds
    return 1 - comb(n - c, k) / comb(n, k)

n, c = 8, 6                     # one task run 8 times, 6 successes
for k in (1, 2, 4, 8):
    print(k, round(pass_hat_k(n, c, k), 3), round(pass_at_k(n, c, k), 3))

# long horizons: per step success compounds
for p, steps in ((0.99, 50), (0.95, 20), (0.90, 10)):
    print(p, steps, round(p ** steps, 3))
```

```text
1 0.75 0.75
2 0.536 0.964
4 0.214 1.0
8 0.0 1.0
0.99 50 0.605
0.95 20 0.358
0.9 10 0.349
```

A 99 percent reliable step gives 60.5 percent success over 50 steps. The first block previews tau-bench's metrics: a task solved 6 times out of 8 has pass at 4 of 1.0 but pass hat 4 of 0.214, because pass at k asks whether any attempt succeeds and pass hat k asks whether all of them do.

## Guardrails and prompt injection

**Prompt injection** is input that makes the model follow an attacker's instructions instead of the developer's. **Direct** injection comes from the user typing it. **Indirect** injection (Greshake et al., 2023) hides instructions in content the system retrieves or reads: a web page, an email, a PDF, a tool result, a code comment. Their paper demonstrated data theft, spreading through email (worming) and manipulating answers against real systems including Bing's GPT-4 chat, and argued that LLM integrated applications blur the line between data and instructions. OWASP's 2025 Top 10 for LLM applications lists prompt injection first (LLM01), alongside improper output handling (LLM05), excessive agency (LLM06), system prompt leakage (LLM07) and unbounded consumption (LLM10).

Why no defence is complete: instructions and data arrive in the same channel, tokens, and the model is trained to follow instructions wherever they appear. Every detector is a classifier with a nonzero miss rate, and attackers adapt with paraphrase, encoding and multi turn setups. So you design for the case where injection succeeds and limit what it can do.

Layered defences:

- **Least privilege.** Give the agent only the tools and scopes the task needs. An agent that reads untrusted web pages should not also hold the ability to send email or move money.
- **Separate trust levels.** Quarantine untrusted content: a model that reads it returns data in a constrained format to a controller that holds the privileges.
- **Human confirmation** for irreversible or external actions.
- **Output handling.** Treat model output as untrusted input to downstream systems: escape it, validate structured outputs, never render markdown images or links to arbitrary domains, since a URL with data in its query string is an exfiltration channel.
- **Input and output classifiers** run in parallel with the main call. Anthropic notes that a separate guardrail call screening inputs tends to beat asking one call to do both jobs.
- **Delimit and label untrusted content** in the prompt, and tell the model it is data. This helps and is easily bypassed, so it is never the only layer.
- **Sandbox** code execution and browsing; keep secrets out of prompts; rate limit and cap spend.
- **Red team continuously** and track attack success rate as an eval.

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=6136s) closes with safety.

## Human in the loop design

Put humans where their judgment changes outcomes and their time is used well. Four placements:

1. **Approval gates** before irreversible, expensive or external actions (refunds, sending messages, deploying code, deleting data). tau-bench's airline policy, for example, requires explicit user confirmation before booking changes, and its reward notes an agent can match the final database while skipping that confirmation.
2. **Escalation** when confidence is low, evidence is missing, or the agent hits a budget or a loop, with the trace attached so the human does not start from zero.
3. **Sampled review** of production outputs to estimate quality and find new failure modes.
4. **Expert labelling** for golden sets and judge calibration, the input every automated evaluator depends on.

Design rules: tier actions by risk and gate only the risky ones, or reviewers learn to approve everything (automation bias); show the evidence, the diff and the proposed action on one screen; make approve, edit and reject one click each; log decisions so edits become training and eval data, as Rechat's labelling tool did. Measure approval rate, edit rate, override rate and time to decision; an approval rate near 100 percent means either the gate is unnecessary or reviewers stopped reading. MCP's elicitation primitive and Anthropic's checkpoints for autonomous agents are protocol and product forms of the same idea.

## Evals and golden sets

Evals are how an LLM product improves at all. Hamel Husain's "Your AI Product Needs Evals" (2024) describes Rechat's assistant plateauing until the team built evaluation into its process, and lays out three levels: unit tests (assertions run on every change), human and model evaluation of logged traces (run on a cadence), and A/B tests of user outcomes (only for mature products). Without them every prompt change is a guess and fixes regress silently.

Start from error analysis. Read real traces, write down what went wrong in plain words, group the notes into failure modes, count them, and build evaluators for the frequent and costly ones. Hamel's heuristic is to keep reading until you stop learning anything new. Shankar et al.'s "Who Validates the Validators?" (2024) adds **criteria drift**: people need criteria to grade outputs, but grading outputs is how they discover the criteria, so rubrics must be revised as you label.

A **golden set** is a versioned collection of inputs with expected behaviour. Each case should carry:

- the input and any context (user role, documents, conversation history);
- the expected behaviour class: answer, answer with hedging, refuse, or block;
- reference evidence ids, and a reference answer or a rubric when exact answers do not exist;
- tags for feature, scenario, persona and difficulty;
- a one line reason the case exists, so nobody deletes it as redundant.

Sources, in order of value: real production traffic sampled across features, production failures, expert written hard cases, adversarial and injection cases, unanswerable questions, and LLM generated synthetic inputs to cover gaps (reviewed by a person). Keep the mix of behaviour classes deliberate. A set with no refusal cases cannot catch a system that never refuses.

Size it from the decision it supports. With 100 cases at a 70 percent pass rate, the 95 percent Wilson interval is 0.60 to 0.78, so 100 cases cannot detect a 5 point change. Start with 30 to 100 cases to find failure modes, then grow toward hundreds per important slice. Hold out part of the set that is never used while tuning prompts, and never paste golden cases into few shot examples.

## Offline and online evaluation

**Offline** evaluation runs a fixed set before release: deterministic assertions (valid JSON, required citation format, no internal ids leaked, length limits), retrieval metrics against qrels, judge based checks, and the golden set pass rate. It is reproducible and cheap enough to run on every change.

**Online** evaluation measures real traffic after release: explicit feedback (thumbs, ratings), implicit signals (copying the answer, rephrasing and retrying, abandoning, escalating to a human), sampled human review, judges run on sampled production traces, guardrail trigger rates, and latency and cost. A/B tests measure whether a change moves user outcomes, which offline sets cannot do. Shadow deployments (new version runs silently on live traffic) and canaries (a small share of traffic) sit between the two.

The two disagree because production drifts away from any fixed set: new topics, new user behaviour, new documents. Close the loop: every production failure that a reviewer confirms becomes a golden case, so the offline set keeps tracking reality. Implicit signals are noisy and biased (users rarely click thumbs down on a confidently wrong answer they believe), so they find candidates for review rather than serving as ground truth.

> **Watch:** [CME295 Lecture 8](https://www.youtube.com/watch?v=8fNP4N46RRo&t=1104s) covers rule based metrics and why BLEU and ROUGE fall short for open ended output.

## LLM as a judge and its biases

An LLM judge grades outputs that have no exact answer. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., 2023) defined three modes: **pairwise** comparison (which of two answers is better, or a tie), **single answer grading** (a score for one answer), and **reference guided** grading (the judge sees a reference solution). Pairwise is more stable for subjective qualities but grows quadratically with the number of systems; single answer grading is cheaper and scales, but absolute scores drift more when the judge changes. Eugene Yan's summary of the literature (2024) gives the usual rule: direct scoring for objective criteria such as faithfulness or policy violations, pairwise for subjective ones such as tone, and binary outputs wherever possible.

The measured biases, all from the MT-Bench paper with 2023 models:

- **Position bias.** With two similar answers swapped, GPT-4 gave a consistent verdict 65 percent of the time; Claude-v1 only 23.8 percent, favouring the first position 75 percent of the time.
- **Verbosity bias.** A "repetitive list" attack padded answers with rephrased copies of their own list items. It fooled Claude-v1 and GPT-3.5 91.3 percent of the time and GPT-4 8.7 percent.
- **Self enhancement bias.** GPT-4 rated its own answers with about a 10 percent higher win rate than humans did, Claude-v1 about 25 percent, though the authors said their data could not establish the effect firmly.
- **Weak grading of maths and reasoning.** GPT-4 marked wrong answers as correct on 14 of 20 math judgments with the default prompt, 6 of 20 with a chain of thought judge, and 3 of 20 with a reference guided judge, even on problems it could solve alone.

Mitigations: run pairwise judgments in both orders and count a win only when both agree (otherwise a tie); give few shot examples (GPT-4's consistency rose from 65 to 77.5 percent, at about 4 times the prompt cost); have the judge solve the problem first or give it a reference; control for length by comparing length matched outputs or instructing against it; use a judge from a different model family than the system under test; and score against an explicit rubric with structured output. Yan also reports panels of smaller judges correlating better with humans than a single GPT-4 judge at about one seventh of the cost.

The headline agreement result: on MT-Bench, GPT-4 agreed with expert humans 85 percent of the time on non tie votes, above the 81 percent agreement between humans. Read the caveat with it: agreement rose from about 70 percent to nearly 100 percent as the quality gap between the two models grew. Judges are weakest on close comparisons, which is exactly where you need them to decide between two prompt versions.

> **Watch:** [CME295 Lecture 8](https://www.youtube.com/watch?v=8fNP4N46RRo&t=1680s) introduces LLM as a judge, then covers [position, verbosity and self enhancement bias](https://www.youtube.com/watch?v=8fNP4N46RRo&t=2327s) and [best practices](https://www.youtube.com/watch?v=8fNP4N46RRo&t=2842s).

## Calibrating a judge against human labels

A judge is a classifier, so calibrate it like one. Hamel Husain's "critique shadowing" (2024) is a good recipe: a principal domain expert labels outputs pass or fail and writes a critique for each; you split the labels into roughly 10 to 20 percent for few shot examples in the judge prompt, 40 to 45 percent dev for iterating on the prompt, and 40 to 45 percent test, run once on the frozen prompt. He reports reaching over 90 percent agreement with the expert in three iterations on one project, and recommends about 100 labelled examples per failure mode, warning that below about 60 the confidence intervals are too wide to use.

Raw agreement misleads when classes are imbalanced. Report the judge's recall on real failures, its true negative rate on real passes, and **Cohen's kappa**, which corrects agreement for chance:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}, \qquad p_e = \sum_{c} p_{\text{human}}(c)\, p_{\text{judge}}(c)
$$

where $p_o$ is observed agreement and $p_e$ the agreement expected if both raters labelled at random with their own marginal rates. Yan cites the common reading that 0.21 to 0.40 is fair and 0.41 to 0.60 moderate agreement. For ordinal scales use weighted kappa or rank correlations (Spearman, Kendall), remembering that correlations run higher than kappa because they ignore chance. For more than two raters use Fleiss' kappa or Krippendorff's alpha.

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint

human = np.array([0] * 20 + [1] * 80)                    # 0 = fail, 1 = pass (expert labels)
judge = np.array([0] * 14 + [1] * 6 + [0] * 6 + [1] * 74)

print(confusion_matrix(human, judge))                     # rows human, columns judge
agree = (human == judge).mean()
p_e = 0.2 * 0.2 + 0.8 * 0.8                               # both say fail + both say pass by chance
print("agreement", agree, "chance", round(p_e, 2), "kappa", round((agree - p_e) / (1 - p_e), 3), round(cohen_kappa_score(human, judge), 3))
print("always pass judge kappa", cohen_kappa_score(human, np.ones(100, dtype=int)))

tpr_fail = 14 / 20                                        # share of real failures the judge catches
tnr_fail = 74 / 80                                        # share of real passes the judge lets through
print("failure recall", tpr_fail, "95% Wilson CI", np.round(proportion_confint(14, 20, method="wilson"), 3))

# correct a judged pass rate on new traffic for the judge's known error rates (Rogan-Gladen)
sens_pass, spec_pass = tnr_fail, tpr_fail                 # same numbers seen from the pass side
observed_pass = 0.85
true_pass = (observed_pass + spec_pass - 1) / (sens_pass + spec_pass - 1)
print("corrected pass rate", round(true_pass, 3))
```

```text
[[14  6]
 [ 6 74]]
agreement 0.88 chance 0.68 kappa 0.625 0.625
always pass judge kappa 0.0
failure recall 0.7 95% Wilson CI [0.481 0.855]
corrected pass rate 0.88
```

The judge agrees 88 percent of the time, but chance alone gives 68 percent, so kappa is 0.625; a judge that passes everything gets 80 percent agreement and kappa 0. It catches 70 percent of real failures, and with only 20 failures the interval on that recall runs from 0.48 to 0.85, which is Hamel's point about sample size. Once you know the judge's error rates you can correct a judged pass rate on new traffic: observed 85 percent becomes an estimated 88 percent with the Rogan-Gladen formula $p = (p_{obs} + t - 1)/(s + t - 1)$, where $s$ and $t$ are sensitivity and specificity for the pass class.

Recalibrate whenever the judge model, its prompt or the product changes, and expect criteria drift as labellers see new outputs. Shankar et al. frame the target as coverage (the share of bad outputs the evaluator fails) and false failure rate (the share of good outputs it wrongly fails), the same two numbers in different words.

> **Watch:** [CME295 Lecture 8](https://www.youtube.com/watch?v=8fNP4N46RRo&t=428s) covers inter rater agreement metrics.

## Regression gates in CI

A regression gate turns evals into a release rule. A typical gate has tiers:

1. Deterministic tests that must all pass (schemas, citation format, no leaked secrets, refusal on known out of scope requests).
2. Must pass golden cases with zero tolerance: safety blocks, refusals when evidence is missing, high value customer flows.
3. Aggregate metrics compared with the last released baseline on the same cases: pass rate, faithfulness, recall at k, with a tolerance.
4. Budgets: p95 latency, tokens and cost per case.

Make runs reproducible: pin the judge model version, judge prompt and temperature, cache system outputs by input hash, and run several samples on cases known to be flaky. Compare paired results, because the same case under old and new versions is the unit of analysis.

```python
from scipy.stats import binomtest

# 200 golden cases scored pass/fail under the old and the new prompt
both_pass, old_only, new_only, both_fail = 150, 18, 8, 24
n = both_pass + old_only + new_only + both_fail
old_rate = (both_pass + old_only) / n
new_rate = (both_pass + new_only) / n
p = binomtest(new_only, old_only + new_only, 0.5).pvalue    # exact McNemar on discordant pairs
print(f"old {old_rate:.2f} new {new_rate:.2f} discordant {old_only}/{new_only} p={p:.3f}")

must_pass_failures = ["g017_refuse_when_no_evidence"]       # zero tolerance cases
blocked = bool(must_pass_failures) or new_rate < old_rate - 0.02
print("gate:", "BLOCK" if blocked else "PASS")
```

```text
old 0.84 new 0.79 discordant 18/8 p=0.076
gate: BLOCK
```

A 5 point drop on 200 cases is not significant at 0.05 by an exact McNemar test (18 cases regressed, 8 improved, p = 0.076). Statistics alone would let it through, which is why the gate also blocks on any must pass failure and on a fixed tolerance, and why someone reads the 18 regressions. Two rules matter more than the mechanics. Never lower a threshold to make the gate pass; find the cause or document the failure. And keep known failures visible in the report, because a suite where everything always passes is not measuring anything.

## Benchmarks and contamination

Public benchmarks measure general capability on a fixed task: MMLU for knowledge, AIME for competition maths, SWE-bench for coding, HarmBench for safety, tau-bench for tool using agents (the set CME295 Lecture 8 surveys). They tell you little about your product, and they decay.

**Contamination** means benchmark items, or close paraphrases, were in the training data, so a score measures memory rather than skill. Signs: the model reproduces reference answers verbatim, accuracy drops sharply on paraphrased or newly written items, and performance is much higher on items published before the training cutoff than after. Defences:

- **Time splits.** Evaluate on items created after the model's cutoff. SWE-bench was built so new task instances can be collected continually from GitHub for exactly this reason.
- **Private or rotating test sets**, and canary strings in published data so training pipelines can exclude it.
- **Overlap checks** (n gram matching against training corpora when you control them) and membership inference style tests when you do not.
- **Perturbation tests**: rename variables, change numbers, reorder options.

Your own golden set gets contaminated too: tuning prompts on the test cases, pasting cases into few shot examples, fine tuning on logs that include eval inputs, or generating references with the same model you evaluate. Benchmarks also carry construction bias, as BEIR showed with lexically pooled relevance judgments. Saturation is the other failure: once top systems cluster near the ceiling, differences are noise.

> **Watch:** [CME295 Lecture 8](https://www.youtube.com/watch?v=8fNP4N46RRo&t=5030s) surveys benchmarks across knowledge, reasoning, coding and safety.

## Agent benchmarks and long horizon tasks

**SWE-bench** (Jimenez et al., ICLR 2024) has 2,294 tasks from real GitHub issues and merged pull requests across 12 Python repositories. A task gives the model the issue text and the repository; the model produces a patch; the patch passes if it applies and the tests that the reference PR turned from failing to passing (fail to pass tests) now pass, along with the existing tests (a median of 51 per task) that guard against breaking other behaviour. Reference fixes edit 1.7 files, 3.0 functions and 32.8 lines on average. In the 2023 paper, Claude 2 with BM25 retrieval resolved 1.96 percent. A 300 task Lite subset exists, and a human validated subset, SWE-bench Verified, is what Anthropic's agent post reports against.

**tau-bench** (Yao et al., 2024) tests tool agents talking to a simulated user (gpt-4-0613) under a written domain policy, in a retail domain (115 tasks) and an airline domain (50 tasks). Grading compares the final database state with the annotated goal state and checks that required information reached the user, so many different conversations can succeed. Its reliability metric is **pass hat k**, the chance that all $k$ independent trials succeed, estimated per task from $n$ trials with $c$ successes:

$$
\text{pass}^k = \mathbb{E}_{\text{task}}\left[\frac{\binom{c}{k}}{\binom{n}{k}}\right], \qquad \text{pass@}k = 1 - \mathbb{E}_{\text{task}}\left[\frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

gpt-4o with function calling scored about 61 percent pass hat 1 on retail and 35 percent on airline, and pass hat 8 on retail fell below 25 percent. Pass at k rewards discovery (useful when a verifier picks the winner); pass hat k rewards consistency (what a customer facing agent needs). The paper also notes that a perfect database match is necessary but not sufficient, since an agent can act without the confirmation the policy requires.

**Long horizon evaluation** asks how long a task an agent can complete. CS329A Part 8 covers METR's time horizon method, which measures the human task length a model completes at 50 and 80 percent reliability and finds it roughly doubling every seven months, from seconds for GPT-2 in 2019 to nearly an hour for Claude 3.7 Sonnet in 2025; and OpenAI's GDPval, where model work is compared against professionals across 44 occupations, with win rates rising from 12.4 percent for GPT-4o to 47.6 percent for Claude Opus 4.1 (2025).

For your own agents, evaluate both the outcome (final state, task success, cost per success) and the trajectory (tool call accuracy, policy compliance, number of steps, loops), run each task several times and report pass hat k, and read transcripts, because two agents with the same success rate can fail in completely different ways.

> **Watch:** [CME295 Lecture 8](https://www.youtube.com/watch?v=8fNP4N46RRo&t=3615s) covers agent evaluation, [SWE-bench](https://www.youtube.com/watch?v=8fNP4N46RRo&t=5637s) and [tau-bench](https://www.youtube.com/watch?v=8fNP4N46RRo&t=6051s); [CS329A Part 8](https://www.youtube.com/watch?v=8JAqLnTaZu4) covers time horizons and long horizon failures.

## Observability, latency and cost

**Tracing** is the debugger for LLM systems. Record a trace per request with a span per step (rewrite, retrieve, rerank, each LLM call, each tool call) holding inputs, outputs, retrieved ids and scores, prompt and model versions, tokens, latency and cost. Standard tracing such as OpenTelemetry works; the MCP docs point new servers to it for logging. Dashboards worth having: p50, p95 and p99 latency, time to first token, tokens and cost per request and per resolved task, empty retrieval rate, refusal rate, guardrail trigger rate, judge pass rate on sampled traffic. Redact personal data in traces, since traces are a copy of everything users said.

**Latency** is a budget split across stages. A target like 800 ms end to end (an example from an interview prep write up) forces choices: retrieve in parallel, keep reranking to a shortlist, stream tokens, run independent tool calls concurrently, and keep long stable prefixes cacheable.

**Caching**, cheapest first:

- Prompt (prefix) caching reuses the computed prefix of a repeated prompt; Anthropic reported more than 2x lower latency and up to 90 percent lower cost for long cached prompts (2024).
- Exact response caching for repeated identical requests.
- Semantic caching returns a stored answer when a new query embeds close to an old one. It risks answering a different question ("refund for monthly" served the "refund for annual" answer), serving stale answers, and leaking answers across users with different permissions, so scope it per permission set and give it a time to live.
- Embedding and retrieval caches for popular queries.

**Routing and cascades** send easy requests to a small model and hard ones to a large one. If 70 percent of traffic goes to a model at one tenth the price, cost falls to $0.7 \times 0.1 + 0.3 \times 1 = 0.37$ of the all large baseline, a 63 percent saving, provided the router's misroutes are measured. CS329A Part 9 presents intelligence per watt results in which local models of 20B parameters or fewer handled 88.7 percent of real chatbot queries. A cascade tries the small model first and escalates when a verifier or confidence check fails.

**Token budgets** keep agents bounded: cap retrieved context (top k times chunk size), compact history, cap steps and tool calls, and set a spend limit per task that triggers escalation instead of silent overspend. OWASP's unbounded consumption entry (LLM10) is the security view of the same rule.

> **Watch:** [CME295 Lecture 7](https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=2460s) covers prompt caching, and [CS329A Part 9](https://www.youtube.com/watch?v=AyO6wyu4DEg) covers intelligence per watt and local inference.

## What interviewers probe

These chains come from the question banks and write ups used for this module ([DataCamp's RAG questions](https://www.datacamp.com/blog/rag-interview-questions), [techinterview.org on applied AI loops in 2026](https://www.techinterview.org/post/3233476824/ai-engineer-interview-rag-agents-evals/?format=md)) and from the lectures. Expect each opener to be followed down at least three levels.

- **"Walk me through what happens between pressing enter and getting an answer over a 200 page PDF."** Then: why that chunk size, what breaks at chunk edges, would you even retrieve if it fits in context, how you would know retrieval is working as a number.
- **"You raised top k from 5 to 20. What did that cost?"** Recall rises; context tokens, latency and distractors rise too; the lost in the middle curve says more is not free; rerank instead.
- **"Dense or BM25?"** Then: why BM25 survives (BEIR), what RRF does and why $k = 60$, what a cross encoder costs per query, why the reranker cannot fix first stage misses.
- **"Write nDCG."** Then: linear or exponential gain, where the ideal DCG comes from, what happens with unjudged documents, how you would compare two systems over 200 queries.
- **"How is a dense retriever trained?"** Then: in batch negatives and why bigger batches help, hard negatives, false negatives, domain adaptation with synthetic queries.
- **"HNSW or IVF PQ for 100M vectors?"** Then: memory per vector, what M and efSearch do, how you pick nprobe, how you delete a document, what filtering does to the graph.
- **"The bot answers confidently from a stale document."** Then: freshness pipeline, deletion through derived artifacts, trust flags as hard filters, refusal gates.
- **"You use an LLM judge. How do you know it is any good?"** Then: agreement with expert labels, why raw agreement misleads, kappa, sample size per class, position and verbosity bias controls, what happens when the judge model is upgraded.
- **"Your agent loops. What do you do?"** Then: how you trace which step failed, what you cap (steps, tokens, spend, repeated calls), when you escalate to a human.
- **"Would you build this as a multi agent system?"** Then: token cost (about 15 times chat in Anthropic's report), shared state, a single agent baseline at equal budget.
- **"How do you defend against prompt injection in retrieved documents?"** Then: why no filter is complete, least privilege, confirmation for side effects, output handling and exfiltration through URLs.
- **"Your eval pass rate dropped 5 points. Ship or not?"** Then: paired test on the same cases, must pass cases, reading the regressions, never moving the threshold.
- **"pass at k or pass hat k for a support agent?"** Then: why reliability over trials matters, what tau-bench's database check does and does not prove.

## Sources

- [Anthropic: Building effective agents (2024)](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic: How we built our multi agent research system (2025)](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Anthropic: Introducing Contextual Retrieval (2024)](https://www.anthropic.com/engineering/contextual-retrieval)
- [Yao et al.: ReAct (2022)](https://arxiv.org/abs/2210.03629)
- [Schick et al.: Toolformer (2023)](https://arxiv.org/abs/2302.04761)
- [Lewis et al.: Retrieval Augmented Generation for Knowledge Intensive NLP Tasks (2020)](https://arxiv.org/abs/2005.11401)
- [Karpukhin et al.: Dense Passage Retrieval (2020)](https://arxiv.org/pdf/2004.04906)
- [Thakur et al.: BEIR (2021)](https://arxiv.org/pdf/2104.08663)
- [Khattab and Zaharia: ColBERT (2020)](https://arxiv.org/pdf/2004.12832)
- [Cormack, Clarke and Buttcher: Reciprocal Rank Fusion (2009)](http://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
- [Malkov and Yashunin: HNSW](https://arxiv.org/pdf/1603.09320)
- [FAISS wiki: Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [Okapi BM25 (Wikipedia, citing Robertson and Zaragoza 2009)](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Liu et al.: Lost in the Middle (2023)](https://arxiv.org/pdf/2307.03172)
- [Es et al.: RAGAS (2023)](https://arxiv.org/pdf/2309.15217)
- [Zheng et al.: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (2023)](https://arxiv.org/pdf/2306.05685)
- [Shankar et al.: Who Validates the Validators? (2024)](https://arxiv.org/pdf/2404.12272)
- [Hamel Husain: Your AI Product Needs Evals (2024)](https://hamel.dev/blog/posts/evals/)
- [Hamel Husain: Using LLM as a judge, critique shadowing (2024)](https://hamel.dev/blog/posts/llm-judge/)
- [Eugene Yan: Evaluating the Effectiveness of LLM Evaluators (2024)](https://eugeneyan.com/writing/llm-evaluators/)
- [Jimenez et al.: SWE-bench (2023)](https://arxiv.org/pdf/2310.06770)
- [Yao et al.: tau-bench (2024)](https://arxiv.org/pdf/2406.12045)
- [Model Context Protocol: architecture overview](https://modelcontextprotocol.io/docs/learn/architecture)
- [OWASP Top 10 for LLM Applications (2025)](https://genai.owasp.org/llm-top-10/)
- [Greshake et al.: Indirect Prompt Injection (2023)](https://arxiv.org/pdf/2302.12173)
