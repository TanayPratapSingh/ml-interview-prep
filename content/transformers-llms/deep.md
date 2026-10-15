# Transformers and LLMs: architecture, pretraining and inference

This is the research lab depth version of the module. It covers how text becomes tokens, how a decoder only transformer turns tokens into next token probabilities, how to count its parameters, FLOPs and memory, what the scaling laws say about how big to make it and how long to train it, and how inference engines make it fast. Post training (SFT, LoRA, RLHF, DPO, GRPO, reasoning) and RAG, agents and evals live in their own modules. Every number names the paper or config it came from, and every formula and code block was run in Python before it went in.

## Tokenisation

A tokeniser maps a string to a list of integer ids from a fixed vocabulary and back. It is trained separately from the model, on its own corpus, with its own algorithm, and after training it exposes two functions: `encode` (string to ids) and `decode` (ids to string). Karpathy's "Let's build the GPT Tokenizer" argues that a large share of odd LLM behaviour traces back to this stage.

### Why subwords

Three granularities are possible, and the CME295 VIP cheatsheet lays out the trade off:

| Unit | Vocabulary | Sequence length | Unknown words | Problem |
| --- | --- | --- | --- | --- |
| Word | Huge (every inflection) | Short | Map to `[UNK]` | "love", "loved", "loving" share nothing |
| Character or byte | Tiny (256 bytes) | Very long | None | Attention cost grows with the square of length, and each unit carries little meaning |
| Subword | 30K to 200K | Medium | None with byte fallback | Splits are statistical, not linguistic |

Subwords keep frequent words whole and break rare words into reusable pieces, so the model never meets an unknown word and sequences stay short.

### BPE step by step

Byte pair encoding (Sennrich et al. 2015 for NLP, popularised for LLMs by GPT-2) is a greedy compression algorithm:

1. Start with a base vocabulary. In byte level BPE this is the 256 possible byte values, so any UTF-8 string is representable.
2. Count every adjacent pair of symbols in the training corpus.
3. Merge the most frequent pair into a new symbol, give it the next id, and record the merge rule.
4. Repeat until the vocabulary reaches the target size. Vocabulary size equals base size plus number of merges.

Encoding new text replays the learned merges in the order they were learned (lowest merge id first), not by current frequency. This is the version in Karpathy's minbpe, rewritten here and run:

```python
from collections import Counter

def get_stats(ids):
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train_bpe(text, n_merges):
    ids = list(text.encode("utf-8"))      # start from raw bytes, ids 0 to 255
    merges = {}
    for k in range(n_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)  # most frequent adjacent pair
        merges[pair] = 256 + k            # mint a new token id
        ids = merge(ids, pair, 256 + k)
    return ids, merges

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:            # nothing left that we learned to merge
            break
        ids = merge(ids, pair, merges[pair])
    return ids

ids, merges = train_bpe("aaabdaaabac", 3)
print(ids)                                # [258, 100, 258, 97, 99]
print(merges)                             # {(97, 97): 256, (256, 97): 257, (257, 98): 258}
print(encode("aaabdaaabac", merges) == ids)
print(len("héllo".encode("utf-8")), "bytes for 5 characters")
```

The output `[258, 100, 258, 97, 99]` matches the example in the minbpe README. Note the last line: "é" is two bytes in UTF-8, which is why byte level tokenisers spend more tokens on non Latin scripts.

### Byte level BPE and the regex pre split

GPT-2's vocabulary is 50,257 tokens: 256 byte tokens, 50,000 merges and one `<|endoftext|>` special token (Hugging Face tokenizer docs, and the comment in build-nanogpt's `GPTConfig`). Before merging, GPT-2 and GPT-4 split text with a regex into chunks of letters, digits, punctuation and whitespace, and BPE never merges across chunk boundaries. Without that split, BPE would learn tokens like "dog." and "dog!" as separate entries. The GPT-4 pattern in minbpe also splits digits into groups of at most three, which gives numbers a more regular tokenisation. Special tokens such as `<|endoftext|>` are handled outside BPE, and minbpe and tiktoken refuse to parse them out of user text unless you opt in, because a user who can inject special tokens can forge document boundaries or chat roles.

### WordPiece, Unigram and SentencePiece

| Algorithm | Builds the vocabulary by | Used by | Tokenises new text by |
| --- | --- | --- | --- |
| BPE | Merging the most frequent pair | GPT family, Llama, Mistral, Qwen | Replaying merges in order (deterministic) |
| WordPiece | Merging the pair with the highest $\frac{\text{freq}(ab)}{\text{freq}(a)\,\text{freq}(b)}$ | BERT, DistilBERT, Electra | Greedy longest match, continuation pieces marked `##` |
| Unigram LM | Starting large and pruning the tokens whose removal hurts corpus likelihood least (often 10 to 20 percent per round) | T5, ALBERT, XLNet (via SentencePiece) | Most probable segmentation (Viterbi), and it can sample segmentations as regularisation |

SentencePiece is a library, not an algorithm. It runs BPE or Unigram directly on the raw stream, treats the space as an ordinary symbol written `▁`, and so needs no language specific pre tokeniser, which matters for Chinese and Japanese. Llama 2 used SentencePiece BPE with byte fallback, and Llama 3 moved to a tiktoken based BPE with 128K tokens: 100K from tiktoken plus 28K extra non English tokens (Llama 3 paper, Meta 2024).

The WordPiece score is pointwise mutual information in disguise: it merges pairs that co occur more than their individual frequencies predict, so it prefers "g"+"s" (seen together every time "s" appears) over a frequent pair of two very common symbols.

### Vocabulary size trade offs

A bigger vocabulary compresses text into fewer tokens, so each forward pass covers more text and the effective context is longer. Llama 3's 128K tokeniser raised English compression from 3.17 to 3.94 characters per token compared with Llama 2's tokeniser (Llama 3 paper). The costs: the embedding and output matrices grow linearly with vocabulary size (at 128,000 tokens and width 4,096, each matrix is 524M parameters, about 6.5 percent of an 8B model each), the softmax over the vocabulary gets more expensive, and rare tokens get few gradient updates so their embeddings stay poorly trained. A practical detail from build-nanogpt: Karpathy pads GPT-2's 50,257 up to 50,304, a multiple of 64 (and of 128), because GPU kernels handle such shapes more efficiently; the extra rows are never used.

### Why tokenisation causes odd behaviour

- **Spelling and counting.** "How many r's in strawberry" fails because the model sees two or three tokens, not ten letters. It has to have memorised the spelling of each token.
- **Arithmetic.** Numbers split inconsistently ("1234" vs "12" "34" depending on context in older tokenisers), so digit positions do not line up. The GPT-4 split pattern, which caps number chunks at three digits, makes this more regular without removing it.
- **Non English cost.** Languages underrepresented in the tokeniser corpus fragment into more tokens per word, so they cost more, fill context faster and get less compute per word.
- **Leading spaces.** " hello" and "hello" are different tokens. A prompt ending in a trailing space pushes the model into a rare token boundary and degrades completions.
- **Glitch tokens.** A token that appeared in the tokeniser's corpus but almost never in the model's corpus (Karpathy's example is "SolidGoldMagikarp", a Reddit username) has an untrained embedding, and prompting with it produces bizarre output.
- **Code.** GPT-2 tokenised each space of Python indentation separately, wasting context. GPT-4's tokeniser merges runs of whitespace.

> **Interview trap:** Saying the tokeniser is learned jointly with the model. It is trained first, frozen, and the model only ever sees integer ids.

> **Watch:** [Let's build the GPT Tokenizer, BPE walkthrough](https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s), then [the quirks section](https://www.youtube.com/watch?v=zduSFxRajkE&t=6701s). [CME295 Lecture 1, tokenization](https://www.youtube.com/watch?v=Ub3GoFaUcds&t=1377s).

## Embeddings and the output layer

The embedding layer is a lookup table $E \in \mathbb{R}^{V \times d}$: token id $i$ becomes row $E_i$. That is the same as multiplying a one hot vector by $E$, which is why gradients only touch the rows of tokens present in the batch. GPT-2 then adds a learned position vector from a second table of shape (1024, 768); Llama style models add nothing here and inject position inside attention instead (see positional information).

At the top, the final hidden state $h_t \in \mathbb{R}^d$ is mapped to logits over the vocabulary by an unembedding matrix $W_U \in \mathbb{R}^{d \times V}$, and the loss is the cross entropy of the true next token:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log \operatorname{softmax}(h_t W_U)_{x_{t+1}}
$$

Perplexity is $e^{\mathcal{L}}$. A useful sanity check from Karpathy's GPT-2 reproduction: at initialisation the model should be close to uniform, so the loss should start near $\ln V$, which is $\ln 50257 \approx 10.82$ for GPT-2. A starting loss far above that means the logits are too large at init.

### Weight tying

**Weight tying** sets $W_U = E^\top$, so the same matrix reads tokens in and writes them out (Press and Wolf 2016; the original transformer also shared these weights and scaled embeddings by $\sqrt{d_{\text{model}}}$, Vaswani et al. 2017). The argument: a token that should be predicted in a context should have an output vector similar to its input vector, and the output side gets a gradient for every vocabulary row at every position, so tying gives the rarely seen input rows a much better training signal. nanoGPT ties them (`self.transformer.wte.weight = self.lm_head.weight`). In GPT-2 small the shared matrix is $50257 \times 768 \approx 38.6$M of the 124.4M parameters, so tying saves about 31 percent of the model.

Large models usually do not tie. Mistral 7B, Mixtral 8x7B, Qwen2.5-7B and Qwen3-8B all ship `"tie_word_embeddings": false` in their Hugging Face configs. At 7B scale the embedding is a few percent of parameters, the saving is small, and separate matrices give the model more freedom. Small models (where the embedding is a large share) are where tying pays.

### Initialisation detail worth knowing

GPT-2 and nanoGPT initialise weights from a normal with standard deviation 0.02 and scale the output projection of every residual branch (`c_proj`) by $1/\sqrt{2 n_{\text{layer}}}$. Each block adds two branches to the residual stream, so after $L$ blocks the stream is a sum of $2L$ terms; scaling each by $1/\sqrt{2L}$ keeps the variance of the stream roughly constant with depth.

> **Watch:** [Let's reproduce GPT-2, parameter sharing wte and lm_head](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3974s) and [model initialisation](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4427s). [CME295 Lecture 1, word representation](https://www.youtube.com/watch?v=Ub3GoFaUcds&t=1828s).

## Scaled dot product attention

Attention lets every position build its output as a weighted average of value vectors from other positions, with weights set by how well its query matches their keys. For input $X \in \mathbb{R}^{T \times d}$ and learned projections:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,\qquad \operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

where $M$ is a mask of zeros and $-\infty$ (Vaswani et al. 2017). Read it as a soft dictionary lookup: the query asks a question, every key advertises what its position contains, the dot product scores the match, softmax turns scores into weights that sum to one, and the output mixes the values by those weights. Karpathy frames it as communication: tokens are nodes that send value messages along edges weighted by query key affinity.

### Deriving it from the averaging trick

Karpathy's "Let's build GPT" builds attention in four steps, and reproducing them is a common whiteboard exercise:

1. The weakest aggregation: each token averages all previous tokens' vectors (a for loop).
2. The same thing as a matrix multiply: a lower triangular matrix with rows normalised to sum to one, times $X$.
3. The same thing with softmax: fill the upper triangle with $-\infty$, put zeros elsewhere, softmax each row. You get uniform weights over the past.
4. Make the weights data dependent: replace the zeros with $QK^\top$ scores, so each token chooses how much to take from each past token.

### Why divide by the square root of d_k

If the components of $q$ and $k$ are independent with mean 0 and variance 1, then $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$ (footnote 1 of Vaswani et al. 2017). Each product $q_i k_i$ has variance $\mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$, and the $d_k$ independent terms add. With $d_k = 128$ the raw scores have standard deviation about 11, so the softmax puts almost all its mass on the largest score. The softmax Jacobian is $\operatorname{diag}(p) - pp^\top$, which goes to zero when $p$ is close to one hot, so gradients through attention vanish and learning stalls. Dividing by $\sqrt{d_k}$ brings the variance back to 1. It is the square root, and not $d_k$, because variance scales with the square of a constant: $\operatorname{Var}(s/c) = \operatorname{Var}(s)/c^2$.

```python
import numpy as np

rng = np.random.default_rng(0)

# 1. The variance argument: q.k has variance d_k when entries are iid with mean 0, variance 1
for d_k in (16, 64, 256):
    q = rng.standard_normal((100_000, d_k))
    k = rng.standard_normal((100_000, d_k))
    dots = (q * k).sum(axis=1)
    print(d_k, round(dots.var(), 1), round((dots / np.sqrt(d_k)).var(), 2))

# 2. What happens to softmax without the scaling
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # subtract the max for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

d_k = 256
scores = rng.standard_normal(8) * np.sqrt(d_k)  # unscaled scores have std of about 16
p_raw, p_scaled = softmax(scores), softmax(scores / np.sqrt(d_k))
print("max weight unscaled", p_raw.max().round(3), "scaled", p_scaled.max().round(3))

# 3. Causal scaled dot product attention with an optional padding mask
def attention(Q, K, V, causal=True, key_padding=None):
    T, d_k = Q.shape[-2], Q.shape[-1]
    S = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)          # (T, T) scores
    if causal:
        S = np.where(np.tril(np.ones((T, T), bool)), S, -np.inf)
    if key_padding is not None:                          # True where the key is padding
        S = np.where(key_padding[None, :], -np.inf, S)
    return softmax(S) @ V

T, d = 5, 8
X = rng.standard_normal((T, d))
Wq, Wk, Wv = (rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3))
out = attention(X @ Wq, X @ Wk, X @ Wv)
# causality check: changing the last token must not change earlier outputs
X2 = X.copy(); X2[-1] += 10
out2 = attention(X2 @ Wq, X2 @ Wk, X2 @ Wv)
print("earlier rows unchanged:", np.allclose(out[:-1], out2[:-1]))
pad = np.array([False, False, False, True, True])         # last two keys are padding
out3 = attention(X @ Wq, X @ Wk, X @ Wv, causal=False, key_padding=pad)
print(out3.shape)
```

Running it prints variances of about 16, 64 and 255 before scaling and about 1.0 after, a maximum softmax weight of 1.0 unscaled against 0.417 scaled, and `True` for the causality check. The subtraction of the row max in `softmax` changes nothing mathematically and prevents overflow; interviewers who ask you to write softmax look for it.

### Causal and padding masks

The **causal mask** sets score $(i, j)$ to $-\infty$ for $j > i$ before the softmax, so position $i$ only sees positions up to $i$. This is what lets one forward pass over a sequence of length $T$ produce $T$ training examples at once (predict token 2 from 1, token 3 from 1 to 2, and so on), and it is also what makes the KV cache valid at inference: past keys and values never change when new tokens arrive. Masking after the softmax would be wrong, because the weights would no longer sum to one.

A **padding mask** removes pad positions from the key side so real tokens never attend to them. Rows belonging to pad queries still produce outputs; they are ignored by the loss (`ignore_index` in PyTorch's cross entropy, which nanoGPT sets to -1). Two practical details: if an entire row is masked the softmax computes $0/0$ and returns NaN, so kernels guard against fully masked rows; and batched generation usually pads on the left so every sequence's last real token sits at the same position. Llama 3 also masks attention between different documents packed into the same training sequence, which the paper reports mattered little for standard pretraining and a lot for long context continued pretraining.

### Cost

Computing $QK^\top$ costs $O(T^2 d)$ FLOPs and the score matrix needs $O(T^2)$ memory per head, against $O(T d^2)$ for the projections and the FFN (Vaswani et al. 2017, Table 1). Per layer, the two attention matmuls ($QK^\top$ and the product with $V$) cost about $4T^2d$ FLOPs, against about $24Td^2$ for the four projections and a 4x wide FFN, so attention dominates only once $T$ exceeds about $6d$ (4,608 tokens at GPT-2 small's width of 768). Its advantages over recurrence: every pair of positions is one step apart (path length $O(1)$ instead of $O(T)$), and all positions are computed in parallel during training.

> **Interview trap:** "Attention is permutation invariant." Without positional information, self attention is permutation equivariant: shuffle the inputs and the outputs shuffle the same way. With a causal mask it is no longer even that, because the mask itself encodes order.

> **Watch:** [Let's build GPT, the self attention crux](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s) and [why divide by sqrt(head_size)](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4616s). [CME295 Lecture 1, self attention](https://www.youtube.com/watch?v=Ub3GoFaUcds&t=4007s).

## Multi head attention, MQA and GQA

**Multi head attention** runs $h$ attention operations in parallel on lower dimensional projections and concatenates the results:

$$
\operatorname{head}_i = \operatorname{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}),\qquad \operatorname{MHA}(X) = [\operatorname{head}_1;\dots;\operatorname{head}_h]\,W_O
$$

with $d_{\text{head}} = d/h$. The original base transformer used $d = 512$, $h = 8$, $d_k = d_v = 64$ (Vaswani et al. 2017). In code nobody loops over heads: nanoGPT computes one fused $d \times 3d$ projection, reshapes to `(B, n_head, T, head_size)`, and treats heads as an extra batch dimension.

Why several heads instead of one big one: a single softmax produces one weighting per query, so one head can attend sharply to one thing at a time. Separate heads can track different relations at once (the previous token, the matching bracket, the subject of the sentence), and the cost is the same as one head of full width because the total projection size is unchanged.

### Parameter count

Without biases MHA has four $d \times d$ matrices, $4d^2$ parameters, independent of $h$. For GPT-2 small ($d = 768$, with biases) that is $768 \cdot 2304 + 2304 + 768 \cdot 768 + 768 = 2{,}362{,}368$ per layer. Changing $h$ changes how the width is sliced, not the parameter count.

### MQA and GQA

At inference the KV cache, not the weights, is what limits batch size and context length (see the inference section), and its size is proportional to the number of key value heads.

- **Multi query attention** (Shazeer 2019) keeps $h$ query heads but a single shared key head and value head. The KV cache shrinks by a factor of $h$. The cost is some quality loss and occasional training instability.
- **Grouped query attention** (Ainslie et al. 2023) uses $g$ key value heads, each shared by $h/g$ query heads. $g = h$ is MHA and $g = 1$ is MQA. The paper converts existing MHA checkpoints by mean pooling the key and value heads within each group and uptraining with 5 percent of the original pretraining compute, and reports quality close to MHA at speed close to MQA.

Llama 3 uses 8 key value heads at every size (Llama 3 paper, Table 3). For the 8B model, 32 query heads share 8 KV heads in groups of 4, so the cache is 4 times smaller than MHA. The attention weights shrink too: $W_K$ and $W_V$ become $4096 \times 1024$ instead of $4096 \times 4096$, so attention costs 41.9M parameters per layer instead of 67.1M.

| Variant | Query heads | KV heads | KV cache per token per layer | Quality |
| --- | --- | --- | --- | --- |
| MHA | $h$ | $h$ | $2 h\, d_{\text{head}}$ values | Baseline |
| GQA | $h$ | $g$ | $2 g\, d_{\text{head}}$ values | Close to MHA |
| MQA | $h$ | 1 | $2 d_{\text{head}}$ values | Measurably lower |

**Multi head latent attention** (DeepSeek-V2, 2024) goes further: it caches one compressed latent vector per token and reconstructs per head keys and values from it with learned up projections, trading extra compute for a smaller cache. CME295 Lecture 3 covers it alongside PagedAttention.

> **Interview trap:** Saying GQA reduces FLOPs. It barely changes compute, because the query side and the score computation are unchanged. It reduces memory traffic, and decode is memory bound, which is why it speeds up generation.

> **Watch:** [CME295 Lecture 2, sharing attention heads](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=3338s). [Let's build GPT, multi headed self attention](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4919s).

## Positional information

Self attention treats its input as a set: without position information, "dog bites man" and "man bites dog" produce the same multiset of outputs. Recurrent and convolutional networks get order for free from their structure, transformers must be told.

### Absolute: sinusoidal and learned

The original transformer adds a fixed vector to each token embedding:

$$
PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

Each pair of dimensions is a clock at a different frequency, from fast (wavelength $2\pi$) to very slow. For any fixed offset $k$, $PE_{pos+k}$ is a linear function of $PE_{pos}$ (a rotation in each pair), which the authors hoped would make relative attention easy to learn and might allow extrapolation (Vaswani et al. 2017). They also report that learned position embeddings gave nearly identical results. GPT-2 and BERT use learned absolute embeddings, a table with one row per position (1,024 for GPT-2, 512 for BERT), which by construction cannot represent positions past the table.

```python
import numpy as np
d, base = 16, 10000.0
def pe(pos):
    i = np.arange(d // 2)
    w = base ** (-2 * i / d)
    out = np.empty(d); out[0::2] = np.sin(pos * w); out[1::2] = np.cos(pos * w)
    return out
# PE(pos + k) is a fixed linear map (a rotation per pair) of PE(pos)
k = 5
w = base ** (-2 * np.arange(d // 2) / d)
M = np.zeros((d, d))
for i, wi in enumerate(w):
    c, s = np.cos(k * wi), np.sin(k * wi)
    M[2*i:2*i+2, 2*i:2*i+2] = [[c, s], [-s, c]]
print(np.allclose(M @ pe(7), pe(12)), np.allclose(M @ pe(40), pe(45)))
```

This prints `True True`: the same matrix $M$ shifts any position by 5.

### Relative: T5 bias and ALiBi

T5 adds a learned scalar bias to each attention score that depends on the bucketed distance $i - j$. **ALiBi** (Press, Smith and Lewis 2021) drops learned parameters entirely: it adds $-m_h \cdot (i - j)$ to the score, a penalty linear in distance with a fixed slope per head (a geometric sequence, so some heads look far and others near). The paper reports that a 1.3B model trained on length 1,024 and tested on 2,048 matched the perplexity of a sinusoidal model trained on 2,048, while training 11 percent faster and using 11 percent less memory.

### RoPE and the rotation math

**Rotary position embedding** (Su et al. 2021, RoFormer) rotates queries and keys instead of adding anything to the embeddings. In two dimensions, treat the projected query as a complex number and multiply by $e^{im\theta}$:

$$
f_q(x_m, m) = (W_q x_m)\,e^{im\theta},\qquad f_k(x_n, n) = (W_k x_n)\,e^{in\theta}
$$

The real part of $f_q \overline{f_k}$ is the dot product, and the phases combine into $e^{i(m-n)\theta}$, so the score depends on the positions only through the offset $m - n$. In $d$ dimensions, split the vector into $d/2$ pairs and rotate pair $i$ by angle $m\theta_i$ with $\theta_i = 10000^{-2(i-1)/d}$, which is a block diagonal orthogonal matrix $R_{\Theta,m}$. Because $R_{\Theta,m}^\top R_{\Theta,n} = R_{\Theta,n-m}$:

$$
(R_{\Theta,m} W_q x_m)^\top (R_{\Theta,n} W_k x_n) = x_m^\top W_q^\top R_{\Theta,n-m} W_k x_n
$$

Properties interviewers ask about:

- It is applied to $q$ and $k$ only, inside every attention layer, never to $v$. Position changes who attends to whom, not what is passed along.
- Rotations preserve norms, so RoPE does not change the scale of the scores.
- The paper shows a long term decay: the bound on the score shrinks as relative distance grows.
- It is computed element wise, never as a matrix: $x \odot \cos + \operatorname{rotate}(x) \odot \sin$, where rotate maps each pair $(x_1, x_2)$ to $(-x_2, x_1)$.
- It caches cleanly: a key's rotation depends only on its own position, so cached keys stay valid.

```python
import numpy as np

def rope(x, pos, base=10000.0):
    """Rotate pairs (x[2i], x[2i+1]) by angle pos * theta_i, theta_i = base^(-2i/d)."""
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)       # d/2 frequencies
    ang = pos * theta
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

rng = np.random.default_rng(1)
q, k = rng.standard_normal(64), rng.standard_normal(64)
# the score depends only on the offset n - m, not on m and n separately
s1 = rope(q, 3) @ rope(k, 10)       # offset 7
s2 = rope(q, 103) @ rope(k, 110)    # offset 7, shifted by 100
s3 = rope(q, 3) @ rope(k, 11)       # offset 8
print(np.isclose(s1, s2), np.isclose(s1, s3))
print("norm preserved:", np.isclose(np.linalg.norm(rope(q, 57)), np.linalg.norm(q)))
theta = 10000 ** (-np.arange(0, 128, 2) / 128)
print("fastest and slowest wavelength in tokens:", round(2 * np.pi / theta[0], 2), round(2 * np.pi / theta[-1]))
theta5 = 500000 ** (-np.arange(0, 128, 2) / 128)
print("slowest wavelength with base 500000:", round(2 * np.pi / theta5[-1]))
```

It prints `True False`, then confirms the norm is preserved, then shows that with head dimension 128 and base 10,000 the wavelengths run from 6.28 tokens to about 54,410 tokens. Raising the base to 500,000, as Llama 3 does, stretches the slowest wavelength to about 2.56 million tokens.

### Length extrapolation

A model trained on length $L$ has never seen rotation angles (or table rows) beyond $L$, and perplexity usually blows up past it. The common responses:

- **Bigger base at training time.** Llama 3 raised the RoPE base to 500,000 and extended context from 8K to 128K in six stages using about 800B tokens (Llama 3 paper). In practice "long context" is mostly a continued pretraining recipe, not an architecture change.
- **Position interpolation** (Chen et al. 2023) squeezes the new, longer range of positions into the trained range by dividing positions by the extension factor, then fine tunes briefly. NTK aware scaling and YaRN instead stretch the low frequencies while leaving the high frequency pairs, which encode local order, nearly untouched.
- **ALiBi** extrapolates without tuning because the bias is defined for any distance, at the cost of a strong recency bias.

> **Interview trap:** Saying RoPE is added to the embeddings at the input like sinusoidal encodings. It is applied to queries and keys in every layer and nothing is added to the residual stream.

> **Watch:** [CME295 Lecture 2, sinusoidal embeddings](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=936s), [T5 bias and ALiBi](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=1556s), [RoPE](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=1862s).

## The transformer block

A modern decoder block is two residual sublayers, each wrapped in a normalisation:

```text
x = x + Attention(Norm(x))     # tokens exchange information
x = x + FFN(Norm(x))           # each token is processed on its own
```

This is the forward pass of nanoGPT's `Block`, and Llama, Mistral and Qwen follow the same shape with RMSNorm, RoPE inside attention and a SwiGLU FFN. The stack ends with one final norm before the unembedding.

### The residual stream view

Treat $x$ as a shared stream of width $d$ that runs from the embedding to the unembedding. Every sublayer reads from it through a norm and adds its output back. Nothing overwrites the stream, so information written by layer 2 is still available to layer 30, and the gradient of the loss reaches every layer through the identity path without passing through any nonlinearity. That identity path is how transformers avoid vanishing gradients at depth. Attention is the only operation that moves information between positions; the FFN, the norms and the residual additions all act on each position separately. This is the framing used in mechanistic interpretability work on transformer circuits, and it is also why the GPT-2 initialisation scales residual projections by $1/\sqrt{2L}$.

### Pre norm versus post norm

The original transformer applied $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$, the **post norm** layout (Vaswani et al. 2017). GPT-2 and almost every LLM since use **pre norm**: $x + \operatorname{Sublayer}(\operatorname{Norm}(x))$, plus a final norm. In post norm the residual path itself goes through a normalisation at every layer, so the identity shortcut is broken and gradients near the output are large at initialisation; Xiong et al. (2020) analyse this and show it is why post norm transformers need a careful learning rate warmup. Pre norm keeps a clean identity path, trains stably at depth and tolerates larger learning rates. The trade off sometimes cited is that pre norm lets later layers contribute less (the stream grows and each new branch is a smaller share of it), and well tuned post norm models can end slightly better when they train at all. For an interview: pre norm wins because it is stable at scale.

### LayerNorm, RMSNorm and why not BatchNorm

LayerNorm normalises each token vector across its $d$ features, $\gamma \odot (x - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$. **RMSNorm** (Zhang and Sennrich 2019) drops the mean subtraction and the bias: $\gamma \odot x / \sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}$. It is cheaper, has one fewer reduction, and matches LayerNorm in quality, so Llama, Mistral and Qwen use it (their configs carry an `rms_norm_eps`). BatchNorm normalises each feature across the batch, which couples examples, depends on batch statistics that differ between training and inference, and breaks with variable length sequences and batch size 1. Karpathy's GPT lecture builds LayerNorm by transposing the BatchNorm reduction from the batch axis to the feature axis.

```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def rms_norm(x, gamma, eps=1e-5):
    return gamma * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)

def silu(z):
    return z / (1 + np.exp(-z))

def swiglu_ffn(x, W_gate, W_up, W_down):
    return (silu(x @ W_gate) * (x @ W_up)) @ W_down

rng = np.random.default_rng(0)
d = 4096
x = rng.standard_normal((2, d)) * 3 + 1
print(np.round(rms_norm(x, np.ones(d)).std(-1), 3), np.round(layer_norm(x, np.ones(d), 0).std(-1), 3))

# parameter matching: a GELU FFN with hidden 4d has 8 d^2 weights,
# a SwiGLU FFN has 3 matrices, so hidden 8d/3 gives the same count
print(2 * d * 4 * d, 3 * d * int(8 * d / 3))
print("Llama 3 8B and Mistral 7B use hidden", 14336, "=", 14336 / d, "x d")
x_small = rng.standard_normal((3, 64))
W_gate, W_up, W_down = rng.standard_normal((64, 170)), rng.standard_normal((64, 170)), rng.standard_normal((170, 64))
print(swiglu_ffn(x_small, W_gate, W_up, W_down).shape)
```

The first line shows RMSNorm output with a standard deviation of about 0.95 against exactly 1 for LayerNorm: RMSNorm rescales but does not centre, so a nonzero mean survives.

### The FFN and SwiGLU

The FFN is a two layer MLP applied to every position independently: $d \to 4d \to d$ with GELU in GPT-2 (nanoGPT's `MLP`). It holds two thirds of each GPT-2 block's parameters (4.72M of 7.09M). One reading is that attention routes information and the FFN stores and transforms it, acting like a key value memory over patterns.

**SwiGLU** (Shazeer 2020, "GLU Variants Improve Transformer") replaces it with a gated unit:

$$
\operatorname{FFN}_{\text{SwiGLU}}(x) = W_{\text{down}}\big(\operatorname{SiLU}(W_{\text{gate}}x) \odot W_{\text{up}}x\big),\qquad \operatorname{SiLU}(z) = z\,\sigma(z)
$$

One branch decides how much of the other to let through, element by element. Because there are three matrices instead of two, the hidden width is cut to about $\tfrac{8}{3}d$ to keep the parameter count of a $4d$ GELU FFN. Llama 3 8B and Mistral 7B go wider than that, $14{,}336 = 3.5d$ (Llama 3 paper Table 3 and the Mistral config), and their FFNs hold about 81 percent of each block's parameters.

### Other small choices

Modern blocks drop biases in linear layers (nanoGPT notes `bias=False` is "a bit better and faster") and use no dropout during pretraining on large corpora, because a single pass over trillions of tokens does not overfit. GPT-2 itself used dropout 0.1 (its config sets `attn_pdrop` and `resid_pdrop` to 0.1).

> **Interview trap:** "The FFN mixes information across tokens." It does not. Only attention moves information between positions.

> **Watch:** [Let's build GPT, residual connections](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5208s) and [layernorm](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5571s). [CME295 Lecture 2, layer normalization](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=2622s).

## Encoders, decoders and objectives

The same block supports three families, which differ in the attention mask and the training objective:

| Family | Example | Attention | Pretraining objective | Good at |
| --- | --- | --- | --- | --- |
| Encoder only | BERT (Devlin et al. 2018) | Bidirectional | Masked language modelling (MLM), plus next sentence prediction in the original | Classification, embeddings, retrieval, rerankers |
| Decoder only | GPT series, Llama, Mistral | Causal | Causal language modelling (CLM): predict each next token | Generation, in context learning, everything general |
| Encoder decoder | T5 (Raffel et al. 2019), the original transformer | Bidirectional encoder, causal decoder with cross attention | Span corruption | Input to output tasks such as translation and summarisation |

**MLM.** BERT picks 15 percent of positions; of those it replaces 80 percent with `[MASK]`, 10 percent with a random token and leaves 10 percent unchanged, then predicts the originals. The mix exists because `[MASK]` never appears at fine tuning time, so the model must also learn good representations for unmasked tokens. The weakness: only the selected 15 percent of positions produce a loss, so each sequence gives less training signal than CLM, where every position does.

**CLM.** Maximise $\sum_t \log p_\theta(x_t \mid x_{<t})$. One causal forward pass over $T$ tokens yields $T$ predictions, generation is the training task itself, and the KV cache makes inference incremental. CS229 Lecture 13 builds LLMs from exactly this next word prediction loss.

**Span corruption.** T5 replaces random spans of the input (15 percent of tokens, mean span length 3) with sentinel tokens and trains the decoder to output the dropped spans, each preceded by its sentinel. Cross attention is how the decoder reads the encoder: queries come from the decoder, keys and values from the encoder output.

Why decoder only won for LLMs: one stack instead of two, every token is a training target, any task can be posed as text continuation, the KV cache makes serving simple, and the scaling laws were measured on it. The CME295 cheatsheet notes that encoder decoder models are also autoregressive and share many traits with decoder only models; T5 style models remain a good choice when inputs are long and outputs short, because the encoder runs once. Encoder only models remain the default for embeddings and rerankers. A 2025 to 2026 trend covered in CME295 Lecture 9 is diffusion LLMs, which generate by iteratively denoising a fully masked sequence instead of left to right.

> **Interview trap:** Calling BERT a generative model. MLM gives it no natural way to sample a sequence left to right.

> **Watch:** [CME295 Lecture 2, transformer based models](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=3762s) and [BERT deep dive](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=4298s). [Let's build GPT, encoder vs decoder](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6159s). [CS229 Lecture 13, LLMs and next word prediction](https://www.youtube.com/watch?v=lNTajqxxOn4).

## Counting parameters and FLOPs

### Parameters

Per block, ignoring biases and norms:

- Attention: $W_Q$ and $W_O$ are $d \times d$; $W_K$ and $W_V$ are $d \times (g\, d_{\text{head}})$. With MHA ($g\,d_{\text{head}} = d$) that is $4d^2$.
- FFN: $8d^2$ for a $4d$ GELU MLP, $3\,d\,d_{\text{ff}}$ for SwiGLU.

So a GPT style block is about $12d^2$ and the non embedding count is $N \approx 12\,L\,d^2$ (Kaplan et al. 2020, Eq. 2.1). Add $V d$ for the embedding, another $Vd$ if the output layer is untied, and the position table if learned.

> **Worked example:** GPT-2 small, from nanoGPT's config ($L = 12$, $d = 768$, $V = 50{,}257$, 1,024 positions, tied). Each block has 7,087,872 parameters with biases, the 12 blocks 85.05M (the $12Ld^2$ shortcut gives 84.93M), token embeddings 38.60M, positions 0.79M, final norm 1,536. Total 124,439,808, the "124M" model.

The same count for Mistral 7B, with values copied from its `config.json`:

```python
# Values copied from mistralai/Mistral-7B-v0.1 config.json on Hugging Face
cfg = dict(hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
           num_key_value_heads=8, intermediate_size=14336, vocab_size=32000,
           tie_word_embeddings=False)

def count_params(c):
    d, L = c["hidden_size"], c["num_hidden_layers"]
    h, kv, f, V = c["num_attention_heads"], c["num_key_value_heads"], c["intermediate_size"], c["vocab_size"]
    hd = d // h                                     # head dimension, 128 here
    attn = d * h * hd + 2 * d * kv * hd + h * hd * d  # W_Q, W_K, W_V (shared by groups), W_O
    mlp = 3 * d * f                                  # SwiGLU: gate, up and down matrices
    norms = 2 * d                                    # two RMSNorm weight vectors per block
    emb = V * d * (1 if c["tie_word_embeddings"] else 2)
    return L * (attn + mlp + norms) + emb + d        # + final norm

def kv_cache_bytes(c, seq_len, batch=1, bytes_per=2):
    hd = c["hidden_size"] // c["num_attention_heads"]
    return 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * hd * bytes_per * seq_len * batch

print(f"{count_params(cfg):,} parameters")                      # 7,241,732,096
print(kv_cache_bytes(cfg, 1) // 1024, "KiB per token")           # 128
print(kv_cache_bytes(cfg, 8192) / 2**30, "GiB for one 8K sequence") # 1.0
mha = dict(cfg, num_key_value_heads=32)
print(kv_cache_bytes(mha, 1) // 1024, "KiB per token if it used full MHA")  # 512
```

7,241,732,096 parameters, which is why it is called a 7B model. Swapping in Llama 3 8B's shape from the Llama 3 paper (identical blocks, vocabulary 128,000) gives 8.03B; the extra 0.8B is almost all the bigger vocabulary across two untied matrices, which make up 13 percent of the model.

### FLOPs: the 6ND rule

A matrix multiply with $N$ weights costs about $2N$ FLOPs per token on the forward pass (one multiply and one add per weight). The backward pass costs about twice the forward, because it computes gradients with respect to both activations and weights. Training therefore costs about $6N$ FLOPs per token, and a whole run about

$$
C \approx 6ND
$$

for $N$ parameters and $D$ training tokens (Kaplan et al. 2020; Hoffmann et al. 2022 report that their detailed count differs from $6ND$ by only about 1 to 10 percent). Inference costs about $2N$ per generated token. The rule ignores attention's own $T^2$ term; nanoGPT's MFU estimate, following the PaLM paper's appendix, uses $6N + 12\,L\,H\,d_{\text{head}}\,T$ per token, and for GPT-2 small at $T = 1024$ the attention term adds about 15 percent.

> **Worked example:** Llama 3 405B on 15.6T tokens: $6 \times 405\text{e}9 \times 15.6\text{e}12 = 3.79\text{e}25$, matching the 3.8e25 FLOPs the Llama 3 paper reports. Chinchilla, 70B on 1.4T tokens: 5.9e23. Karpathy's GPT-2 124M on 10B FineWeb EDU tokens: 7.4e18.

To turn FLOPs into time, divide by achieved throughput. **Model FLOPs utilisation** (MFU) is achieved FLOPs over the hardware peak; nanoGPT measures it against the A100's 312 TFLOPS bf16 peak, and Llama 3 405B ran at 38 to 43 percent BF16 MFU, about 380 to 430 TFLOPs per GPU on up to 16,384 GPUs (Llama 3 paper, Table 4). At 400 TFLOPs on 16,384 GPUs, 3.8e25 FLOPs is about 67 days of uninterrupted compute, an idealised floor that ignores failures and restarts.

### Training memory

With mixed precision Adam, each parameter needs about 16 bytes: 2 for bf16 weights, 2 for bf16 gradients, and 12 of fp32 optimiser state (a master copy of the weights plus Adam's first and second moments). This is the accounting the ZeRO paper uses to motivate sharding optimiser state across data parallel workers. An 8B model needs about 128 GB before activations, which is why even an 8B model is trained across several GPUs.

> **Watch:** [CME295 Lecture 4, FLOPs and FLOPS](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=806s) and [data parallelism with ZeRO](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=1869s).

## Scaling laws

Scaling laws are empirical power laws that predict loss from model size, data and compute, so you can choose a configuration before spending the budget.

### Kaplan et al. 2020

Fit on decoder only transformers, with $N$ counting non embedding parameters:

$$
L(N) = \left(\frac{N_c}{N}\right)^{0.076},\qquad L(D) = \left(\frac{D_c}{D}\right)^{0.095},\qquad L(C_{\min}) = \left(\frac{C_c}{C_{\min}}\right)^{0.050}
$$

Their main recommendations: shape (depth versus width) matters little within a wide range, larger models are more sample efficient, and for a fixed compute budget the optimum is a very large model trained on relatively little data and stopped well before convergence, with $N_{\text{opt}} \propto C^{0.73}$ and $D_{\text{opt}} \propto C^{0.27}$. Under that rule, 10 times more compute means a 5.4 times bigger model and only 1.9 times more data. Models of the GPT-3 era (175B parameters) were sized in this spirit: big models, comparatively few tokens.

### Chinchilla (Hoffmann et al. 2022)

DeepMind trained over 400 models from 70M to over 16B parameters on 5B to 500B tokens and fitted the optimum three ways. All three put the exponents near one half: $N_{\text{opt}} \propto C^{a}$, $D_{\text{opt}} \propto C^{b}$ with $(a, b) = (0.50, 0.50)$, $(0.49, 0.51)$ and $(0.46, 0.54)$. Parameters and tokens should grow together; 10 times more compute means about 3.2 times more of each. Their Table 3 implies roughly 20 tokens per parameter (1B parameters for 20.2B tokens, 10B for 205.1B, 67B for 1.5T). They tested it by training **Chinchilla**, 70B parameters on 1.4T tokens (exactly 20 per parameter), with the same compute as the 280B Gopher. Chinchilla beat Gopher, GPT-3, Jurassic-1 and Megatron-Turing NLG, reaching 67.5 percent on MMLU, more than 7 points above Gopher.

Why Kaplan and Chinchilla disagree: Chinchilla attributes the difference mainly to Kaplan using a fixed learning rate schedule length for all runs, so intermediate checkpoints were judged before their cosine schedule had decayed, which understates what more data buys. Chinchilla matched the cosine cycle to each run's length and reports that overshooting the training length by more than 25 percent noticeably degrades performance.

The parametric fit is $L(N, D) = E + A/N^{0.34} + B/D^{0.28}$ with $E = 1.69$, $A = 406.4$, $B = 410.7$. $E$ is the irreducible loss (the entropy of text), and the other two terms are the penalties for a finite model and finite data. A detail that shows you have run the numbers: minimising this fit at Table 3's 67B budget (5.76e23 FLOPs) gives about 32B parameters and 3T tokens, roughly 90 tokens per parameter, so the "20 tokens per parameter" rule comes from the first two approaches, not from the parametric fit. When sources disagree like this, quote the rule with its approach.

### What changed: over trained small models

Chinchilla optimises training compute only. A model that will serve billions of requests pays inference cost per token forever, so it is worth spending extra training compute on a smaller model. Llama 3 8B was trained on about 15T tokens, about 1,875 tokens per parameter, 94 times the Chinchilla ratio. The Llama 3 paper states it trained its smaller models "much longer than is compute optimal", and that they beat compute optimal models at the same inference budget. Its own scaling law fit suggested a 402B model on 16.55T tokens for the 3.8e25 budget, close to the 405B it trained. Loss keeps falling well past 20 tokens per parameter, with diminishing returns per FLOP, which is what the $B/D^{0.28}$ data term of the Chinchilla fit predicts.

> **Interview trap:** Saying "Chinchilla says models should be trained on 20 tokens per parameter." It says that is the training compute optimum. Production models deliberately go far past it.

> **Watch:** [CME295 Lecture 4, scaling laws and the Chinchilla law](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=994s).

## Pretraining data

Data quality, not architecture, is where most of the gains between model generations come from; the Llama 3 paper attributes its improvements over Llama 2 primarily to data quality and diversity plus scale, with the architecture nearly unchanged.

### The pipeline

Llama 3's web pipeline (Llama 3 paper, Section 3.1) is a good template to recite:

1. **Extraction.** HTML to clean text, keeping structure such as code and math.
2. **Deduplication at three levels.** URL level (keep the most recent version of each page), document level (global MinHash to remove near duplicates), and line level (ccNet style, dropping lines that appear more than 6 times in each bucket of 30M documents, which removes boilerplate such as cookie banners and navigation).
3. **Heuristic filters.** Remove documents dominated by repeated n grams, adult sites found by "dirty word" counting, and documents whose token distribution is an outlier by KL divergence.
4. **Model based quality filters.** fastText and RoBERTa based classifiers trained to predict which pages a stronger model (Llama 2) judges high quality. FineWeb EDU, which build-nanogpt trains on, applies the same idea with a classifier for educational value.
5. **Domain classification and mixing.** Llama 3's final mix is roughly 50 percent general knowledge, 25 percent math and reasoning, 17 percent code and 8 percent multilingual tokens.

### Why deduplicate

Duplicates waste compute, push the model to memorise and regurgitate, overweight whatever happens to be copied often (licences, templates, spam), and increase the chance that test data leaks in. MinHash makes near duplicate detection cheap: represent each document by its set of word n grams (shingles), hash the set with many independent hash functions, and keep the minimum of each. The probability that two documents share a minimum equals their Jaccard similarity, so the share of matching minimums estimates it. Locality sensitive hashing over bands of the signature then finds candidate pairs without comparing every pair.

```python
import hashlib
import numpy as np

def shingles(text, n=5):
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

def minhash(sh, num_perm=128):
    sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for s in sh:
        for i in range(num_perm):   # one salted hash per "permutation"
            h = int.from_bytes(hashlib.blake2b(f"{i}:{s}".encode(), digest_size=8).digest(), "little")
            sig[i] = min(sig[i], h)
    return sig

a = "the quick brown fox jumps over the lazy dog while the cat sleeps on the warm mat by the door"
b = "the quick brown fox jumps over the lazy dog while the cat naps on the warm mat by the door"
A, B = shingles(a), shingles(b)
true_j = len(A & B) / len(A | B)
est_j = np.mean(minhash(A) == minhash(B))   # share of matching minimums estimates Jaccard
print(round(true_j, 3), round(est_j, 3))
```

It prints a true Jaccard similarity of 0.524 and a MinHash estimate of 0.531 from 128 hashes. One changed word in the middle of a sentence destroys five 5 word shingles, which is why the similarity is far below one.

### Mixtures, annealing and packing

The mixture is a set of sampling weights per source, tuned with small proxy models and scaling law experiments. High quality sources are upsampled (seen more than once), noisy web text is seen once or not at all. Llama 3 ends pretraining with an **annealing** phase: over the final 40M tokens the learning rate decays linearly to zero on upsampled high quality data, and the final model averages checkpoints from that phase. Documents are packed end to end into fixed length sequences with an end of text separator, so no compute is wasted on padding; Llama 3 additionally masks attention across document boundaries inside a sequence.

### Contamination

**Contamination** means benchmark test items, or close paraphrases, appear in the pretraining data, so a score measures recall instead of ability. The usual checks are n gram overlap between benchmark items and the corpus, comparing accuracy on items flagged as seen against unseen items, and testing on benchmarks released after the data cutoff. The Llama 3 paper includes a contamination analysis for its benchmarks. Mitigations: deduplicate the corpus against known test sets, keep private held out sets, and treat any public benchmark older than the data cutoff with suspicion.

> **Watch:** [Let's reproduce GPT-2, datasets used in GPT-2, GPT-3 and FineWeb EDU](https://www.youtube.com/watch?v=l8pRSuU81PU&t=11421s). [CME295 Lecture 4, pretraining](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=439s).

## Training at scale

### Optimiser settings

The standard recipe is AdamW (Adam with weight decay decoupled from the gradient update, Loshchilov and Hutter). build-nanogpt copies GPT-3's settings: $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, weight decay 0.1 applied only to two dimensional tensors (matmul weights and embeddings, not biases or norm gains), and gradient norm clipping at 1.0. The original transformer used $\beta_2 = 0.98$ and $\epsilon = 10^{-9}$ (Vaswani et al. 2017). A lower $\beta_2$ than the textbook 0.999 makes the second moment adapt faster when gradient scale shifts, which helps stability in long runs. Clipping caps the update size when one batch produces a huge gradient.

### Warmup and cosine decay

```text
lr(step) = max_lr * (step + 1) / warmup                      if step < warmup
         = min_lr + 0.5 * (1 + cos(pi * progress)) * (max_lr - min_lr)   afterwards
```

build-nanogpt uses max 6e-4, min 6e-5 (10 percent of max), 715 warmup steps and 19,073 total steps at 524,288 tokens per step, which is about 375M warmup tokens and 10B tokens in total. Llama 3 405B used a peak of 8e-5, 8,000 warmup steps and cosine decay to 8e-7 over 1.2M steps (Llama 3 paper). Warmup exists because at initialisation Adam's moment estimates are poor and the loss surface is sharp, so full sized steps can blow the model up; it matters most for post norm models (Xiong et al. 2020). Cosine decay anneals the step size so the model settles into a low loss region, and Chinchilla showed the schedule length must match the run. Since 2024 many labs also use warmup, stable, decay schedules (a long constant phase then a short decay), which let you branch a checkpoint and decay it at any point instead of committing to a length in advance.

Batch size is set in tokens: 0.5M for GPT-2 small (from the GPT-3 paper, via build-nanogpt), and Llama 3 405B ramped from 4M to 8M to 16M tokens for stability early and efficiency later. Gradient accumulation reaches large batches on small hardware; the loss must be divided by the number of micro steps, or the gradient is a sum instead of a mean (a bug Karpathy walks through).

### bf16 and mixed precision

| Format | Exponent bits | Mantissa bits | Largest value | Spacing at 1.0 |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 23 | about 3.4e38 | 1.2e-7 |
| fp16 | 5 | 10 | 65,504 | 0.00098 |
| bf16 | 8 | 7 | about 3.39e38 | 0.0078 |

fp16 has precision but little range: small gradients underflow and large activations overflow, so fp16 training needs dynamic loss scaling. bf16 keeps fp32's exponent range, so no loss scaling is needed, and gives up precision instead. That is why the weights are kept in fp32 for the update: in bf16, $1.0 + 0.001$ rounds back to $1.0$ because the spacing near 1 is 0.0078, so small updates would vanish. Chinchilla trained with bf16 forward and backward passes and an fp32 copy of the weights in the optimiser; build-nanogpt wraps the forward pass in `torch.autocast(dtype=torch.bfloat16)` and enables TF32 for remaining fp32 matmuls. FP8 training on Hopper and later GPUs pushes this further in 2025 to 2026 frontier runs.

### Loss spikes

A loss spike is a sudden jump in training loss that either recovers or diverges. Common causes: a learning rate too high for the current curvature, a bad or repetitive data batch, attention logits growing until the softmax saturates, and Adam's second moment underestimating a suddenly large gradient. The responses, roughly in order: gradient clipping, a lower peak learning rate or longer warmup, smaller batches early, rewinding to a checkpoint before the spike and skipping the offending batches, and architectural fixes such as normalising queries and keys (QK norm) or adding a small auxiliary loss on the log of the softmax normaliser (z loss). Llama 3 reports that its recipe with an early batch size ramp saw few loss spikes and needed no interventions.

### Parallelism in one paragraph

Data parallelism replicates the model and splits the batch; ZeRO and FSDP shard optimiser state, gradients and weights across those replicas so no GPU holds all 16 bytes per parameter. Tensor parallelism splits individual matmuls across GPUs inside a node. Pipeline parallelism places groups of layers on different GPUs and streams micro batches through them. Expert parallelism places MoE experts on different GPUs, and context parallelism splits long sequences. Llama 3 405B combined tensor 8, pipeline 16, context parallel up to 16 and data parallel up to 128 (Llama 3 paper, Table 4).

> **Watch:** [Let's reproduce GPT-2, AdamW and gradient clipping](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8095s), [warmup and cosine decay](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8466s), [float16, gradient scalers, bfloat16](https://www.youtube.com/watch?v=l8pRSuU81PU&t=5978s). [CME295 Lecture 4, mixed precision training](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=3360s).

## Prefill, decode and the KV cache

Generation has two phases (CME295 cheatsheet). **Prefill** runs the whole prompt through the model in one parallel forward pass and writes every layer's keys and values into the cache. **Decode** then produces one token per forward pass, appending that token's keys and values to the cache. Time to first token is dominated by prefill; time per output token is dominated by decode.

### Why the KV cache is valid and what it saves

Under a causal mask, the key and value of token $j$ depend only on tokens $1..j$, so they never change when later tokens arrive. Without a cache, step $t$ would recompute keys and values for all $t$ previous tokens; with it, step $t$ computes one new query, key and value per layer and attends over the stored ones. Past queries are never needed, because only the newest position's output is used. Per generated token, compute drops from re-running the whole prefix to about $2N$ FLOPs plus an attention read over the cache. An encoder with bidirectional attention cannot reuse its cache this way, because a new token changes every earlier token's representation.

### The memory formula

$$
\text{KV bytes} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times \text{bytes per value} \times T \times B
$$

The 2 is one tensor for keys and one for values; $L$ layers, $n_{\text{kv}}$ key value heads, sequence length $T$, batch $B$.

> **Worked example:** Llama 3 8B: 32 layers, 8 KV heads, head dimension $4096/32 = 128$ (Llama 3 paper, Table 3; Mistral 7B's config has the same shape). In bf16 that is $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes, 128 KiB per token. An 8,192 token sequence needs 1 GiB; the full 128K context needs 16 GiB, as much as the 16 GB of bf16 weights; a batch of 32 sequences at 8K needs 32 GiB. With full MHA (32 KV heads) every figure would be four times larger, 512 KiB per token. Llama 3 70B (80 layers, 8 KV heads, head dimension 128) needs 320 KiB per token and 40 GiB for one 128K sequence. One caveat on Mistral 7B: its config sets `sliding_window` to 4096, and with a rolling buffer its cache stops growing at 4,096 tokens per sequence.

The cache, not the weights, is usually what caps batch size and context length on a serving GPU. Ways to shrink it: GQA or MQA (fewer KV heads), MLA (cache a compressed latent), quantising the cache to int8 or fp8, sliding window attention for some layers, and evicting or offloading old entries. Prompt caching reuses the cache for a shared prefix such as a long system prompt across requests.

### Why decode is memory bound

At batch size 1, each decode step must read every weight from HBM once and does about 2 FLOPs per weight (one multiply, one add), so its arithmetic intensity is about 1 FLOP per byte in bf16. An A100 offers about 312 TFLOPS of bf16 compute (the figure nanoGPT uses) and 1.5 to 2.0 TB/s of HBM bandwidth (FlashAttention paper), so it needs about 156 FLOPs per byte to be compute bound. Decode is two orders of magnitude below that, so the GPU sits waiting on memory.

> **Worked example:** an 8B model in bf16 is 16 GB of weights. At 2.0 TB/s, reading them once takes 8 ms, so batch 1 decode tops out near 125 tokens per second no matter how many FLOPs the GPU has. Batching 32 requests reads the weights once for 32 tokens, which is why throughput rises almost linearly with batch size until compute or KV cache memory runs out. Quantising the weights to 4 bits cuts the read to about 4 GB and raises the ceiling about four times.

Prefill processes hundreds or thousands of tokens per weight read, so it is compute bound. This asymmetry is behind most serving designs: batching decode, splitting long prefills into chunks so they do not stall running decodes, and in some systems running prefill and decode on separate GPU pools.

> **Interview trap:** "The KV cache saves memory." It spends memory to save compute. It grows linearly with sequence length and batch size and is often the largest thing on the GPU.

> **Watch:** [CME295 Lecture 3, inference optimisations with KV cache](https://www.youtube.com/watch?v=Q5baLehv5So&t=5100s) and [PagedAttention and MLA](https://www.youtube.com/watch?v=Q5baLehv5So&t=5589s).

## FlashAttention and serving systems

### FlashAttention and IO awareness

Standard attention writes the $T \times T$ score matrix to HBM, reads it back for the softmax, writes the probabilities, and reads them again to multiply by $V$. Those transfers, not the FLOPs, dominate run time: an A100 has 40 to 80 GB of HBM at 1.5 to 2.0 TB/s but only 192 KB of SRAM per streaming multiprocessor (108 of them) at about 19 TB/s (Dao et al. 2022). **FlashAttention** tiles $Q$, $K$ and $V$ into blocks that fit in SRAM, computes each block's scores and softmax contribution on chip, and never materialises the full matrix. The softmax over a row is assembled block by block with the online softmax trick: keep a running maximum $m$ and running normaliser $\ell$, and rescale the partial output whenever the maximum grows.

```python
import numpy as np

rng = np.random.default_rng(0)
T, d, block = 1024, 64, 128
q = rng.standard_normal(d)
K = rng.standard_normal((T, d))
V = rng.standard_normal((T, d))

# reference: materialise every score, then softmax
s = K @ q / np.sqrt(d)
w = np.exp(s - s.max()); ref = (w / w.sum()) @ V

# FlashAttention style: one pass over key blocks, keeping a running max m,
# a running normaliser l and an unnormalised running output acc
m, l, acc = -np.inf, 0.0, np.zeros(d)
for start in range(0, T, block):
    sb = K[start:start + block] @ q / np.sqrt(d)
    m_new = max(m, sb.max())
    scale = np.exp(m - m_new)              # rescale what we accumulated so far
    pb = np.exp(sb - m_new)
    l = l * scale + pb.sum()
    acc = acc * scale + pb @ V[start:start + block]
    m = m_new
print(np.allclose(acc / l, ref))           # exact, not an approximation
```

It prints `True`. The paper proves standard attention needs $\Theta(Td + T^2)$ HBM accesses and FlashAttention $\Theta(T^2 d^2 / M)$ for SRAM size $M$, with only $O(T)$ extra memory. The backward pass recomputes the blocks from the saved statistics instead of storing the $T \times T$ matrix, so FLOPs actually go up (75.2 against 66.6 GFLOPs in the paper's GPT-2 medium benchmark) while wall clock time goes down. Reported speedups: 15 percent end to end on BERT large, 3 times on GPT-2 at length 1K, 2.4 times on Long Range Arena. It is exact attention, so it changes speed and memory, never outputs (up to floating point order). In build-nanogpt, switching to `F.scaled_dot_product_attention` with `is_causal=True` dropped the step time from about 130 ms to 96 ms.

### Continuous batching

Static batching starts a batch together and waits for the longest sequence to finish, so slots that finish early sit idle and new requests queue. **Continuous batching** (iteration level scheduling, introduced by the Orca system) re-forms the batch at every decode step: finished sequences leave and waiting requests join immediately. Throughput rises sharply under variable output lengths, and it is standard in vLLM, TensorRT-LLM, SGLang and TGI.

### PagedAttention

Continuous batching needs memory that can grow and shrink per request. Reserving a contiguous KV region for each request's maximum length wastes most of it through internal fragmentation (reserved but unused) and external fragmentation (gaps between regions). **PagedAttention** (Kwon et al. 2023, vLLM) borrows virtual memory from operating systems: the cache is split into fixed size blocks of a few tokens, a block table maps each sequence's logical blocks to physical blocks anywhere in memory, and the attention kernel follows the table. Waste falls to near zero, and blocks can be shared copy on write between sequences with a common prefix or between parallel samples of one prompt. The paper reports 2 to 4 times the throughput of FasterTransformer and Orca at the same latency.

> **Watch:** [CME295 Lecture 4, Flash Attention](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=2306s). [Let's reproduce GPT-2, flash attention](https://www.youtube.com/watch?v=l8pRSuU81PU&t=7218s).

## Quantisation

Quantisation stores numbers in fewer bits. For LLM inference the main target is the weights, because decode time is set by how many bytes are read per token: an 8B model is about 16 GB in bf16, 8 GB in int8 and 4 GB in int4 (plus scales). Symmetric absmax quantisation maps a tensor to integers with one scale, $q = \operatorname{round}(x/s)$ with $s = \max|x| / (2^{b-1}-1)$, and dequantises as $qs$.

```python
import numpy as np

rng = np.random.default_rng(0)
w = rng.standard_normal(4096).astype(np.float32) * 0.02

def absmax_quant(x, bits):
    qmax = 2 ** (bits - 1) - 1                 # 127 for int8, 7 for int4
    scale = np.abs(x).max() / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return q * scale                           # dequantised values

for bits in (8, 4):
    err = np.abs(absmax_quant(w, bits) - w).mean()
    print(f"int{bits} mean abs error {err:.2e}")

# one outlier stretches the scale and wipes out resolution for everything else
w_out = w.copy(); w_out[0] = 1.0
err = np.abs(absmax_quant(w_out, 8)[1:] - w_out[1:]).mean()
print(f"int8 with one outlier {err:.2e}")
err4 = np.abs(absmax_quant(w_out, 4)[1:] - w_out[1:]).mean()
print(f"int4 with one outlier {err4:.2e}")

# group wise scales (one scale per 128 values) contain the damage
g = w_out.reshape(-1, 128)
deq = np.vstack([absmax_quant(row, 4) for row in g]).ravel()
print(f"int4 group size 128 with outlier {np.abs(deq[1:] - w_out[1:]).mean():.2e}")
```

The output shows the central problem. Mean error is 1.5e-4 at int8 and 2.8e-3 at int4; a single outlier raises int8 error 13 times (2.0e-3) and int4 error to 1.6e-2, while one scale per group of 128 values brings int4 back to 2.4e-3. LLM activations have a few feature dimensions with very large magnitudes, which is why per channel and per group scales, and outlier aware methods, exist.

- **Weight only int8 and int4 (W8A16, W4A16)** keep activations in bf16 and dequantise weights inside the matmul kernel. This is the common serving setup because decode is memory bound.
- **GPTQ** (Frantar et al. 2022) quantises one layer at a time using approximate second order (Hessian) information from a small calibration set: after rounding each weight it updates the not yet quantised weights to compensate for the error. It reaches 3 or 4 bits with negligible accuracy loss, quantises a 175B model in about four GPU hours, and reports end to end speedups of about 3.25 times on A100 and 4.5 times on A6000 over fp16.
- **AWQ** (Lin et al. 2023) observes that protecting about 1 percent of salient weights greatly reduces error, and that saliency is set by activation magnitude, not weight magnitude. Instead of keeping those channels in high precision, it scales them up before quantisation (and the matching activations down), an exact transformation that avoids mixed precision kernels. It needs no backpropagation or reconstruction, so it overfits the calibration set less.
- **W8A8 and FP8** quantise activations too, which speeds up compute bound prefill; FP8 is native on Hopper and later GPUs.

> **Watch:** [CME295 Lecture 4, quantisation](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=3157s).

## Speculative decoding

Decode leaves the GPU's compute mostly idle, so verifying several tokens costs about the same as generating one. **Speculative decoding** uses a cheap draft model $q$ to propose $\gamma$ tokens, then runs the target model $p$ once over all of them in parallel and keeps a prefix (Leviathan et al. 2022; Chen et al. 2023 independently). The acceptance rule makes the output distribution exactly the target's:

- Accept draft token $x$ with probability $\min\!\left(1, \frac{p(x)}{q(x)}\right)$.
- On the first rejection, sample a replacement from the residual $p'(x) = \operatorname{norm}(\max(0, p(x) - q(x)))$ and discard the rest of the draft.
- If all $\gamma$ are accepted, sample one bonus token from $p$ at the next position, which the target pass already computed.

```python
import numpy as np

rng = np.random.default_rng(0)
p = np.array([0.50, 0.30, 0.15, 0.05])   # target model distribution
q = np.array([0.25, 0.25, 0.25, 0.25])   # draft model distribution

def speculative_sample(p, q, rng):
    x = rng.choice(len(q), p=q)               # the draft proposes x
    if rng.random() < min(1.0, p[x] / q[x]):  # accept with probability min(1, p/q)
        return x, True
    residual = np.maximum(p - q, 0.0)         # on rejection, resample from norm(max(0, p - q))
    return rng.choice(len(p), p=residual / residual.sum()), False

draws = [speculative_sample(p, q, rng) for _ in range(200_000)]
xs = np.array([d[0] for d in draws]); acc = np.mean([d[1] for d in draws])
print("empirical", np.bincount(xs, minlength=4) / len(xs))
print("target   ", p)
print("acceptance rate", round(acc, 3), "theory", np.minimum(p, q).sum())
```

The empirical distribution matches the target to three decimals, and the acceptance rate is $\sum_x \min(p(x), q(x)) = 0.70$, which is one minus the total variation distance between the two distributions. With per token acceptance rate $\alpha$, the expected tokens per target pass is $\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$, and with draft cost ratio $c$ the expected speedup is $\frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}$ (Leviathan et al. 2022, Theorem 3.8).

> **Worked example:** $\alpha = 0.8$, $\gamma = 4$, $c = 0.05$: $3.36$ tokens per target pass and a $2.8$ times speedup. At $\alpha = 0.6$ the speedup drops to $1.9$. The paper reports 2 to 3 times on T5-XXL with identical outputs.

It works best when output is predictable (code, extraction, low temperature) and at small batch sizes. At large batch sizes decode becomes compute bound, the spare FLOPs disappear and the draft becomes overhead, which is the standard follow up. Variants drop the separate draft model: Medusa adds extra prediction heads, EAGLE drafts from the target's own hidden states, and n gram or prompt lookup drafting copies spans from the context.

## Decoding strategies

The model gives a distribution over the next token; the decoding strategy picks from it.

- **Greedy** takes the argmax. Deterministic and cheap, but locally optimal choices lead to repetition loops and bland text.
- **Beam search** keeps the $k$ best partial sequences by total log probability. Good for short, constrained outputs such as translation; for open ended text it favours generic, repetitive sequences, because the highest probability text is not the most humanlike.
- **Temperature** divides logits by $T$ before the softmax: $p_i \propto \exp(z_i / T)$. $T < 1$ sharpens, $T > 1$ flattens, and $T \to 0$ is greedy.
- **Top k** keeps the $k$ most likely tokens and renormalises. A fixed $k$ is too many when the model is confident and too few when it is not.
- **Top p (nucleus)** (Holtzman et al. 2019) keeps the smallest set of tokens whose cumulative probability reaches $p$, so the candidate set adapts to the model's confidence.
- **Min p** (Nguyen et al. 2024) keeps tokens with probability at least $p_{\min}$ times the top token's probability, which scales the cutoff with confidence and stays sensible at high temperatures.
- **Repetition penalties** push down tokens already generated: the CTRL style penalty divides their positive logits (and multiplies negative ones) by a factor above 1, while frequency and presence penalties subtract an amount proportional to the token's count or to whether it appeared at all.

```python
import numpy as np

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def filter_logits(logits, temperature=1.0, top_k=None, top_p=None, min_p=None):
    z = logits / temperature
    p = softmax(z)
    keep = np.ones_like(p, dtype=bool)
    if top_k is not None:                       # keep the k most likely tokens
        keep &= p >= np.sort(p)[-top_k]
    if top_p is not None:                       # smallest set whose mass reaches top_p
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        n = np.searchsorted(cum, top_p) + 1
        mask = np.zeros_like(keep); mask[order[:n]] = True
        keep &= mask
    if min_p is not None:                       # keep tokens with p >= min_p * max(p)
        keep &= p >= min_p * p.max()
    q = np.where(keep, p, 0.0)
    return q / q.sum()

logits = np.array([4.0, 3.5, 2.0, 1.0, 0.5, -1.0])
np.set_printoptions(precision=3, suppress=True)
print("T=1.0      ", filter_logits(logits))
print("T=0.5      ", filter_logits(logits, temperature=0.5))
print("T=2.0      ", filter_logits(logits, temperature=2.0))
print("top_k=2    ", filter_logits(logits, top_k=2))
print("top_p=0.9  ", filter_logits(logits, top_p=0.9))
print("min_p=0.1  ", filter_logits(logits, min_p=0.1))
```

At $T = 1$ the top token has 0.547; at $T = 0.5$ it rises to 0.72, at $T = 2$ it falls to 0.381. Top p 0.9 keeps three tokens (cumulative mass 0.547, 0.879, 0.953), and min p 0.1 keeps the same three, because the cutoff is $0.1 \times 0.547$. Temperature 0 is greedy, but greedy is not always reproducible in production: batched floating point arithmetic can change the argmax when two logits are nearly tied. Structured or guided decoding masks invalid tokens at each step so the output must match a grammar or JSON schema.

> **Watch:** [CME295 Lecture 3, greedy decoding and beam search](https://www.youtube.com/watch?v=Q5baLehv5So&t=2314s), [sampling methods](https://www.youtube.com/watch?v=Q5baLehv5So&t=2796s), [temperature](https://www.youtube.com/watch?v=Q5baLehv5So&t=3144s).

## Long context

Context length is the maximum number of tokens the model can attend over. Three things limit it: attention compute grows with $T^2$ in prefill, the KV cache grows linearly with $T$ (16 GiB per 128K sequence for Llama 3 8B), and the model only handles positions it was trained on.

Techniques, grouped by the limit they attack:

- **Positions.** Raise the RoPE base, interpolate positions, or use NTK aware or YaRN scaling, then continue pretraining on long documents. Llama 3 went from 8K to 128K in six stages with about 800B tokens (Llama 3 paper).
- **Compute and memory.** FlashAttention makes exact attention fit in linear memory. Sparse and local patterns restrict which pairs interact: sliding window attention (Mistral 7B's config uses a 4,096 token window), interleaving local and global layers, or a few always visible "sink" tokens at the start. Context parallelism splits a long sequence across GPUs (Llama 3 used it for the 128K stage).
- **Cache size.** GQA, MLA, KV quantisation and eviction.
- **Alternatives to attention.** State space models and linear attention variants carry a fixed size state instead of a growing cache, and hybrid models mix them with a few attention layers.

Two cautions. First, a model that accepts 128K tokens may not use them well: Liu et al. 2023 ("Lost in the Middle") found accuracy is highest when the relevant fact is at the start or end of the context and drops in the middle, and the CME295 cheatsheet calls the general degradation as the window fills "context rot". Needle in a haystack tests only check retrieval of one planted fact, so multi fact and reasoning benchmarks are needed. Second, long prefills are expensive: a 100K token prompt costs far more than its decode, which is why prompt caching of shared prefixes matters in production.

> **Watch:** [CME295 Lecture 2, sparse attention](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=3039s).

## Mixture of experts

A **mixture of experts** layer replaces the single FFN with $E$ expert FFNs and a router that sends each token to $k$ of them. The output is the gate weighted sum $\hat{y} = \sum_{i \in \text{top}k} G(x)_i E_i(x)$ (CME295 cheatsheet). Attention, embeddings and norms stay dense and shared.

### Total versus active parameters

Every expert must sit in memory, but each token only runs through $k$ of them, so compute per token follows active parameters and memory follows total parameters.

> **Worked example:** Mixtral 8x7B's config: 32 layers, width 4,096, 8 experts each a SwiGLU FFN with hidden size 14,336, top 2 routing, GQA with 8 KV heads, vocabulary 32,000. Each expert is $3 \times 4096 \times 14336 = 176.2$M parameters per layer. Total: 46.7B parameters. Active per token: attention, two experts per layer, router, norms and embeddings, 12.9B. It is not 8 times 7B (56B) because attention and embeddings are shared, and it computes like a 13B dense model while needing memory for 47B.

### Routing and load balancing

The router is a linear layer giving logits over experts; softmax, take the top $k$, renormalise the selected gates. Left alone, routing collapses: a few experts get most tokens, get better, and attract even more, while the rest starve. With expert parallelism an overloaded expert also becomes a straggler, and experts with a fixed capacity drop overflow tokens. The fix is an auxiliary load balancing loss, in the Switch Transformer form $\alpha E \sum_i f_i P_i$, where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ the mean router probability for it. It equals $\alpha$ under uniform routing and grows when both concentrate on the same experts. Mixtral's Hugging Face config sets `router_aux_loss_coef` to 0.02.

```python
import numpy as np

rng = np.random.default_rng(0)
n_tokens, d, n_experts, k = 512, 32, 8, 2
x = rng.standard_normal((n_tokens, d))
W_router = rng.standard_normal((d, n_experts)) * 0.1

logits = x @ W_router
probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
topk = np.argsort(-probs, axis=1)[:, :k]                 # each token picks its 2 best experts
gates = np.take_along_axis(probs, topk, 1)
gates /= gates.sum(1, keepdims=True)                     # renormalise the 2 gate weights

# Switch Transformer style auxiliary loss: n_experts * sum_i f_i * P_i
f = np.bincount(topk[:, 0], minlength=n_experts) / n_tokens   # share of tokens whose first choice is i
P = probs.mean(0)                                             # mean router probability for i
aux = n_experts * np.sum(f * P)
print("tokens per expert (first choice):", np.bincount(topk[:, 0], minlength=n_experts))
print("aux loss", round(aux, 3), "(1.0 when perfectly balanced)")
collapsed_f = np.eye(n_experts)[0]; collapsed_P = np.eye(n_experts)[0]
print("aux loss if every token goes to expert 0:", n_experts * np.sum(collapsed_f * collapsed_P))
```

With a random router the loss is 1.007 against 8.0 for full collapse. Other choices interviewers raise: fine grained experts plus always on shared experts (the DeepSeek MoE design), auxiliary loss free balancing by adjusting per expert router biases (DeepSeek-V3, late 2024), expert capacity and token dropping, and the communication cost of the all to all exchange that sends tokens to the GPUs holding their experts. MoE wins when you are compute bound in training and can afford memory; it is awkward on a single small GPU, where all experts still need to fit.

> **Watch:** [CME295 Lecture 3, mixture of experts](https://www.youtube.com/watch?v=Q5baLehv5So&t=457s) and [MoE in LLMs](https://www.youtube.com/watch?v=Q5baLehv5So&t=947s).

## In context learning and emergence

**In context learning** is a pretrained model doing a new task from instructions or examples placed in the prompt, with no weight update. GPT-3 made it famous with zero shot, one shot and few shot prompting. The model is still only predicting the next token; the prompt shapes which continuation is likely.

Three explanations worth knowing (CS229 Lecture 14 covers transformers and in context learning; Tengyu Ma, who teaches it with Chris Ré, is one of the authors of the Bayesian account below):

- **Implicit Bayesian inference** (Xie et al. 2021). Pretraining documents each have a latent "concept" (a topic, format or task). The examples in a prompt are evidence about which concept generated this document, and the model's prediction averages over concepts weighted by that evidence.
- **Induction heads** (Olsson et al. 2022). Mechanistic work on transformers finds pairs of attention heads that implement "find the earlier occurrence of the current token and copy what came after it", which supports pattern completion from examples; their appearance during training coincides with a jump in in context learning ability.
- **Attention as an algorithm** (Garg et al. 2022, von Oswald et al. 2022). Constructions and experiments on linear regression tasks show a transformer can implement steps resembling gradient descent or ridge regression in its forward pass on the in context examples.

Practical facts: few shot performance is sensitive to example order, format and label balance, and Min et al. (2022) found that randomising the labels in demonstrations often hurts less than you would expect, so models lean on the format and label space of examples as much as on their correctness.

**Emergent abilities** (Wei et al. 2022) are abilities that are near chance for smaller models and jump sharply above some scale. Schaeffer, Miranda and Koyejo (2023, "Are Emergent Abilities of Large Language Models a Mirage?") argue the jump is often created by the metric: all or nothing metrics such as exact match turn a smooth improvement in per token accuracy into a sudden step, while continuous metrics on the same outputs improve smoothly. The defensible interview position: loss scales smoothly and predictably, some downstream metrics look discontinuous, and part of that is measurement.

> **Watch:** [CS229 Lecture 14, transformers and in context learning](https://www.youtube.com/watch?v=pwQ0l4hFCVI). [CME295 Lecture 3, in context learning](https://www.youtube.com/watch?v=Q5baLehv5So&t=4449s).

## Why models hallucinate

A **hallucination** is a fluent, confident output that is false or unsupported. At the model level the causes are:

1. **The objective rewards plausibility, not truth.** Next token prediction maximises likelihood of text that looks like the training data. A false sentence in the style of a true one gets high probability.
2. **Statistical pressure on rare facts.** Kalai, Nachum, Vempala and Zhang (2025, "Why Language Models Hallucinate") frame generation errors through binary classification: if valid and invalid statements cannot be told apart from the training data, pretraining pressure will produce some invalid ones. It follows that facts seen once or a few times in the corpus (an obscure person's birthday) are where errors concentrate, because nothing in the data separates the true value from a plausible one.
3. **Evaluation rewards guessing.** The same paper argues hallucinations persist after post training because most benchmarks grade answers right or wrong with no credit for abstaining, so a model that guesses scores higher than one that says it does not know. Their proposed fix is to change the scoring of existing benchmarks to stop penalising uncertainty.
4. **Exposure bias and snowballing.** At inference the model conditions on its own earlier tokens; one wrong token becomes context it then stays consistent with.
5. **Decoding.** High temperature or wide top p samples from the low probability tail, where errors live.
6. **Knowledge limits.** Facts after the data cutoff, facts lost to compression of a huge corpus into fixed weights, and contradictions in the training data.

Mitigations at this layer: retrieval and citation (covered in the RAG module), lower temperature for factual tasks, calibrated abstention trained in post training, and checking consistency across several samples (if the model gives different answers each time, it probably does not know).

> **Interview trap:** "Hallucination is a bug that more data will fix." It follows from the objective and the incentives. More data helps on common facts and does little for facts seen once.

## What interviewers probe

These chains come from the question banks in the sources (KalyanKS, Let's Data Science, techinterview.org) and a Netflix ML engineer screen written up on PracHub. Expect each first question to be followed by the next ones in the chain.

- **Attention.** Write scaled dot product attention. Why divide by the square root of $d_k$ and not $d_k$? What breaks if you mask after the softmax? What is the complexity, and in which regime does attention dominate the FFN? Write softmax so it does not overflow.
- **KV cache.** What exactly is cached and why is it valid? Why are past queries not cached? Size the cache for an 8B model at 128K context. Why is decode memory bound and prefill compute bound? Name four ways to shrink the cache and what each costs.
- **Heads.** Why multiple heads? Does changing the head count change the parameter count? How do MQA and GQA trade quality for memory, and does GQA reduce FLOPs?
- **Positions.** Why do transformers need positional information when RNNs do not? Derive why RoPE depends only on relative position. Why is RoPE applied to queries and keys but not values? How do you extend an 8K model to 128K?
- **Tokenisation.** Walk through BPE training and encoding. Why byte level? What does a bigger vocabulary buy and cost? Why does the model fail at counting letters?
- **Architecture.** Pre norm versus post norm and why warmup matters for one of them. Why RMSNorm? Why SwiGLU and why is its hidden size about $\tfrac{8}{3}d$? Why are decoder only models dominant?
- **Compute.** Derive $6ND$. Count the parameters of a model from its config. Estimate training time from FLOPs and MFU. How much memory does training an 8B model with Adam need?
- **Scaling.** What did Chinchilla change relative to Kaplan and why did the two disagree? Why are production models trained far past 20 tokens per parameter?
- **Serving.** Explain continuous batching and PagedAttention. Why is FlashAttention faster if it does more FLOPs? Does speculative decoding change the output distribution, and when does it stop helping? GPTQ versus AWQ.
- **MoE.** Why do total and active parameters differ? What happens without a load balancing loss? When is MoE the wrong choice?
- **Behaviour.** What is in context learning and how might it work? Are emergent abilities real? Why do models hallucinate, and why does post training not remove it?

## Sources

- [CME295 VIP cheatsheet: Transformers and Large Language Models (Afshine and Shervine Amidi)](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models)
- [nanoGPT model.py (Karpathy)](https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py)
- [build-nanogpt train_gpt2.py (Karpathy)](https://raw.githubusercontent.com/karpathy/build-nanogpt/master/train_gpt2.py)
- [minbpe README (Karpathy)](https://raw.githubusercontent.com/karpathy/minbpe/master/README.md)
- [Vaswani et al. 2017, Attention Is All You Need](https://arxiv.org/html/1706.03762v7)
- [Kaplan et al. 2020, Scaling Laws for Neural Language Models](https://ar5iv.labs.arxiv.org/html/2001.08361)
- [Hoffmann et al. 2022, Training Compute Optimal Large Language Models (Chinchilla)](https://ar5iv.labs.arxiv.org/html/2203.15556)
- [Su et al. 2021, RoFormer: Rotary Position Embedding](https://ar5iv.labs.arxiv.org/html/2104.09864)
- [Press, Smith and Lewis 2021, ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409)
- [Ainslie et al. 2023, GQA](https://arxiv.org/abs/2305.13245)
- [Dao et al. 2022, FlashAttention](https://ar5iv.labs.arxiv.org/html/2205.14135)
- [Kwon et al. 2023, PagedAttention and vLLM](https://arxiv.org/abs/2309.06180)
- [Leviathan, Kalman and Matias 2022, Fast Inference from Transformers via Speculative Decoding](https://ar5iv.labs.arxiv.org/html/2211.17192)
- [Llama Team, Meta 2024, The Llama 3 Herd of Models](https://arxiv.org/html/2407.21783v3)
- [Frantar et al. 2022, GPTQ](https://arxiv.org/abs/2210.17323)
- [Lin et al. 2023, AWQ](https://arxiv.org/abs/2306.00978)
- [Schaeffer, Miranda and Koyejo 2023, Are Emergent Abilities of LLMs a Mirage?](https://arxiv.org/abs/2304.15004)
- [Kalai et al. 2025, Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664)
- [Hugging Face Transformers docs, tokenisation algorithms](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Mistral 7B v0.1 config.json](https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json)
- [Mixtral 8x7B v0.1 config.json](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/resolve/main/config.json)
- [LLM Interview Questions and Answers Hub (KalyanKS NLP)](https://github.com/KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub)
- [Let's Data Science, 50 LLM and AI engineer interview questions for 2026](https://letsdatascience.com/blog/50-llm-and-ai-engineer-interview-questions-for-2026)
- [techinterview.org, What LLM interviewers actually ask about transformers](https://www.techinterview.org/post/3233476821/what-llm-interviewers-ask-about-transformers/)
- [PracHub, Netflix ML engineer screen on attention and the KV cache](https://prachub.com/interview-questions/transformer-attention-computing-q-k-and-v-and-how-a-kv-cache-speeds-up-decoding)
