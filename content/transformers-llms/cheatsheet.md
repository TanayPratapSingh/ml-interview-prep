# Transformers and LLMs cheatsheet

The formulas, numbers and one line answers for tokenisation, attention, positions, the block, compute, scaling, data, training and inference. Every number names its source; the worked figures were computed from published configs.

## Core formulas

| Thing | Formula |
| --- | --- |
| Attention | $\operatorname{softmax}(QK^\top/\sqrt{d_k} + M)\,V$, mask $M$ is 0 or $-\infty$ |
| Why the square root | $\operatorname{Var}(q\cdot k) = d_k$ for unit variance entries (Vaswani et al. 2017), so divide by $\sqrt{d_k}$ to get variance 1 |
| Next token loss | $-\frac{1}{T}\sum_t \log p(x_{t+1}\mid x_{\le t})$; perplexity $= e^{\text{loss}}$; loss at init $\approx \ln V$ (10.82 for GPT-2) |
| Params per block | about $12d^2$ (MHA + 4x GELU FFN); stack $\approx 12Ld^2$ (Kaplan et al. 2020) |
| SwiGLU FFN | $W_{\text{down}}(\operatorname{SiLU}(W_g x)\odot W_u x)$, hidden $\approx \tfrac{8}{3}d$ to match params; Llama 3 8B uses $3.5d$ |
| RMSNorm | $\gamma\odot x/\sqrt{\operatorname{mean}(x^2)+\epsilon}$, no centring, no bias |
| Training FLOPs | $C \approx 6ND$; inference $\approx 2N$ per token |
| KV cache bytes | $2 \cdot L \cdot n_{kv} \cdot d_{\text{head}} \cdot \text{bytes} \cdot T \cdot B$ |
| RoPE | rotate pair $i$ by $m\theta_i$, $\theta_i = b^{-2i/d}$, so $q_m^\top k_n$ depends on $n-m$ only (Su et al. 2021) |
| ALiBi | add $-m_h(i-j)$ to scores, slope per head (Press et al. 2021) |
| Chinchilla fit | $L = 1.69 + 406.4/N^{0.34} + 410.7/D^{0.28}$ (Hoffmann et al. 2022) |
| Speculative accept | accept with $\min(1, p/q)$, else resample from $\operatorname{norm}(\max(0,p-q))$; output is exactly $p$ |
| Expected tokens per target pass | $(1-\alpha^{\gamma+1})/(1-\alpha)$ (Leviathan et al. 2022) |
| MoE balance loss | $\alpha E\sum_i f_i P_i$, equals $\alpha$ when uniform (Switch Transformer form) |

## Numbers worth memorising

- GPT-2 small: 12 layers, width 768, 12 heads, vocabulary 50,257 (256 bytes + 50,000 merges + 1 special), context 1,024, 124,439,808 parameters, tied embeddings (nanoGPT).
- Original transformer base: $d = 512$, 8 heads, $d_k = 64$, FFN 2,048, 6 layers, Adam $\beta_2 = 0.98$, 4,000 warmup steps (Vaswani et al. 2017).
- Llama 3 8B: 32 layers, width 4,096, FFN 14,336, 32 query heads, 8 KV heads, vocabulary 128,000, RoPE base 500,000, about 15T tokens (Llama 3 paper).
- Mistral 7B from its config: 7,241,732,096 parameters, 8 KV heads, sliding window 4,096, untied embeddings.
- KV cache, Llama 3 8B in bf16: 128 KiB per token, 1 GiB per 8K tokens, 16 GiB at 128K. Full MHA would be 4 times that. Llama 3 70B: 320 KiB per token.
- Mixtral 8x7B from its config: 46.7B total, 12.9B active, 8 experts, top 2.
- Chinchilla: 70B on 1.4T tokens, about 20 tokens per parameter, MMLU 67.5 percent. Kaplan said $N \propto C^{0.73}$; Chinchilla says about $C^{0.5}$ for both.
- Llama 3 405B: 15.6T tokens, 3.8e25 FLOPs (6ND gives 3.79e25), 38 to 43 percent MFU.
- Llama 3 8B: about 1,900 tokens per parameter, roughly 94 times Chinchilla's ratio.
- Training memory with mixed precision Adam: 16 bytes per parameter, so 128 GB for 8B.
- A100: 312 TFLOPS bf16, 1.5 to 2.0 TB/s HBM, 192 KB SRAM per SM (FlashAttention paper).
- Batch 1 decode of an 8B bf16 model at 2 TB/s: about 8 ms per token, at most about 125 tokens per second.
- fp16 max 65,504; bf16 has fp32's range and 7 mantissa bits (spacing 0.0078 at 1.0).
- build-nanogpt: AdamW (0.9, 0.95), eps 1e-8, weight decay 0.1, clip 1.0, max lr 6e-4 to min 6e-5, 715 warmup steps, 0.5M token batches, 10B tokens.

## Decision tables

| Model family | Mask | Objective | Use |
| --- | --- | --- | --- |
| Encoder only (BERT) | Bidirectional | MLM, 15 percent masked | Classification, embeddings, rerankers |
| Decoder only (GPT, Llama) | Causal | Next token | Generation, general LLMs |
| Encoder decoder (T5) | Bidirectional + causal + cross attention | Span corruption | Translation, summarisation |

| Attention variant | KV heads | Cache | Quality |
| --- | --- | --- | --- |
| MHA | $h$ | Largest | Baseline |
| GQA | $g$ | $h/g$ times smaller | Close to MHA |
| MQA | 1 | $h$ times smaller | Lower |

| Tokeniser | Merge rule | Users |
| --- | --- | --- |
| BPE (byte level) | Most frequent pair | GPT, Llama, Mistral, Qwen |
| WordPiece | Highest freq(ab) over freq(a) times freq(b) | BERT |
| Unigram | Prune tokens that least hurt likelihood | T5, ALBERT (via SentencePiece) |

| Sampler | Keeps |
| --- | --- |
| Greedy | The argmax (temperature 0) |
| Top k | The $k$ most likely |
| Top p | Smallest set with mass at least $p$ |
| Min p | Tokens with $p \ge p_{\min}\cdot p_{\max}$ |

| Bottleneck | Phase | Fix |
| --- | --- | --- |
| Compute | Prefill, training | FlashAttention, FP8, chunked prefill |
| Memory bandwidth | Decode | Batching, quantisation, GQA, speculative decoding |
| Memory capacity | Long context, big batches | PagedAttention, KV quantisation, MLA, sliding windows |

## If asked X, say Y

- **Why divide by the square root of d_k?** Dot products of unit variance vectors have variance $d_k$; unscaled, softmax saturates and its gradient vanishes.
- **Why is decode memory bound?** Each token reads every weight once for about 2 FLOPs per weight, about 1 FLOP per byte, while an A100 needs about 156 to be compute bound.
- **What does the KV cache cost?** Memory linear in length and batch. It is often bigger than the weights at long context.
- **Does GQA cut FLOPs?** Barely. It cuts cache size and memory traffic.
- **Why RoPE on q and k but not v?** Position should change who attends to whom, not the content passed on. The rotation cancels to a relative offset in the dot product.
- **Pre norm or post norm?** Pre norm: a clean residual path, stable at depth, less dependent on warmup.
- **Why decoder only?** Every token is a target, generation is the training task, one stack, simple caching.
- **Chinchilla in one line?** Scale parameters and tokens together, about 20 tokens per parameter for compute optimal training; production models overtrain small models for cheaper inference.
- **Why is FlashAttention faster with more FLOPs?** HBM traffic dominates; tiling in SRAM avoids writing the $T \times T$ matrix.
- **Does speculative decoding change outputs?** No, the acceptance rule makes the distribution exactly the target's. It stops paying off at large batch sizes.
- **GPTQ versus AWQ?** GPTQ: layer wise second order error compensation. AWQ: scale up the 1 percent of channels with large activations before rounding.
- **Total versus active MoE parameters?** Memory pays for all experts, compute only for the top $k$.
- **Why do models hallucinate?** The objective rewards plausible text, rare facts look like fakes, and benchmarks reward guessing over abstaining (Kalai et al. 2025).
- **Are emergent abilities real?** Loss scales smoothly; sharp jumps often come from all or nothing metrics (Schaeffer et al. 2023).
- **Why does the model miscount letters?** It sees tokens, not characters.

## Traps

- Masking after the softmax (rows no longer sum to one).
- Calling attention permutation invariant (it is equivariant, and causal masking breaks even that).
- Saying the tokeniser is trained with the model.
- Saying the FFN mixes tokens.
- Quoting 20 tokens per parameter as what labs actually do.
- Saying the KV cache saves memory.
