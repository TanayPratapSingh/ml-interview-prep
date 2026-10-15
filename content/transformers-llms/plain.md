# Transformers and LLMs, explained plainly

A large language model is a very big next word predictor. It reads text as a list of numbered pieces, passes them through a stack of identical layers that let every piece look at the earlier pieces, and outputs a probability for every possible next piece. This version explains each part with intuition and small examples, and covers how these models are built, trained and made fast. Training models to follow instructions (post training) and retrieval systems are in other modules.

## Tokenisation

A model cannot read letters. A **tokeniser** chops text into pieces called tokens and gives each piece an id number from a fixed list, the vocabulary. The model only ever sees those numbers.

Whole words make the list too long (every form of every word needs its own entry) and leave nothing for new words. Single characters make the list tiny but the sequences very long. The compromise is **subwords**: common words stay whole, rare words split into reusable chunks, so "unbelievably" might become "un", "believ", "ably".

The most common way to build the list is **byte pair encoding** (BPE). Start with the 256 possible bytes, so any text in any language can be written. Find the pair of neighbouring symbols that appears most often in a big pile of text, glue it into a new symbol, and repeat until the list is as long as you want.

> **Worked example:** take the text "aaabdaaabac". The most frequent pair is "a a", so it becomes a new symbol. Do that three times and the 11 bytes compress to 5 tokens. GPT-2's vocabulary is 50,257 tokens: 256 bytes, 50,000 learned merges and one end of text marker.

Other schemes exist. WordPiece (used by BERT) merges the pair that is most surprising given how common its parts are, and Unigram (used by T5) starts with a huge list and prunes it.

Where it breaks: many odd model behaviours come from here. Asked how many r's are in "strawberry", the model sees two or three tokens, not letters. Numbers split unevenly, which makes arithmetic hard. Languages that were rare in the tokeniser's training text split into many more tokens, so they cost more and fit less into the context.

> **Watch:** [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s) and [CME295 Lecture 1, tokenization](https://www.youtube.com/watch?v=Ub3GoFaUcds&t=1377s).

## Embeddings and the output layer

Each token id is turned into a list of numbers, a vector, by looking it up in a big table called the **embedding**. Think of it as giving every token a set of coordinates in a space where tokens used in similar ways end up near each other. GPT-2 small uses 768 numbers per token.

At the other end, the model has to turn its final vector back into a score for every token in the vocabulary. That is done by a second table of the same shape. Many smaller models simply reuse the first table for this, called **weight tying**: the vector for reading the word "cat" is also the vector for predicting "cat". In GPT-2 small that one table is 38.6 million of its 124 million numbers, so sharing it saves about a third of the model. Large models such as Llama 3 and Mistral keep two separate tables, because at their size the saving is small.

Training measures how surprised the model is by the true next token (the cross entropy loss). A fresh, untrained model should be equally unsure about all 50,257 options, which gives a starting loss of about 10.8. If yours starts far higher, something is wrong with the setup.

> **Watch:** [Let's reproduce GPT-2, sharing the embedding and output weights](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3974s).

## Scaled dot product attention

Attention is how tokens gather information from other tokens. Picture each token holding up three cards. The **query** says what it is looking for, the **key** says what it offers, and the **value** is the content it will hand over. Each token compares its query with every other token's key; a good match means a high score. The scores are turned into percentages that add up to 100, and the token's new vector is a blend of everyone's values weighted by those percentages.

In "the animal didn't cross the street because it was tired", the token "it" should send most of its attention to "animal". Attention learns to do that from data.

Two details matter in interviews.

**The scaling.** A score is a sum of many small products, one per dimension. Add up 128 random products and the total swings much more widely than one product does: its spread grows with the square root of the number of terms. Big swings make the percentages lopsided, close to 100 percent on one token and 0 on the rest, and the model stops learning because nudging the scores no longer changes anything. Dividing every score by the square root of the dimension puts the spread back to a sensible size. In a test with 256 dimensions, the biggest weight was 1.0 without scaling and 0.42 with it.

**The causal mask.** When predicting the next word, a token must not peek at the words after it. Before the percentages are computed, every future position gets a score of minus infinity, which becomes exactly 0 percent. Padding (filler added to make sequences the same length) is blanked out the same way.

Where it breaks: every token compares with every other token, so doubling the text length quadruples the work.

> **Watch:** [Let's build GPT, self attention](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s) and [why the scores are scaled](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4616s).

## Multi head attention, MQA and GQA

One attention operation produces one blend per token, so it can focus on one kind of relationship at a time. **Multi head attention** runs several smaller attention operations side by side (12 in GPT-2 small, each 64 numbers wide) and glues the results together. One head might track the previous word, another the subject of the sentence. It costs the same as one big head because the total width is unchanged.

While generating text, the model stores every past token's keys and values so it does not recompute them (the KV cache, below). That store gets huge. Two tricks shrink it:

- **Multi query attention** (MQA): all query heads share a single set of keys and values.
- **Grouped query attention** (GQA): query heads share keys and values in groups. Llama 3 8B has 32 query heads and 8 key value heads, so groups of four share, and the store is four times smaller than with one set per head.

The analogy: 32 students each with their own textbook, against 8 shared textbooks with four students at each. Quality barely drops and memory falls a lot. GQA saves memory, not much arithmetic, and that is why it speeds up generation, which is limited by memory.

> **Watch:** [CME295 Lecture 2, sharing attention heads](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=3338s).

## Positional information

Attention by itself has no sense of order. Without help, "dog bites man" and "man bites dog" look like the same bag of words. Models need to be told where each token sits.

- **Sinusoidal** (the original transformer): add to each token a pattern of waves at different speeds, like the hands of many clocks, so each position has a unique signature.
- **Learned** (GPT-2, BERT): a table with one learned vector per position. Simple, but it has no row for position 1,025 if it was trained on 1,024.
- **RoPE** (rotary, used by Llama, Mistral, Qwen): rotate each token's query and key by an angle proportional to its position. When two rotated vectors are compared, only the difference in their angles matters, so the score depends on how far apart the tokens are, not where they sit. The values are not rotated: position decides who listens to whom, not what is said.
- **ALiBi**: subtract a penalty from the score that grows with distance, so nearby tokens are favoured.

> **Worked example:** with RoPE, a query at position 3 and a key at position 10 give exactly the same score as the same query at 103 and key at 110, because both pairs are 7 apart.

Where it breaks: models trained on short texts do poorly on longer ones, because they have never seen those angles or positions. Llama 3 handled this by increasing RoPE's base frequency and then training on progressively longer texts, from 8,000 up to 128,000 tokens.

> **Watch:** [CME295 Lecture 2, RoPE](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=1862s).

## The transformer block

A transformer is the same block stacked many times: 12 in GPT-2 small, 32 in Llama 3 8B. Each block does two things in order.

1. **Attention**: tokens exchange information with earlier tokens.
2. **Feed forward network** (FFN): each token is processed on its own by a small two layer network, like each person thinking privately after a group discussion.

Around each step sits a **residual connection**: the block adds its result to the running vector instead of replacing it. Picture a shared notebook passed up the stack. Each layer reads the notebook and writes notes in the margin; nothing is erased. This keeps information from early layers available to late ones and gives the training signal a straight path down through dozens of layers.

Each step also starts with a **normalisation** that rescales the token's vector to a standard size so numbers do not drift too large or too small. Modern models normalise before each step ("pre norm"), which trains more stably than the original transformer's choice of normalising after ("post norm"). Most now use RMSNorm, a cheaper version that only rescales and skips recentring.

Llama and Mistral also replace the plain FFN with **SwiGLU**, which has a "gate": one part of the network decides, number by number, how much of another part's output to let through. It has three weight tables instead of two, so the hidden layer is made a bit narrower to keep the size the same.

Where it breaks: the FFN never mixes tokens. If an interviewer asks how information moves between words, the answer is attention, and only attention.

> **Watch:** [Let's build GPT, residual connections](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5208s).

## Encoders, decoders and objectives

The same blocks come in three arrangements, distinguished by who may look at whom and what the model is trained to predict.

| Type | Example | Looks at | Trained to | Best for |
| --- | --- | --- | --- | --- |
| Encoder only | BERT | The whole sentence, both directions | Fill in hidden words | Classifying text, search embeddings |
| Decoder only | GPT, Llama | Only earlier words | Predict the next word | Writing, chat, general use |
| Encoder decoder | T5 | Encoder sees all input, decoder sees earlier output plus the encoder | Rewrite blanked out spans | Translation, summarisation |

BERT hides 15 percent of words and guesses them from both sides, which makes it good at understanding but gives it no natural way to write left to right. GPT predicts every next word, so every position in every sentence is a training example, and generating text is the training task itself. That simplicity, plus the fact that any task can be phrased as "continue this text", is why nearly all large models today are decoder only.

> **Watch:** [CME295 Lecture 2, BERT deep dive](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=4298s). [CS229 Lecture 13, next word prediction](https://www.youtube.com/watch?v=lNTajqxxOn4).

## Counting parameters and FLOPs

A **parameter** is one learned number. Most of them sit in the weight tables of the attention and FFN parts of each block. A rough rule for GPT style models: each block holds about 12 times the width squared, so the whole stack is about 12 times layers times width squared.

> **Worked example:** GPT-2 small has 12 blocks of width 768. Twelve times 12 times 768 squared is 85 million, and adding the token and position tables gives the famous 124 million. Doing the same sum for Mistral 7B from its published settings gives 7.24 billion.

A **FLOP** is one arithmetic operation. Running the model forward costs about 2 FLOPs per parameter per token (one multiply, one add). Training also runs backward, which costs about twice as much again, so training costs about 6 FLOPs per parameter per token. For a whole run that is **6 times parameters times training tokens**.

Check it on Llama 3's largest model: 6 times 405 billion parameters times 15.6 trillion tokens is 3.8 times 10 to the 25 FLOPs, which is the number Meta reports. Training also needs memory: about 16 bytes per parameter with the usual optimiser, so 128 GB for an 8 billion parameter model before counting anything else.

> **Watch:** [CME295 Lecture 4, FLOPs](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=806s).

## Scaling laws

Scaling laws are measured rules for how much better a model gets with more parameters, more data and more compute. Loss falls smoothly and predictably, like a straight line on a log log plot, so labs fit the line on small models and predict the big one before paying for it.

The two famous results disagree on how to split a budget:

- **Kaplan et al. 2020** (OpenAI): spend most extra compute on a bigger model and train it on relatively little data.
- **Chinchilla** (Hoffmann et al. 2022, DeepMind): grow model size and data equally. That works out to roughly 20 training tokens per parameter. Their 70 billion parameter Chinchilla, trained on 1.4 trillion tokens, beat the 280 billion parameter Gopher that used the same compute.

The analogy: with a fixed budget for a student, Kaplan says hire a genius and give them a few books; Chinchilla says hire someone smaller and give them the whole library.

Where it changed: Chinchilla minimises training cost only. A model that answers millions of questions pays for every answer, and smaller models are cheaper to run. So labs now train small models far past 20 tokens per parameter. Llama 3 8B saw about 15 trillion tokens, around 1,900 per parameter.

> **Watch:** [CME295 Lecture 4, scaling laws and Chinchilla](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=994s).

## Pretraining data

Pretraining reads trillions of tokens of web pages, code, books, papers and maths. The quality of that pile matters more than most architecture tweaks.

The cleaning steps, as in Llama 3:

1. Pull clean text out of web pages.
2. Remove duplicates: the same URL, near identical documents, and lines repeated across many pages (cookie banners, menus).
3. Drop junk with simple rules: very repetitive pages, adult sites, pages with unusual word statistics.
4. Score the rest with small classifier models trained to spot high quality text, and keep the best.
5. Mix sources in chosen proportions. Llama 3 used about half general knowledge, a quarter maths and reasoning, 17 percent code and 8 percent other languages.

Duplicates are removed because they waste compute and make the model memorise and parrot. **Contamination** is when benchmark test questions leak into the training pile; then a high score means the model remembers the answer, not that it can solve the problem.

## Training at scale

The training loop is: take a batch of text, predict every next token, measure the error, compute how each parameter should change, and update. A few settings are near universal.

- **AdamW** optimiser, which adapts the step size per parameter and shrinks weights slightly each step (weight decay).
- **Warmup then cosine decay**: the learning rate (step size) ramps up from almost zero over the first few hundred or thousand steps, then glides down along a cosine curve. Early on the model is fragile and big steps can wreck it; at the end, small steps let it settle.
- **Gradient clipping**: if one batch produces a huge update, cap it.
- **bf16**: store and compute numbers in a 16 bit format. Ordinary 16 bit numbers (fp16) top out at 65,504 and small values round to zero, so training needs tricks. bf16 keeps the same range as 32 bit numbers and gives up some precision, which training tolerates.

> **Worked example:** Karpathy's GPT-2 reproduction uses a peak learning rate of 0.0006, warms up for 715 steps, then decays to 0.00006 over about 19,000 steps, processing about half a million tokens per step and 10 billion in total.

Where it breaks: a **loss spike** is a sudden jump in error, from a bad batch or a step that was too big. Teams lower the learning rate, rewind to a checkpoint before the spike and skip the bad data.

> **Watch:** [Let's reproduce GPT-2, learning rate schedule](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8466s).

## Prefill, decode and the KV cache

Answering a prompt has two phases. **Prefill** reads the whole prompt at once, in parallel. **Decode** then writes the answer one token at a time, each step feeding the new token back in.

Without a shortcut, every decode step would redo the work for all earlier tokens. The **KV cache** stores each earlier token's keys and values, which never change because earlier tokens cannot see later ones. Each step then only computes the newest token. It is like keeping your notes from earlier pages instead of rereading the whole book every time you write a sentence.

The price is memory, and it adds up.

> **Worked example:** Llama 3 8B stores 131,072 bytes (128 KiB) of cache per token: 2 (keys and values) times 32 layers times 8 key value heads times 128 numbers times 2 bytes per number. A 128,000 token conversation needs 16 GiB of cache, about the size of the model's own weights.

Decode is slow for a surprising reason: the GPU has plenty of arithmetic power but must read all 16 GB of weights from memory for every single token. At about 2 TB per second, that is 8 milliseconds per token, so about 125 tokens per second at most for one user. Serving many users at once reads the weights once for all of them, which is why batching helps so much.

> **Watch:** [CME295 Lecture 3, KV cache](https://www.youtube.com/watch?v=Q5baLehv5So&t=5100s).

## FlashAttention and serving systems

**FlashAttention** computes exactly the same attention, faster. A GPU has a little very fast memory on the chip and a lot of slower memory beside it. Normal attention writes a huge table of scores to the slow memory and reads it back several times. FlashAttention works on small tiles that fit in the fast memory and never writes the full table. It does slightly more arithmetic but moves far less data, and moving data was the bottleneck. It reported a 3 times speedup on GPT-2.

Serving systems add two more ideas:

- **Continuous batching**: instead of waiting for every request in a batch to finish, the server swaps finished requests out and new ones in at every step, like a restaurant seating new diners as soon as a table frees up.
- **PagedAttention** (vLLM): instead of reserving one big block of memory per request for the longest possible answer, store the cache in small pages placed anywhere, like a computer's virtual memory. Almost no memory is wasted, so more requests fit at once. vLLM reported 2 to 4 times the throughput of earlier systems.

> **Watch:** [CME295 Lecture 4, Flash Attention](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=2306s).

## Quantisation

**Quantisation** stores each weight in fewer bits, like saving a photo at lower quality. An 8 billion parameter model takes 16 GB in 16 bit numbers, 8 GB in 8 bit integers and about 4 GB in 4 bit integers. Because decoding is limited by how many bytes must be read per token, smaller weights mean faster generation as well as fitting on smaller GPUs.

The catch is that each group of numbers shares one scale, and a single very large value stretches the scale so everything else gets rounded coarsely. In a test, one outlier made 8 bit rounding error 13 times worse. LLMs have a few such outliers, so practical methods are smarter than plain rounding:

- **GPTQ** quantises one layer at a time and nudges the remaining weights to cancel the rounding errors already made. It reaches 3 to 4 bits with little accuracy loss.
- **AWQ** notices that about 1 percent of weights matter most, the ones that multiply large inputs, and scales those up before rounding so they keep their precision.

## Speculative decoding

A big model takes about as long to check five guessed tokens as to write one, because decode time is spent reading weights, not doing arithmetic. **Speculative decoding** lets a small, fast draft model guess the next few tokens, then the big model checks them all in one pass and keeps the ones it agrees with.

The clever part is the checking rule. Each guess is accepted with a probability based on how much more the big model likes that token than the small one does. When a guess is rejected, a replacement is drawn in a way that exactly corrects the bias. The result: the text comes out with exactly the same probabilities as if the big model had written it alone, only faster.

> **Worked example:** if the big model agrees with each draft token 80 percent of the time, drafts are 4 tokens long and the draft costs 5 percent of a big model pass, you get about 3.4 tokens per big pass and about a 2.8 times speedup. At 60 percent agreement the speedup falls to about 1.9.

Where it breaks: when a server already batches many users, the GPU is busy and the draft model's work is no longer free.

## Decoding strategies

The model outputs probabilities; a decoding strategy picks the actual token.

- **Greedy**: always take the most likely token. Predictable but prone to dull loops.
- **Beam search**: track several candidate sentences and keep the best overall. Good for translation, bland for open writing.
- **Temperature**: sharpen or flatten the probabilities before sampling. Low temperature means safe and repetitive, high means varied and risky. Temperature 0 is greedy.
- **Top k**: sample only from the k most likely tokens.
- **Top p (nucleus)**: sample from the smallest group of tokens whose probabilities add to p, so the group shrinks when the model is confident and grows when it is not.
- **Min p**: keep tokens at least some fraction as likely as the top token.
- **Repetition penalties**: lower the score of tokens already used.

> **Worked example:** six tokens with top probability 0.55. At temperature 0.5 the top token rises to 0.72; at temperature 2 it drops to 0.38. Top p of 0.9 keeps three tokens, because the first two only cover 0.88.

> **Watch:** [CME295 Lecture 3, sampling methods](https://www.youtube.com/watch?v=Q5baLehv5So&t=2796s).

## Long context

The context window is how many tokens the model can consider at once; the CME295 cheatsheet puts typical windows at tens of thousands to millions of tokens. Longer contexts cost in three ways: attention work grows with the square of length, the KV cache grows with length, and the model must have been trained on such lengths. Fixes include adjusting the position scheme and training further on long documents, limiting some layers to a sliding window of recent tokens, and shrinking the cache.

Where it breaks: fitting a document in the window is not the same as using it. Models find facts at the start or end of a long prompt more reliably than facts buried in the middle, and quality tends to fall as the window fills up.

## Mixture of experts

A **mixture of experts** model replaces each block's single feed forward network with several "expert" networks and a small router that sends each token to just a couple of them. Think of a hospital: every patient sees two specialists, not the entire staff.

> **Worked example:** Mixtral 8x7B has 8 experts per layer and sends each token to 2. From its published settings, that is 46.7 billion parameters in total but only 12.9 billion used per token. It runs about as fast as a 13 billion parameter model while needing memory for 47 billion. It is not 56 billion (8 times 7) because the attention parts are shared.

Where it breaks: left alone, the router sends most tokens to a few favourite experts, which then improve and attract even more. Training adds a small penalty for uneven traffic to keep all experts busy.

> **Watch:** [CME295 Lecture 3, mixture of experts](https://www.youtube.com/watch?v=Q5baLehv5So&t=457s).

## In context learning and emergence

**In context learning** is a model doing a new task just from examples in the prompt, with no retraining. Show it three English to French pairs and it translates the fourth. One explanation: during pretraining the model learned to work out what kind of document it is reading, and examples are strong clues. Another: some attention heads learn to find an earlier match of the current pattern and copy what followed it.

**Emergent abilities** are skills that seem to switch on suddenly past a certain model size. A 2023 paper argued much of that suddenness comes from how we score: an all or nothing grade (right answer or not) turns a steady improvement into a jump, while partial credit scores show smooth progress.

> **Watch:** [CS229 Lecture 14, transformers and in context learning](https://www.youtube.com/watch?v=pwQ0l4hFCVI).

## Why models hallucinate

A **hallucination** is a confident, fluent answer that is wrong. It follows from how the model is built:

- It is trained to produce text that looks likely, not text that is true, and a false sentence can look just as likely.
- Facts that appeared only once or twice in training are hard to tell apart from plausible fakes.
- A 2025 OpenAI paper (Kalai et al.) points out that most tests give no credit for "I don't know", so guessing scores better than admitting uncertainty, and training learns to guess.
- Once it writes one wrong word, it tends to stay consistent with it.
- Sampling at high temperature reaches for unlikely, often wrong, tokens.

The analogy is a student taking a multiple choice exam with no penalty for wrong answers: guessing is always the best strategy. Fixes include giving the model sources to quote, lowering temperature for factual questions, and rewarding it for saying when it is unsure.
