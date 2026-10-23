# Transformers and LLMs, explained plainly

A large language model (LLM) is a very big next word predictor. It reads text as a list of numbered pieces. It passes them through a stack of identical layers, which let every piece look at the earlier pieces. Then it outputs a probability for every possible next piece. A probability is a number from 0 to 1 that says how likely something is. The design behind this is called a transformer, a type of neural network (a model made of many simple connected units that learn from data). This guide explains each part with intuition and small examples. It covers how these models are built, trained and made fast. Training models to follow instructions (post training) and retrieval systems are in other modules. The lecture links point to CME295, Stanford University's course on transformers and large language models, and to CS229, its machine learning course.

## Tokenisation

A model cannot read letters. A **tokeniser** is a program that chops text into pieces called tokens. It gives each piece an id number from a fixed list called the vocabulary. The model only ever sees those numbers.

Whole words make the list too long, because every form of every word needs its own entry. Whole words also leave nothing for new words. Single characters make the list tiny, but they make the sequences very long. The compromise is **subwords**. Common words stay whole, and rare words split into reusable chunks. For example, "unbelievably" might become "un", "believ", "ably".

The most common way to build the list is **byte pair encoding** (BPE). A byte is the basic unit of computer data. Start with the 256 possible bytes, so any text in any language can be written. Find the pair of neighbouring symbols that appears most often in a big pile of text. Glue that pair into a new symbol, and repeat until the list is as long as you want.

> **Worked example:** take the text "aaabdaaabac". The most frequent pair is "a a", so it becomes a new symbol. Do that three times and the 11 bytes compress to 5 tokens. The vocabulary of GPT-2 (generative pre trained transformer 2, an early OpenAI language model) is 50,257 tokens. That is 256 bytes, 50,000 learned merges and one end of text marker.

Other schemes exist. WordPiece merges the pair that is most surprising given how common its parts are. BERT (bidirectional encoder representations from transformers, a model that reads text in both directions) uses it. Unigram starts with a huge list and prunes it. T5 (text to text transfer transformer, a Google model that treats every task as turning text into text) uses it.

Where it breaks: many odd model behaviours come from here. Asked how many r's are in "strawberry", the model sees two or three tokens, not letters. Numbers split unevenly, which makes arithmetic hard. Languages that were rare in the tokeniser's training text split into many more tokens. They then cost more and fit less into the context. The context is the text the model can consider at once.

> **Watch:** [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s) and [CME295 Lecture 1, tokenization](https://www.youtube.com/watch?v=Ub3GoFaUcds&t=1377s).

## Embeddings and the output layer

Each token id is turned into a list of numbers, called a vector, by looking it up in a big table called the **embedding**. Think of it as giving every token a set of coordinates in a space. Tokens used in similar ways end up near each other. GPT-2 small uses 768 numbers per token.

At the other end, the model has to turn its final vector back into a score for every token in the vocabulary. That is done by a second table of the same shape. Many smaller models simply reuse the first table for this. This is called **weight tying**. The vector for reading the word "cat" is also the vector for predicting "cat". In GPT-2 small, that one table is 38.6 million of its 124 million numbers. So sharing it saves about a third of the model. Large models such as Llama 3 (Meta's open model family) and Mistral (a model family from the company Mistral AI) keep two separate tables. At their size the saving is small.

Training measures how surprised the model is by the true next token. This measure is the cross entropy loss. A loss is a single number that says how wrong the model is, and training tries to make it smaller. A fresh, untrained model should be equally unsure about all 50,257 options. That gives a starting loss of about 10.8. If yours starts far higher, something is wrong with the setup.

> **Watch:** [Let's reproduce GPT-2, sharing the embedding and output weights](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3974s).

## Scaled dot product attention

Attention is how tokens gather information from other tokens. Picture each token holding up three cards. The **query** says what it is looking for. The **key** says what it offers. The **value** is the content it will hand over. Each token compares its query with every other token's key. A good match means a high score. The scores are turned into percentages that add up to 100. The token's new vector is a blend of everyone's values, weighted by those percentages.

In "the animal didn't cross the street because it was tired", the token "it" should send most of its attention to "animal". Attention learns to do that from data.

Two details matter in interviews.

**The scaling.** A score is a sum of many small products, one per dimension. This sum is called a dot product. A dimension is one of the numbers in the vector. Add up 128 random products and the total swings much more widely than one product does. Its spread grows with the square root of the number of terms. Big swings make the percentages lopsided, close to 100 percent on one token and 0 on the rest. Then the model stops learning, because nudging the scores no longer changes anything. Dividing every score by the square root of the dimension puts the spread back to a sensible size. In a test with 256 dimensions, the biggest weight was 1.0 without scaling and 0.42 with it.

**The causal mask.** When predicting the next word, a token must not peek at the words after it. Before the percentages are computed, every future position gets a score of minus infinity. That becomes exactly 0 percent. Padding is blanked out the same way. Padding is filler added to make sequences the same length.

Where it breaks: every token compares with every other token. So doubling the text length quadruples the work.

> **Watch:** [Let's build GPT, self attention](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s) and [why the scores are scaled](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4616s).

## Multi head attention, MQA and GQA

One attention operation produces one blend per token, so it can focus on one kind of relationship at a time. **Multi head attention** runs several smaller attention operations side by side. Each one is called a head. GPT-2 small has 12 heads, each 64 numbers wide. The results are glued together. One head might track the previous word, and another the subject of the sentence. It costs the same as one big head because the total width is unchanged.

While generating text, the model stores every past token's keys and values so it does not recompute them. This store is the KV (key value) cache, explained below. The store gets huge. Two tricks shrink it.

- **Multi query attention** (MQA): all query heads share a single set of keys and values.
- **Grouped query attention** (GQA): query heads share keys and values in groups. Llama 3 8B has 8 billion parameters, where a parameter is one learned number inside the model. It has 32 query heads and 8 key value heads. So groups of four share, and the store is four times smaller than with one set per head.

The analogy: 32 students each with their own textbook, against 8 shared textbooks with four students at each. Quality barely drops, and memory falls a lot. GQA saves memory, not much arithmetic. That is why it speeds up generation, which is limited by memory.

> **Watch:** [CME295 Lecture 2, sharing attention heads](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=3338s).

## Positional information

Attention by itself has no sense of order. Without help, "dog bites man" and "man bites dog" look like the same bag of words. Models need to be told where each token sits. There are four common ways.

- **Sinusoidal** (the original transformer): add to each token a pattern of waves at different speeds, like the hands of many clocks. Each position then has a unique signature.
- **Learned** (GPT-2, BERT): a table with one learned vector per position. It is simple, but it has no row for position 1,025 if it was trained on 1,024.
- **RoPE** (rotary position embedding, used by Llama, Mistral and Qwen): rotate each token's query and key by an angle proportional to its position. When two rotated vectors are compared, only the difference in their angles matters. So the score depends on how far apart the tokens are, not where they sit. The values are not rotated. Position decides who listens to whom, not what is said. Qwen is a model family from Alibaba.
- **ALiBi** (attention with linear biases): subtract a penalty from the score that grows with distance, so nearby tokens are favoured.

> **Worked example:** with RoPE, a query at position 3 and a key at position 10 give a certain score. The same query at 103 and the key at 110 give exactly the same score. The reason is that both pairs are 7 apart.

Where it breaks: models trained on short texts do poorly on longer ones, because they have never seen those angles or positions. Llama 3 handled this by increasing RoPE's base frequency, which sets how fast the angles turn. It then trained on progressively longer texts, from 8,000 up to 128,000 tokens.

> **Watch:** [CME295 Lecture 2, RoPE](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=1862s).

## The transformer block

A transformer is the same block stacked many times: 12 in GPT-2 small, 32 in Llama 3 8B. Each block does two things in order.

1. **Attention**: tokens exchange information with earlier tokens.
2. **Feed forward network** (FFN): each token is processed on its own by a small two layer network. It is like each person thinking privately after a group discussion.

Around each step sits a **residual connection**. The block adds its result to the running vector instead of replacing it. Picture a shared notebook passed up the stack. Each layer reads the notebook and writes notes in the margin, and nothing is erased. This keeps information from early layers available to late ones. It also gives the training signal a straight path down through dozens of layers.

Each step also starts with a **normalisation**. This rescales the token's vector to a standard size, so numbers do not drift too large or too small. Modern models normalise before each step ("pre norm"). That trains more stably than the original transformer's choice of normalising after ("post norm"). Most now use RMSNorm (root mean square normalisation). It is a cheaper version that only rescales and skips recentring.

Llama and Mistral also replace the plain FFN with **SwiGLU** (Swish gated linear unit). It has a "gate". One part of the network decides, number by number, how much of another part's output to let through. It has three weight tables instead of two. So the hidden layer is made a bit narrower to keep the size the same.

Where it breaks: the FFN never mixes tokens. If an interviewer asks how information moves between words, the answer is attention, and only attention.

> **Watch:** [Let's build GPT, residual connections](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5208s).

## Encoders, decoders and objectives

The same blocks come in three arrangements. They differ in who may look at whom and in what the model is trained to predict. The thing a model is trained to predict is called its objective. An encoder reads the whole input and builds a rich summary of it. A decoder writes output one token at a time.

| Type | Example | Looks at | Trained to | Best for |
| --- | --- | --- | --- | --- |
| Encoder only | BERT | The whole sentence, both directions | Fill in hidden words | Classifying text, search embeddings |
| Decoder only | GPT, Llama | Only earlier words | Predict the next word | Writing, chat, general use |
| Encoder decoder | T5 | Encoder sees all input, decoder sees earlier output plus the encoder | Rewrite blanked out spans | Translation, summarisation |

BERT hides 15 percent of words and guesses them from both sides. This makes it good at understanding. But it has no natural way to write left to right. GPT predicts every next word. So every position in every sentence is a training example, and generating text is the training task itself. That simplicity helps. Another reason is that any task can be phrased as "continue this text". Together these are why nearly all large models today are decoder only.

> **Watch:** [CME295 Lecture 2, BERT deep dive](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=4298s). [CS229 Lecture 13, next word prediction](https://www.youtube.com/watch?v=lNTajqxxOn4).

## Counting parameters and FLOPs

A **parameter** is one learned number. Most of them sit in the weight tables of the attention and FFN parts of each block. Here is a rough rule for GPT style models. Each block holds about 12 times the width squared. So the whole stack is about 12 times layers times width squared. Width is the length of the vector that each token carries through the model.

> **Worked example:** GPT-2 small has 12 blocks of width 768. Twelve times 12 times 768 squared is 85 million. Adding the token and position tables gives the famous 124 million. Doing the same sum for Mistral 7B from its published settings gives 7.24 billion.

A **FLOP** is one floating point operation, which means one arithmetic operation on a decimal number. Running the model forward costs about 2 FLOPs per parameter per token: one multiply and one add. Training also runs backward, which costs about twice as much again. So training costs about 6 FLOPs per parameter per token. For a whole run that is **6 times parameters times training tokens**.

Check it on Llama 3's largest model. Six times 405 billion parameters times 15.6 trillion tokens is 3.8 times 10 to the 25 FLOPs. That is the number Meta reports. Training also needs memory. It takes about 16 bytes per parameter with the usual optimiser. An optimiser is the algorithm that updates the parameters. So an 8 billion parameter model needs 128 GB (gigabytes) before counting anything else.

> **Watch:** [CME295 Lecture 4, FLOPs](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=806s).

## Scaling laws

Scaling laws are measured rules for how much better a model gets with more parameters, more data and more compute. Compute means the amount of calculation spent on training. Loss falls smoothly and predictably, like a straight line on a log log plot. A log log plot is a graph where equal steps along each axis mean multiplying by the same factor. So labs fit the line on small models. Then they predict the big one before paying for it.

The two famous results disagree on how to split a budget.

- **Kaplan et al. 2020** (OpenAI): spend most extra compute on a bigger model, and train it on relatively little data.
- **Chinchilla** (Hoffmann et al. 2022, DeepMind): grow model size and data equally. That works out to roughly 20 training tokens per parameter. Their 70 billion parameter Chinchilla, trained on 1.4 trillion tokens, beat the 280 billion parameter Gopher that used the same compute.

The analogy: with a fixed budget for a student, Kaplan says hire a genius and give them a few books. Chinchilla says hire someone smaller and give them the whole library.

Where it changed: Chinchilla minimises training cost only. A model that answers millions of questions pays for every answer, and smaller models are cheaper to run. So labs now train small models far past 20 tokens per parameter. Llama 3 8B saw about 15 trillion tokens, around 1,900 per parameter.

> **Watch:** [CME295 Lecture 4, scaling laws and Chinchilla](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=994s).

## Pretraining data

Pretraining is the first, long stage of training. The model reads trillions of tokens of web pages, code, books, papers and maths. The quality of that pile matters more than most architecture tweaks. Architecture means the design of the model.

The cleaning steps, as in Llama 3:

1. Pull clean text out of web pages.
2. Remove duplicates: the same URL (web address), near identical documents, and lines repeated across many pages, such as cookie banners and menus.
3. Drop junk with simple rules: very repetitive pages, adult sites, pages with unusual word statistics.
4. Score the rest with small classifier models trained to spot high quality text, and keep the best. A classifier is a model that sorts items into categories.
5. Mix sources in chosen proportions. Llama 3 used about half general knowledge, a quarter maths and reasoning, 17 percent code and 8 percent other languages.

Duplicates are removed because they waste compute and make the model memorise and parrot. **Contamination** is when benchmark test questions leak into the training pile. A benchmark is a standard test used to compare models. A high score then means the model remembers the answer, not that it can solve the problem.

## Training at scale

The training loop is simple. Take a batch of text, which is a group of examples processed together. Predict every next token. Measure the error. Compute how each parameter should change. Then update. A few settings are near universal.

- **AdamW** optimiser (Adam, short for adaptive moment estimation, with weight decay): it adapts the step size per parameter. It also shrinks weights slightly each step, which is called weight decay.
- **Warmup then cosine decay**: the learning rate (step size) ramps up from almost zero over the first few hundred or thousand steps. Then it glides down along a cosine curve. Early on, the model is fragile and big steps can wreck it. At the end, small steps let it settle.
- **Gradient clipping**: if one batch produces a huge update, cap it. A gradient is the direction and size of the change that would reduce the error.
- **bf16**: store and compute numbers in a 16 bit format. The name is short for brain floating point 16. Ordinary 16 bit numbers (fp16) top out at 65,504, and small values round to zero. So training needs tricks. bf16 keeps the same range as 32 bit numbers and gives up some precision. Training tolerates that.

> **Worked example:** Andrej Karpathy's lecture that reproduces GPT-2 uses a peak learning rate of 0.0006. It warms up for 715 steps. Then it decays to 0.00006 over about 19,000 steps. It processes about half a million tokens per step and 10 billion in total.

Where it breaks: a **loss spike** is a sudden jump in error, from a bad batch or a step that was too big. Teams lower the learning rate, rewind to a checkpoint before the spike and skip the bad data. A checkpoint is a saved copy of the model partway through training.

> **Watch:** [Let's reproduce GPT-2, learning rate schedule](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8466s).

## Prefill, decode and the KV cache

Answering a prompt has two phases. **Prefill** reads the whole prompt at once, in parallel, which means all pieces at the same time. **Decode** then writes the answer one token at a time. Each step feeds the new token back in.

Without a shortcut, every decode step would redo the work for all earlier tokens. The **KV cache** stores each earlier token's keys and values. These never change, because earlier tokens cannot see later ones. Each step then only computes the newest token. It is like keeping your notes from earlier pages instead of rereading the whole book every time you write a sentence.

The price is memory, and it adds up.

> **Worked example:** Llama 3 8B stores 131,072 bytes (128 KiB) of cache per token. That is 2 (keys and values) times 32 layers times 8 key value heads times 128 numbers times 2 bytes per number. A 128,000 token conversation needs 16 GiB of cache. That is about the size of the model's own weights. KiB and GiB are binary units, where each step up is 1,024 times larger.

Decode is slow for a surprising reason. The GPU (graphics processing unit, the chip that does the heavy arithmetic) has plenty of arithmetic power. But it must read all 16 GB of weights from memory for every single token. At about 2 TB (terabytes) per second, that is 8 milliseconds per token. So it manages about 125 tokens per second at most for one user. Serving many users at once reads the weights once for all of them. That is why batching helps so much.

> **Watch:** [CME295 Lecture 3, KV cache](https://www.youtube.com/watch?v=Q5baLehv5So&t=5100s).

## FlashAttention and serving systems

**FlashAttention** computes exactly the same attention, faster. A GPU has a little very fast memory on the chip and a lot of slower memory beside it. Normal attention writes a huge table of scores to the slow memory and reads it back several times. FlashAttention works on small tiles that fit in the fast memory and never writes the full table. It does slightly more arithmetic but moves far less data. Moving data was the bottleneck, which means it was the slowest part and limited the whole speed. FlashAttention reported a 3 times speedup on GPT-2.

Serving systems add two more ideas. Serving means running a trained model to answer requests.

- **Continuous batching**: instead of waiting for every request in a batch to finish, the server swaps finished requests out and new ones in at every step. It is like a restaurant seating new diners as soon as a table frees up.
- **PagedAttention** (used in vLLM, an open source serving library): store the cache in small pages placed anywhere. The old way reserved one big block of memory per request, sized for the longest possible answer. Paging works like a computer's virtual memory. Almost no memory is wasted, so more requests fit at once. vLLM reported 2 to 4 times the throughput of earlier systems. Throughput is how many requests the system handles per second.

> **Watch:** [CME295 Lecture 4, Flash Attention](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=2306s).

## Quantisation

**Quantisation** stores each weight in fewer bits, like saving a photo at lower quality. A bit is a single 0 or 1. An 8 billion parameter model takes 16 GB in 16 bit numbers. It takes 8 GB in 8 bit integers (whole numbers) and about 4 GB in 4 bit integers. Decoding is limited by how many bytes must be read per token. So smaller weights mean faster generation as well as fitting on smaller GPUs.

The catch is that each group of numbers shares one scale. A single very large value stretches the scale, so everything else gets rounded coarsely. In a test, one outlier made 8 bit rounding error 13 times worse. An outlier is a value far larger than the rest. LLMs have a few such outliers. So practical methods are smarter than plain rounding.

- **GPTQ** (a quantisation method built for GPT style models) quantises one layer at a time. It nudges the remaining weights to cancel the rounding errors already made. It reaches 3 to 4 bits with little accuracy loss.
- **AWQ** (activation aware weight quantisation) notices that about 1 percent of weights matter most. These are the ones that multiply large inputs. It scales those up before rounding so they keep their precision.

## Speculative decoding

A big model takes about as long to check five guessed tokens as to write one. The reason is that decode time is spent reading weights, not doing arithmetic. **Speculative decoding** lets a small, fast draft model guess the next few tokens. Then the big model checks them all in one pass and keeps the ones it agrees with.

The clever part is the checking rule. Each guess is accepted with a probability based on how much more the big model likes that token than the small one does. When a guess is rejected, a replacement is drawn in a way that exactly corrects the bias. The bias here is the lean toward the small model's guesses. The result is that the text comes out with exactly the same probabilities as if the big model had written it alone, only faster.

> **Worked example:** suppose the big model agrees with each draft token 80 percent of the time. Drafts are 4 tokens long, and the draft costs 5 percent of a big model pass. Then you get about 3.4 tokens per big pass and about a 2.8 times speedup. At 60 percent agreement the speedup falls to about 1.9.

Where it breaks: when a server already batches many users, the GPU is busy and the draft model's work is no longer free.

## Decoding strategies

The model outputs probabilities. A decoding strategy picks the actual token from them.

- **Greedy**: always take the most likely token. It is predictable but prone to dull loops.
- **Beam search**: track several candidate sentences and keep the best overall. It is good for translation and bland for open writing.
- **Temperature**: sharpen or flatten the probabilities before sampling. Sampling means picking at random in proportion to the probabilities. Low temperature means safe and repetitive. High temperature means varied and risky. Temperature 0 is greedy.
- **Top k**: sample only from the k most likely tokens.
- **Top p (nucleus)**: sample from the smallest group of tokens whose probabilities add to p. The group shrinks when the model is confident and grows when it is not.
- **Min p**: keep tokens at least some fraction as likely as the top token.
- **Repetition penalties**: lower the score of tokens already used.

> **Worked example:** six tokens with top probability 0.55. At temperature 0.5 the top token rises to 0.72. At temperature 2 it drops to 0.38. Top p of 0.9 keeps three tokens, because the first two only cover 0.88.

> **Watch:** [CME295 Lecture 3, sampling methods](https://www.youtube.com/watch?v=Q5baLehv5So&t=2796s).

## Long context

The context window is how many tokens the model can consider at once. The CME295 cheatsheet puts typical windows at tens of thousands to millions of tokens. Longer contexts cost in three ways. Attention work grows with the square of the length, so doubling the length quadruples the work. The KV cache grows with the length. The model must also have been trained on such lengths.

There are three kinds of fix. One is adjusting the position scheme and training further on long documents. Another is limiting some layers to a sliding window of recent tokens. The third is shrinking the cache.

Where it breaks: fitting a document in the window is not the same as using it. Models find facts at the start or end of a long prompt more reliably than facts buried in the middle. Quality also tends to fall as the window fills up.

## Mixture of experts

A **mixture of experts** model replaces each block's single feed forward network with several "expert" networks and a small router. The router sends each token to just a couple of them. Think of a hospital: every patient sees two specialists, not the entire staff.

> **Worked example:** Mixtral 8x7B (a mixture of experts model from Mistral AI) has 8 experts per layer and sends each token to 2. From its published settings, that is 46.7 billion parameters in total but only 12.9 billion used per token. It runs about as fast as a 13 billion parameter model while needing memory for 47 billion. It is not 56 billion (8 times 7) because the attention parts are shared.

Where it breaks: left alone, the router sends most tokens to a few favourite experts. These then improve and attract even more. Training adds a small penalty for uneven traffic to keep all experts busy.

> **Watch:** [CME295 Lecture 3, mixture of experts](https://www.youtube.com/watch?v=Q5baLehv5So&t=457s).

## In context learning and emergence

**In context learning** is a model doing a new task just from examples in the prompt, with no retraining. Show it three English to French pairs and it translates the fourth. There are two explanations. In one, the model learned during pretraining to work out what kind of document it is reading, and examples are strong clues. In the other, some attention heads learn to find an earlier match of the current pattern and copy what followed it.

**Emergent abilities** are skills that seem to switch on suddenly past a certain model size. A 2023 paper argued that much of that suddenness comes from how we score. An all or nothing grade (right answer or not) turns a steady improvement into a jump. Partial credit scores show smooth progress.

> **Watch:** [CS229 Lecture 14, transformers and in context learning](https://www.youtube.com/watch?v=pwQ0l4hFCVI).

## Why models hallucinate

A **hallucination** is a confident, fluent answer that is wrong. It follows from how the model is built. There are five causes.

- It is trained to produce text that looks likely, not text that is true. A false sentence can look just as likely.
- Facts that appeared only once or twice in training are hard to tell apart from plausible fakes.
- A 2025 OpenAI paper (Kalai et al.) points out that most tests give no credit for "I don't know". So guessing scores better than admitting uncertainty, and training learns to guess.
- Once it writes one wrong word, it tends to stay consistent with it.
- Sampling at high temperature reaches for unlikely, often wrong, tokens.

The analogy is a student taking a multiple choice exam with no penalty for wrong answers. Guessing is always the best strategy. Fixes include giving the model sources to quote, lowering temperature for factual questions, and rewarding it for saying when it is unsure.
