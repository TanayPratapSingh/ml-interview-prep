# Post training and reasoning

Post training is everything that happens to a language model after next token pretraining: supervised fine tuning on demonstrations, preference optimisation (RLHF with PPO, DPO and its variants), reinforcement learning on verifiable rewards (GRPO, Dr. GRPO, the DeepSeek R1 recipe), and the inference time machinery that turns extra compute into better answers (sampling, voting, verifiers, search). This guide gives the derivations interviewers ask you to reproduce (policy gradient, PPO clipping, Bradley Terry, the DPO closed form, the GRPO advantage and the two biases Dr. GRPO removes), the numbers worth quoting with the paper each came from, and the failure modes that separate someone who has run these methods from someone who has read about them. Every formula and worked example was checked numerically in Python.

## Why post training exists

A pretrained base model is a document completer. It was trained to minimise cross entropy on internet text, so given "Write a poem about cheese" it continues the most likely document, which may be a list of more writing prompts. An assistant model answers the request, follows a format, refuses some things, and stops at the end of its turn. Post training is the set of stages that converts the first into the second, and, since 2024, the stage where reasoning ability is trained with reinforcement learning.

### Base model versus assistant

The base model already contains most of the knowledge and skill. What it lacks is the behaviour policy: knowing that the text after the prompt should be a helpful reply in a particular voice, that it should end, and which of the many plausible continuations people actually want. You can coax a base model into acting like an assistant with a few shot prompt that looks like a transcript between a human and an assistant, but that is brittle. Karpathy's State of GPT talk makes the same point: base models want to complete documents, not answer questions, and they keep more output diversity (entropy) than tuned models, which is why he still prefers a base model when you want many more items like a handful of examples.

> **Watch:** [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=636s) on base models not being assistants, then [the reward modelling segment](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=768s) and [why RLHF beats SFT](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=1036s).

### The standard pipeline

The InstructGPT paper (Ouyang et al., 2022) fixed the three step recipe most interviewers still have in mind:

1. **SFT.** Fine tune GPT-3 on labeller written demonstrations (about 13k prompts).
2. **Reward model.** Labellers rank 4 to 9 model outputs per prompt; a 6B reward model learns to score them (about 33k prompts).
3. **PPO.** Optimise the SFT policy against the reward model with a KL penalty to the SFT model (about 31k prompts, no human labels needed at this stage).

The headline result was that outputs from the 1.3B InstructGPT model were preferred to the 175B GPT-3, and 175B InstructGPT outputs beat 175B GPT-3 outputs 85 plus or minus 3 percent of the time. Modern pipelines keep the shape but swap parts: DPO often replaces the reward model plus PPO pair, AI feedback replaces some human labels, and reasoning models add a large RL stage with rule checked rewards.

| Stage | Data | Signal | Typical compute |
| --- | --- | --- | --- |
| Pretraining | Trillions of web tokens | Next token likelihood | Thousands of GPUs, months |
| SFT | 1k to 1M curated (prompt, response) pairs | Imitate the response | A few GPUs, hours to days |
| Preference tuning | (prompt, chosen, rejected) triples | Prefer chosen over rejected | Similar to SFT (DPO) or more (PPO) |
| RL with verifiable rewards | Prompts with checkable answers | Reward from a checker | Large; dominated by generation |

### What post training changes

The LIMA paper (Zhou et al., 2023) fine tuned a 65B LLaMA on only about 1,000 carefully chosen examples, with no RL, and its answers were judged equivalent to or better than GPT-4 in 43 percent of comparisons. The authors argue that nearly all knowledge comes from pretraining and alignment mostly teaches format and style; this is often called the superficial alignment hypothesis. Treat it as a strong statement about chat style, not about reasoning: the 2025 reasoning results show RL can substantially change how a model uses its knowledge on math and code, even if (as later sections discuss) it may mostly sharpen abilities already present.

> **Interview trap:** Saying fine tuning "adds knowledge" as its main purpose. SFT and preference tuning mostly change behaviour. Facts that change or that the model never saw are better served by retrieval.

## Supervised fine tuning

SFT is ordinary next token training on a curated dataset of (prompt, ideal response) pairs, with the loss counted only on the response tokens. Nothing changes algorithmically from pretraining; you swap the dataset and mask the loss.

### The objective

For a dataset of pairs $(x, y)$ with response tokens $y_1, \dots, y_T$:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}} \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})
$$

This is maximum likelihood, equivalently minimising the forward KL from the data distribution to the model, so SFT is mode covering: the model is pushed to put probability on every demonstrated response. It is also behaviour cloning in RL language: the model imitates an expert policy under teacher forcing, always conditioned on the expert's own previous tokens, never on its own mistakes.

> **Watch:** [CME295 Lecture 4, supervised fine tuning](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=3751s) and [instruction tuning](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=4161s). [CS229 Lecture 16](https://www.youtube.com/watch?v=hHC-SF3utxg&t=3874s) writes the same instruction tuning loss as the likelihood of the answer given the instruction.

### Data formats

Hugging Face TRL, the most common open library, accepts four shapes, and the shape decides where loss goes:

| Format | Example record | Default loss |
| --- | --- | --- |
| Language modelling | `{"text": "..."}` | Every token |
| Conversational | `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` | Every token unless `assistant_only_loss=True` |
| Prompt completion | `{"prompt": "...", "completion": "..."}` | Completion only (TRL default) |
| Conversational prompt completion | prompt and completion as message lists | Completion only |

Instruction tuning is SFT where the data is many tasks phrased as instructions (FLAN style collections, or human written chats); the goal is general instruction following rather than one task.

### Chat templates

A chat model never sees a list of messages. It sees one token sequence in which roles are marked by control tokens, for example `<|im_start|>user ... <|im_end|>` in ChatML style models or `[INST] ... [/INST]` in Mistral. The template lives in the tokenizer, and `apply_chat_template` renders it. Three practical rules from the Hugging Face docs: use the exact template the model was trained with (two models fine tuned from the same base can use different control tokens, and the wrong one degrades output badly); set `add_generation_prompt=True` at inference so the sequence ends with the assistant header; and do not let the tokenizer add a second BOS token when you tokenise an already rendered template. When you train a base model into a chat model you choose the template, and you must make the end of turn token the token generation stops on, or the model will ramble past its answer.

### Loss masking on the prompt

Labels for prompt tokens are set to an ignore index (`-100` in PyTorch's cross entropy and in TRL), so only response tokens contribute gradient. Reasons: the prompt is given at inference time, so learning to predict user text wastes capacity and can teach the model to write the user's side of the conversation; with long prompts and short answers the unmasked loss is dominated by prompt tokens; and for multi turn chats you usually want loss only on assistant turns. Some work trains on prompts too as a mild regulariser, so this is a default, not a law.

```python
import numpy as np

prompt = [1, 15, 27, 8, 4]   # rendered system + user turn + assistant header
answer = [61, 62, 63, 2]     # assistant reply ending with the end of turn token
input_ids = prompt + answer
labels = [-100] * len(prompt) + answer   # -100 means "no loss here"

# position t predicts token t + 1, so shift labels left by one
targets = np.array(labels[1:])
rng = np.random.default_rng(0)
logits = rng.normal(size=(len(input_ids) - 1, 100))
logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

keep = targets != -100
nll = -logp[np.arange(len(targets))[keep], targets[keep]]
print(keep.sum(), round(nll.mean(), 4))   # 4 supervised tokens, mean NLL
```

The four supervised positions are the ones that predict the three answer tokens and the end of turn token. Forgetting the end of turn token in the labels is a classic bug: the model never learns to stop.

### Data quality over quantity

Two results are worth quoting. LIMA (above): 1,000 curated examples were enough for a strong chat model. QLoRA (Dettmers et al., 2023): a 9k example dataset (OASST1) beat a 450k example subsample of FLAN v2 on chatbot quality, which the authors read as quality mattering more than size. Practical consequences: deduplicate, remove wrong or low effort answers, keep a deliberate mix of tasks and lengths, and filter for answers the model could plausibly produce. John Schulman's argument, summarised in Chip Huyen's RLHF post, is that cloning answers that rely on knowledge the model does not have teaches it to bluff, which is one mechanism for hallucination.

### Practical details

- **Learning rate.** Full fine tuning uses small rates (TRL's SFT default is 2e-5); adapters use larger ones (TRL suggests about 1e-4 for LoRA) because only new parameters train.
- **Packing.** Concatenating short examples into full length sequences removes padding waste; with packing you need attention that does not cross example boundaries, or at least you accept some contamination across examples.
- **Epochs.** TRL's default is 3 epochs; on small curated sets watch held out loss and, more importantly, held out generations, because overfitting shows up as memorised phrasing.
- **Evaluation.** Training loss falls smoothly while answer quality can get worse, so evaluate on generations, not only on loss.

### What SFT cannot do

SFT only ever says "make this response more likely". It never says "this response is worse than that one", so it cannot use negative examples, it cannot express a ranking among acceptable answers, and it is capped by the quality of the demonstrations. Because it is teacher forced, it never trains on the model's own mistakes (exposure bias). These gaps are what preference optimisation and RL fill. The comparison asymmetry is the core argument: a person who cannot write a great haiku can still tell which of two haikus is better, so comparisons extract signal beyond what demonstrations can.

## Parameter efficient fine tuning

PEFT methods freeze the pretrained weights and train a small number of new parameters. They cut optimiser memory, make checkpoints tiny, and let one base model serve many tasks. LoRA is the one interviewers expect you to derive.

### LoRA: the update

For a frozen weight $W_0 \in \mathbb{R}^{d \times k}$, LoRA (Hu et al., 2021) learns a low rank update $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ and $r \ll \min(d, k)$:

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

Only $A$ and $B$ get gradients. The forward cost of the extra path is $O(r(d + k))$ per token, small next to $O(dk)$.

### Initialisation and the alpha over r scale

$A$ starts as a small random Gaussian and $B$ starts at zero, so $\Delta W = 0$ at step zero and training begins exactly at the pretrained model. Both cannot be zero: if $A = 0$ and $B = 0$ the gradient with respect to each is zero ($\partial L / \partial B = g\,(Ax)^\top$ and $\partial L / \partial A = B^\top g\, x^\top$), so nothing would move. With $B = 0$ and random $A$, $B$ receives gradient on the first step and $A$ starts moving after that.

The $\alpha / r$ factor keeps the update's scale roughly stable when you change $r$; the paper sets $\alpha$ to the first $r$ tried and does not tune it, noting that with Adam, tuning $\alpha$ is close to tuning the learning rate. Many recipes simply set $\alpha = r$ or $\alpha = 2r$ and tune the learning rate.

### Parameter count worked example

One 4096 by 4096 projection has 16,777,216 weights. A rank 8 LoRA on it has $8 \times (4096 + 4096) = 65{,}536$, or 0.39 percent. For a Llama-shaped 7B model (32 layers, hidden size 4096, MLP width 11008, four attention projections and three MLP projections per layer), the linear layers hold about 6.48 billion weights:

| Target modules | Rank | Trainable parameters | Share of linear weights |
| --- | --- | --- | --- |
| q and v only | 8 | 4.19M | 0.065% |
| q and v only | 16 | 8.39M | 0.13% |
| All seven linear layers | 16 | 39.98M | 0.62% |
| All seven linear layers | 64 | 159.9M | 2.5% |

```python
import numpy as np

d, k, r, alpha = 4096, 4096, 8, 16
print(d * k, r * (d + k), round(100 * r * (d + k) / (d * k), 3))  # full, LoRA, percent

# merged weights give the same output as the two path forward pass
rng = np.random.default_rng(0)
d, k = 6, 5
W0 = rng.normal(size=(d, k))
A = rng.normal(size=(r, k)) * 0.01   # Gaussian init
B = np.zeros((d, r))                 # zero init, so the update starts at 0
x = rng.normal(size=k)
print(np.allclose(W0 @ x + (alpha / r) * B @ (A @ x), W0 @ x))   # True at init
B = rng.normal(size=(d, r))          # pretend training moved B
W_merged = W0 + (alpha / r) * B @ A
print(np.allclose(W0 @ x + (alpha / r) * B @ (A @ x), W_merged @ x))
```

On GPT-3 175B the LoRA paper reports up to 10,000 times fewer trainable parameters, training memory down from 1.2 TB to 350 GB, checkpoints down from about 350 GB to 35 MB (rank 4 on q and v), and about 25 percent higher training throughput than full fine tuning.

### Rank, alpha and target modules

- **Rank.** The LoRA paper found ranks as low as 1 or 2 competitive on GPT-3 for its tasks, and that the top singular directions learned at rank 8 and rank 64 overlap heavily. Higher ranks help for larger distribution shifts (new languages, code, long reasoning traces) and cost little.
- **Target modules.** Under a fixed 18M parameter budget on GPT-3, adapting $W_q$ and $W_v$ at rank 4 beat $W_q$ alone at rank 8 (73.7 versus 70.4 on WikiSQL). QLoRA later found that matching full fine tuning on large models required LoRA on all linear layers, and that with all layers adapted the rank mattered little. Current default: all linear layers.
- **Learning rate.** Usually about ten times the full fine tuning rate.

### Merging and serving

Because $W = W_0 + \frac{\alpha}{r} BA$ has the same shape as $W_0$, you can fold the update in after training and serve at exactly the base model's latency. Switching tasks means subtracting one $BA$ and adding another, or keeping adapters separate and batching requests for many adapters against one base model. Adapter methods that insert layers in series cannot be merged and add latency.

### Why a low rank update is enough

Earlier work showed that fine tuning objectives can be optimised in a random low dimensional subspace of parameters (intrinsic dimension). LoRA hypothesises the weight change during adaptation also has low intrinsic rank, and its analysis found $\Delta W$ amplifies directions already present but not emphasised in $W_0$ (amplification factor about 21.5 at rank 4) rather than repeating its top singular directions. The limit: a task that needs a lot of genuinely new information (continued pretraining on a new domain) may not fit in a small rank, so compare against full fine tuning when the shift is large.

### QLoRA

QLoRA (Dettmers et al., 2023) stores the frozen base model in 4 bits and trains LoRA adapters in 16 bits on top, with three ideas:

1. **NF4 (4 bit NormalFloat).** Pretrained weights are roughly zero mean normal, so the 16 code values are placed at quantiles of a standard normal, scaled into $[-1, 1]$, which spreads values evenly in probability mass. The construction is asymmetric (more levels on one side) so that zero is exactly representable. Weights are quantised in blocks of 64 with one absmax scale per block.
2. **Double quantisation.** The per block FP32 scales cost $32/64 = 0.5$ bits per parameter. Quantising those scales to 8 bits in blocks of 256 costs $8/64 + 32/(64 \times 256) \approx 0.127$ bits, saving 0.373 bits per parameter, about 3 GB on a 65B model.
3. **Paged optimisers.** Optimiser state uses NVIDIA unified memory and pages to CPU RAM during memory spikes (long sequences with gradient checkpointing) instead of crashing.

Computation happens in BF16: each matmul dequantises the needed block to BF16, multiplies, and adds the LoRA path. Gradients flow through the frozen quantised weights into the adapters only. The paper reports finetuning a 65B model on a single 48 GB GPU, down from more than 780 GB for 16 bit full fine tuning, and a Guanaco 65B model reaching 99.3 percent of ChatGPT's score on the Vicuna benchmark (GPT-4 as judge) after 24 hours on one GPU.

### Memory arithmetic

A useful back of envelope for full fine tuning with Adam in mixed precision: BF16 weights (2 bytes) plus BF16 gradients (2) plus FP32 master weights (4) plus Adam's two moments (4 + 4) is 16 bytes per parameter before activations. A 7B model needs about 112 GB, which is why it does not fit on one 80 GB card without sharding or offload. The same model's weights alone are 14 GB in BF16 and about 3.5 GB in 4 bits, and LoRA's optimiser state is tiny, which is why QLoRA fits on a consumer GPU.

### Adapters, prefix tuning and prompt tuning

| Method | What is trained | Mergeable | Notes |
| --- | --- | --- | --- |
| Adapters (Houlsby et al., 2019) | Small bottleneck MLPs inserted after attention and MLP blocks | No | Adds serial latency |
| Prefix tuning (Li and Liang, 2021) | Trainable key and value vectors prepended at every layer | No | Uses up context length |
| Prompt tuning (Lester et al., 2021) | Trainable embeddings prepended to the input only | No | Fewest parameters of all |
| LoRA | Low rank update to existing weight matrices | Yes | Default choice |
| QLoRA | LoRA on a 4 bit frozen base | Yes, after dequantising | Default when memory bound |

Raschka's Machine Learning Q and AI chapter on fine tuning lays out the same families with pseudocode.

## Choosing how to adapt a model

The decision depends on what is missing: knowledge, behaviour, or capability, and how much data and compute you have.

| Option | Changes weights | Good for | Weak at | Cost |
| --- | --- | --- | --- | --- |
| Prompting and few shot | No | Format, tone, simple tasks, fast iteration | Long instructions get ignored; context cost per call | Lowest setup, highest per call tokens |
| RAG | No | Fresh, private or citable facts | Teaching a style or a skill | Index plus retrieval latency |
| LoRA or QLoRA | A small adapter | Consistent format, domain style, tool use patterns, distilling a bigger model's behaviour | Large knowledge injection | Hours on one GPU |
| Full fine tuning | All weights | Large distribution shift, continued pretraining, maximum quality | Forgetting, cost, serving one model per task | Many GPUs |
| Preference tuning or RL | All or adapters | Ranking between acceptable answers, safety, reasoning with checkable rewards | Needs preference data or a reliable reward | Highest engineering effort |

A good interview answer goes in order: try prompting with a strong model and an eval set; add retrieval if the failures are missing facts; fine tune with LoRA when failures are behavioural and you have a few thousand good examples; go to full fine tuning or RL only with clear evidence and budget. Fine tuning a small model can also cut latency and cost once the behaviour is pinned down.

### Knowledge versus behaviour

Fine tuning is good at teaching how to answer and poor at teaching what is true. Schulman's argument (summarised by Chip Huyen) is that cloning answers that go beyond what the model knows teaches it to answer confidently past its knowledge, while retrieval keeps facts updatable and citable. A common combination: RAG for knowledge, a light fine tune so the model uses retrieved context and cites it in the house format.

### Catastrophic forgetting

Training on a narrow distribution moves weights that earlier abilities relied on, so performance on other tasks drops. Mitigations: mix in a slice of general or pretraining data (replay), use a lower learning rate and fewer steps, use PEFT (the frozen base limits drift, though adapters can still override behaviour), regularise toward the original weights or outputs (a KL term to the reference model plays this role in RLHF), and evaluate on a broad regression suite, not only the target task.

### The alignment tax

InstructGPT found PPO training lowered scores on public NLP benchmarks (SQuAD, DROP, HellaSwag, WMT French to English) and called this the alignment tax. Mixing pretraining gradients into PPO (PPO-ptx) largely removed the regressions without hurting labeller preference. The general lesson: any post training stage should be checked against a regression suite, and replay is the cheapest fix.

## RL foundations

Interviews for post training roles assume you can derive the policy gradient and explain why baselines and advantages exist. Everything in PPO, RLHF and GRPO is built from these pieces.

### The MDP vocabulary

A Markov decision process has states $s$, actions $a$, a transition distribution $P(s' \mid s, a)$, a reward $r(s, a)$ and a discount $\gamma \in [0, 1]$. A policy $\pi_\theta(a \mid s)$ maps states to action distributions. A trajectory is $\tau = (s_0, a_0, r_0, s_1, \dots)$, its return is $G(\tau) = \sum_t \gamma^t r_t$, and the objective is $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$.

### LLM generation as an MDP

For a language model the state is the prompt plus the tokens generated so far, the action is the next token, the transition is deterministic (append the token), and the episode ends at the end of turn token or a length limit. In RLHF and RLVR the reward is usually given once, at the end of the sequence ($r_T = R(x, y)$, all other $r_t = 0$, $\gamma = 1$), possibly plus a per token KL penalty. Because the reward arrives only at the end, you can also view the whole response as a single action in a contextual bandit, which is the view DPO and GRPO's outcome rewards effectively take. Credit assignment, deciding which tokens deserved the reward, is the central difficulty.

### Value, Q and advantage

- State value: $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$, expected return from $s$.
- Action value: $Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$.
- Advantage: $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$, how much better action $a$ is than the policy's average action in $s$. Its expectation under the policy is zero.

### The policy gradient theorem

The trajectory probability is $p_\theta(\tau) = \rho(s_0) \prod_t \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$. Using the log derivative trick $\nabla_\theta p = p \nabla_\theta \log p$:

$$
\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) G(\tau)\, d\tau = \mathbb{E}_{\tau \sim \pi_\theta}\big[ G(\tau)\, \nabla_\theta \log p_\theta(\tau) \big]
$$

and since the initial state and transition terms do not depend on $\theta$, their gradients vanish:

$$
\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

So you never need a model of the environment, only the ability to sample from the policy and score its log probabilities. Two refinements follow. First, causality: an action cannot affect rewards that came before it, so replacing $G(\tau)$ by the reward to go $\hat G_t = \sum_{t' \ge t} \gamma^{t'-t} r_{t'}$ keeps the expectation and lowers variance. Second, a baseline (next subsection). The general form, with the same expectation for every choice listed in Spinning Up, is

$$
\nabla_\theta J = \mathbb{E}\Big[\sum_t \Phi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\Big], \quad \Phi_t \in \{G(\tau),\ \hat G_t,\ \hat G_t - b(s_t),\ Q^\pi(s_t, a_t),\ A^\pi(s_t, a_t)\}
$$

### REINFORCE

REINFORCE (Williams, 1992) is the Monte Carlo estimator of that gradient: sample trajectories, compute returns, and ascend $\sum_t \hat G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$. In code the "loss" is $-\sum_t \hat G_t \log \pi_\theta(a_t \mid s_t)$ with the returns treated as constants. It is unbiased and on policy (samples must come from the current policy) and very high variance, because a single return multiplies every token's gradient.

### Baselines and variance

Subtracting any $b(s_t)$ that does not depend on the action leaves the gradient unbiased because of the expected grad log prob lemma: $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a \mid s)] = \nabla_\theta \sum_a \pi_\theta(a \mid s) = \nabla_\theta 1 = 0$. So $\mathbb{E}[b(s)\nabla \log \pi] = 0$. The variance falls because the multiplier $\hat G_t - b$ is centred: without a baseline, if all rewards are positive every sampled action gets pushed up and only the relative sizes carry information. The best simple baseline is $V^\pi(s)$, which turns the multiplier into an advantage estimate. Checked on a three action bandit:

```python
import numpy as np

rng = np.random.default_rng(0)
theta = np.array([0.2, -0.1, 0.4])      # logits of a 3 action softmax policy
R = np.array([1.0, 0.0, 2.0])           # reward of each action
p = np.exp(theta) / np.exp(theta).sum()
exact = p * (R - p @ R)                 # true gradient of expected reward

a = rng.choice(3, size=200_000, p=p)
score = np.eye(3)[a] - p                # gradient of log pi(a) wrt logits
g_plain = score * R[a][:, None]
g_base = score * (R[a] - p @ R)[:, None]   # baseline = expected reward
print(exact.round(3), g_plain.mean(0).round(3), g_base.mean(0).round(3))
print(round(g_plain.var(0).sum(), 3), round(g_base.var(0).sum(), 3))
```

Both estimators match the exact gradient (about $[-0.055, -0.291, 0.345]$), and the baseline cuts the summed variance from 0.879 to 0.237. GRPO's group mean and RLOO's leave one out mean are exactly this trick with the baseline estimated from other samples of the same prompt.

### Actor critic and GAE

An actor critic method learns a value network $V_\phi$ (the critic) alongside the policy (the actor). The one step TD residual $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ is a low variance, biased advantage estimate; the Monte Carlo $\hat G_t - V_\phi(s_t)$ is unbiased (given the baseline) but high variance. Generalised Advantage Estimation (Schulman et al., 2016) interpolates with an exponentially weighted sum:

$$
\hat A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

$\lambda = 0$ gives the TD residual and $\lambda = 1$ gives the Monte Carlo advantage. Computed backwards in one pass:

```python
import numpy as np

rewards = np.array([0.0, 0.0, 1.0])          # reward only at the last step
values = np.array([0.5, 0.6, 0.8, 0.0])      # V(s_0..s_3), terminal value 0
gamma = 1.0

def gae(lam):
    adv, running = np.zeros(3), 0.0
    for t in reversed(range(3)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]   # TD residual
        running = delta + gamma * lam * running
        adv[t] = running
    return adv

print(gae(0.0))    # equals the TD residuals
print(gae(1.0))    # equals Monte Carlo return minus V
print(gae(0.95).round(4))
```

The outputs are $[0.1, 0.2, 0.2]$, $[0.5, 0.4, 0.2]$ (return 1 minus each value) and $[0.4705, 0.39, 0.2]$. In LLM RLHF with PPO, the value head is typically a scalar head on a copy of the policy or reward model, and $\gamma = 1$ with $\lambda$ near 0.95 to 1 is common.

### On policy, off policy and importance sampling

The policy gradient expectation is under the current policy. Reusing samples from an older policy $\pi_{\text{old}}$ requires the importance weight $\rho_t = \pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$, giving the surrogate $\mathbb{E}_{\pi_{\text{old}}}[\rho_t \hat A_t]$, whose gradient at $\theta = \theta_{\text{old}}$ equals the policy gradient. Far from $\theta_{\text{old}}$ the weights explode and the estimate is useless, which is the problem PPO addresses. DPO and SFT on a fixed dataset are offline; PPO and GRPO are on policy with a few gradient steps per batch, so they are "mostly on policy".

## PPO

PPO (Schulman et al., 2017) maximises the importance weighted surrogate but removes the incentive to move the policy far from the one that collected the data, which lets you take several minibatch epochs per batch of samples.

### Trust regions

TRPO maximises $\mathbb{E}[\rho_t \hat A_t]$ subject to $\mathbb{E}_s[\mathrm{KL}(\pi_{\text{old}}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s))] \le \delta$, solved with conjugate gradients and a line search. It works but is complex. PPO gets similar stability with a first order method.

### The clipped objective

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t(\theta) \hat A_t,\ \mathrm{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon)\, \hat A_t\big)\Big], \quad \epsilon \approx 0.2
$$

Read it by the sign of the advantage. If $\hat A_t > 0$ the objective rises with $\rho_t$ until $1 + \epsilon$, then goes flat, so there is no gradient for pushing a good action further. If $\hat A_t < 0$ it rises as $\rho_t$ falls until $1 - \epsilon$, then goes flat. The min makes the objective a pessimistic lower bound on the unclipped surrogate: when the change makes things worse (ratio below $1 - \epsilon$ with positive advantage, or above $1 + \epsilon$ with negative advantage) the unclipped term is kept, so the gradient still pulls back.

```python
import numpy as np

def ppo_clip_objective(ratio, adv, eps=0.2):
    return np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)

for ratio, adv in [(1.1, 2.0), (1.5, 2.0), (0.5, 2.0), (0.7, -1.0), (1.3, -1.0)]:
    h = 1e-6
    slope = (ppo_clip_objective(ratio + h, adv) - ppo_clip_objective(ratio - h, adv)) / (2 * h)
    print(ratio, adv, round(float(ppo_clip_objective(ratio, adv)), 3), round(float(slope), 3))
```

| Ratio | Advantage | Objective | Slope in ratio | Meaning |
| --- | --- | --- | --- | --- |
| 1.1 | +2 | 2.2 | 2 | Inside the band, normal gradient |
| 1.5 | +2 | 2.4 | 0 | Already up enough, clipped |
| 0.5 | +2 | 1.0 | 2 | Good action made rarer, gradient restores it |
| 0.7 | -1 | -0.8 | 0 | Bad action already down enough, clipped |
| 1.3 | -1 | -1.3 | -1 | Bad action made likelier, gradient pushes it down |

At the first minibatch step $\rho_t = 1$ exactly, so a ratio far from 1 before any update signals a bug (mismatched log probs between the sampler and the trainer, different dropout, or a precision mismatch).

### Full loss: value and entropy

With a shared network the loss combines three terms: $L = -L^{\text{CLIP}} + c_1 (V_\phi(s_t) - V_t^{\text{target}})^2 - c_2 H[\pi_\theta(\cdot \mid s_t)]$. The value term trains the critic, and the entropy bonus discourages premature collapse. In LLM RLHF the critic is usually a separate model and the entropy bonus is often zero, with the KL penalty to the reference policy doing the regularising instead.

### The KL penalty variant

The PPO paper also tested a penalty form, $\mathbb{E}[\rho_t \hat A_t - \beta\, \mathrm{KL}(\pi_{\text{old}} \,\|\, \pi_\theta)]$, with $\beta$ adapted to hit a target KL, and found clipping worked better on its benchmarks. Do not confuse the two KLs in LLM RLHF: PPO's clip (or penalty) keeps each update close to $\pi_{\text{old}}$, the policy that sampled the batch; the RLHF KL term keeps the whole run close to $\pi_{\text{ref}}$, the frozen SFT model.

> **Watch:** [CME295 Lecture 5, PPO](https://www.youtube.com/watch?v=PmW_TMQ3l0I&t=3234s), [advantage and value function](https://www.youtube.com/watch?v=PmW_TMQ3l0I&t=3476s), and [the clip and KL penalty variants](https://www.youtube.com/watch?v=PmW_TMQ3l0I&t=3819s).

## RLHF end to end

RLHF turns human comparisons into a reward model, then optimises the policy against that reward with a penalty for drifting from the SFT model.

### Preference data

Labellers see a prompt and two or more responses sampled from the current model and pick the better one (or rank them). Comparisons are used instead of absolute scores because people agree more on relative judgements than on a 1 to 10 scale, and because judging is easier than writing. InstructGPT collected rankings of $K$ = 4 to 9 responses, giving $\binom{K}{2}$ pairs per prompt; inter labeller agreement was about 73 percent, which caps how accurate any reward model can be. Data quality issues to name: label noise, labeller drift, length preference (longer answers win), and sampling responses from a model that is not the one you will train (off distribution pairs).

> **Watch:** [CME295 Lecture 5, preference data collection](https://www.youtube.com/watch?v=PmW_TMQ3l0I&t=691s) and [RLHF overview](https://www.youtube.com/watch?v=PmW_TMQ3l0I&t=1063s).

### The Bradley Terry reward model

The Bradley Terry model says each response has a latent score and the probability one beats another is a logistic function of the score difference:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big), \qquad \mathcal{L}_{\text{RM}} = -\mathbb{E}\big[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\big]
$$

The reward model is usually the SFT model with the LM head replaced by a scalar head that reads the last token's hidden state. Only differences matter, so rewards are identified up to a constant per prompt; many pipelines normalise rewards before RL. Worked example: rewards 2.0 and 0.5 give $\sigma(1.5) = 0.818$ and loss 0.201; equal rewards give loss $\ln 2 = 0.693$. InstructGPT trained all $\binom{K}{2}$ pairs from one prompt in the same batch element, because shuffling them as independent examples made the reward model overfit, and used a 6B reward model because 175B training was unstable.

> **Watch:** [CME295 Lecture 5, Bradley Terry formulation](https://www.youtube.com/watch?v=PmW_TMQ3l0I&t=1786s).

### RL against the reward model with a KL anchor

The RLHF objective is

$$
\max_{\pi_\theta}\ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] - \beta\, \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
$$

In PPO implementations the KL enters the reward per token: $r_t = -\beta \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}$ at every token, plus $r_\phi(x, y)$ added at the last token. InstructGPT's PPO-ptx also adds $\gamma\, \mathbb{E}_{x \sim D_{\text{pretrain}}}[\log \pi_\theta(x)]$. The KL term does three jobs: it keeps the policy where the reward model was trained (the reward model is only accurate near the SFT distribution), it preserves fluency and diversity, and it makes the optimum well defined (the next section derives it in closed form).

### Four models in memory

PPO based RLHF holds four networks: the policy being trained, the frozen reference (SFT) model for the KL term, the frozen reward model, and the trained value model, often the size of the policy. Generation (sampling rollouts) usually dominates wall clock time. This is the cost DPO and GRPO attack: DPO drops the reward and value models and the sampling loop; GRPO drops the value model.

### KL estimators

The exact sequence KL is intractable, so implementations estimate it from sampled tokens. With $\rho = \pi_{\text{ref}}(y_t) / \pi_\theta(y_t)$ for tokens sampled from $\pi_\theta$:

- $k_1 = -\log \rho$: unbiased, but often negative per sample and high variance.
- $k_3 = \rho - 1 - \log \rho$: unbiased (because $\mathbb{E}_{\pi_\theta}[\rho] = 1$) and always non negative (since $\log z \le z - 1$).

DeepSeekMath uses $k_3$ in GRPO. A quick simulation with a true KL of 0.004 gave standard deviations of 0.090 for $k_1$ (negative 58 percent of the time) and 0.006 for $k_3$, both with means matching the truth.

## Reward hacking and overoptimisation

The reward model is a proxy for human preference, so optimising it hard eventually exploits its errors. This is Goodhart's law: past some point proxy reward keeps rising while true quality falls.

### What it looks like

Classic symptoms: longer and longer answers (length correlates with preference in the data), sycophancy (agreeing with the user), confident tone without content, list formatting everywhere, repeated phrases the reward model likes, and in RLVR, gaming the checker (printing a guess in every format the extractor accepts, or exploiting weak unit tests).

### Scaling laws for overoptimisation

Gao, Schulman and Hilton (2022) used a large "gold" reward model to label data for smaller proxy reward models, then optimised against the proxies and measured gold reward as a function of $d = \sqrt{\mathrm{KL}(\pi \,\|\, \pi_{\text{init}})}$:

$$
R_{\text{BoN}}(d) = d(\alpha_{\text{BoN}} - \beta_{\text{BoN}} d), \qquad R_{\text{RL}}(d) = d(\alpha_{\text{RL}} - \beta_{\text{RL}} \log d)
$$

Gold reward rises, peaks, and falls, and the coefficients improve smoothly with reward model size. They found the KL penalty in their RL setup acted like early stopping rather than moving the gold reward frontier, and that larger policies gained less from optimisation but did not overoptimise faster.

### Best of N and its KL

Best of N sampling (draw $N$, keep the highest reward) is the simplest optimiser against a reward model, and its divergence from the base policy has a closed form, $\mathrm{KL}_{\text{BoN}} = \log N - \frac{N - 1}{N}$ nats: 0.64 for $N = 4$, 1.84 for $N = 16$, 3.17 for $N = 64$. It grows only logarithmically in $N$, which is why best of N is a strong, hard to hack baseline. (Derivation: if the reward has a continuous distribution, the selected sample's quantile has density $N u^{N-1}$ on $[0, 1]$, and the KL to uniform is $\log N + (N - 1)\mathbb{E}[\log U] = \log N - (N - 1)/N$; a numerical integral gives 1.8351 for $N = 16$.)

### Mitigations

- Keep the KL penalty, and early stop on a held out gold signal (human eval or a stronger judge), not on proxy reward.
- Ensemble reward models, or penalise reward uncertainty.
- Refresh the reward model on samples from the current policy (iterated RLHF, as in DeepSeekMath's iterative GRPO).
- Control length: length normalised rewards, length penalties, or length controlled win rates in evaluation.
- For rule based rewards, harden the checker (strict answer format, hidden tests) and audit high reward samples by hand.

> **Watch:** [CME295 Lecture 5, challenges of RLHF](https://www.youtube.com/watch?v=PmW_TMQ3l0I&t=4580s) and [best of N](https://www.youtube.com/watch?v=PmW_TMQ3l0I&t=4963s).

## DPO

Direct Preference Optimization (Rafailov et al., 2023) shows that the KL constrained RLHF objective has a closed form optimum, that the reward can be rewritten in terms of that optimal policy, and that plugging this into the Bradley Terry likelihood gives a simple classification loss on preference pairs. No reward model, no sampling during training, no value network.

### Step 1: the optimal policy in closed form

For a fixed prompt $x$, maximise $\mathbb{E}_{y \sim \pi}[r(x, y)] - \beta\, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ over distributions $\pi(\cdot \mid x)$. Rewrite the objective as

$$
-\beta\, \mathbb{E}_{y \sim \pi}\Big[\log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x) \exp(r(x, y)/\beta)}\Big] = -\beta\, \mathrm{KL}\big(\pi \,\|\, \pi^*\big) + \beta \log Z(x)
$$

where

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x) \exp\big(r(x, y)/\beta\big), \qquad Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\big(r(x, y)/\beta\big)
$$

$Z(x)$ does not depend on $\pi$ and the KL is minimised (at zero) by $\pi = \pi^*$. The optimum reweights the reference policy by exponentiated reward: small $\beta$ means sharper reweighting. Numerically, maximising this objective over a five outcome distribution with BFGS recovered $\pi^*$ to within $10^{-5}$.

### Step 2: invert it to express the reward

Take logs and rearrange:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Any reward can be written as $\beta$ times a log ratio of its optimal policy to the reference, plus a term that depends only on the prompt. $Z(x)$ is a sum over all possible responses, so it cannot be computed, which is why RLHF needs RL in the first place.

### Step 3: plug into Bradley Terry, and Z cancels

Bradley Terry only uses the difference $r(x, y_w) - r(x, y_l)$. Both responses share the prompt, so $\beta \log Z(x)$ cancels:

$$
P(y_w \succ y_l \mid x) = \sigma\Big(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)
$$

Parameterise $\pi^*$ by the policy $\pi_\theta$ and fit by maximum likelihood on the preference data:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)\Big]
$$

The quantity $\hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is the implicit reward; the paper's title says the language model "is secretly a reward model". The paper also proves that rewards differing by a function of $x$ alone induce the same preferences and the same optimal policy, so nothing is lost by this parameterisation, and that the same substitution works for Plackett Luce rankings over more than two responses.

> **Watch:** [CME295 Lecture 5, DPO](https://www.youtube.com/watch?v=PmW_TMQ3l0I&t=5387s).

### Worked example

Sequence log probabilities are sums of token log probabilities. Suppose $\log \pi_\theta(y_w) = -12$, $\log \pi_{\text{ref}}(y_w) = -13$, $\log \pi_\theta(y_l) = -15$, $\log \pi_{\text{ref}}(y_l) = -14$ and $\beta = 0.1$. The chosen log ratio is $+1$, the rejected is $-1$, the margin is $0.1 \times 2 = 0.2$, and the loss is $-\log \sigma(0.2) = 0.598$. At initialisation $\pi_\theta = \pi_{\text{ref}}$, so every margin is 0 and the loss is exactly $\ln 2 = 0.693$; a training curve that does not start there means the reference log probs are wrong. With $\beta = 0.5$ and a log ratio difference of 1, the loss is 0.474.

### The gradient and its weighting

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \mathbb{E}\Big[\sigma\big(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w)\big)\big(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\big)\Big]
$$

It raises the chosen response's likelihood and lowers the rejected one's, weighted by how wrong the implicit reward currently is: pairs the model already ranks correctly by a wide margin get almost no weight. The paper reports that dropping this weight (plain unlikelihood training on rejected responses) makes the model degenerate. A finite difference check on a toy softmax policy over six responses:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Bradley Terry reward model loss for one pair
r_w, r_l = 2.0, 0.5
print(round(-np.log(sigmoid(r_w - r_l)), 4))   # 0.2014

# DPO on a toy policy: softmax over 6 candidate responses
rng = np.random.default_rng(1)
theta, ref_logits, beta, w, l = rng.normal(size=6), rng.normal(size=6), 0.1, 1, 4

def logp(t):
    return t - np.log(np.exp(t).sum())

def dpo_loss(t):
    lp, lr = logp(t), logp(ref_logits)
    h = beta * ((lp[w] - lr[w]) - (lp[l] - lr[l]))
    return -np.log(sigmoid(h))

def dpo_grad(t):
    lp, lr = logp(t), logp(ref_logits)
    p = np.exp(lp)
    h = beta * ((lp[w] - lr[w]) - (lp[l] - lr[l]))
    grad_logp_w = np.eye(6)[w] - p   # gradient of log softmax
    grad_logp_l = np.eye(6)[l] - p
    return -beta * sigmoid(-h) * (grad_logp_w - grad_logp_l)

eps = 1e-6
fd = np.array([(dpo_loss(theta + eps * e) - dpo_loss(theta - eps * e)) / (2 * eps)
               for e in np.eye(6)])
print(np.abs(dpo_grad(theta) - fd).max() < 1e-8)   # True
```

The analytic and finite difference gradients agree to about $10^{-11}$.

### What beta does

$\beta$ is the KL strength from the original objective. Small $\beta$ allows the policy to move far from the reference to fit preferences; large $\beta$ keeps it close. Values around 0.1 are the usual starting point (the paper swept 0.05 to 5 on sentiment). In the loss, $\beta$ also scales the margin inside the sigmoid, so it changes how quickly pairs saturate.

### DPO versus PPO

| | PPO based RLHF | DPO |
| --- | --- | --- |
| Models in memory | Policy, reference, reward, value | Policy, reference (log probs can be precomputed) |
| Data | Prompts plus a reward model; samples generated online | Fixed preference pairs (offline) |
| Stability | Many hyperparameters, sensitive | Supervised style training, stable |
| Exploration | Learns from its own samples | Only learns from the pairs given |
| Reward reuse | Reward model can score anything, including new samples | No explicit reward model |
| Results reported | Strong at scale (InstructGPT, ChatGPT) | Matched or beat PPO in the paper: about 61 percent GPT-4 judged win rate against reference summaries on TL;DR versus about 57 percent for PPO |

The trade off interviewers want: DPO is cheaper and simpler and works well when you have good preference pairs close to your model's distribution; online methods win when you can generate fresh samples and score them reliably, because the model then learns from its own mistakes. The usual fix for DPO's offline limitation is to make it iterative: generate new pairs from the current policy each round, label them, and retrain.

### Failure modes

- **Off distribution pairs.** If the pairs were sampled from a different model, the implicit reward is fitted where the policy puts little mass; running SFT on chosen responses first helps.
- **Both likelihoods fall.** The loss only depends on the margin, so it can be minimised by pushing the rejected log prob down faster than the chosen one, while the chosen response's absolute probability also falls. Watch both log probs in training logs.
- **Overfitting deterministic preferences.** If every pair is labelled the same way, the Bradley Terry optimum needs an infinite reward gap, and the KL term no longer restrains the policy (the IPO paper's argument).
- **Length exploitation.** Preferred responses are often longer, and DPO learns that.

## DPO variants

Each variant fixes one weakness of DPO. You should know the problem each solves and the shape of its loss.

| Method | Fixes | Loss idea | Needs reference model | Data |
| --- | --- | --- | --- | --- |
| IPO (Azar et al., 2023) | Overfitting when preferences are near deterministic | Squared loss pulling the log ratio margin to a finite target $1/(2\tau)$ | Yes | Pairs |
| KTO (Ethayarajh et al., 2024) | Needs pairs; ignores asymmetric value of gains and losses | Prospect theory value of each output relative to a KL reference point | Yes | Single outputs labelled good or bad |
| ORPO (Hong et al., 2024) | Separate SFT stage and reference model | SFT loss plus a log odds ratio penalty | No | Pairs |
| SimPO (Meng et al., 2024) | Reward that does not match how generation scores text; length exploitation | Length normalised log likelihood as reward, with a target margin | No | Pairs |

### IPO

With $h_\pi(y_w, y_l, x) = \log \frac{\pi(y_w \mid x)\, \pi_{\text{ref}}(y_l \mid x)}{\pi(y_l \mid x)\, \pi_{\text{ref}}(y_w \mid x)}$, IPO minimises $\mathbb{E}[(h_\pi - \frac{1}{2\tau})^2]$. A squared loss has a finite target, so the margin stops growing once it reaches $1/(2\tau)$, and the policy stays regularised toward the reference even when every label agrees. It is the identity mapping special case of the paper's general $\Psi$PO objective.

### KTO

KTO uses the implied reward $r_\theta = \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ and a reference point $z_0$, a batch estimate of the KL between policy and reference (computed from mismatched prompt and response pairs in the microbatch, clamped at zero, with no gradient). Desirable outputs score $\lambda_D \sigma(\beta(r_\theta - z_0))$, undesirable ones $\lambda_U \sigma(\beta(z_0 - r_\theta))$, and the loss is $\mathbb{E}[\lambda_y - v(x, y)]$. Its practical selling point is data: thumbs up and thumbs down logs are far easier to collect than matched pairs. The paper reports matching or beating DPO from 1B to 30B parameters, with a default $\beta$ of 0.1 and $\lambda_D n_D / (\lambda_U n_U)$ in $[1, 3/2]$.

### ORPO

ORPO adds an odds ratio term to plain SFT: $\mathcal{L} = \mathcal{L}_{\text{SFT}} + \lambda\, \mathcal{L}_{\text{OR}}$ with $\mathcal{L}_{\text{OR}} = -\log \sigma\big(\log \frac{\text{odds}(y_w \mid x)}{\text{odds}(y_l \mid x)}\big)$ and $\text{odds}(y \mid x) = \frac{P(y \mid x)}{1 - P(y \mid x)}$, where $P$ is the length normalised likelihood. One stage, no reference model, so half the forward passes of DPO. The paper used $\lambda$ between 0.1 and 0.25.

### SimPO

SimPO's reward is the average token log probability, $\frac{\beta}{|y|} \log \pi_\theta(y \mid x)$, which is what sampling and ranking actually use, and the loss adds a target margin $\gamma$: $-\log \sigma\big(\frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma\big)$. The authors found only about half of DPO training triples were ranked correctly by average log likelihood, which motivated the change, and report gains over DPO of up to 6.4 points on AlpacaEval 2 and 7.5 on Arena Hard, with $\beta$ of 2.0 to 2.5 and $\gamma$ of 0.5 to 1.5 working well.

> **Interview trap:** Calling these "better DPO". Each trades something: reference free methods lose the explicit anchor to the starting model, and all offline methods still only learn from the data you give them.

## RLAIF and Constitutional AI

RLAIF replaces human preference labels with labels from a model. Constitutional AI (Bai et al., 2022) is the best known version and has two phases:

1. **Supervised phase.** Sample responses to harmful prompts, ask the model to critique its own response against a written principle from the constitution and then revise it, and fine tune on the revisions.
2. **RL phase.** Sample pairs of responses from the fine tuned model, ask a model which better satisfies a principle, train a preference model on those AI labels, and run RL against it.

The claims: a harmless assistant that engages with difficult requests and explains objections instead of refusing evasively, with far fewer human labels, and chain of thought in the judging step improving label quality and transparency. Humans still write the constitution and still supply helpfulness data in the original paper. The risks are the judge's own biases (position bias, preference for its own style, length bias) being baked into the reward, and less independent signal: a model judging its own outputs can miss its own blind spots. AI judges and rubric based graders are now common for tasks without a checkable answer; the safe practice is to calibrate them against a smaller set of human labels.

> **Watch:** [CS329A Part 4, learning from feedback with tools and code](https://www.youtube.com/watch?v=Lxh9RF5S-K0), which covers Constitutional AI alongside execution feedback (RLEF).

## GRPO

Group Relative Policy Optimization was introduced in the DeepSeekMath paper (Shao et al., 2024) as a PPO variant that drops the value network. For each prompt it samples a group of responses and uses the group's own reward statistics as the baseline.

### The objective

For a question $q$, sample $G$ outputs $\{o_1, \dots, o_G\}$ from $\pi_{\theta_{\text{old}}}$ and maximise

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big\{\min\big(\rho_{i,t} \hat A_{i,t},\ \mathrm{clip}(\rho_{i,t}, 1 - \epsilon, 1 + \epsilon) \hat A_{i,t}\big) - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \,\|\, \pi_{\text{ref}}]\Big\}\Bigg]
$$

with $\rho_{i,t} = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})$. Three differences from PPO for LLMs:

1. **Advantage from the group.** With outcome rewards $r_i$, every token of output $i$ gets $\hat A_{i,t} = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}$.
2. **KL in the loss, not the reward.** The KL to the reference is added directly to the objective using the $k_3$ estimator $\frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1$, which keeps the advantage computation clean.
3. **Per response length averaging.** Each response's token terms are averaged by its own length $|o_i|$, then responses are averaged. Dr. GRPO later showed this term and the std division introduce biases.

For process rewards (a reward per reasoning step), DeepSeekMath normalises step rewards across the group and gives each token the sum of normalised rewards of the steps at or after it. Its reported settings: group size 64, policy learning rate $10^{-6}$, KL coefficient 0.04, max length 1024, batch size 1024, one policy update per exploration stage.

> **Watch:** [CME295 Lecture 6, GRPO](https://www.youtube.com/watch?v=k5Fh-UgTuCo&t=3464s) and [GRPO compared with PPO](https://www.youtube.com/watch?v=k5Fh-UgTuCo&t=3963s); [CS329A Part 6](https://www.youtube.com/watch?v=yVnmHSAy3ck) covers DeepSeekMath and DAPO.

### A worked group

Four samples for one math question, one correct: $\mathbf{r} = [1, 0, 0, 0]$. The mean is 0.25. The standard deviation is 0.5 with the sample (Bessel corrected) formula, which is PyTorch's default `std`, or 0.433 with the population formula; implementations differ, so say which. With the sample std the advantages are $[1.5, -0.5, -0.5, -0.5]$; with the population std, $[1.732, -0.577, -0.577, -0.577]$. They sum to zero either way: the correct answer is pushed up, the three wrong ones down.

```python
import numpy as np

rewards = np.array([1.0, 0.0, 0.0, 0.0])   # G = 4 samples, one correct
lengths = np.array([200, 120, 300, 600])   # tokens per response

mean, std = rewards.mean(), rewards.std(ddof=1)   # torch.std default is the sample std
adv_grpo = (rewards - mean) / (std + 1e-4)
adv_dr = rewards - mean
print(adv_grpo.round(3), adv_dr)

# per token weight each response's tokens receive in the loss
print((adv_grpo / lengths).round(5))    # GRPO: divide by own length
print((adv_dr / 1024).round(5))         # Dr. GRPO: divide by a constant budget

# Dr. GRPO advantage equals leave one out (RLOO) up to G / (G - 1)
G = len(rewards)
rloo = np.array([rewards[i] - np.delete(rewards, i).mean() for i in range(G)])
print(np.allclose(rloo, adv_dr * G / (G - 1)))
```

The per token weights under GRPO are $[0.0075, -0.00417, -0.00167, -0.00083]$: the 600 token wrong answer is penalised per token five times less than the 120 token wrong answer. Under Dr. GRPO every wrong token gets the same $-0.00024$.

### Why no value network

PPO's critic must predict expected reward from every prefix of a reasoning chain, which is hard to learn, costs a model the size of the policy, and with a reward only at the end mostly propagates that final reward backwards anyway. Sampling several answers per prompt gives a direct Monte Carlo estimate of the prompt's expected reward, which serves as the baseline. The group mean includes the sample itself, but $r_i - \mathrm{mean}(\mathbf{r}) = \frac{G-1}{G}\big(r_i - \mathrm{mean}_{j \ne i}(r_j)\big)$, so it is the leave one out baseline (independent of sample $i$, hence unbiased) times a constant. The cost moves from memory to generation: you need $G$ rollouts per prompt.

### Why it suits verifiable rewards

With a binary correct or incorrect reward, the group mean is the model's empirical pass rate on that prompt, so the advantage says "better or worse than your own average on this exact question". That is a well calibrated, per prompt signal that needs no learned reward model and no critic. Rule based rewards are also cheap to compute for many samples and are harder to hack than a neural reward model, though not impossible to hack.

### Groups with no signal

If all $G$ samples are correct or all are wrong, every advantage is zero and the prompt contributes no policy gradient (only the KL term acts). With the std in the denominator this is also a division by zero, guarded by a small epsilon. As training progresses more prompts become all correct, so the effective batch shrinks; DAPO's dynamic sampling filters such prompts out and keeps sampling until the batch is full of informative groups. Curriculum (choosing prompts the model solves sometimes) does the same job upstream.

### GRPO versus PPO

| | PPO for LLMs | GRPO |
| --- | --- | --- |
| Baseline | Learned value network, GAE | Mean reward of the group |
| Models trained | Policy and value | Policy only |
| Samples per prompt | Often 1 | $G$ (DeepSeekMath used 64; TRL's default is 8) |
| KL to reference | Per token penalty inside the reward | Separate loss term ($k_3$), often set to zero in 2025 recipes |
| Credit assignment | Per token advantages from the critic | Same advantage on every token of a response (outcome reward) |
| Best fit | Dense or learned rewards, long horizons | Sparse, verifiable rewards on prompts with mixed success |

## Dr. GRPO

"Understanding R1-Zero-Like Training: A Critical Perspective" (Liu et al., 2025) argues that GRPO has an optimisation bias that inflates response length, especially for wrong answers, and proposes Dr. GRPO ("GRPO done right"), which removes two normalisation terms. If you replicated this paper, be exact about what each term does.

> **Watch:** [CME295 Lecture 6, length bias](https://www.youtube.com/watch?v=k5Fh-UgTuCo&t=4574s) and [DAPO and Dr. GRPO](https://www.youtube.com/watch?v=k5Fh-UgTuCo&t=5129s).

### Bias 1: dividing by the response length

In GRPO, each token of response $i$ carries weight $\hat A_i / |o_i|$, so each response's total weight is $\hat A_i$ regardless of length.

- **Correct responses** ($\hat A_i > 0$): short ones get a larger per token push, so the policy is pulled toward short correct answers.
- **Incorrect responses** ($\hat A_i < 0$): long ones get a smaller per token penalty, so being wrong at length is punished less per token than being wrong briefly. The policy drifts toward longer incorrect answers.

The paper's reading: part of the steady growth in response length during R1-Zero-style training is this artefact, not emerging reasoning. It shows that with Dr. GRPO, lengths of correct responses stay similar while incorrect responses become much shorter, improving token efficiency at similar accuracy.

### Bias 2: dividing by the group's reward std

Dividing by $\mathrm{std}(\mathbf{r})$ rescales each question's update by $1/\mathrm{std}$. With binary rewards the std is smallest when nearly all samples agree, so questions that are almost always solved or almost never solved get their advantages multiplied up relative to questions the model solves half the time. With $G = 16$ and the sample std:

| Correct out of 16 | Std | Advantage of a correct sample, GRPO | Same, Dr. GRPO | Scale factor $1/\mathrm{std}$ |
| --- | --- | --- | --- | --- |
| 1 | 0.250 | 3.75 | 0.9375 | 4.0 |
| 4 | 0.447 | 1.68 | 0.75 | 2.24 |
| 8 | 0.516 | 0.97 | 0.50 | 1.94 |
| 15 | 0.250 | 0.25 | 0.0625 | 4.0 |

The paper calls this a question level difficulty bias: the objective weights questions differently depending on how consistent their rewards are, rather than treating each question's policy gradient on equal terms.

### The fix

Dr. GRPO drops both terms:

$$
\tilde A_{i,t} = r_i - \mathrm{mean}(\mathbf{r}), \qquad \mathcal{J}_{\text{Dr.GRPO}} = \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min\big(\rho_{i,t} \tilde A_{i,t},\ \mathrm{clip}(\rho_{i,t}, 1 - \epsilon, 1 + \epsilon) \tilde A_{i,t}\big)\Big]
$$

In code the per sequence masked mean is replaced by a masked sum divided by a constant, `(loss * mask).sum(-1) / MAX_TOKENS`, where the constant is the generation budget and stays fixed for the whole run, so it only rescales the learning rate. The paper notes the resulting advantage equals the RLOO (leave one out) estimator up to the factor $G/(G-1)$, which the code above confirms. It also points out that popular open PPO implementations (trl, OpenRLHF, verl and others) had the same per response length normalisation, which does not appear in the original PPO objective; they speculate it came from pretraining code, where averaging over a fixed context is harmless.

### Loss aggregation, three ways

| Aggregation | Per token weight | Effect on length |
| --- | --- | --- |
| GRPO: mean over each response, then mean over responses | $\hat A_i / (G\,\vert o_i\vert )$ | Favours short correct and long incorrect responses |
| DAPO: mean over all tokens in the batch | $\hat A_i / \sum_j \vert o_j\vert$ | Every token in the batch equal, but the denominator still depends on this batch's lengths |
| Dr. GRPO: sum over tokens divided by a constant | $\tilde A_i / (G \cdot L_{\max})$ | Every token equal across batches; matches the unnormalised policy gradient up to a constant |

TRL's documentation (2026) describes DAPO's token level loss as reducing the length bias without fully removing it and Dr. GRPO's constant as removing it, exposes both as `loss_type` options ("dapo" is the current default), and lets you turn the std scaling off with `scale_rewards`.

### What the paper claimed

- **Recipe.** A minimal R1-Zero-style run with Dr. GRPO on Qwen2.5-Math-7B, using MATH questions at levels 3 to 5 and 8 A100 GPUs, produced Oat-Zero-7B at 43.3 percent on AIME 2024, which the authors reported as a new state of the art for 7B models at the time.
- **Base models already reason.** Qwen2.5 models answer well with no prompt template at all, and Qwen2.5-Math-7B averaged 38.2 across five benchmarks with no template versus 23.8 with a 4 shot prompt; the authors suspect pretraining on concatenated question and answer text.
- **The aha moment predates RL.** Base models including DeepSeek-V3-Base already produce self reflection words like "wait" and "aha", and in their analysis more self reflection in R1-Zero outputs did not correlate with higher accuracy.
- **Pretraining sets the ceiling.** On Llama-3.2-3B, RL alone moved the five benchmark average from 3.3 to 6.8, while math continued pretraining before RL reached 14.8 (FineMath) and 20.7 (NuminaQA).
- **Templates and question sets matter.** A template that mismatches the base model can destroy its reasoning before RL restores it, which looks like RL "teaching" reasoning.

### DAPO

DAPO (Yu et al., 2025), the other common GRPO fix, combines four changes: Clip-Higher (asymmetric clip, $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$, so low probability tokens can grow and entropy does not collapse), dynamic sampling (drop all correct or all wrong groups), token level loss aggregation, and overlong reward shaping (a soft penalty, or a masked loss, for truncated responses). It also removes the KL penalty, arguing that long chain of thought training should be free to move far from the initial model. It reports 50 on AIME 2024 with Qwen2.5-32B, above the 47 of DeepSeek-R1-Zero-Qwen-32B, in half the training steps.

## Verifiable rewards and DeepSeek R1

Reinforcement learning with verifiable rewards (RLVR) uses a program, not a learned model, to score outputs: exact match on a math answer, unit tests for code, a format check. DeepSeek R1 (DeepSeek-AI, 2025) made it the default recipe for reasoning models.

### The R1-Zero recipe

- Start from the base model (DeepSeek-V3-Base) with no SFT.
- Prompt with a template that asks for reasoning inside `<think>` tags and the answer inside `<answer>` tags.
- Reward = accuracy (rule checked final answer, or compiled and tested code) plus format (tags present). No neural reward model, because the authors found neural reward models prone to reward hacking during large scale RL and costly to retrain.
- Optimise with GRPO.

AIME 2024 pass@1 rose from 15.6 percent to 71.0 percent over training, and 86.7 percent with majority voting over 64 samples (the first arXiv version; the current arXiv revision reports 77.9 percent pass@1 for R1-Zero). Average response length grew from hundreds to thousands of tokens without being asked to, and an intermediate checkpoint showed an "aha moment" where it paused and re-evaluated its approach. Drawbacks: poor readability and language mixing in the chain of thought.

### The full R1 pipeline

1. **Cold start SFT** on thousands of long, readable chain of thought examples.
2. **Reasoning RL** with GRPO and an added language consistency reward (share of target language words in the chain of thought).
3. **Rejection sampling and SFT**: sample from the RL checkpoint, keep good outputs, and fine tune on about 800k examples (about 600k reasoning, 200k non reasoning such as writing and factual QA).
4. **A second RL stage** over all scenarios: rule rewards for reasoning, reward models for helpfulness and harmlessness.

DeepSeek-R1 reports 79.8 percent pass@1 on AIME 2024 (OpenAI o1-1217: 79.2) and 97.3 percent on MATH-500.

> **Watch:** [CME295 Lecture 6, scaling with RL](https://www.youtube.com/watch?v=k5Fh-UgTuCo&t=2887s) and [the DeepSeek R1 recipe](https://www.youtube.com/watch?v=k5Fh-UgTuCo&t=5378s).

### Does RL create new abilities?

The evidence points toward RL mostly sharpening what the base model can already do. DeepSeekMath found RL improved Maj@K but not Pass@K on GSM8K and MATH, which the authors read as RL making correct answers already in the model's top K more likely rather than adding fundamental capability. CS329A's train time scaling lecture ends on the same open question. Dr. GRPO's base model findings (reasoning and self reflection already present before RL) point the same way. Counterpoint for an interview: at large scale and with long training, RL clearly changes behaviour that matters (longer search, self verification, much higher pass@1), and pass@1 is what users experience.

## Test time compute

Test time compute scaling spends more inference computation per question (longer reasoning, more samples, verification, search) to get better answers from a fixed model. It is the inference side counterpart of pretraining scaling laws, and the two trade against each other.

> **Watch:** [CS329A Part 2, test time compute scaling](https://www.youtube.com/watch?v=-Ggc37xLj_Y) and its discussion of [the generation verification gap](https://www.youtube.com/watch?v=-Ggc37xLj_Y&t=1064s).

### Chain of thought

A transformer does a fixed amount of computation per token, so a hard problem answered in one token gets one forward pass of thought. Writing intermediate steps gives the model more serial computation and a scratchpad it can condition on. Karpathy's advice in State of GPT is to spread reasoning across more tokens. Reasoning models internalise this: RL rewards whatever chain of thought leads to correct answers, and the chains get long. Two caveats: the written chain is not guaranteed to be the computation that produced the answer (faithfulness), and longer is not always better (overthinking on easy problems wastes tokens).

### Best of N and majority voting

- **Best of N.** Sample $N$ answers, score each with a verifier or reward model, return the best. Quality depends entirely on the verifier.
- **Majority voting (self consistency).** Sample $N$ chains of thought and return the most common final answer. No verifier needed, only a way to extract and compare answers. Wang et al. (2022) report gains over single chain of thought decoding of 17.9 points on GSM8K, 11.0 on SVAMP and 12.2 on AQuA.
- **Weighted voting.** Sum verifier scores over samples that share a final answer and pick the answer with the largest total.

Majority voting helps when wrong answers are scattered and the right one is the plurality. In a simulation where the model is right 40 percent of the time and wrong answers spread evenly over six values, voting accuracy rose from 0.39 with one sample to 0.86 with 15 and 0.995 with 45. When one wrong answer is more likely than the right one (a systematic error at 45 percent versus 40 percent correct), voting converges to the wrong answer and accuracy fell to 0.36 at 45 samples. Voting amplifies whatever the plurality is.

### Repeated sampling and coverage

Large Language Monkeys (Brown et al., 2024) measured coverage, the fraction of problems solved by at least one of $k$ samples, and found it grows steadily over four orders of magnitude of samples, often log linear and well fit by an exponentiated power law $c \approx \exp(a k^b)$. On SWE-bench Lite, DeepSeek-Coder-V2-Instruct went from 15.9 percent with one sample to 56 percent with 250, above the 43 percent single sample state of the art at the time. Where an automatic checker exists (unit tests, proofs), coverage converts into solved problems; where it does not, selection is the bottleneck.

### The generation verification gap

The same paper shows the gap. On MATH with Llama-3-8B-Instruct, coverage rose from 82.9 percent at 100 samples to 98.44 percent at 10,000, but majority voting and reward model selection stayed near 40.5 to 41.4 percent and plateaued after about 100 samples. Generating a right answer is easier than picking it. CS329A's verification lecture treats closing this gap (better verifiers, ensembles of weak verifiers such as Weaver) as the central problem. Verifiers are imperfect even for code: the paper found about 11.3 percent of SWE-bench Lite problems have flaky tests.

### Sequential revision versus parallel sampling

Parallel sampling explores different approaches independently. Sequential revision lets the model condition on its previous attempt and fix it. Snell et al. (2024) found easier questions benefit most from sequential revisions (the first attempt is close and needs polishing), while harder questions do best with a mix that includes more parallel sampling (the first approach is often wrong and needs replacing).

### Compute optimal allocation

Snell et al. define a compute optimal strategy that picks the method and its hyperparameters per prompt based on estimated difficulty (binning the base model's pass rate into five levels, using either ground truth or a verifier's predicted scores). On MATH with PaLM 2-S*, it matched best of N while using about four times less test time compute. In a FLOPs matched comparison, test time compute on a smaller model beat a model about 14 times larger on problems where the small model already had non trivial success, but pretraining was the better use of compute on the hardest questions or when inference load is high relative to training (the ratio $R$ of inference tokens to pretraining tokens is large). The interview takeaway: test time compute is not free and not universal; it helps most where the model is partly competent.

### Search over reasoning

With a process reward model scoring partial solutions, you can search over steps:

- **Beam search.** Keep the top $M$ partial solutions by PRM score, expand each with several next steps, repeat. Snell et al. found it beats best of N at low budgets and on hard questions, but on easy questions it degrades as compute grows, likely from overoptimising the PRM.
- **Lookahead search.** Score a step by rolling out further first; it underperformed at equal generation budget in that paper.
- **Tree search (MCTS, tree of thoughts).** Expand, evaluate and back up values over a tree of partial chains. Powerful in games, but the DeepSeek R1 authors report it did not scale for them (below).

> **Interview trap:** Saying test time scaling "always" beats a bigger model. The Snell result is conditional on difficulty and on the inference to training compute ratio.

## Verifiers and process rewards

A verifier scores candidate solutions. An outcome reward model (ORM) scores only the final result; a process reward model (PRM) scores each step.

### ORM versus PRM

| | ORM | PRM |
| --- | --- | --- |
| Label | Is the final answer right | Is each step right |
| Label cost | Cheap (compare to the reference answer) | Expensive (humans or rollouts per step) |
| Credit assignment | Coarse; a right answer reached by wrong steps gets full credit | Pinpoints the first bad step |
| Use | Reranking, RL reward | Reranking, guiding beam or tree search, dense RL reward |
| Risk | Rewards lucky or flawed reasoning | Step segmentation is ambiguous; PRMs can be hacked |

The first ORM work, Cobbe et al. (2021), introduced GSM8K and showed that training a verifier and reranking many samples beat simply fine tuning harder.

> **Watch:** [CS329A Part 3, verification](https://www.youtube.com/watch?v=p7TdPUcPoik), which covers GSM8K verifiers, Let's Verify Step by Step, Math-Shepherd and Weaver.

### Let's Verify Step by Step

Lightman et al. (2023) compared outcome and process supervision on MATH with models fine tuned from GPT-4. They released PRM800K: about 800k step labels (positive, negative, neutral) on about 75k solutions to about 12k problems. Using best of 1860 selection on a 500 problem test subset, the PRM solved 78.2 percent, the ORM 72.4 percent and majority voting 69.6 percent. Active learning (labelling convincing wrong solutions) was about 2.6 times more data efficient. They argue process supervision has a negative alignment tax: it is both safer (rewarding endorsed reasoning) and better.

### Scoring a solution with a PRM

A PRM outputs a probability that each step is correct. Lightman et al. score a solution as the product, the probability that every step is correct; Snell et al. used the prediction at the final step. Minimum over steps is a third common choice. For steps scored 0.95, 0.9, 0.4, 0.92, the product is 0.315, the minimum 0.4 and the last step 0.92: the last step rule would miss the bad third step entirely.

### Automatic step labels

Human step labels do not scale. Math-Shepherd (Wang et al., 2023) labels a step by rolling out several completions from it and checking how often they reach the right answer (a Monte Carlo estimate of the step's value), and Lightman et al. also used a large PRM to label data for small ones. These labels are noisy, and they conflate "this step is wrong" with "this step makes success less likely".

### Why R1 did not use a PRM or MCTS

The DeepSeek R1 paper lists both as unsuccessful attempts. PRMs: fine grained steps are hard to define in general reasoning, judging an intermediate step is hard (model labels are unreliable and human labels do not scale), and a model based PRM invites reward hacking and needs retraining. MCTS: the token level search space is vastly larger than chess, so capping expansions traps search in local optima, and the value model that guides search is hard to train well. Their conclusion was that simple outcome rewards plus RL scaled better.

## Distilling reasoning

Distillation trains a smaller student on outputs from a stronger teacher. For reasoning models the student is fine tuned (plain SFT) on the teacher's full chains of thought plus answers.

### What R1 showed

DeepSeek fine tuned Qwen2.5 (1.5B, 7B, 14B, 32B) and Llama (8B, 70B) models on the roughly 800k samples curated with DeepSeek-R1, with SFT only and no RL. DeepSeek-R1-Distill-Qwen-7B reached 55.5 percent pass@1 on AIME 2024 and Distill-Qwen-32B 72.6 percent. Running large scale R1-Zero-style RL directly on Qwen2.5-32B-Base reached only 47.0 percent. The paper's conclusion: distilling a stronger model into a small one beats RL on the small one, and small models trained with RL alone need enormous compute and may still fall short. Distillation transfers the search strategy the teacher discovered with RL; the student does not have to discover it.

### STaR and rejection sampling

STaR (Zelikman et al., 2022) is self distillation: sample rationales, keep those reaching the right answer, fine tune on them, and for failures give the answer as a hint and ask for a rationale (rationalisation). Rejection sampling fine tuning (RFT), R1's stage 3, and "SFT on filtered model outputs" are the same family. In the DeepSeekMath unified view, these are policy gradient methods whose gradient coefficient is 1 for kept samples and 0 for discarded ones: they reinforce correct answers but never push down wrong ones.

> **Watch:** [CS329A Part 6](https://www.youtube.com/watch?v=yVnmHSAy3ck&t=39s) opens with STaR before DeepSeekMath and DAPO.

### Caveats

Students copy the teacher's habits, including verbosity and errors; long traces make training and inference expensive; and if the teacher saw the benchmark, the student inherits the contamination. Licences of teacher model outputs also matter in practice.

## Evaluating reasoning models

Reasoning evaluations are small, noisy and easy to fool yourself with. The main tools are the right metric, enough samples and a careful answer checker.

### pass@k and its unbiased estimator

pass@k is the probability that at least one of $k$ samples is correct. Estimating it by drawing exactly $k$ samples is noisy, so the standard estimator (Chen et al., 2021, used by Large Language Monkeys) draws $n \ge k$ samples, counts $c$ correct, and computes

$$
\widehat{\text{pass@}k} = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}
$$

which is the probability that a random subset of $k$ of the $n$ samples contains at least one correct one. With $n = 10$, $c = 2$: pass@1 = 0.2 and pass@5 = $1 - 56/252 = 0.778$. The tempting plug in estimate $1 - (1 - c/n)^k$ gives 0.672 on the same data, and it is biased low on average (Jensen's inequality, since $(1 - \hat p)^k$ is convex in $\hat p$). In a simulation with true per sample accuracy 0.2, so true pass@5 = 0.672, the combinatorial estimator from 10 samples averaged 0.672 while the plug in averaged 0.593.

```python
from math import comb, log

def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

print(round(pass_at_k(10, 2, 1), 4), round(pass_at_k(10, 2, 5), 4))
print(round(1 - (1 - 2 / 10) ** 5, 4))      # plug in estimate, biased

def best_of_n_kl(n):
    return log(n) - (n - 1) / n               # nats, Gao et al. (2022)

print([round(best_of_n_kl(n), 3) for n in (4, 16, 64)])
```

> **Watch:** [CME295 Lecture 6, the pass@k metric](https://www.youtube.com/watch?v=k5Fh-UgTuCo&t=1924s), including how temperature trades pass@1 against pass@k.

### pass@1 with sampling, maj@k and cons@k

DeepSeek R1 reports pass@1 as the mean correctness over $k$ sampled responses per question (4 to 64, temperature 0.6, top p 0.95, up to 32,768 tokens) instead of greedy decoding, which they found more repetitive and more variable across checkpoints. maj@k or cons@k is majority voting accuracy over $k$ samples. Always state which metric, $k$, temperature, and token budget you used; a pass@64 number compared with someone else's pass@1 is meaningless. RL tends to raise pass@1 and maj@k much more than pass@k at large $k$.

### Small benchmarks

AIME 2024 has 30 problems, so one problem is 3.3 points and the standard error of an accuracy near 50 percent is about 9 points from a single pass. Average over many samples per problem, report seeds and confidence intervals, and do not claim a 2 point gain.

### Contamination

Benchmarks leak into pretraining data (GSM8K and MATH have been public for years), and contamination inflates scores silently. Defences: evaluate on problems released after the model's data cutoff (new AIME years, live coding contests), check n gram overlap with training data where you can, compare against perturbed or paraphrased versions, and be suspicious of large gains on old benchmarks with no gains on fresh ones. The Dr. GRPO finding that Qwen2.5-Math does well with no template, possibly from pretraining on question answer text, is a reminder that the base model's data shapes what RL appears to achieve.

### Answer extraction

Scores depend on parsing. Failures: the answer is not in `\boxed{}` or the expected tag, equivalent forms are marked wrong (0.5 versus 1/2, units, LaTeX variants), the checker takes the first number instead of the final one, and truncated responses at the token limit are scored as wrong (or worse, partially parsed). In RL the extractor is also the reward function, so a lenient extractor is a reward hacking target and a strict one gives false negatives. Use a symbolic equivalence checker where possible and audit a sample of disagreements by hand.

## What interviewers probe

- **"Walk me through RLHF."** Then: why a KL term; what happens with $\beta = 0$; why pairwise comparisons; what the Bradley Terry loss is; which models sit in memory; where it fails (reward hacking, labeller noise, length).
- **"Derive DPO."** Then: why $Z(x)$ cancels; what the implicit reward is; what $\beta$ does; why the chosen log prob can fall; when you would still use PPO or GRPO.
- **"Explain PPO's clip."** Then: what happens for negative advantages; which policy is "old" versus "reference"; why the ratio should be 1 at the first minibatch; GAE and $\lambda$.
- **"What is GRPO and why no critic?"** Then: compute advantages for a given group; what happens when all rewards are equal; why it fits verifiable rewards; std choice (sample or population).
- **"What did Dr. GRPO change?"** Then: which way each bias pushes length for correct and incorrect answers; why dividing by a constant is fine; the RLOO connection; what you would measure to confirm the length claim (length of incorrect responses over training, accuracy at matched tokens).
- **"LoRA math."** Then: parameter count for a given layer; why $B = 0$ at init; what $\alpha$ does; merging; QLoRA's NF4 and double quantisation; memory for full fine tuning.
- **"Fine tune or RAG?"** Then: knowledge versus behaviour; forgetting; evaluation plan.
- **"How does test time compute scale?"** Then: majority voting versus best of N; the generation verification gap; when a bigger model is better; ORM versus PRM.
- **"How would you evaluate a reasoning model?"** Then: pass@k estimator; why sample instead of greedy; contamination; answer extraction; variance on 30 problems.
- **Coding rounds** (reported in the techinterview.org and PracHub write ups): implement the DPO loss, the PPO clip objective, GRPO advantages, or a reward model evaluation harness.

## Sources

- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/html/2203.02155)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/html/2305.18290)
- [DeepSeekMath, the paper that introduced GRPO](https://arxiv.org/html/2402.03300)
- [DeepSeek-R1 (current arXiv revision)](https://arxiv.org/html/2501.12948)
- [DeepSeek-R1 (first arXiv version)](https://arxiv.org/html/2501.12948v1)
- [Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)](https://arxiv.org/html/2503.20783)
- [DAPO: an open source LLM reinforcement learning system at scale](https://arxiv.org/html/2503.14476)
- [LoRA: Low Rank Adaptation of Large Language Models](https://arxiv.org/html/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/html/2305.14314)
- [Let's Verify Step by Step](https://arxiv.org/html/2305.20050)
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/html/2407.21787)
- [Scaling LLM Test Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/html/2408.03314)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)](https://arxiv.org/html/2310.12036)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/html/2402.01306)
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/html/2403.07691)
- [SimPO: Simple Preference Optimization with a Reference Free Reward](https://arxiv.org/html/2405.14734)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Self Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [OpenAI Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [OpenAI Spinning Up: Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Chip Huyen: RLHF](https://huyenchip.com/2023/05/02/rlhf.html)
- [Hugging Face TRL: GRPO Trainer documentation](https://huggingface.co/docs/trl/grpo_trainer)
- [Hugging Face TRL: SFT Trainer documentation](https://huggingface.co/docs/trl/sft_trainer)
- [Hugging Face Transformers: chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Sebastian Raschka, Machine Learning Q and AI, chapter 18](https://sebastianraschka.com/books/ml-q-and-ai-chapters/ch18/)
- [Stanford CS329A Self Improving AI Agents syllabus](https://cs329a.stanford.edu/)
- [RLHF and post training interview topics, 2026 (techinterview.org)](https://www.techinterview.org/post/3233475333/rlhf-post-training-interview-topics-2026/)
- [LLM post training interview questions: SFT, DPO, PPO and GRPO trade offs (PracHub)](https://prachub.com/resources/llm-post-training-interview-questions-sft-dpo-ppo-and-grpo-trade-offs)
