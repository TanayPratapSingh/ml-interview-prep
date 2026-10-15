# Post training and reasoning cheatsheet

One page for the ten minutes before the interview: the formulas you may be asked to write, the numbers worth quoting (with their paper), and the one line answers to the questions that come up most.

## Pipeline in one breath

Base model (document completer) then SFT (imitate demonstrations, loss on response tokens only) then preference tuning (RLHF with reward model plus PPO, or DPO) then, for reasoning models, RL with verifiable rewards (GRPO family) plus distillation into smaller models. Knowledge comes mostly from pretraining; post training changes behaviour.

## Formulas to write from memory

| Name | Formula |
| --- | --- |
| SFT | $-\sum_t \log \pi_\theta(y_t \mid x, y_{<t})$, prompt labels set to -100 |
| LoRA | $h = W_0x + \frac{\alpha}{r}BAx$, $B \in \mathbb{R}^{d\times r}$ zero init, $A \in \mathbb{R}^{r\times k}$ Gaussian; params $r(d+k)$ |
| Policy gradient | $\nabla J = \mathbb{E}[\sum_t \hat A_t \nabla\log\pi_\theta(a_t \mid s_t)]$; baseline unbiased since $\mathbb{E}[\nabla\log\pi] = 0$ |
| GAE | $\hat A_t = \sum_l (\gamma\lambda)^l\delta_{t+l}$, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$; $\lambda=0$ TD, $\lambda=1$ Monte Carlo |
| PPO clip | $\mathbb{E}[\min(\rho_t\hat A_t, \mathrm{clip}(\rho_t, 1\pm\epsilon)\hat A_t)]$, $\rho_t = \pi_\theta/\pi_{\text{old}}$, $\epsilon = 0.2$ |
| Bradley Terry | $P(y_w \succ y_l) = \sigma(r_w - r_l)$; RM loss $-\log\sigma(r_w - r_l)$ |
| RLHF objective | $\max \mathbb{E}[r_\phi(x,y)] - \beta\,\mathrm{KL}(\pi_\theta \,\Vert \, \pi_{\text{ref}})$ |
| Optimal policy | $\pi^* = \pi_{\text{ref}}\,e^{r/\beta}/Z(x)$, so $r = \beta\log\frac{\pi^*}{\pi_{\text{ref}}} + \beta\log Z(x)$ |
| DPO | $-\log\sigma\big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)$; $Z$ cancels within a prompt |
| DPO gradient | $-\beta\,\sigma(\hat r_l - \hat r_w)(\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l))$ |
| IPO | $(h_\pi - \frac{1}{2\tau})^2$ |
| SimPO | $-\log\sigma(\frac{\beta}{\vert y_w\vert }\log\pi(y_w) - \frac{\beta}{\vert y_l\vert }\log\pi(y_l) - \gamma)$ |
| GRPO advantage | $\hat A_i = (r_i - \mathrm{mean}(\mathbf r))/\mathrm{std}(\mathbf r)$, same for every token of $o_i$ |
| GRPO loss shape | $\frac{1}{G}\sum_i \frac{1}{\vert o_i\vert }\sum_t \{\text{clip term} - \beta\,\mathrm{KL}\}$ |
| Dr. GRPO | $\tilde A_i = r_i - \mathrm{mean}(\mathbf r)$; masked sum over tokens divided by a constant (generation budget) |
| RLOO link | $r_i - \bar r = \frac{G-1}{G}(r_i - \bar r_{-i})$ |
| k3 KL | $\rho - 1 - \log\rho$, $\rho = \pi_{\text{ref}}/\pi_\theta$; unbiased, non negative |
| Best of N KL | $\log N - \frac{N-1}{N}$ (1.84 nats at $N=16$) |
| pass@k | $1 - \binom{n-c}{k}/\binom{n}{k}$ |

## The two Dr. GRPO biases, exactly

| Term in GRPO | What it does | Effect | Dr. GRPO |
| --- | --- | --- | --- |
| $1/\vert o_i\vert$ | Per token weight $\hat A_i/\vert o_i\vert$; each response's total weight is $\hat A_i$ regardless of length | Short correct answers pushed up harder; long incorrect answers penalised less per token, so wrong answers grow longer | Masked sum divided by a fixed constant |
| $1/\mathrm{std}(\mathbf r)$ | Scales each question's update by $1/\mathrm{std}$ | Near all right or all wrong questions (low std) upweighted; question difficulty bias | Dropped; advantage is $r_i - \mathrm{mean}$ |

Group of 16, binary reward: 1 correct gives std 0.25, scale 4.0; 8 correct gives std 0.516, scale 1.94. Group $[1,0,0,0]$: sample std 0.5, advantages $[1.5, -0.5, -0.5, -0.5]$; population std 0.433 gives $[1.73, -0.58, \dots]$; Dr. GRPO gives $[0.75, -0.25, \dots]$. Paper result: Oat-Zero-7B (Qwen2.5-Math-7B, MATH levels 3 to 5, 8 A100s) at 43.3 percent AIME 2024.

## Worked numbers

- LoRA, 4096 by 4096 at $r=8$: 65,536 versus 16,777,216, 0.39 percent. 7B Llama shape, all linear at $r=16$: about 40M, 0.62 percent.
- Full fine tuning memory: 16 bytes per parameter (2 weights, 2 grads, 4 master, 8 Adam), 7B is about 112 GB. QLoRA weights about 3.6 GB.
- Double quantisation: $32/64 = 0.5$ bits to $8/64 + 32/(64\cdot256) = 0.127$ bits, saving 0.373 bits per parameter, about 3 GB on 65B.
- Reward model: rewards 2.0 and 0.5 give loss 0.201; equal rewards give $\ln 2 = 0.693$.
- DPO: log ratios $+1$ and $-1$ at $\beta = 0.1$ give margin 0.2, loss 0.598; loss starts at 0.693.
- PPO, $\epsilon=0.2$: ratio 1.5 with $A=+2$ clipped (slope 0); ratio 0.5 with $A=+2$ unclipped (slope 2).
- pass@k with $n=10$, $c=2$: pass@1 0.2, pass@5 0.778; plug in gives 0.672 and is biased low.
- AIME 2024: 30 problems, 3.3 points each, about 9 point standard error per pass at 50 percent.

## Numbers worth quoting

| Claim | Source |
| --- | --- |
| 1.3B InstructGPT preferred to 175B GPT-3; 85 percent win at 175B; labeller agreement about 73 percent | InstructGPT, 2022 |
| 1,000 curated SFT examples for a strong chat model | LIMA, 2023 |
| 9k OASST1 beat 450k FLAN v2; 65B finetuned on one 48 GB GPU | QLoRA, 2023 |
| 10,000 times fewer trainable params; 350 GB to 35 MB checkpoint on GPT-3 | LoRA, 2021 |
| DPO about 61 percent versus PPO about 57 percent on TL;DR (GPT-4 judged) | DPO, 2023 |
| GRPO: $G=64$, lr 1e-6, $\beta=0.04$; RL lifts Maj@K not Pass@K | DeepSeekMath, 2024 |
| R1-Zero AIME pass@1 15.6 to 71.0, cons@64 86.7 (first arXiv version) | DeepSeek R1, 2025 |
| R1 79.8 AIME, 97.3 MATH-500; distill 32B 72.6 versus RL 32B 47.0 | DeepSeek R1, 2025 |
| DAPO 50 AIME on Qwen2.5-32B; clip 0.2 low, 0.28 high | DAPO, 2025 |
| PRM 78.2, ORM 72.4, majority 69.6 at best of 1860 | Let's Verify Step by Step, 2023 |
| SWE-bench Lite 15.9 to 56 percent with 250 samples | Large Language Monkeys, 2024 |
| Coverage 98.4 percent at 10k samples, voting about 41 percent | Large Language Monkeys, 2024 |
| Compute optimal about 4 times cheaper than best of N; can beat a 14 times larger model | Snell et al., 2024 |

## Decision table

| You have | Use |
| --- | --- |
| Good demonstrations | SFT (LoRA if memory bound) |
| Offline preference pairs | DPO, or IPO, SimPO, ORPO |
| Thumbs up and down only | KTO |
| A scorer for new samples | PPO, RLOO, GRPO, online DPO |
| A verifiable checker | GRPO style RLVR (Dr. GRPO or DAPO fixes), rejection sampling fine tuning |
| A stronger teacher | Distil its reasoning traces |
| Missing or changing facts | RAG, not fine tuning |

## If asked X, say Y

- **Why the KL term?** Reward model is only valid near the SFT distribution; KL delays hacking, keeps fluency, and gives the closed form optimum.
- **Old versus reference policy?** Old sampled this batch and sets PPO's ratio; reference is the frozen SFT model in the KL penalty.
- **Why does Z cancel in DPO?** Bradley Terry uses reward differences for the same prompt, and $\beta\log Z(x)$ depends only on the prompt.
- **DPO loss falling but quality not?** Both log probs can fall; log chosen and rejected separately.
- **Why no critic in GRPO?** Group mean is a Monte Carlo value estimate per prompt; critic is costly and hard to learn on long chains.
- **All rewards equal in a group?** Zero advantage, no gradient; filter (DAPO dynamic sampling) or pick medium difficulty prompts.
- **Does RL add abilities?** Evidence says mostly sharpens: Maj@K up, Pass@K flat; base models already reflect.
- **Why did R1 skip PRMs and MCTS?** Steps hard to define and label, PRMs get hacked; token search space too large and value model hard to train.
- **Test time compute or bigger model?** Test time wins where the model is partly competent and inference volume is low; pretraining wins on the hardest problems.
- **Fine tune or RAG for internal docs?** RAG for knowledge and citations; a light fine tune for format and grounded use of context.
- **How do you evaluate?** Held out win rates with length control, regression suite, safety both ways, sampled pass@1 with stated settings, fresh problems, never the training reward.

## Traps

- Saying PPO clipping blocks the gradient in both directions outside the band.
- Saying std normalisation upweights medium difficulty questions (it upweights easy and hard ones).
- Saying Dr. GRPO removes the group mean (it removes the std and the length division).
- Comparing pass@64 with someone else's pass@1, or a 2 point gain on 30 problems.
- Using the training reward model as the evaluation metric.
