# Post training and reasoning

A freshly pretrained language model is like a brilliant reader who has never had a conversation: it can continue any text, but it does not know it is supposed to answer you. Post training is the set of steps that turn it into a helpful assistant, and more recently into a model that reasons carefully through hard problems. This guide explains each step with everyday analogies and small worked numbers, in the same order as the technical version, so you can switch between them.

## Why post training exists

A base model has read a huge amount of text and learned to predict what comes next. Ask it "Write a poem about cheese" and it may continue with "Write a poem about bread", because lists of writing prompts are common on the internet. It is a document completer, not an assistant.

Post training teaches it the job. The classic recipe from OpenAI's InstructGPT paper has three steps:

1. **Show it examples.** People write ideal answers to thousands of prompts, and the model is trained to imitate them.
2. **Ask people to compare.** The model writes several answers to the same prompt and people rank them. A separate model, the reward model, learns to predict those rankings as a score.
3. **Practise against the scorer.** The model writes answers, the reward model scores them, and the model is nudged toward higher scores, while a leash keeps it close to where it started.

The striking result: people preferred answers from a 1.3 billion parameter InstructGPT model over the original 175 billion parameter GPT-3. Teaching the job mattered more than size.

> **Remember:** Most knowledge comes from pretraining. Post training mostly changes behaviour: format, tone, what to refuse, when to stop, and how carefully to reason.

A useful analogy is a new hire with a great education. Pretraining is the education. Post training is onboarding: here is how we answer customers, here is the format, here is what we never do.

## Supervised fine tuning

Supervised fine tuning, SFT, is training by example. You collect pairs of a prompt and a great answer, and train the model to predict the answer word by word, exactly as in pretraining. The only differences are the data and one detail: the model is graded only on the answer, not on the prompt. The prompt is given at answer time, so there is no point learning to predict it.

> **Worked example:** A prompt of five tokens and an answer of four tokens make a nine token sequence. The training labels for the five prompt positions are set to a special "ignore" value, so only the four answer tokens count. One of those four must be the "end of turn" token, or the model never learns when to stop talking.

Two practical ideas matter a lot:

- **Chat templates.** The model does not see a list of messages; it sees one long string with special marker tokens saying who is speaking. Each model family has its own markers. Use the wrong ones and quality drops sharply.
- **Quality beats quantity.** The LIMA paper made a good chatbot from only about 1,000 carefully chosen examples. The QLoRA paper found 9,000 good examples beat 450,000 mediocre ones. The model copies whatever you show it, mistakes included.

SFT has a ceiling. It can only say "do more of this". It never says "this answer is worse than that one", and it can never be better than the people who wrote the examples. Comparing is easier than creating: you may not be able to write a great haiku, but you can usually tell which of two haikus is better. That is why the later steps use comparisons.

> **Interview trap:** Saying fine tuning is the main way to add knowledge. It mostly teaches behaviour. Facts that change belong in retrieval.

## Parameter efficient fine tuning

Training every weight in a big model is expensive. Each weight needs memory for itself, its gradient, and the optimiser's bookkeeping, about 16 bytes per weight in a common setup. A 7 billion parameter model therefore needs about 112 GB just for that, more than one large GPU holds.

LoRA avoids most of this. It freezes the original weights and learns a small correction for each weight matrix, built as the product of two thin matrices. Picture a 4096 by 4096 matrix, about 16.8 million numbers. Instead of changing all of them, LoRA learns a 4096 by 8 matrix and an 8 by 4096 matrix, 65,536 numbers in total, about 0.4 percent. Their product has the right shape to add to the original.

> **Worked example:** For a 7B model, putting rank 16 LoRA on every linear layer trains about 40 million numbers, around 0.6 percent of the model. The saved adapter is a few hundred megabytes at most, and many adapters can share one base model.

Three details interviewers like:

- One of the two matrices starts at zero, so at the start the model is exactly the original. If both started at zero, neither would ever learn.
- After training you can add the correction into the original weights, so there is no slowdown when serving.
- The size of the correction, the rank, is often small, like 8 or 16, and works surprisingly well, because fine tuning mostly adjusts what the model already knows.

QLoRA goes further. It stores the frozen model in 4 bits per weight instead of 16, using a number format shaped like the bell curve that weights tend to follow, and trains the small LoRA matrices on top. It even compresses the extra scaling numbers that compression needs. The headline: fine tuning a 65 billion parameter model on one 48 GB GPU.

Older relatives exist: adapters (small extra layers inside each block), prefix tuning and prompt tuning (learned extra tokens). LoRA won because it can be merged away and adds no cost at answer time.

## Choosing how to adapt a model

Think of a ladder. Climb only when the step below clearly fails on a test set.

1. **Prompting.** Write better instructions and examples. Cheapest to try.
2. **Retrieval (RAG).** If the model lacks facts, especially changing or private ones, look them up and show them to the model. You also get citations.
3. **LoRA fine tuning.** If the model knows enough but behaves wrong, such as format, tone, or tool use, and you have a few thousand good examples.
4. **Full fine tuning or reinforcement learning.** For big changes, such as a new domain, or reasoning with answers you can check automatically.

Fine tuning has a side effect called catastrophic forgetting: train hard on legal contracts and the model may get worse at chatting or coding. Defences are mixing general data back in, training gently, freezing most weights with LoRA, and testing broadly. InstructGPT saw a version of this, an "alignment tax" on some benchmarks, and fixed most of it by mixing ordinary pretraining text into training.

## RL foundations

Reinforcement learning, RL, is learning by trial and score. An agent sees a situation, picks an action, and eventually gets a reward. Its strategy, the policy, gives probabilities for each action. The goal is a strategy with the highest expected total reward.

For a language model, the situation is the prompt plus the words written so far, the action is the next word, and the reward usually comes once at the end: for example 1 if the final math answer is right, 0 if wrong. The hard part is credit: which of the hundreds of words deserved the reward?

The core recipe, called the policy gradient, is simple to say: make the actions you took more likely if the outcome was good, less likely if it was bad. The basic algorithm is REINFORCE. Its weakness is noise. One lucky outcome pushes every action in that episode.

The fix is a baseline: judge each outcome against what you expected. If you usually score 0.7 on a question and this attempt scored 1, push it up; if it scored 0, push it down. This does not change what the method learns on average, but it removes a lot of noise.

> **Worked example:** In a three option game with rewards 1, 0 and 2, sampled estimates of the gradient matched the exact answer with or without a baseline, but subtracting the expected reward cut the noise by almost four times.

Two more terms. The **advantage** of an action is how much better it was than average in that situation. A **critic**, or value model, is a second network that predicts expected reward from any situation, so you can estimate advantages step by step. GAE is a standard way to blend the critic's quick guesses with actual outcomes.

## PPO

PPO is the workhorse RL method behind the original ChatGPT style training. Its idea is caution. Big policy updates can wreck a model, so PPO limits how far each update can move.

It compares how likely an action is under the new policy versus the policy that collected the data. This ratio starts at 1. If an action was better than expected, PPO rewards raising its probability, but only until the ratio reaches about 1.2. If it was worse, PPO rewards lowering it, but only down to about 0.8. Beyond those limits there is no extra reward for moving further, so the model has no reason to overshoot. If an update accidentally moved the wrong way, PPO still pulls it back.

> **Worked example:** With a good action (advantage 2), a ratio of 1.1 still gets a full push; at 1.5 the push stops. If a good action's ratio fell to 0.5, the push to restore it continues. Only moves in the direction the advantage wanted are capped.

Keep two anchors apart. The **old policy** is the model that produced this batch, and PPO's limit keeps each update near it. The **reference policy** is the frozen starting model, and a separate penalty keeps the whole training run near it.

## RLHF end to end

RLHF, reinforcement learning from human feedback, puts the pieces together.

First, collect comparisons. The model writes several answers per prompt and people rank them. Comparisons are used because people agree with each other more on "A is better than B" than on scores out of 10, though even then InstructGPT's labellers agreed only about 73 percent of the time.

Second, train a reward model. It reads a prompt and answer and outputs one number. The Bradley Terry model links scores to preferences: the chance A beats B depends only on the difference in their scores.

> **Worked example:** If the reward model scores the preferred answer 2.0 and the other 0.5, the gap is 1.5 and the predicted chance the preferred one wins is about 82 percent. The training loss is about 0.20. If both scores were equal, the chance would be 50 percent and the loss about 0.69.

Third, run PPO to maximise the reward model's score minus a penalty for drifting from the starting model. That penalty matters: the reward model only learned to judge normal looking answers, so without a leash the model wanders into strange outputs that fool it.

The cost is heavy. PPO based RLHF keeps four models in memory: the model being trained, a value model, the reward model and the frozen starting model.

## Reward hacking and overoptimisation

A reward model is a stand in for human judgement, and stand ins have blind spots. Push hard enough and the model learns to please the scorer instead of people. This is Goodhart's law: once a measure becomes a target, it stops being a good measure.

Classic signs: answers getting longer and longer, flattering the user, confident tone without substance, heavy formatting. With automatic checkers, the model might list several answers hoping one matches, or write code that only passes the visible tests.

A study by Gao, Schulman and Hilton measured this. As training pushed the model further from its start, the true quality first rose, then peaked and fell, while the reward model's score kept climbing. Bigger reward models delayed the peak.

Best of N sampling is a gentle alternative: generate N answers and keep the best scoring one. It moves the model's behaviour only a little, which makes it harder to hack, at the cost of N times the generation.

> **Remember:** Judge training with an independent check, such as human ratings or a stronger judge, never the reward the model was trained on.

## DPO

DPO, direct preference optimisation, skips the reward model and the reinforcement learning loop. The authors noticed that the RLHF goal has an exact best answer: take the starting model and boost each answer by how much reward it deserves. Turning that around, the reward can be written in terms of the model itself. When you compare two answers to the same prompt, a messy term that is impossible to compute appears in both and cancels out.

What is left is a simple training rule on pairs: make the preferred answer more likely and the rejected answer less likely, both measured relative to the frozen starting model, and focus on pairs the model currently gets wrong.

> **Worked example:** Suppose training has made the preferred answer a bit more likely than the starting model did, and the rejected answer a bit less likely, giving a combined score of 2. With the usual strength setting of 0.1, the margin is 0.2 and the loss is about 0.60. At the very start every margin is zero and the loss is about 0.69 for every pair, a handy bug check.

DPO is cheap and stable, like ordinary fine tuning. Its limits: it only learns from the pairs you give it and never explores, it works best when the pairs look like your model's own answers, and it can lower the probability of both answers as long as the rejected one falls faster.

## DPO variants

Each variant fixes one weakness.

| Method | What it fixes |
| --- | --- |
| IPO | DPO over fitting when the data is completely one sided; it aims for a fixed gap instead of an infinite one |
| KTO | The need for matched pairs; it learns from single answers marked good or bad, like thumbs up and down |
| ORPO | The separate fine tuning stage and the reference model; it folds both into one loss |
| SimPO | A mismatch between DPO's score and how text is generated; it uses average per word likelihood and no reference model, which also curbs long answers |

None of them explores. All are only as good as their data.

## RLAIF and Constitutional AI

Human labels are slow and expensive, so RLAIF uses a capable model as the judge. Show it two answers and some guidelines, ask which is better, and train on its choices.

Constitutional AI, from Anthropic, is the best known version. People write a short list of principles, the constitution. First the model critiques and rewrites its own answers to difficult prompts according to those principles, and is fine tuned on the rewrites. Then an AI judge compares pairs of answers using the principles, and those judgements train a reward model for reinforcement learning. The goal was an assistant that is harmless but still engages and explains rather than flatly refusing.

The risk is that the judge's quirks, such as liking longer answers or the first answer shown, get trained into the model. Check AI judges against a smaller set of human labels.

## GRPO

GRPO, from the DeepSeekMath paper, is PPO without the value model. Instead of a second network guessing how good each partial answer is, GRPO asks the model the same question several times and uses the group's own average score as the yardstick.

> **Worked example:** Four answers to a math question score 1, 0, 0, 0. The average is 0.25. Each score minus the average, divided by the spread of the scores (0.5), gives 1.5 for the correct answer and minus 0.5 for each wrong one. Every word of the correct answer gets pushed up, every word of the wrong ones pushed down, and the pushes add to zero.

This fits problems with checkable answers perfectly. The group average is simply how often the model gets this exact question right, so the signal says "better or worse than your usual on this question". No reward model or value model is needed.

One catch: if all answers in a group are right, or all wrong, everyone equals the average and the question teaches nothing. As the model improves, more questions become all right, so methods like DAPO drop those questions and sample more useful ones.

## Dr. GRPO

The "Understanding R1-Zero-Like Training" paper found two hidden biases in GRPO's arithmetic and removed them; the fixed version is called Dr. GRPO, for "GRPO done right".

**Bias one: dividing by the answer's length.** GRPO averages each answer's word level pushes over that answer's own length. Every answer gets the same total push, but a long answer spreads it thinly. For correct answers, short ones get a stronger push per word. For wrong answers, long ones get a weaker penalty per word, so rambling while wrong is punished less than being wrong briefly. Over training, wrong answers grow longer, which can look like the model is "learning to think".

> **Worked example:** One correct answer of 200 words and wrong answers of 120, 300 and 600 words. Under GRPO, each word of the 600 word wrong answer is penalised five times less than each word of the 120 word one. Under Dr. GRPO, every wrong word gets the same penalty, so the long wrong answer gets the largest total penalty.

**Bias two: dividing by the spread of scores.** GRPO divides each question's advantages by how varied its scores are. Questions where almost all answers agree have tiny spread, so their signal gets multiplied up. With 16 answers, a question with only 1 correct answer gets multiplied by 4, while a 50/50 question gets multiplied by about 1.9. Very easy and very hard questions end up weighted more than medium ones.

Dr. GRPO simply uses each score minus the group average, and divides the summed word level losses by one fixed number for the whole run. The paper reported that this kept accuracy while making wrong answers much shorter. With it they trained a 7 billion parameter model to 43.3 percent on the AIME 2024 math exam using eight GPUs. They also found base models already show the famous "aha" self checking behaviour before any reinforcement learning, and that the prompt template and pretraining data shape results a great deal.

## Verifiable rewards and DeepSeek R1

When a program can check the answer, such as comparing a math result with the solution or running unit tests, you do not need a learned reward model at all. This is reinforcement learning with verifiable rewards.

DeepSeek's R1-Zero took a pretrained base model, with no example answers at all, told it to think inside special tags before answering, and rewarded it only for correct, well formatted answers, using GRPO. Its accuracy on the AIME 2024 math exam rose from about 16 percent to 71 percent on a single attempt, and its reasoning grew from hundreds to thousands of words. It sometimes stopped, said "wait", and reconsidered, which the authors called an aha moment. The downside: its reasoning was hard to read and mixed languages.

The full R1 fixed this in four stages: a small set of clean reasoning examples to start, reasoning reinforcement learning with a reward for staying in one language, a large round of fine tuning on about 800,000 of its own best outputs plus general data, and a final reinforcement learning round for helpfulness and safety. R1 scored 79.8 percent on AIME 2024.

Does this teach new abilities, or sharpen existing ones? The evidence leans toward sharpening. DeepSeekMath found reinforcement learning improved majority vote accuracy but not the chance that some sample among many is right. Sharpening still matters: single attempt accuracy is what users feel.

## Test time compute

Instead of a bigger model, you can let the same model work harder when answering. This is test time compute.

- **Think longer.** Writing out steps gives the model more computation and a scratchpad. A model does a fixed amount of work per word, so more words means more thinking.
- **Try many times.** Sample many answers. Majority voting picks the most common final answer; best of N uses a checker or scorer to pick the best.
- **Revise.** Let the model look at its previous attempt and fix it.
- **Search.** Explore partial solutions step by step, keeping the promising ones.

> **Worked example:** A model right 40 percent of the time, with wrong answers scattered over six values, reaches about 86 percent with 15 votes and over 99 percent with 45. But if one particular wrong answer appears 45 percent of the time, voting converges on that wrong answer and accuracy drops. Voting amplifies whatever answer is most common.

Two key findings. The Large Language Monkeys study found that the chance at least one of many tries is right keeps rising with more tries, but picking the right one without a perfect checker is the bottleneck: on one math benchmark, a right answer appeared for over 98 percent of problems with 10,000 tries, yet voting picked correctly only about 41 percent of the time. And a 2024 study by Snell and colleagues found that tailoring the strategy to each question's difficulty matched a standard approach with about four times less computation, and could beat a model 14 times larger, but only on questions the small model could sometimes solve. For the hardest questions, a bigger model was still better.

## Verifiers and process rewards

A verifier scores answers. An outcome verifier looks only at the final answer. A process verifier scores each step, so it can point to where reasoning goes wrong, like a teacher marking each line of working rather than just the final number.

OpenAI's "Let's Verify Step by Step" paper collected about 800,000 human judgements of individual steps. Picking the best of 1,860 sampled solutions, the step level verifier chose correctly 78 percent of the time, versus 72 percent for the final answer verifier and 70 percent for majority voting.

> **Worked example:** A solution's steps are scored 0.95, 0.9, 0.4 and 0.92. Multiplying gives about 0.31, the lowest score is 0.4, and the last step alone says 0.92. Only the last rule misses the bad third step.

Step labels are expensive, so methods like Math-Shepherd label a step by continuing the solution from it several times and seeing how often it reaches the right answer. Interestingly, DeepSeek reported that process reward models and tree search did not work well for training R1 at scale: steps are hard to define, step labels are unreliable, and learned step scorers get gamed.

## Distilling reasoning

Distillation means a small student model learns by copying a strong teacher. For reasoning, the student is fine tuned on the teacher's complete worked solutions, including the long thinking.

DeepSeek did this with about 800,000 examples from R1 and got strong small models. The striking comparison: running full reinforcement learning on a 32 billion parameter base model reached 47 percent on AIME 2024, while simply fine tuning the same model on R1's solutions reached about 73 percent. Discovering a good reasoning strategy is expensive; copying one is cheap.

A related self teaching trick: generate many answers, keep only the correct ones, and fine tune on them. STaR adds a twist, giving the model the right answer as a hint on problems it failed and training on its explanation.

## Evaluating reasoning models

Reasoning scores are easy to fool yourself with.

- **pass@k** is the chance that at least one of k tries is right. The standard way to estimate it draws more tries than k and counts combinations, because the shortcut formula comes out too low on average.

> **Worked example:** With 10 tries and 2 correct, pass@1 is 0.2. For pass@5, there are 252 ways to pick 5 tries and 56 of them contain only wrong ones, so pass@5 is 1 minus 56/252, about 0.78. The shortcut 1 minus 0.8 to the fifth power gives 0.67.

- **Small tests are noisy.** AIME 2024 has 30 problems, so one problem is worth 3.3 points and a single run varies by about 9 points either way. Never trust a 2 point gain on it alone.
- **Contamination.** If test questions leaked into training data, a high score means memory, not skill. Test on problems released after the model's training data was collected.
- **Answer extraction.** A grader must find the final answer in a long response. It can miss answers in the wrong format, mark 0.5 and 1/2 as different, or grab the wrong number. In reinforcement learning the grader is the reward, so its mistakes get trained in.
- **State your settings.** Number of samples, randomness setting, and maximum length all change the score, so report them with every number.
