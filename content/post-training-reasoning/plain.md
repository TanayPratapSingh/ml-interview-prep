# Post training and reasoning

A freshly pretrained language model is like a brilliant reader who has never had a conversation. Pretrained means it was trained once on huge amounts of text to predict the next word. It can continue any text, but it does not know it is supposed to answer you. Post training is the set of steps that turn it into a helpful assistant. More recently, post training also turns it into a model that reasons carefully through hard problems. This guide explains each step with everyday analogies and small worked numbers. It follows the same order as the technical version, so you can switch between them.

## Why post training exists

A base model has read a huge amount of text and learned to predict what comes next. Ask it "Write a poem about cheese" and it may continue with "Write a poem about bread". That happens because lists of writing prompts are common on the internet. It is a document completer, not an assistant.

Post training teaches it the job. The classic recipe from OpenAI's InstructGPT paper has three steps.

1. **Show it examples.** People write ideal answers to thousands of prompts, and the model is trained to imitate them.
2. **Ask people to compare.** The model writes several answers to the same prompt, and people rank them. A separate model, the reward model, learns to predict those rankings as a score.
3. **Practise against the scorer.** The model writes answers and the reward model scores them. The model is nudged toward higher scores, while a leash keeps it close to where it started.

Here is the striking result. People preferred answers from a 1.3 billion parameter InstructGPT model. They rated them above the original 175 billion parameter GPT-3 (generative pre trained transformer 3). A parameter is one learned number inside a model. Teaching the job mattered more than size.

> **Remember:** Most knowledge comes from pretraining. Post training mostly changes behaviour: format, tone, what to refuse, when to stop, and how carefully to reason.

A useful analogy is a new hire with a great education. Pretraining is the education. Post training is onboarding: here is how we answer customers, here is the format, here is what we never do.

## Supervised fine tuning

Supervised fine tuning, SFT, is training by example. Fine tuning means training an existing model further. Supervised means each example comes with the right answer. You collect pairs of a prompt and a great answer. You then train the model to predict the answer word by word, exactly as in pretraining. There are two differences: the data, and one detail. The model is graded only on the answer, not on the prompt. The prompt is given at answer time, so there is no point learning to predict it.

> **Worked example:** A prompt of five tokens and an answer of four tokens make a nine token sequence. A token is a small chunk of text. The training labels for the five prompt positions are set to a special "ignore" value, so only the four answer tokens count. One of those four must be the "end of turn" token. Otherwise the model never learns when to stop talking.

Two practical ideas matter a lot.

- **Chat templates.** The model does not see a list of messages. It sees one long string with special marker tokens saying who is speaking. Each model family has its own markers. Use the wrong ones and quality drops sharply.
- **Quality beats quantity.** The LIMA paper made a good chatbot from only about 1,000 carefully chosen examples. LIMA stands for "less is more for alignment". The QLoRA (quantised low rank adaptation) paper found 9,000 good examples beat 450,000 mediocre ones. The model copies whatever you show it, mistakes included.

SFT has a ceiling. It can only say "do more of this". It never says "this answer is worse than that one". It can also never be better than the people who wrote the examples. Comparing is easier than creating. You may not be able to write a great haiku, but you can usually tell which of two haikus is better. That is why the later steps use comparisons.

> **Interview trap:** Saying fine tuning is the main way to add knowledge. It mostly teaches behaviour. Facts that change belong in retrieval. Retrieval means looking facts up in documents and showing them to the model.

## Parameter efficient fine tuning

Training every weight in a big model is expensive. A weight is one of the model's learned numbers. Each weight needs memory for itself, its gradient, and the optimiser's bookkeeping. The gradient is the signal that says how to change the weight. The optimiser is the algorithm that applies the changes. A common setup uses about 16 bytes per weight. A 7 billion parameter model therefore needs about 112 GB (gigabytes) just for that. This is more than one large GPU (graphics processing unit) holds.

LoRA (low rank adaptation) avoids most of this. It freezes the original weights and learns a small correction for each weight matrix. A matrix is a grid of numbers. The correction is built as the product of two thin matrices. Picture a 4096 by 4096 matrix, about 16.8 million numbers. Instead of changing all of them, LoRA learns a 4096 by 8 matrix and an 8 by 4096 matrix. That is 65,536 numbers in total, about 0.4 percent. Their product has the right shape to add to the original.

> **Worked example:** For a 7B model, putting rank 16 LoRA on every linear layer trains about 40 million numbers, around 0.6 percent of the model. The saved adapter is a few hundred megabytes at most. Many adapters can share one base model. Here 7B means 7 billion parameters. The rank is the size of the thin middle dimension shared by the two matrices.

Three details interviewers like:

- One of the two matrices starts at zero, so at the start the model is exactly the original. If both started at zero, neither would ever learn.
- After training you can add the correction into the original weights, so there is no slowdown when serving. Serving means answering requests.
- The size of the correction, the rank, is often small, like 8 or 16. It works surprisingly well, because fine tuning mostly adjusts what the model already knows.

QLoRA (quantised LoRA) goes further. It stores the frozen model in 4 bits per weight instead of 16. A bit is a single 0 or 1. It uses a number format shaped like the bell curve that weights tend to follow. Then it trains the small LoRA matrices on top. It even compresses the extra scaling numbers that compression needs. The headline: fine tuning a 65 billion parameter model on one 48 GB GPU.

Older relatives exist: adapters (small extra layers inside each block), prefix tuning and prompt tuning (learned extra tokens). LoRA won because it can be merged away and adds no cost at answer time.

## Choosing how to adapt a model

Think of a ladder. Climb only when the step below clearly fails on a test set. A test set is a set of questions you keep aside to measure quality.

1. **Prompting.** Write better instructions and examples. It is cheapest to try.
2. **Retrieval (RAG).** If the model lacks facts, look them up and show them to the model. This matters most for changing or private facts. You also get citations. RAG stands for retrieval augmented generation.
3. **LoRA fine tuning.** Use it if the model knows enough but behaves wrong. Examples are format, tone or tool use. You also need a few thousand good examples.
4. **Full fine tuning or reinforcement learning.** Use these for big changes, such as a new domain, or reasoning with answers you can check automatically.

Fine tuning has a side effect called catastrophic forgetting. Train hard on legal contracts and the model may get worse at chatting or coding. Defences are mixing general data back in, training gently, freezing most weights with LoRA, and testing broadly. InstructGPT saw a version of this, an "alignment tax" on some benchmarks. It fixed most of it by mixing ordinary pretraining text into training. A benchmark is a standard test used to compare models.

## RL foundations

Reinforcement learning, RL, is learning by trial and score. An agent sees a situation, picks an action, and eventually gets a reward. Its strategy, the policy, gives probabilities for each action. The goal is a strategy with the highest expected total reward. Expected means the average over many tries.

For a language model, the situation is the prompt plus the words written so far. The action is the next word. The reward usually comes once at the end. For example, it is 1 if the final math answer is right and 0 if wrong. The hard part is credit: which of the hundreds of words deserved the reward?

The core recipe is called the policy gradient, which is the direction in which to change the policy to raise the expected reward. The recipe is simple to say. Make the actions you took more likely if the outcome was good, and less likely if it was bad. The basic algorithm is REINFORCE. Its weakness is noise. One lucky outcome pushes every action in that episode. An episode is one full attempt from start to finish.

The fix is a baseline: judge each outcome against what you expected. If you usually score 0.7 on a question and this attempt scored 1, push it up. If it scored 0, push it down. This does not change what the method learns on average, but it removes a lot of noise.

> **Worked example:** In a three option game with rewards 1, 0 and 2, sampled estimates of the gradient matched the exact answer with or without a baseline. But subtracting the expected reward cut the noise by almost four times.

Two more terms. The **advantage** of an action is how much better it was than average in that situation. A **critic**, or value model, is a second network that predicts expected reward from any situation. With it you can estimate advantages step by step. GAE (generalised advantage estimation) is a standard way to blend the critic's quick guesses with actual outcomes.

## PPO

PPO (proximal policy optimisation) is the workhorse RL method behind the training of the original ChatGPT, OpenAI's chat assistant. Its idea is caution. Big policy updates can wreck a model, so PPO limits how far each update can move.

It compares how likely an action is under the new policy versus under the policy that collected the data. This ratio starts at 1. If an action was better than expected, PPO rewards raising its probability, but only until the ratio reaches about 1.2. If it was worse, PPO rewards lowering it, but only down to about 0.8. Beyond those limits there is no extra reward for moving further, so the model has no reason to overshoot. If an update accidentally moved the wrong way, PPO still pulls it back.

> **Worked example:** With a good action (advantage 2), a ratio of 1.1 still gets a full push. At 1.5 the push stops. If a good action's ratio fell to 0.5, the push to restore it continues. Only moves in the direction the advantage wanted are capped.

Keep two anchors apart. The **old policy** is the model that produced this batch, and PPO's limit keeps each update near it. The **reference policy** is the frozen starting model, and a separate penalty keeps the whole training run near it.

## RLHF end to end

RLHF (reinforcement learning from human feedback) puts the pieces together.

First, collect comparisons. The model writes several answers per prompt and people rank them. Comparisons are used because people agree with each other more on "A is better than B" than on scores out of 10. Even then, InstructGPT's labellers agreed only about 73 percent of the time.

Second, train a reward model. It reads a prompt and answer and outputs one number. The Bradley Terry model links scores to preferences. It says the chance that A beats B depends only on the difference in their scores.

> **Worked example:** If the reward model scores the preferred answer 2.0 and the other 0.5, the gap is 1.5. The predicted chance the preferred one wins is about 82 percent. The training loss is about 0.20. A loss is a number that training tries to shrink. If both scores were equal, the chance would be 50 percent and the loss about 0.69.

Third, run PPO to maximise the reward model's score minus a penalty for drifting from the starting model. That penalty matters. The reward model only learned to judge normal looking answers. Without a leash, the model wanders into strange outputs that fool it.

The cost is heavy. PPO based RLHF keeps four models in memory: the model being trained, a value model, the reward model and the frozen starting model.

## Reward hacking and overoptimisation

A reward model is a stand in for human judgement, and stand ins have blind spots. Push hard enough and the model learns to please the scorer instead of people. This is Goodhart's law: once a measure becomes a target, it stops being a good measure.

Classic signs are answers getting longer and longer, flattering the user, confident tone without substance, and heavy formatting. With automatic checkers, the model might list several answers hoping one matches. Or it might write code that only passes the visible tests.

A study by Gao, Schulman and Hilton measured this. As training pushed the model further from its start, the true quality first rose, then peaked and fell. The reward model's score kept climbing. Bigger reward models delayed the peak.

Best of N sampling is a gentle alternative. Generate N answers and keep the best scoring one. It moves the model's behaviour only a little, which makes it harder to hack. The cost is N times the generation.

> **Remember:** Judge training with an independent check, such as human ratings or a stronger judge, never the reward the model was trained on.

## DPO

DPO (direct preference optimisation) skips the reward model and the reinforcement learning loop. The authors noticed that the RLHF goal has an exact best answer: take the starting model and boost each answer by how much reward it deserves. Turning that around, the reward can be written in terms of the model itself. When you compare two answers to the same prompt, a messy term that is impossible to compute appears in both and cancels out.

What is left is a simple training rule on pairs. Make the preferred answer more likely and the rejected answer less likely. Measure both relative to the frozen starting model. Focus on pairs the model currently gets wrong.

> **Worked example:** Suppose training has made the preferred answer a bit more likely than the starting model did, and the rejected answer a bit less likely. This gives a combined score of 2. With the usual strength setting of 0.1, the margin is 0.2 and the loss is about 0.60. At the very start every margin is zero and the loss is about 0.69 for every pair. That is a handy bug check.

DPO is cheap and stable, like ordinary fine tuning. It has three limits. It only learns from the pairs you give it and never explores. It works best when the pairs look like your model's own answers. And it can lower the probability of both answers, as long as the rejected one falls faster.

## DPO variants

Each variant fixes one weakness of DPO. The four below are IPO (identity preference optimisation), KTO (Kahneman Tversky optimisation), ORPO (odds ratio preference optimisation) and SimPO (simple preference optimisation).

| Method | What it fixes |
| --- | --- |
| IPO | DPO overfitting, which means memorising the data too closely, when the data is completely one sided. IPO aims for a fixed gap between the two answers instead of an infinite one |
| KTO | The need for matched pairs. KTO learns from single answers marked good or bad, like thumbs up and down |
| ORPO | The separate fine tuning stage and the reference model, which is the frozen starting model. ORPO folds both into one loss |
| SimPO | A mismatch between DPO's score and how text is generated. SimPO uses the average likelihood per word and needs no reference model, which also curbs long answers |

None of them explores new answers on its own. All are only as good as their data.

## RLAIF and Constitutional AI

Human labels are slow and expensive. So RLAIF (reinforcement learning from AI feedback, where AI means artificial intelligence) uses a capable model as the judge. Show it two answers and some guidelines, ask which is better, and train on its choices.

Constitutional AI, from Anthropic, is the best known version. People write a short list of principles, the constitution. First, the model critiques and rewrites its own answers to difficult prompts according to those principles, and is fine tuned on the rewrites. Then an AI judge compares pairs of answers using the principles. Those judgements train a reward model for reinforcement learning. The goal was an assistant that is harmless but still engages and explains rather than flatly refusing.

The risk is that the judge's quirks get trained into the model. Examples are liking longer answers or the first answer shown. Check AI judges against a smaller set of human labels.

## GRPO

GRPO (group relative policy optimisation), from the DeepSeekMath paper, is PPO without the value model. A second network would normally guess how good each partial answer is. Instead, GRPO asks the model the same question several times and uses the group's own average score as the yardstick.

> **Worked example:** Four answers to a math question score 1, 0, 0, 0. The average is 0.25. Take each score minus the average, then divide by the standard deviation of the scores, a measure of spread, which is 0.5. This gives 1.5 for the correct answer and minus 0.5 for each wrong one. Every word of the correct answer gets pushed up, and every word of the wrong ones gets pushed down. The pushes add to zero.

This fits problems with checkable answers perfectly. The group average is simply how often the model gets this exact question right. So the signal says "better or worse than your usual on this question". No reward model or value model is needed.

One catch: if all answers in a group are right, or all wrong, everyone equals the average and the question teaches nothing. As the model improves, more questions become all right. So methods like DAPO (decoupled clip and dynamic sampling policy optimisation) drop those questions and sample more useful ones.

## Dr. GRPO

The "Understanding R1-Zero-Like Training" paper found two hidden biases in GRPO's arithmetic and removed them. The fixed version is called Dr. GRPO, for "GRPO done right".

**Bias one: dividing by the answer's length.** GRPO averages each answer's word level pushes over that answer's own length. Every answer gets the same total push, but a long answer spreads it thinly. For correct answers, short ones get a stronger push per word. For wrong answers, long ones get a weaker penalty per word. So rambling while wrong is punished less than being wrong briefly. Over training, wrong answers grow longer, which can look like the model is "learning to think".

> **Worked example:** One correct answer of 200 words and wrong answers of 120, 300 and 600 words. Under GRPO, each word of the 600 word wrong answer is penalised five times less than each word of the 120 word one. Under Dr. GRPO, every wrong word gets the same penalty. So the long wrong answer gets the largest total penalty.

**Bias two: dividing by the spread of scores.** GRPO divides each question's advantages by how varied its scores are. Questions where almost all answers agree have tiny spread, so their signal gets multiplied up. With 16 answers, a question with only 1 correct answer gets multiplied by 4. A 50/50 question gets multiplied by about 1.9. Very easy and very hard questions end up weighted more than medium ones.

Dr. GRPO simply uses each score minus the group average. It divides the summed word level losses by one fixed number for the whole run. The paper reported that this kept accuracy while making wrong answers much shorter. With it they trained a 7 billion parameter model to 43.3 percent on the AIME 2024 math exam using eight GPUs. AIME is the American Invitational Mathematics Examination. They also found two more things. Base models already show the famous "aha" self checking behaviour before any reinforcement learning. And the prompt template and pretraining data shape results a great deal.

## Verifiable rewards and DeepSeek R1

When a program can check the answer, you do not need a learned reward model at all. Examples are comparing a math result with the solution or running unit tests. This is reinforcement learning with verifiable rewards.

DeepSeek's R1-Zero took a pretrained base model with no example answers at all. It told the model to think inside special tags before answering. It rewarded the model only for correct, well formatted answers, using GRPO. Accuracy on the AIME 2024 math exam rose from about 16 percent to 71 percent on a single attempt. The reasoning grew from hundreds to thousands of words. The model sometimes stopped, said "wait", and reconsidered. The authors called this an aha moment. The downside is that its reasoning was hard to read and mixed languages.

The full R1 fixed this in four stages.

1. A small set of clean reasoning examples to start.
2. Reasoning reinforcement learning, with a reward for staying in one language.
3. A large round of fine tuning on about 800,000 of its own best outputs plus general data.
4. A final reinforcement learning round for helpfulness and safety.

R1 scored 79.8 percent on AIME 2024.

Does this teach new abilities, or sharpen existing ones? The evidence leans toward sharpening. DeepSeekMath found that reinforcement learning improved majority vote accuracy, but not the chance that some sample among many is right. Sharpening still matters. Single attempt accuracy is what users feel.

## Test time compute

Instead of a bigger model, you can let the same model work harder when answering. This is test time compute. There are four ways.

- **Think longer.** Writing out steps gives the model more computation and a scratchpad. A model does a fixed amount of work per word, so more words means more thinking.
- **Try many times.** Sample many answers. Majority voting picks the most common final answer. Best of N uses a checker or scorer to pick the best.
- **Revise.** Let the model look at its previous attempt and fix it.
- **Search.** Explore partial solutions step by step, keeping the promising ones.

> **Worked example:** A model is right 40 percent of the time, with wrong answers scattered over six values. It reaches about 86 percent with 15 votes and over 99 percent with 45. But suppose one particular wrong answer appears 45 percent of the time. Then voting converges on that wrong answer and accuracy drops. Voting amplifies whatever answer is most common.

Two key findings follow. The Large Language Monkeys study found that the chance at least one of many tries is right keeps rising with more tries. But picking the right one without a perfect checker is the bottleneck. On one math benchmark, a right answer appeared for over 98 percent of problems with 10,000 tries. Yet voting picked correctly only about 41 percent of the time.

A 2024 study by Snell and colleagues tailored the strategy to each question's difficulty. This matched a standard approach with about four times less computation. It could also beat a model 14 times larger, but only on questions the small model could sometimes solve. For the hardest questions, a bigger model was still better.

## Verifiers and process rewards

A verifier scores answers. An outcome verifier looks only at the final answer. A process verifier scores each step. It can point to where reasoning goes wrong, like a teacher marking each line of working rather than just the final number.

OpenAI's "Let's Verify Step by Step" paper collected about 800,000 human judgements of individual steps. The task was to pick the best of 1,860 sampled solutions. The step level verifier chose correctly 78 percent of the time. The final answer verifier managed 72 percent, and majority voting 70 percent.

> **Worked example:** A solution's steps are scored 0.95, 0.9, 0.4 and 0.92. There are three ways to combine them. Multiplying gives about 0.31. Taking the lowest score gives 0.4. Using the last step alone gives 0.92. Only the last step rule misses the bad third step.

Step labels are expensive. So methods like Math-Shepherd label a step by continuing the solution from it several times and seeing how often it reaches the right answer. Interestingly, DeepSeek reported that process reward models and tree search did not work well for training R1 at scale. A process reward model scores individual steps. Tree search explores many branching solution paths. DeepSeek gave three reasons. Steps are hard to define. Step labels are unreliable. The model finds ways to fool learned step scorers.

## Distilling reasoning

Distillation means a small student model learns by copying a strong teacher. For reasoning, the student is fine tuned on the teacher's complete worked solutions, including the long thinking.

DeepSeek did this with about 800,000 examples from R1 and got strong small models. The striking comparison is this. Running full reinforcement learning on a 32 billion parameter base model reached 47 percent on AIME 2024. Simply fine tuning the same model on R1's solutions reached about 73 percent. Discovering a good reasoning strategy is expensive. Copying one is cheap.

A related self teaching trick: generate many answers, keep only the correct ones, and fine tune on them. STaR (self taught reasoner) adds a twist. It gives the model the right answer as a hint on problems it failed, and trains on its explanation.

## Evaluating reasoning models

Reasoning scores are easy to fool yourself with. Five habits protect you.

- **pass@k** is the chance that at least one of k tries is right. The standard way to estimate it draws more tries than k and counts combinations. The shortcut formula comes out too low on average.

> **Worked example:** With 10 tries and 2 correct, pass@1 is 0.2. For pass@5, there are 252 ways to pick 5 tries, and 56 of them contain only wrong ones. So pass@5 is 1 minus 56/252, about 0.78. The shortcut, 1 minus 0.8 to the fifth power, gives 0.67.

- **Small tests are noisy.** AIME 2024 has 30 problems, so one problem is worth 3.3 points. A single run varies by about 9 points either way. Never trust a 2 point gain on it alone.
- **Contamination.** If test questions leaked into training data, a high score means memory, not skill. Test on problems released after the model's training data was collected.
- **Answer extraction.** A grader must find the final answer in a long response. It can miss answers in the wrong format, mark 0.5 and 1/2 as different, or grab the wrong number. In reinforcement learning the grader is the reward, so its mistakes get trained in.
- **State your settings.** Number of samples, the randomness setting (called temperature), and maximum length all change the score. So report them with every number.
