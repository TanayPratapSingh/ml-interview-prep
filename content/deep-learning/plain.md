# Deep Learning: The Plain Version

Deep learning builds a function out of many simple adjustable pieces and tunes them by following slopes. This guide explains each piece with an everyday picture and one small worked number. It then says where the idea stops working. The flashcards in this module ask the same things as interview questions, and the deep guide adds the derivations, which are the step by step maths.

## The neuron and the MLP

A neuron is a small calculator. It weighs its inputs, adds them up and squashes the total into a limited range. Picture a bouncer who scores a guest on age and dress code. The bouncer adds a personal bias, which is a fixed nudge up or down. The guest gets in only if the score is high enough. The weights and the bias are the numbers that training adjusts.

A layer is many neurons reading the same inputs. A multilayer perceptron (MLP) stacks layers, with each layer feeding the next. Take a network that reads 784 pixels and has hidden layers of 256 and 128 units. A unit is one neuron. The network also has 10 outputs. It has 235,146 adjustable numbers. It breaks on data with structure, such as images and sentences. The reason is that nothing in its design says that neighbours matter.

> **Remember:** Without a squashing step between layers, two layers collapse into one, because you can multiply the two weight matrices (the grids of weights) together.

## Universal approximation and its limits

A network with one hidden layer can copy any smooth curve if you give it enough units. This is the universal approximation theorem. Imagine building a mountain range from adjustable ramps. With enough ramps you can match any outline. In a small test with random ramps, the error was 0.60 with 2 ramps and 0.0001 with 200.

The theorem only says the right weights exist. It does not say how many units you need. It does not say how to find the weights. It does not say whether the result works on new data. Depth is the practical answer to the size problem. Stacking layers multiplies the number of bends instead of adding them.

## Computational graphs and backprop

Every calculation can be drawn as a graph of tiny steps. This is a computational graph. Backpropagation (backprop) walks that graph backwards. It finds how each input nudges the result.

Take L = (a times b plus c) times f. Let a = 2, b = -3, c = 10 and f = -2, so L = -8. Working backwards, the gradient of a is 6. A gradient is the slope: how much the result changes per small change in an input. So nudging a up by 0.01 raises L by about 0.06.

When a value is used twice, its gradients add up. So b = a + a has slope 2, not 1. One backward pass costs about two forward passes, however many parameters there are. A parameter is one learned number.

> **Interview trap:** Overwriting a gradient instead of adding to it silently breaks any value that feeds two operations.

## Backprop in matrix form

Real networks process a whole batch at once, so the same recipe is written with matrices. A batch is a group of examples handled together. For a layer Z = XW + b, the weight gradient is X transposed times the incoming gradient. The input gradient is the incoming gradient times W transposed. Transposed means a matrix flipped so its rows become columns.

A shape check finds the formulas for you, because a gradient always has the shape of its variable.

Now take softmax with cross entropy. Softmax turns raw scores, called logits, into probabilities that add to 1. Cross entropy measures how surprised the model is by the true answer. The gradient at the logits is just the predicted probabilities minus the one hot answer. A one hot answer has a 1 for the right class and 0 elsewhere. For example, scores of (2, 1, 0.1) with the first class correct give the gradient (-0.341, 0.242, 0.099). Compare a hand written backward pass with a numerical estimate before you trust it.

## Activation functions

An activation is the bend that makes stacking layers worthwhile. Sigmoid and tanh squash everything into a small range. Sigmoid is an S shaped curve from 0 to 1, and tanh is an S shaped curve from minus 1 to 1. Their curves go flat at the ends, and a flat curve passes almost no gradient back. The sigmoid's slope is at most 0.25.

ReLU (rectified linear unit) clips negatives to zero. This keeps the slope at 1 for positive inputs. But it lets a unit die if its input is always negative. GELU (Gaussian error linear unit) and SiLU (sigmoid linear unit) are smooth versions that let a little negative signal through. Transformers, the design behind modern language models, use them.

At the output, pick the activation that matches the answer. Use softmax when exactly one class is right. Use independent sigmoids when several labels can be true.

## Loss functions

A loss scores how wrong the model is. Cross entropy is the natural score for classes. It comes from asking for the model that makes the observed labels most probable. Squared error instead assumes Gaussian noise, which means bell shaped random error. When a sigmoid output is confidently wrong, cross entropy gives a gradient near 1. Squared error gives one about 400 times smaller.

Computing softmax naively overflows, which means the numbers get too big to store. The exponential of 1000 is infinity. So libraries subtract the largest score first. A fresh model with K classes should start with a loss near ln K. Here ln is the natural logarithm. For 27 characters, that is 3.296.

> **Say it like this:** "Choosing a loss means choosing a noise model: cross entropy for categories, squared error for Gaussian noise."

## Vanishing and exploding gradients

In a deep network, the signal travelling backwards is multiplied by one factor per layer. If the factors are a bit under 1, the signal fades. If they are a bit over 1, it explodes. With 0.9 over 50 layers, you keep 0.005 of it. With 1.1 over 50 layers, you get 117 times it.

Fading means the early layers never learn. Exploding means the loss spikes or turns into nan, which stands for not a number and comes from broken arithmetic. Clipping the length of the gradient tames the spikes. Good initialisation tackles the cause, and so do normalisation and skip connections. Initialisation is how you pick the starting weights. Normalisation rescales activations. Skip connections are shortcuts around layers. Each is explained below.

## Initialisation

Starting weights must be random, and of the right size. If every weight starts equal, all the units in a layer copy each other forever. If the weights are too small, the signal shrinks layer by layer. If they are too large, tanh units get stuck at plus or minus one.

In a 50 layer test, weights of 0.01 shrank the signal to about 10 to the minus 33. He scaling for ReLU kept it near 0.74. He scaling is named after Kaiming He, who proposed it. The rule is to choose the weight variance so that each layer keeps the signal size. Variance is a measure of how spread out the weights are. For ReLU, the variance is 2 divided by the number of inputs.

## Optimisers

Training is walking downhill on a loss surface in fog, feeling only the local slope. An optimiser is the rule for each step. Plain SGD (stochastic gradient descent) steps against the slope measured on a small random batch. Momentum adds the inertia of a heavy ball. Steady directions speed up and zig zags cancel. On a narrow valley it cut 653 steps to 95.

Adam (adaptive moment estimation) gives each weight its own step size from the history of its gradients. This makes it forgiving to tune. AdamW also shrinks weights separately from the gradient, which is called weight decay. Adam's averages start at zero, so it corrects them early on. Without the correction, its very first step would be 3.16 times too large.

## Learning rate schedules

The learning rate is the step length, and it matters more than almost any other setting. If it is too high, the loss bounces or explodes. If it is too low, the loss crawls. A sweep finds the sensible range. In a sweep, you raise the rate exponentially while you watch the loss.

Big models start with a short warmup, where the rate ramps from nearly zero to its peak. Then they decay it, often along a cosine curve. In the GPT-2 (generative pre trained transformer 2) reproduction, the rate rises to 0.0006 over 715 steps and falls to a tenth of that. GPT-2 is an early large language model from OpenAI. Decay helps because small steps late in training calm the noise.

## Normalisation layers

A normalisation layer rescales activations so each layer sees numbers of a steady size. Batch norm does this across the examples in a batch. A feature with values 1, 2, 3 and 4 becomes about minus 1.34 up to plus 1.34. At test time it uses running averages instead. Forgetting to switch modes is a classic bug.

Layer norm works within a single example, which suits transformers. RMSNorm (root mean square normalisation) skips the mean to save work. Batch norm struggles with tiny batches, because the variance of one example is zero.

## Regularisation

Regularisation stops a network from memorising its training set. Dropout silences random units during training, so no unit can lean on one particular partner. Test time uses all of them. Weight decay nudges every weight toward zero a little at each step. Early stopping halts when validation stops improving. Validation is a set of held back examples used to check progress.

Augmentation shows the model altered copies of the data, such as flipped images. Label smoothing keeps the model from being absolutely certain. If validation loss is lower than training loss, that is often harmless. The reason is that dropout and augmentation act only during training.

## Residual connections

A residual connection adds a block's input to its output, so the block only has to learn a correction. Gradients then have a clean highway back to the early layers. In a 50 layer test, a plain stack turned a unit gradient into 0.005. The residual version kept it near 1.3.

Transformers use the idea around every sublayer. Most put the norm inside the branch, which is called pre norm. That keeps the main path untouched. The price is a residual stream that grows with depth. The residual stream is the running vector that every layer adds to.

## Convolutional networks

A convolution slides one small filter across an image and reuses the same weights everywhere, like scanning a page with a magnifying glass. A filter is a small grid of weights, such as 3 by 3. The same few numbers detect an edge anywhere. So a 3 by 3 layer from 64 to 128 channels has 73,856 weights. A channel is one map of features, like the red layer of a colour image. A dense layer between the same maps would need about 80 billion.

Stacked 3 by 3 filters see ever larger regions. Pooling or strides shrink the picture. ResNet-50 is a famous 50 layer image network built from residual blocks. It has 25,557,032 weights. Convolutions assume nearby pixels matter most. That fails when distant parts must be related.

## Recurrent networks

An RNN (recurrent neural network) reads a sequence one item at a time. It keeps a memory that it updates at each step, using the same weights every time. The memory is like a note you rewrite after every word. Training unrolls the network over time. This makes a very deep network that multiplies by the same matrix again and again.

In a test over 50 steps, a matrix a little too large gave gradients of 110. A matrix a little too small gave 0.005. An RNN also cannot be run in parallel across time, because each step needs the one before.

## LSTMs and GRUs

An LSTM (long short term memory) network keeps a separate cell state and three gates. The gates decide what to forget, write and show. Think of a notebook with an eraser, a pen and a window. The cell is updated by adding. So the gradient along that path is just the forget gate, with no matrix in the way. A forget gate of 0.99 over 100 steps still leaves 0.37. Starting the forget bias near 1 makes the notebook mostly remember at first.

A GRU (gated recurrent unit) merges the cell and the state and has two gates. For the same sizes it has 295,680 parameters against 394,240 in the LSTM.

## Embeddings

An embedding is a learned address book. Each token id maps to a list of numbers. A token is a small chunk of text, and its id is its number in a fixed list. Looking up a row is the same as multiplying a one hot vector by the table, only much cheaper. Similar tokens end up with similar vectors because the task rewards it.

GPT-2 small's token table has 50,257 rows of 768 numbers. That is about 38.6 million weights. It reuses that table as its output layer, which is called weight tying.

## Autoencoders and VAEs

An autoencoder squeezes data through a narrow code and tries to rebuild it, like describing a face to an artist in a few words. A linear one finds the same subspace as PCA (principal component analysis), a method that keeps the directions of greatest spread. A subspace is a flat slice of the space.

A variational autoencoder (VAE) makes the code a distribution and keeps it close to a standard normal. This gives a space you can sample from. Its score has two parts. One is how well it rebuilds the data. The other is how far its code strays from the prior, the distribution you want the code to resemble. The reparameterisation trick writes a sample as mean plus spread times noise. This lets gradients pass through the sampling step.

## GANs

A GAN (generative adversarial network) is a forger and a detective trained together. The generator turns noise into fake samples. The discriminator tries to tell them from real ones. Each improves against the other. At the ideal end, the fakes match the data and the detective can only guess, giving 50 percent.

In practice, training can be unstable. A generator may also learn only a few kinds of output, which is called mode collapse. There is also no likelihood to tell you how well it is doing. A likelihood is a score for how probable the real data is under the model.

## Diffusion models

A diffusion model learns to clean up noise one small step at a time. Training adds Gaussian noise to real images until nothing is left. Then it teaches a network to predict the noise that was added. To generate, start from pure noise and subtract the predicted noise about a thousand times.

The forward process has a closed form, which is a direct formula with no step by step loop. So training can jump to any noise level at once. In a toy test with data at minus 1 or plus 1, a perfect denoiser sent noise to those two values, half each. Sampling is slow because it needs many network calls.

## Contrastive learning

Contrastive learning teaches an encoder what belongs together without labels. An encoder is a network that turns an input into a vector. Two crops of one photo are a matching pair. Crops of other photos are mismatches. The loss is a softmax that must pick the true partner out of the batch. So a model that tells nothing apart scores exactly ln N, which is 5.545 for 256 candidates.

A low temperature makes hard mismatches count more. Temperature here is a setting that sharpens or flattens the softmax. SimCLR (a simple framework for contrastive learning of visual representations) found two things. Random crop together with colour distortion was crucial. And big batches help.

## Training recipe and debugging

Build up in small safe steps. Look at the data. Make a tiny end to end version. Check the first loss. Then try to overfit one small batch. If the model cannot memorise 32 examples, there is a bug, and tuning will not help.

After that, fit the full data, add regularisation, tune the learning rate, and only then squeeze out extra gains. Watch the loss curves and the gradient norm, which is the overall length of the gradient. If you see nan, find the first bad step. Check the data, the learning rate and the numerics, which means the way numbers are handled. A healthy ratio of update size to weight size is about one thousandth.

## Mixed precision

Mixed precision does the heavy arithmetic in short numbers and keeps the delicate bookkeeping in long ones. fp16 (16 bit floating point) is fairly precise but tops out at 65,504. It also loses anything below about 6e-8. So the loss is multiplied by a big factor before backprop, and the gradients are divided afterwards.

bf16 (brain floating point 16) keeps the range of fp32 with less precision. fp32 is 32 bit floating point, the standard full precision format. bf16 needs no such trick. The weights, gradients and Adam state still cost 16 bytes per parameter, the same as full precision. What shrinks is the activation memory. Activations are the intermediate values saved for the backward pass.

## Distributed training

When one GPU (graphics processing unit) is too small or too slow, split the work. Data parallelism copies the model and splits the batch. It then averages gradients with an all reduce. An all reduce sums values across all GPUs and gives every GPU the result. It costs each GPU about twice the gradient size.

ZeRO (zero redundancy optimizer) and FSDP (fully sharded data parallel) stop every GPU keeping a full copy of the training state. They cut a 7.5 billion parameter model from 120 GB (gigabytes) to about 2 GB per GPU on 64 GPUs. Tensor parallelism splits matrices inside a layer. Pipeline parallelism splits layers into stages. It has an idle bubble of 27 percent for 4 stages and 8 micro batches. Micro batches are small slices of the batch. Gradient accumulation fakes a bigger batch on small memory.

## Attention as a bridge

Attention lets each token look at every other token and take a weighted mix of what it finds. An RNN passes a message down a chain. Attention lets any two positions talk directly in one step. Each token makes a query and compares it with every key. A softmax turns the scores into weights.

The scores are divided by the square root of the key size. The reason is that a dot product of 64 numbers with variance 1 has variance 64. A dot product multiplies two lists of numbers pair by pair and adds the results. Without the division, the softmax would put nearly all the weight on one token, which is called one hot. The price is cost that grows with the square of the length. The Transformers and LLMs module (large language models) takes it from here.
