# Deep Learning: The Plain Version

Deep learning builds a function out of many simple adjustable pieces and tunes them by following slopes. This guide explains each piece with an everyday picture and one small worked number, then says where the idea stops working. The flashcards in this module ask the same things as interview questions, and the deep guide adds the derivations.

## The neuron and the MLP

A neuron is a small calculator that weighs its inputs, adds them up and squashes the total. Picture a bouncer who scores a guest on age and dress code, adds a personal bias, and lets the guest in only if the score is high enough. The weights and the bias are the numbers training adjusts. A layer is many neurons reading the same inputs, and a multilayer perceptron stacks layers. A network that reads 784 pixels, has hidden layers of 256 and 128 units and 10 outputs has 235,146 adjustable numbers. It breaks on data with structure, such as images and sentences, because nothing in its design says that neighbours matter.

> **Remember:** Without a squashing step between layers, two layers collapse into one, because you can multiply the two weight matrices together.

## Universal approximation and its limits

A network with one hidden layer can copy any smooth curve if you give it enough units. Imagine building a mountain range from adjustable ramps: with enough ramps you can match any outline. In a small test with random ramps, the error was 0.60 with 2 ramps and 0.0001 with 200. The theorem only says the right weights exist. It does not say how many units you need, how to find the weights, or whether the result works on new data. Depth is the practical answer to the size problem, because stacking layers multiplies the number of bends instead of adding them.

## Computational graphs and backprop

Every calculation can be drawn as a graph of tiny steps, and backpropagation walks that graph backwards to find how each input nudges the result. Take L = (a times b plus c) times f with a = 2, b = -3, c = 10 and f = -2, so L = -8. Working backwards, the gradient of a is 6, which means nudging a up by 0.01 raises L by about 0.06. When a value is used twice its gradients add up, so b = a + a has slope 2, not 1. One backward pass costs about two forward passes, however many parameters there are.

> **Interview trap:** Overwriting a gradient instead of adding to it silently breaks any value that feeds two operations.

## Backprop in matrix form

Real networks process a whole batch at once, so the same recipe is written with matrices. For a layer Z = XW + b, the weight gradient is X transposed times the incoming gradient, and the input gradient is the incoming gradient times W transposed. A shape check finds the formulas for you, because a gradient always has the shape of its variable. For softmax with cross entropy the gradient at the logits is just the predicted probabilities minus the one hot answer, for example (-0.341, 0.242, 0.099) for scores (2, 1, 0.1). Compare a hand written backward pass with a numerical estimate before you trust it.

## Activation functions

An activation is the bend that makes stacking layers worthwhile. Sigmoid and tanh squash everything into a small range, so their curves go flat at the ends, and a flat curve passes almost no gradient back. The sigmoid's slope is at most 0.25. ReLU clips negatives to zero, which keeps the slope at 1 for positive inputs but lets a unit die if its input is always negative. GELU and SiLU are smooth versions that let a little negative signal through, and transformers use them. At the output, pick the activation that matches the answer: softmax when exactly one class is right, independent sigmoids when several labels can be true.

## Loss functions

A loss scores how wrong the model is. Cross entropy is the natural score for classes, because it comes from asking for the model that makes the observed labels most probable, while squared error assumes Gaussian noise. When a sigmoid output is confidently wrong, cross entropy gives a gradient near 1 and squared error gives one about 400 times smaller. Computing softmax naively overflows, since the exponential of 1000 is infinity, so libraries subtract the largest score first. A fresh model with K classes should start with a loss near ln K, which is 3.296 for 27 characters.

> **Say it like this:** "Choosing a loss means choosing a noise model: cross entropy for categories, squared error for Gaussian noise."

## Vanishing and exploding gradients

In a deep network the signal travelling backwards is multiplied by one factor per layer. If the factors are a bit under 1 the signal fades, and if they are a bit over 1 it explodes. With 0.9 over 50 layers you keep 0.005 of it, and with 1.1 over 50 layers you get 117 times it. Fading means the early layers never learn, and exploding means the loss spikes or turns into nan. Clipping the length of the gradient tames the spikes, while good initialisation, normalisation and skip connections tackle the cause.

## Initialisation

Starting weights must be random, and of the right size. If every weight starts equal, all the units in a layer copy each other forever. If they are too small the signal shrinks layer by layer, and if they are too large tanh units get stuck at plus or minus one. In a 50 layer test, weights of 0.01 shrank the signal to about 10 to the minus 33, while He scaling for ReLU kept it near 0.74. The rule is to choose the weight variance so that each layer keeps the signal size, which for ReLU is 2 divided by the number of inputs.

## Optimisers

Training is walking downhill on a loss surface in fog, feeling only the local slope. Plain SGD steps against the slope measured on a small random batch. Momentum adds the inertia of a heavy ball, so steady directions speed up and zig zags cancel: on a narrow valley it cut 653 steps to 95. Adam gives each weight its own step size from the history of its gradients, which makes it forgiving to tune, and AdamW also shrinks weights separately from the gradient. Adam's averages start at zero, so it corrects them early on, or its very first step would be 3.16 times too large.

## Learning rate schedules

The learning rate is the step length, and it matters more than almost any other setting. Too high and the loss bounces or explodes, too low and it crawls. A sweep that raises the rate exponentially while you watch the loss finds the sensible range. Big models start with a short warmup, where the rate ramps from nearly zero to its peak, and then decay it, often along a cosine curve. In the GPT-2 reproduction the rate rises to 0.0006 over 715 steps and falls to a tenth of that. Decay helps because small steps late in training calm the noise.

## Normalisation layers

A normalisation layer rescales activations so each layer sees numbers of a steady size. Batch norm does this across the examples in a batch, so a feature with values 1, 2, 3 and 4 becomes about minus 1.34 up to plus 1.34. At test time it uses running averages instead, and forgetting to switch modes is a classic bug. Layer norm works within a single example, which suits transformers, and RMSNorm skips the mean to save work. Batch norm struggles with tiny batches, because the variance of one example is zero.

## Regularisation

Regularisation stops a network from memorising its training set. Dropout silences random units during training, so no unit can lean on one particular partner, and test time uses all of them. Weight decay nudges every weight toward zero a little at each step. Early stopping halts when validation stops improving, augmentation shows the model altered copies of the data, and label smoothing keeps it from being absolutely certain. If validation loss is lower than training loss, that is often harmless, because dropout and augmentation act only during training.

## Residual connections

A residual connection adds a block's input to its output, so the block only has to learn a correction. Gradients then have a clean highway back to the early layers. In a 50 layer test a plain stack turned a unit gradient into 0.005, while the residual version kept it near 1.3. Transformers use the idea around every sublayer, and most put the norm inside the branch, which is called pre norm. That keeps the main path untouched, at the price of a residual stream that grows with depth.

## Convolutional networks

A convolution slides one small filter across an image and reuses the same weights everywhere, like scanning a page with a magnifying glass. The same few numbers detect an edge anywhere, so a 3 by 3 layer from 64 to 128 channels has 73,856 weights, where a dense layer between the same maps would need about 80 billion. Stacked 3 by 3 filters see ever larger regions, and pooling or strides shrink the picture. ResNet-50 has 25,557,032 weights. Convolutions assume nearby pixels matter most, which fails when distant parts must be related.

## Recurrent networks

An RNN reads a sequence one item at a time and keeps a memory that it updates at each step, using the same weights every time. The memory is like a note you rewrite after every word. Training unrolls it over time, which makes a very deep network that multiplies by the same matrix again and again. In a test over 50 steps, a matrix a little too large gave gradients of 110, and one a little too small gave 0.005. It also cannot be run in parallel across time, because each step needs the one before.

## LSTMs and GRUs

An LSTM keeps a separate cell state and three gates that decide what to forget, write and show. Think of a notebook with an eraser, a pen and a window. The cell is updated by adding, so the gradient along that path is just the forget gate, with no matrix in the way: 0.99 over 100 steps still leaves 0.37. Starting the forget bias near 1 makes the notebook mostly remember at first. A GRU merges the cell and the state and has two gates, so for the same sizes it has 295,680 parameters against 394,240 in the LSTM.

## Embeddings

An embedding is a learned address book: each token id maps to a list of numbers. Looking up a row is the same as multiplying a one hot vector by the table, only much cheaper. Similar tokens end up with similar vectors because the task rewards it. GPT-2 small's token table has 50,257 rows of 768 numbers, about 38.6 million weights, and it reuses that table as its output layer, which is called weight tying.

## Autoencoders and VAEs

An autoencoder squeezes data through a narrow code and tries to rebuild it, like describing a face to an artist in a few words. A linear one finds the same subspace as PCA. A variational autoencoder makes the code a distribution and keeps it close to a standard normal, which gives a space you can sample from. Its score has two parts: how well it rebuilds the data, and how far its code strays from the prior. The reparameterisation trick writes a sample as mean plus spread times noise, so gradients can pass through the sampling step.

## GANs

A GAN is a forger and a detective trained together. The generator turns noise into fake samples, the discriminator tries to tell them from real ones, and each improves against the other. At the ideal end the fakes match the data and the detective can only guess, giving 50 percent. In practice training can be unstable, and a generator may learn only a few kinds of output, which is called mode collapse. There is also no likelihood to tell you how well it is doing.

## Diffusion models

A diffusion model learns to clean up noise one small step at a time. Training adds Gaussian noise to real images until nothing is left, then teaches a network to predict the noise that was added. To generate, start from pure noise and subtract the predicted noise about a thousand times. The forward process has a closed form, so training can jump to any noise level at once. In a toy test with data at minus 1 or plus 1, a perfect denoiser sent noise to those two values, half each. Sampling is slow because it needs many network calls.

## Contrastive learning

Contrastive learning teaches an encoder what belongs together without labels. Two crops of one photo are a matching pair, and crops of other photos are mismatches. The loss is a softmax that must pick the true partner out of the batch, so a model that tells nothing apart scores exactly ln N, which is 5.545 for 256 candidates. A low temperature makes hard mismatches count more. SimCLR found that random crop together with colour distortion was crucial, and that big batches help.

## Training recipe and debugging

Build up in small safe steps. Look at the data, make a tiny end to end version, check the first loss, and then try to overfit one small batch: if the model cannot memorise 32 examples there is a bug, and tuning will not help. After that, fit the full data, add regularisation, tune the learning rate, and only then squeeze out extra gains. Watch the loss curves and the gradient norm. If you see nan, find the first bad step and check the data, the learning rate and the numerics. A healthy ratio of update size to weight size is about one thousandth.

## Mixed precision

Mixed precision does the heavy arithmetic in short numbers and keeps the delicate bookkeeping in long ones. fp16 is fairly precise but tops out at 65,504 and loses anything below about 6e-8, so the loss is multiplied by a big factor before backprop and the gradients are divided afterwards. bf16 keeps fp32's range with less precision and needs no such trick. The weights, gradients and Adam state still cost 16 bytes per parameter, the same as full precision, and what shrinks is the activation memory.

## Distributed training

When one GPU is too small or too slow, split the work. Data parallelism copies the model and splits the batch, then averages gradients with an all reduce that costs each GPU about twice the gradient size. ZeRO and FSDP stop every GPU keeping a full copy of the training state, cutting a 7.5 billion parameter model from 120 GB to about 2 GB per GPU on 64 GPUs. Tensor parallelism splits matrices inside a layer, and pipeline parallelism splits layers into stages, with an idle bubble of 27 percent for 4 stages and 8 micro batches. Gradient accumulation fakes a bigger batch on small memory.

## Attention as a bridge

Attention lets each token look at every other token and take a weighted mix of what it finds. Where an RNN passes a message down a chain, attention lets any two positions talk directly in one step. Each token makes a query, compares it with every key, and a softmax turns the scores into weights. The scores are divided by the square root of the key size, because a dot product of 64 unit variance numbers has variance 64 and would otherwise make the softmax nearly one hot. The price is cost that grows with the square of the length. The transformers and LLMs module takes it from here.
