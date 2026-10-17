# Deep Learning: The Deep Version

This guide covers deep learning from the neuron to distributed training, with the derivations and numbers interviewers ask for. Every number comes from a script run in NumPy, from a paper's text, or from a named lecture or repository. Attention is only a bridge here, because the transformers and LLMs module covers it in full.

## The neuron and the MLP

A neuron computes $a = \sigma(w^\top x + b)$ with $d + 1$ parameters. A layer stacks $m$ neurons as $a = \sigma(Wx + b)$, one matrix multiply and an element wise function. An MLP with $r$ layers applies $a^{[k]} = \sigma(W^{[k]} a^{[k-1]} + b^{[k]})$, with parameter count $\sum_k (m_{k-1} + 1)\, m_k$. For $784 \to 256 \to 128 \to 10$ the layers hold 200,960, 32,896 and 1,290 parameters, total 235,146. CS229 counts weight matrices as layers, so a two layer network has one hidden layer; always state the convention.

Without a nonlinearity, $W^{[2]}(W^{[1]}x) = (W^{[2]}W^{[1]})x$ collapses to one linear map. Two ReLU units compute XOR: $h_1 = \mathrm{ReLU}(x_1 + x_2)$, $h_2 = \mathrm{ReLU}(x_1 + x_2 - 1)$, output $h_1 - 2h_2$ gives 0, 1, 1, 0 on the four inputs. One hidden unit cannot, because its output is monotone in a single projection. A sigmoid neuron is logistic regression; the last layer of a deep net is a linear model on learned features $\phi_\beta(x)$ (CS229 notes, section 7.2).

> **Watch:** [Karpathy, micrograd, a neuron](https://www.youtube.com/watch?v=VMj-3S1tku0&t=3172s)

## Universal approximation and its limits

For compact $K \subset \mathbb{R}^d$, continuous $f$ and a non polynomial activation, one hidden layer approximates $f$ uniformly to any $\varepsilon$ (Cybenko 1989, Hornik 1991, Leshno et al. 1993). The theorem is about existence. It leaves open the width (possibly exponential in $d$), whether gradient descent finds the weights, and generalisation. In my test, random ReLU features with a least squares output layer fitted $\sin 2x + 0.5\cos 5x$ with test MSE 0.598, 0.467, 0.165, 0.127, 0.0027 and 0.00011 at widths 2, 5, 10, 20, 50 and 200.

Depth buys efficiency. A one hidden layer ReLU net with $w$ units has at most $w + 1$ linear pieces in 1D, but composing the 2 unit tent map $\mathrm{ReLU}(2x) - 2\,\mathrm{ReLU}(2x - 1)$ $k$ times gives $2^k$ pieces from $2k$ units; I counted 2, 4, ..., 128 pieces for $k = 1$ to 7. Depth separation theorems (Telgarsky 2016) and linear region counts (Montufar et al. 2014) make this general. The price is harder optimisation.

## Computational graphs and backprop

A computational graph is a DAG of elementary operations ending in a scalar loss. The forward pass evaluates it and caches intermediates. Backprop visits nodes in reverse topological order and adds (local derivative) times (gradient from above) into each input. Example: $L = (ab + c)f$ at $a = 2, b = -3, c = 10, f = -2$ gives $e = -6$, $d = 4$, $L = -8$ and gradients $\partial L/\partial a = 6$, $\partial L/\partial b = -4$, $\partial L/\partial c = -2$, $\partial L/\partial f = 4$, confirmed by central differences.

Node rules: add distributes the gradient, multiply swaps inputs, max routes to the winner, copy and broadcast sum gradients. Accumulate with `+=`, because a value used twice has gradient equal to the sum over paths: `b = a + a` has derivative 2. Cost: a circuit of size $N$ has its gradient in $O(N)$ time (CS229 theorem 7.4.1); for matmul layers forward costs 2 FLOPs per weight and backward about 4. Reverse mode gives the whole gradient of a scalar in one pass, while forward mode needs one pass per input; numerical differentiation costs $2n$ forward passes (470,292 for the 235,146 parameter MLP) and serves only to check gradients, using central differences, float64 and a relative error below $10^{-7}$.

> **Watch:** [Karpathy, micrograd, backward for each operation](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4142s), [CS229 Lecture 8, backprop](https://www.youtube.com/watch?v=ne2ngVAoMG8&t=1150s)

## Backprop in matrix form

For $Z = XW + \mathbf{1}b^\top$ with upstream $G = \partial J/\partial Z$: $\partial J/\partial W = X^\top G$, $\partial J/\partial b = \mathbf{1}^\top G$, $\partial J/\partial X = G W^\top$. The shape rule (a gradient has its variable's shape) recovers them. For a two layer ReLU network with softmax cross entropy: $dZ_2 = (P - Y)/B$, $dW^{[2]} = A^\top dZ_2$, $dA = dZ_2 W^{[2]\top}$, $dZ_1 = dA \odot \mathbf{1}[Z_1 > 0]$, $dW^{[1]} = X^\top dZ_1$, with bias gradients as column sums. My NumPy implementation matched numerical gradients to about $10^{-11}$.

Softmax plus cross entropy: $J = -z_y + \log\sum_j e^{z_j}$, so $\partial J/\partial z_i = p_i - \mathbb{1}[i = y]$. For $z = (2, 1, 0.1)$ and $y = 0$, $p = (0.659, 0.242, 0.099)$, loss $0.417$, gradient $(-0.341, 0.242, 0.099)$, which sums to zero. The same $\hat y - y$ form appears in linear and logistic regression.

> **Watch:** [Karpathy, backprop ninja, cross entropy backward](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=5191s)

## Activation functions

| | formula | max slope | issue |
| --- | --- | --- | --- |
| sigmoid | $1/(1+e^{-x})$ | 0.25 | saturates, not zero centred |
| tanh | $(e^x - e^{-x})/(e^x + e^{-x})$ | 1 | saturates |
| ReLU | $\max(0, x)$ | 1 | dead units |
| leaky ReLU | $\max(\gamma x, x)$ | 1 | extra hyperparameter |
| GELU | $x\Phi(x)$ | about 1.13 | costlier |
| SiLU | $x\sigma(x)$ | about 1.10 | costlier |

$\sigma' = \sigma(1 - \sigma)$; its slope is 0.0066 at $x = 5$, and ten sigmoid layers give at most $0.25^{10} = 9.5 \times 10^{-7}$ from the activations alone. A dead ReLU has negative pre activation on every example, so its gradient is exactly zero; detect by firing rate per unit. GELU$(1) = 0.841$, SiLU$(1) = 0.731$, GELU$(-1) = -0.159$, SiLU$(-1) = -0.269$; the tanh approximation of GELU differs by at most $4.7 \times 10^{-4}$. SwiGLU gates one projection with SiLU of another.

Output layers: linear with MSE for regression, sigmoid with BCE for binary, softmax with cross entropy for exclusive classes, independent sigmoids for multilabel. Feed logits to fused losses.

## Loss functions

Maximum likelihood minimises $-\frac1n\sum \log p_\theta(y \mid x)$. A categorical model gives cross entropy, Bernoulli gives BCE, Gaussian noise gives $(y - f)^2/2\sigma^2$ plus constants (MSE), Laplace gives absolute error. At logit $-6$ with label 1, cross entropy has logit gradient $-0.9975$, while MSE through a sigmoid has $-0.00246$, about 405 times smaller.

Softmax with temperature: $e^{z_i/T}/\sum_j e^{z_j/T}$; for $z = (2, 1, 0.1)$, $T = 0.5$ gives $(0.864, 0.117, 0.019)$. It is shift invariant, so subtracting the max is legal: $\log\sum e^{z_j} = m + \log\sum e^{z_j - m}$. For logits $(1000, 1001, 1002)$ the naive softmax gives nan; the stable form gives $(0.090, 0.245, 0.665)$ and LSE $1002.4076$. A fresh $K$ class model should start near $\ln K$: 2.303 for 10, 3.296 for 27, 10.82 for 50,257. Scaling a 27 way output layer by 1.0, 0.1 and 0.01 gave initial losses 17.6, 3.66 and 3.298 in my simulation.

$H(p, q) = H(p) + D_{\mathrm{KL}}(p \| q)$ (for $p = (0.7, 0.2, 0.1)$, $q = (0.5, 0.3, 0.2)$: $0.8869 = 0.8018 + 0.0851$ nats). Perplexity is $e^{H(p,q)}$.

> **Watch:** [Karpathy, makemore part 3, fixing the initial loss](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=259s)

## Vanishing and exploding gradients

$\partial J/\partial h_l = (\prod_{k > l} J_k)^\top \partial J/\partial h_L$ with $J_k = \mathrm{diag}(\sigma')W_k$. Factors below 1 shrink the gradient exponentially, above 1 grow it: $0.9^{50} = 0.0052$, $1.1^{50} = 117$. In a 50 layer random Jacobian test the plain scale 0.9 net gave $5 \times 10^{-3}$, scale 1.1 gave 116, and residual blocks gave 1.3.

Fixes: ReLU family (saturation), He or Xavier init (scale), normalisation (drift), residual connections (depth), gates (long sequences), clipping (explosion only). Detect explosions by logging the global gradient norm. Clip by global norm: $g \leftarrow g\min(1, c/\|g\|)$, so $(3, 4)$ with $c = 1$ becomes $(0.6, 0.8)$. Clip after backward and before the step, and unscale first under loss scaling. Karpathy calls clipping a patch on deeper issues; the GPT-3 recipe clips at 1.0.

## Initialisation

Constant weights give identical units and identical gradients forever; zero weights give zero gradient to the earlier layer. For $z = \sum_{i=1}^{n_{in}} w_i x_i$, $\mathrm{Var}(z) = n_{in}\sigma_w^2 E[x^2]$, so linear units want $\sigma_w^2 = 1/n_{in}$, ReLU wants $2/n_{in}$ (He; ReLU halves the second moment), the backward pass wants $1/n_{out}$, and Xavier compromises at $2/(n_{in} + n_{out})$. Uniform bounds are $\sqrt{6/n}$ because $U[-a, a]$ has variance $a^2/3$.

Fifty layer, width 512 experiment, rms of activations at layers 1, 10, 50: tanh with std 0.01 gives 0.217, $3.4 \times 10^{-7}$, $5.3 \times 10^{-33}$; tanh with std 1.0 stays at 0.982 with 92% saturated; tanh with gain 5/3 holds 0.65; ReLU with $1/\sqrt n$ falls to $2.2 \times 10^{-8}$; He holds 0.74. Tanh contracts its input (the std of $\tanh(\mathcal{N}(0,1))$ is 0.628), hence the gain 5/3.

GPT style init uses normal std 0.02, zero bias, and residual projections scaled by $(2N)^{-1/2}$ because each block adds two branches; with $N = 12$ the stream variance is 24 unscaled and 1.00 scaled.

> **Watch:** [Karpathy, makemore part 3, Kaiming init](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=1673s), [GPT-2 reproduction, init](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4427s)

## Optimisers

SGD: $\theta \leftarrow \theta - \eta\hat g_B$, with gradient noise std falling as $s/\sqrt B$ (measured 3.00, 0.75, 0.19 for $B = 1, 16, 256$). Sampling with replacement leaves $1/e \approx 36.8\%$ of examples unseen; epochs sample without replacement. Linear learning rate scaling with warmup (Goyal et al. 2017) is the SGD heuristic for larger batches.

Momentum: $v \leftarrow \beta v + g$, $\theta \leftarrow \theta - \eta v$. On $f = \tfrac12(w_1^2 + 50 w_2^2)$ with $\eta = 0.01$, steps to loss $10^{-6}$: SGD 653, momentum ($\beta = 0.9$) 95, Nesterov 61; SGD diverges above $2/50 = 0.04$. Nesterov uses $\theta \leftarrow \theta - \eta(g + \beta v)$. AdaGrad divides by $\sqrt{\sum g^2}$ (rate decays like $1/\sqrt t$, 0.01 at $t = 10^4$); RMSProp uses an exponential average.

Adam (Kingma and Ba 2015): $m_t = \beta_1 m + (1 - \beta_1)g$, $v_t = \beta_2 v + (1 - \beta_2)g^2$, $\hat m = m/(1 - \beta_1^t)$, $\hat v = v/(1 - \beta_2^t)$, $\theta \leftarrow \theta - \alpha\hat m/(\sqrt{\hat v} + \epsilon)$; defaults $10^{-3}$, 0.9, 0.999, $10^{-8}$. Without correction the first step is $(1 - \beta_1)/\sqrt{1 - \beta_2} = 3.16\alpha$, which is the paper's bound. AdamW (Loshchilov and Hutter) applies $-\eta\lambda\theta$ outside the adaptive update, and reported about 15% lower relative test error than Adam with L2. The GPT-3 style recipe uses $\beta = (0.9, 0.95)$, weight decay 0.1 on tensors with two or more dimensions, clip 1.0; the update to data ratio should sit near $10^{-3}$.

## Learning rate schedules

Learning rate range test: sweep rates exponentially (Karpathy: exponent from $-3$ to 0 over 1000 steps) and pick the valley before instability, about 0.1 for his model. Warmup ramps linearly from near zero, because early gradients are large and Adam's second moment is estimated from few samples; pre norm reduces the need (Xiong et al. 2020). The build-nanogpt schedule: peak $6 \times 10^{-4}$, 715 warmup steps, then

$$\eta_t = \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})\big(1 + \cos(\pi\tfrac{t - W}{T - W})\big),$$

with $\eta_{\min} = 6 \times 10^{-5}$ and $T = 19{,}073$, giving $5.31 \times 10^{-4}$ at step 5000 and $3.25 \times 10^{-4}$ at step 10,000. Decay helps because SGD noise scales with $\eta/B$. Do not decay for continual learning on drifting data, and size the schedule to the training length.

> **Watch:** [Karpathy, GPT-2 reproduction, warmup and cosine decay](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8466s)

## Normalisation layers

Batch norm: $\hat x = (x - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$, $y = \gamma\hat x + \beta$, with statistics per channel over batch and space. In training it uses batch statistics and updates running averages with momentum 0.1; in evaluation it uses the running averages. Failure modes: batch size 1 (zero variance), forgetting `model.eval()`, per device statistics in data parallel training (SimCLR aggregates them as global BN). Batch norm's bias is redundant because mean subtraction cancels it.

Layer norm normalises over features of one example; RMSNorm is $\gamma z/\sqrt{\frac1m\sum z_i^2 + \epsilon}$ with no mean or shift, and is scale invariant; for $z = (3, 1, -2, 6)$ the RMS is 3.5355. Batch norm backward: $dx = \frac{1}{Ns}(N\,d\hat x - \sum d\hat x - \hat x\sum d\hat x\,\hat x)$ with $d\hat x = \gamma\,dy$, checked numerically. Why it helps: stable scale, larger learning rates, and noise from batch coupling; the covariate shift story is contested (Santurkar et al. 2018 point to smoother optimisation).

> **Watch:** [Karpathy, makemore part 3, batch normalisation](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=2440s), [makemore part 4, batch norm backward](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=5797s)

## Regularisation

Inverted dropout multiplies by a mask divided by $1 - p$ in training and does nothing at test time; for $\mathcal{N}(1,1)$ inputs and $p = 0.5$ the mean stays 1.0 while naive dropout gives 0.5, and the variance rises from 1 to 3. The backward pass reuses the cached mask. Weight decay multiplies weights by $(1 - \eta\lambda)$, identical to an L2 penalty in SGD but not in Adam; squared L2 is preferred because its gradient $2\theta$ is smooth, while the unsquared norm has a kink and constant gradient size. Early stopping keeps the best validation checkpoint after a patience window.

Augmentation applies label preserving transforms and must touch only the training split. Label smoothing sets targets to $(1 - \varepsilon)$ plus $\varepsilon/K$ (0.91 and 0.01 for $K = 10$, $\varepsilon = 0.1$), makes the optimal logit gap finite at $\ln(0.91/0.01) = 4.51$, and hurts distillation. Validation loss below training loss is usually benign: regularisation is active only in training, the training average lags, or the validation set is easier; check for leakage.

## Residual connections

$y = x + F(x)$, so $\partial y/\partial x = I + \partial F/\partial x$ and $\partial J/\partial x_0 = \partial J/\partial x_L\prod_k (I + \partial F_k/\partial x_{k-1})$ contains a term with no small Jacobians: a gradient highway. He et al. found deeper plain networks had higher training error, an optimisation problem, and the identity is easy to represent with $F \to 0$. Veit et al. view the network as $2^L$ paths. Post norm, $\mathrm{LN}(x + F(x))$, blocks the identity path and needs warmup; pre norm, $x + F(\mathrm{LN}(x))$, keeps it clean, needs a final norm and lets the stream variance grow with depth (about $2N\sigma^2$). nanoGPT's block is exactly the pre norm form with a final `ln_f`.

> **Watch:** [Karpathy, build GPT, residual connections](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5208s)

## Convolutional networks

A convolution computes $y[i,j,c] = b_c + \sum w[u,v,c',c]\,x[i+u, j+v, c']$ with local connectivity, shared weights and translation equivariance (checked by shifting an image in NumPy). Output size is $\lfloor (n + 2p - k)/s\rfloor + 1$: 224, $k=7$, $s=2$, $p=3$ gives 112. Parameters are $k^2 C_{in}C_{out} + C_{out}$: 73,856 for $3\times3$, $64 \to 128$, against about 80.6 billion for a dense layer between $56\times56$ maps. MACs are $H_{out}W_{out}C_{out}k^2C_{in}$: 231,211,008 for that layer on a $56 \times 56$ output.

Receptive field grows by $(k-1)j$ per layer; stacked $3\times3$ stride 1 layers see 3, 5, 7, 9. Two $3\times3$ layers match one $5\times5$ with $18C^2$ weights instead of $25C^2$ (28% fewer). Max pooling keeps the strongest response and routes gradient to the argmax; global average pooling removes the dense head. A $1\times1$ convolution mixes channels: the bottleneck $256 \to 64 \to 64 \to 256$ has 69,632 weights against 1,179,648 for two $3\times3$ layers at 256 channels. A depthwise separable $3\times3$, $64 \to 128$ has $576 + 8192 = 8768$ weights against 73,728 (8.4 times fewer).

ResNet-50: a $7\times7$ stride 2 stem, then stages of 3, 4, 6 and 3 bottleneck blocks at widths 256, 512, 1024, 2048, global average pooling and a 1000 way linear layer. Counting convolutions and batch norm gives 25,557,032 parameters. Convolutions are equivariant, only approximately invariant after pooling; a dense layer on a $7\times7\times C$ volume equals a $7\times7$ convolution, so classifiers can run convolutionally on larger images.

## Recurrent networks

$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$ with the same weights at every step; a layer has $h(h + d) + h$ parameters, 98,560 for $d = 128$, $h = 256$. Backprop through time unrolls the graph and sums gradient contributions over steps. $\partial h_T/\partial h_k = \prod_{j=k+1}^{T}\mathrm{diag}(1 - h_j^2)W_{hh}$, so a largest singular value below 1 gives exponential vanishing and a spectral radius above 1 allows explosion (Pascanu et al. 2013). In my 50 step test with hidden size 64, the spectral norm of $\partial h_{50}/\partial h_0$ was $3 \times 10^{-15}$, $5.5 \times 10^{-3}$, 0.26, 110 and $3.5 \times 10^{7}$ at recurrent scales 0.5, 0.9, 1.0, 1.5 and 3.0. Truncated BPTT detaches the state every $K$ steps. Remedies: clipping, orthogonal initialisation, gates, truncation, or attention.

## LSTMs and GRUs

With $u = [h_{t-1}, x_t]$: $f = \sigma(W_f u + b_f)$, $i = \sigma(W_i u + b_i)$, $g = \tanh(W_g u + b_g)$, $o = \sigma(W_o u + b_o)$, $c_t = f \odot c_{t-1} + i \odot g$, $h_t = o \odot \tanh(c_t)$. Parameters are $4(h(h+d) + h) = 394{,}240$ for $d = 128$, $h = 256$ (395,264 with PyTorch's two biases per gate). Along the cell path $\partial c_t/\partial c_{t-1} \approx \mathrm{diag}(f_t)$, with no matrix and no tanh derivative: $0.99^{100} = 0.366$, versus $0.9^{100} = 2.7 \times 10^{-5}$. Forget bias near 1 starts the gate at $\sigma(1) = 0.731$. LSTMs still need clipping.

A GRU has an update gate $z$ and reset gate $r$ and $h_t = (1 - z)h_{t-1} + z\tilde h$; it has 3 parameter blocks, 295,680 for the same sizes. Dropout on recurrent connections needs one mask per sequence (variational dropout, Gal and Ghahramani 2016) or is applied only between layers (Zaremba et al. 2014).

## Embeddings

An embedding is $E[i] = e_i^\top E$, a row lookup equal to one hot times a matrix. The backward pass scatter adds gradients into used rows only, and repeated ids sum. GPT-2 small's table has $50257 \times 768 = 38{,}597{,}376$ parameters, or 38,633,472 with the vocabulary padded to 50,304. Weight tying shares the input table with the output head (`wte.weight = lm_head.weight` in nanoGPT), removing about a third of the small model's parameters. Similarity is cosine similarity, and retrieval stores normalised embeddings for inner product search (CS229 notes, section 16.3).

> **Watch:** [Karpathy, makemore part 2, the embedding lookup table](https://www.youtube.com/watch?v=TCH_1BHY58I&t=739s)

## Autoencoders and VAEs

An autoencoder minimises $\|x - g(f(x))\|^2$ through a bottleneck. A linear one learns the PCA subspace: in my run with $k = 2$ both reached MSE 0.9975 and the projectors differed by 0.0000. A VAE uses prior $\mathcal{N}(0, I)$, decoder $p_\theta(x \mid z)$ and encoder $q_\phi(z \mid x)$. By Jensen,

$$\log p_\theta(x) \ge \mathbb{E}_q[\log p_\theta(x \mid z)] - D_{\mathrm{KL}}(q_\phi(z \mid x)\,\|\,p(z)),$$

with gap $D_{\mathrm{KL}}(q \| p(z \mid x))$. Closed form check: $p(z) = \mathcal{N}(0,1)$, $p(x \mid z) = \mathcal{N}(z, 1)$, $x = 1.3$ gives $\log p(x) = -1.688$, ELBO with the exact posterior $-1.688$, and with $q = \mathcal{N}(0,1)$ ELBO $-2.264$, a gap of 0.576 equal to the KL.

Reparameterisation $z = \mu + \sigma\odot\varepsilon$ moves the randomness out of the sampling distribution so gradients flow (CS229 notes, section 11.5). For a diagonal Gaussian $D_{\mathrm{KL}} = \tfrac12\sum(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2)$, 1.4171 for $\mu = (0.5, -1, 0)$, $\sigma = (0.8, 0.3, 1)$. Posterior collapse occurs with powerful decoders; KL annealing and free bits help. A $\beta$ weight trades reconstruction for regular latents. A fixed variance Gaussian decoder predicts means, hence blurry samples.

## GANs

$\min_G\max_D\ \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))]$. The optimal discriminator is $D^* = p_{data}/(p_{data} + p_g)$ and then $V(D^*, G) = -\log 4 + 2\,\mathrm{JSD}(p_{data}\|p_g)$, minimised at $p_g = p_{data}$ with $D^* = 1/2$. For $p_{data} = (0.5, 0.3, 0.2, 0)$ and $p_g = (0.1, 0.2, 0.3, 0.4)$, JSD is 0.2215 and both sides equal $-0.9433$. The non saturating generator loss $-\log D(G(z))$ avoids a vanishing gradient: at a fake logit of $-10$ the minimax gradient is $-4.5\times10^{-5}$ and the non saturating one is $-0.99995$.

Failure modes: mode collapse, oscillation, vanishing gradients with disjoint supports, no likelihood. Mitigations: Wasserstein losses (Arjovsky et al. 2017), spectral normalisation, two time scale updates, FID for evaluation. Compared with VAEs, GANs give sharper samples but no encoder or likelihood.

## Diffusion models

Forward: $x_t = \sqrt{1 - \beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t$. With $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod\alpha_s$, independent Gaussians give $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon$. DDPM uses $T = 1000$, linear $\beta$ from $10^{-4}$ to 0.02, a U Net with group norm, sinusoidal time embeddings and attention at $16\times16$. Computed $\bar\alpha$: 0.9999, 0.8970, 0.0786 and $4.04\times10^{-5}$ at $t = 1, 100, 500, 1000$. The iterative and closed form moments matched at $t = 300$ (mean 1.2615 against 1.2590).

Training: $L_{simple} = \|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\varepsilon, t)\|^2$; sampling: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\varepsilon_\theta) + \sigma_t z$. The CS229 derivation treats $x_{1:T}$ as latents with a fixed encoder, splits the ELBO's KL by the chain rule into one term per step, uses the Gaussian posterior $q(x_{t-1} \mid x_t, x_0)$ whose mean is linear in $x_0$ and $x_t$, and gets $\frac{1}{2\tilde\beta_t}\|\tilde\mu_t - \mu_\theta\|^2$; predicting the noise is equivalent to predicting that mean. The reverse step is Gaussian for small $\beta_t$ and remains noisy because $x_t$ does not determine $x_{t-1}$.

Toy check: with data $\{-1, +1\}$ the exact denoiser is $E[x_0 \mid x_t] = \tanh(\sqrt{\bar\alpha}x_t/(1 - \bar\alpha))$, and 1000 reverse steps from noise gave 50.1% positive and 49.9% negative samples at $|x| = 1$. Faster sampling uses DDIM, distillation and latent diffusion; classifier free guidance uses $\varepsilon_\varnothing + w(\varepsilon_c - \varepsilon_\varnothing)$.

> **Watch:** [CS229 Lecture 11, forward noising](https://www.youtube.com/watch?v=dqUMCzWjZSI&t=464s), [the ELBO](https://www.youtube.com/watch?v=dqUMCzWjZSI&t=3326s), [KL chain rule](https://www.youtube.com/watch?v=dqUMCzWjZSI&t=4024s)

## Contrastive learning

Positives are two augmentations of one example, negatives come from other examples (CS229 notes, section 16.2). InfoNCE is a softmax cross entropy, $\ell_i = -\log\frac{e^{s_{ii^+}}}{\sum_k e^{s_{ik}}}$ with $s = z_i^\top z_k/\tau$ on unit vectors. Constant embeddings give exactly $\ln N$ (5.545 for $N = 256$); unrelated random pairs at $\tau = 0.1$ gave 6.425; aligned pairs gave 4.561, 3.603 and 0.025 at $\tau = 1, 0.5, 0.1$. It lower bounds mutual information by $\ln N - \mathcal{L}$.

SimCLR (Chen et al. 2020): random crop with colour distortion is crucial, a one hidden layer MLP head is discarded after training, embeddings are normalised with a temperature, batch sizes run from 256 to 8192 (16,382 negatives per positive pair at 8192) with LARS and global batch norm, and linear evaluation reached 76.5% top 1 on ImageNet. MoCo uses a queue of negatives; BYOL has no negatives and needs a predictor and stop gradient to avoid collapse.

## Training recipe and debugging

Order: inspect data; build an end to end skeleton with baselines; check the initial loss against $\ln K$; overfit one batch (Karpathy does this in makemore part 2 and the GPT-2 reproduction); fit the training set; regularise; tune; squeeze. Curve reading: train down and validation up is overfitting, both high and flat is underfitting or a low rate, noisy swings mean a high rate, flat from step 0 means a bug.

For nan: find the first bad step, then separate data (nan, inf, bad labels), optimisation (rate, explosion; add warmup and clipping) and numerics (log of zero, division by zero, masks with all minus infinity, fp16 overflow). Track the loss, learning rate and gradient norm each step, and per layer activation histograms, gradient std and the update to data ratio $\log_{10}(\mathrm{std}(\eta\,\text{update})/\mathrm{std}(\text{data}))$, aiming near $-3$.

> **Watch:** [Karpathy, GPT-2 reproduction, overfitting a single batch](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3402s), [makemore part 3, the update to data ratio](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=5995s)

## Mixed precision

| | layout | max | smallest normal | epsilon |
| --- | --- | --- | --- | --- |
| fp32 | 1/8/23 | $3.4\times10^{38}$ | $1.18\times10^{-38}$ | $1.19\times10^{-7}$ |
| fp16 | 1/5/10 | 65,504 | $6.1\times10^{-5}$ | $9.77\times10^{-4}$ |
| bf16 | 1/8/7 | about $3.4\times10^{38}$ | as fp32 | $7.8\times10^{-3}$ |

fp16 has smallest subnormal $5.96\times10^{-8}$, and `float16(1e-8)` is 0. Summing 10,000 copies of $10^{-4}$ in fp16 stalls at 0.25, so accumulate in fp32. Mixed precision (Micikevicius et al. 2018) uses fp32 master weights, loss scaling and fp32 accumulation. Scaling by $2^{16}$ turns $10^{-8}$ into $6.55\times10^{-4}$, recovered as $9.997\times10^{-9}$. The paper picks a constant factor so $s\max|g| < 65{,}504$ (tested 8 to 32K), unscales before clipping, and skips steps on overflow; dynamic scaling halves on overflow and doubles after a run of good steps. bf16 needs none of it.

Model state memory is $2 + 2 + 12 = 16$ bytes per parameter for Adam in mixed precision, equal to fp32 Adam's 16; activations halve. Weight memory grows 50% for the master copy.

## Distributed training

Data parallelism averages gradients with an all reduce; a ring all reduce moves $2\frac{N-1}{N}S$ bytes per GPU (26.2, 29.5 and 30 GB for 15 GB of gradients at $N = 8$, 64 and infinity), and I simulated it in $2(N-1)$ steps. Gradient accumulation divides each micro loss by the number of steps; undivided gradients are exactly $4\times$ too large for 4 steps. Skip the all reduce except on the last micro step (`require_backward_grad_sync` in build-nanogpt).

Memory: 16 bytes per parameter gives 24 GB for 1.5B, 112 GB for 7B, 120 GB for 7.5B. ZeRO (Rajbhandari et al. 2020) shards optimiser states (stage 1), gradients (stage 2) and parameters (stage 3): for 7.5B on 64 GPUs, 120, 31.4, 16.6 and 1.9 GB per GPU, with communication 1x, 1x and 1.5x. FSDP follows the stage 3 idea. Tensor parallelism splits $A$ by columns and $B$ by rows in $\mathrm{act}(XA)B$ with one all reduce (verified numerically). Pipeline parallelism has bubble $(p-1)/(m+p-1)$: 42.9%, 27.3%, 8.6% for $(p, m) = (4,4), (4,8), (4,32)$. Activation checkpointing recomputes activations for about a third more compute.

> **Watch:** [CME295 Lecture 4, data parallelism with ZeRO](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=1869s), [model parallelism](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=2151s), [mixed precision](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=3360s)

## Attention as a bridge

$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^\top/\sqrt{d_k})V$ is a soft dictionary lookup: any two positions connect in one step, training parallelises over the sequence, and cost is $O(n^2 d)$ time with $O(n^2)$ score memory. A causal mask sets future scores to minus infinity. For unit variance entries $\mathrm{Var}(q^\top k) = d_k$, which I measured as 16.0, 63.8 and 510.6 for $d_k = 16, 64, 512$ and about 1 after scaling. Without scaling the softmax saturates: for 8 scores with std 8 the top weight was 0.999 against 0.264 at std 1. See the transformers and LLMs module for multi head attention, positions and the KV cache.

> **Watch:** [Karpathy, build GPT, the trick in self attention](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2831s)

## What interviewers probe

- Backprop: "Derive the gradient of softmax plus cross entropy." Then: "Why p minus y? What changes with label smoothing? How do you check it numerically?"
- Activations: "Why did ReLU replace sigmoid?" Then: "What is a dead ReLU, and why can a saturated tanh recover but a dead ReLU cannot?"
- Initialisation: "Derive He init." Then: "Why does tanh need a gain of 5/3? Why scale transformer residual projections by 1 over root 2N?"
- Optimisers: "Write Adam." Then: "Why bias correction? Why does Adam need two buffers? How does AdamW differ from L2? Why beta2 of 0.95 for LLMs?"
- Normalisation: "Batch norm in train versus eval?" Then: "Why does it fail at batch size 1? Why do transformers use layer norm or RMSNorm? Derive the backward pass."
- Regularisation: "Explain inverted dropout." Then: "Why can validation loss be below training loss? What does weight decay do under Adam?"
- Depth: "Why do residual connections help?" Then: "Pre norm or post norm? What happens to the residual stream variance?"
- CNNs: "Count the parameters of this layer." Then: "Receptive field? Why 3 by 3? What does a 1 by 1 convolution do? Depthwise separable cost?"
- Sequences: "Why do RNNs vanish?" Then: "How does an LSTM help, and why initialise the forget bias to 1? How do you do dropout in an LSTM?"
- Generative models: "Derive the ELBO." Then: "Why reparameterise? What is posterior collapse? Why does the diffusion loss predict noise? GAN versus diffusion trade offs?"
- Systems: "Why is my 7B model out of memory?" Then: "Where do 16 bytes per parameter come from? What does ZeRO stage 3 shard? Pipeline bubble? Why bf16 needs no loss scaling?"
- Debugging: "Loss is NaN." Then: "Overfit one batch first. What do you log? What is a healthy update to data ratio?"

## Sources

- [Stanford CS229 main lecture notes](https://cs229.stanford.edu/main_notes.pdf)
- [Chip Huyen, Machine Learning Interviews book, training neural networks](https://huyenchip.com/ml-interviews-book/contents/8.3-training-neural-networks.html)
- [Chip Huyen, Machine Learning Interviews book, computer vision](https://huyenchip.com/ml-interviews-book/contents/8.2.2-computer-vision.html)
- [Karpathy, micrograd engine.py](https://raw.githubusercontent.com/karpathy/micrograd/master/micrograd/engine.py)
- [Karpathy, build-nanogpt train_gpt2.py](https://raw.githubusercontent.com/karpathy/build-nanogpt/master/train_gpt2.py)
- [Karpathy, nanoGPT model.py](https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py)
- [Karpathy, micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Karpathy, makemore part 1](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Karpathy, makemore part 2](https://www.youtube.com/watch?v=TCH_1BHY58I)
- [Karpathy, makemore part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc)
- [Karpathy, makemore part 4](https://www.youtube.com/watch?v=q8SA3rM6ckI)
- [Karpathy, Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy, Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Stanford CS229 Lecture 7, neural networks architecture](https://www.youtube.com/watch?v=fRM41w9jzQo)
- [Stanford CS229 Lecture 8, backprop](https://www.youtube.com/watch?v=ne2ngVAoMG8)
- [Stanford CS229 Lecture 11, diffusion models](https://www.youtube.com/watch?v=dqUMCzWjZSI)
- [Stanford CS229 Lecture 12, representation learning](https://www.youtube.com/watch?v=_kREM2UAiJ8)
- [Stanford CME295 Lecture 4, LLM training](https://www.youtube.com/watch?v=VlA_jt_3Qc4)
- [Kingma and Ba 2015, Adam](https://arxiv.org/pdf/1412.6980)
- [Loshchilov and Hutter 2019, Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101)
- [Rajbhandari et al. 2020, ZeRO](https://arxiv.org/pdf/1910.02054)
- [Ho, Jain and Abbeel 2020, DDPM](https://arxiv.org/pdf/2006.11239)
- [Micikevicius et al. 2018, Mixed Precision Training](https://arxiv.org/pdf/1710.03740)
- [Chen et al. 2020, SimCLR](https://arxiv.org/pdf/2002.05709)
