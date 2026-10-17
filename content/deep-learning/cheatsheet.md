# Deep Learning Cheatsheet

One page of formulas, defaults and numbers to review before an interview. Every number was computed in NumPy or read from a named paper or lecture, and the cards in this module explain each line.

## Backprop and losses

- Chain rule at a node: gradient in = local derivative times gradient from above. Add node distributes, multiply swaps inputs, max routes, a value used twice sums.
- Linear layer $Z = XW + b$: $dW = X^\top G$, $db = \sum_{rows} G$, $dX = G W^\top$. A gradient has its variable's shape.
- Softmax plus cross entropy: $\partial J/\partial z = p - y$, divided by $B$ for a batch mean. For $z = (2, 1, 0.1)$: $(-0.341, 0.242, 0.099)$.
- Stable softmax: subtract the max; $\mathrm{LSE}(z) = m + \log\sum e^{z - m}$; loss $= -z_y + \mathrm{LSE}$.
- Start loss $\approx \ln K$: 2.303 (10), 3.296 (27), 10.82 (50,257). Far above means large logits.
- Likelihood: categorical gives cross entropy, Gaussian gives MSE, Laplace gives MAE, Bernoulli gives BCE.
- $H(p, q) = H(p) + D_{KL}(p\|q)$; perplexity $= e^{H}$.
- Gradient check: central difference, $h = 10^{-5}$, float64, relative error below $10^{-7}$.

## Activations

| | max slope | note |
| --- | --- | --- |
| sigmoid | 0.25 | slope 0.0066 at $x = 5$, saturates |
| tanh | 1 | zero centred, saturates |
| ReLU | 1 | dead units when always negative |
| GELU, SiLU | 1.13, 1.10 | smooth, small negative dip |

Output layer: linear and MSE, sigmoid and BCE, softmax and CE, independent sigmoids for multilabel.

## Gradients and initialisation

- Gradient through $L$ layers is a product of Jacobians: $0.9^{50} = 0.005$, $1.1^{50} = 117$.
- Clip by global norm: $g \leftarrow g\min(1, c/\|g\|)$, $c = 1$ for transformers; unscale first under loss scaling.
- Never init to a constant: units stay identical.
- Xavier $2/(n_{in} + n_{out})$ for tanh. He $2/n_{in}$ for ReLU. Tanh gain $5/3$.
- GPT init: std 0.02, zero bias, residual projections times $(2N)^{-1/2}$ (variance 24 against 1 at $N = 12$).
- 50 layer test: ReLU with $1/\sqrt n$ shrinks to $2\times10^{-8}$, He holds 0.74.

## Optimisers and schedules

| | update | defaults |
| --- | --- | --- |
| SGD | $\theta - \eta g$ | noise std $\propto 1/\sqrt B$ |
| momentum | $v = \beta v + g$ | $\beta = 0.9$ |
| Nesterov | $\theta - \eta(g + \beta v)$ | |
| AdaGrad | divide by $\sqrt{\sum g^2}$ | rate decays as $1/\sqrt t$ |
| RMSProp | EMA of $g^2$ | $\rho = 0.9$ |
| Adam | $\alpha\hat m/(\sqrt{\hat v} + \epsilon)$ | $10^{-3}$, 0.9, 0.999, $10^{-8}$ |
| AdamW | plus $-\eta\lambda\theta$ outside | $\lambda = 0.1$ for GPT style |

- Bias correction: $\hat m = m/(1 - \beta_1^t)$. Uncorrected first step is $3.16\alpha$.
- LLM recipe: AdamW, betas (0.9, 0.95), clip 1.0, warmup 715 steps to $6\times10^{-4}$, cosine to $6\times10^{-5}$.
- Toy bowl with curvatures 1 and 50, rate 0.01: SGD 653 steps, momentum 95, Nesterov 61.
- Batch up by $k$: rate up by $k$ with warmup (SGD), diminishing returns past the critical batch size.
- Learning rate range test: exponential sweep, pick the valley before instability. Update to data ratio near $10^{-3}$.

## Normalisation, regularisation, residuals

- Batch norm: $\hat x = (x - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$, then $\gamma\hat x + \beta$. Eval uses running stats (momentum 0.1). Fails at batch size 1.
- Layer norm: per example over features. RMSNorm: $\gamma z/\mathrm{RMS}(z)$, no mean. Transformers use LN or RMSNorm.
- Inverted dropout: mask over $(1 - p)$ in training, identity at test. Reuse the mask in backward.
- Weight decay $= (1 - \eta\lambda)\theta$; equals L2 in SGD, not in Adam. Decay matrices, not biases or norms.
- Label smoothing $K = 10$, $\varepsilon = 0.1$: targets 0.91 and 0.01, logit gap 4.51.
- Validation below training loss: regularisation off at eval, epoch average lags, easier split, or leakage.
- Residual $y = x + F(x)$: Jacobian $I + \partial F/\partial x$. Pre norm $x + F(\mathrm{LN}(x))$ needs a final norm.

## CNNs and sequences

- Output size $\lfloor (n + 2p - k)/s\rfloor + 1$. Params $k^2 C_{in}C_{out} + C_{out}$: 73,856 for $3\times3$, $64\to128$.
- Receptive field grows $(k-1)\cdot$jump per layer. Two $3\times3$ equal one $5\times5$ with 28% fewer weights.
- Bottleneck $256\to64\to64\to256$: 69,632 weights. Depthwise separable $64\to128$: 8,768 against 73,728.
- ResNet-50: 25,557,032 parameters, stages 3, 4, 6, 3.
- RNN $h(h+d)+h$; LSTM $4\times$ that (394,240 at $d = 128$, $h = 256$); GRU $3\times$ (295,680).
- LSTM cell path: $\partial c_t/\partial c_{t-1} \approx f_t$; $0.99^{100} = 0.37$. Forget bias near 1. Clip anyway.
- Embedding lookup equals one hot times a matrix; GPT-2 small table is 38,597,376 parameters, tied with the output head.

## Generative models and contrastive learning

- ELBO $= \mathbb{E}_q[\log p(x|z)] - D_{KL}(q\|p(z))$; gap is $D_{KL}(q\|p(z|x))$. Reparameterise $z = \mu + \sigma\varepsilon$. Diagonal Gaussian KL $= \tfrac12\sum(\mu^2 + \sigma^2 - 1 - \log\sigma^2)$.
- GAN: $D^* = p_{data}/(p_{data} + p_g)$, value $-\log 4 + 2\,\mathrm{JSD}$. Use the non saturating loss. Failures: mode collapse, instability.
- Diffusion: $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\varepsilon$; DDPM $T = 1000$, $\beta$ from $10^{-4}$ to $0.02$, $\bar\alpha_T = 4\times10^{-5}$. Train by predicting the noise; sample by 1000 reverse steps.
- InfoNCE: softmax over candidates, chance $\ln N$ (5.545 at 256). SimCLR: crop plus colour, projection head dropped, big batch.

## Debugging and systems

- Order: data, skeleton, initial loss, overfit one batch, fit, regularise, tune.
- NaN: find the first bad step; check data, learning rate and gradient norm, numerics (log 0, all minus infinity mask, fp16).
- fp16 max 65,504, smallest subnormal $5.96\times10^{-8}$, eps $9.8\times10^{-4}$. bf16 has fp32 range, eps $7.8\times10^{-3}$, no loss scaling.
- Loss scaling: scale up, unscale in fp32 before clipping, skip on overflow.
- Memory: Adam mixed precision is $2 + 2 + 12 = 16$ bytes per parameter. 7.5B gives 120 GB.
- ZeRO 1, 2, 3 for 7.5B on 64 GPUs: 31.4, 16.6, 1.9 GB; communication 1x, 1x, 1.5x.
- Ring all reduce: $2(N-1)/N$ times the size per GPU. Accumulate by dividing each loss by the steps.
- Pipeline bubble $(p-1)/(m+p-1)$: 27.3% at $p = 4$, $m = 8$. Tensor parallel: columns then rows, one all reduce.
- Attention: $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$; $\mathrm{Var}(q^\top k) = d_k$.

## Say it like this

> **Say it like this:** "Backprop is the chain rule with caching: forward stores activations, backward multiplies local derivatives by the gradient from above, and one pass gives every parameter's gradient."

> **Say it like this:** "Adam keeps a moving average of the gradient and its square, corrects both for starting at zero, and steps by their ratio, so each parameter moves about the learning rate."

> **Interview trap:** Residual connections fix optimisation of deep networks, not overfitting, and clipping fixes exploding gradients, not vanishing ones.
