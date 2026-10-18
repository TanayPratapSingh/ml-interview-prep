# Andrej Karpathy, Neural Networks: Zero to Hero

This series builds a GPT from a blank notebook. It starts with a scalar autograd engine, grows into character level language models, then a transformer, a tokenizer and a full reproduction of GPT-2, with one talk on how assistants like ChatGPT are trained. Every note below links to the minute of the video where the idea is taught, and the matching flashcards are in the Zero to Hero deck.

## Video 1: Building micrograd

[Watch](https://www.youtube.com/watch?v=VMj-3S1tku0&t=0s) (2:25:52). Original title: The spelled out intro to neural networks and backpropagation, building micrograd.

Karpathy builds a tiny automatic differentiation engine from nothing, then uses it to train a small neural network. You see that backpropagation is the chain rule applied node by node over a recorded expression graph, and that a neural net is just a large expression whose weights we nudge to lower a loss. By the end you have a 100 line engine, a neuron, layer and MLP library on top, and a training loop that matches how PyTorch behaves. The lecture also shows the classic bugs: gradients that must accumulate, and gradients that must be zeroed between steps.

### Key ideas

- [0:25](https://www.youtube.com/watch?v=VMj-3S1tku0&t=25s) micrograd is an autograd engine: it records how a result was computed and backpropagation gives the gradient of the loss with respect to every weight. The engine is about 100 lines and the neural net library on top is about 50.
- [5:24](https://www.youtube.com/watch?v=VMj-3S1tku0&t=324s) The engine works on single numbers on purpose. Tensors only pack many scalars into arrays for speed, and none of the maths changes.
- [8:08](https://www.youtube.com/watch?v=VMj-3S1tku0&t=488s) A derivative is the sensitivity of the output to a tiny nudge of an input. You can estimate it as rise over run with a small h, for example a slope of 14 at x equal to 3 for 3x squared minus 4x plus 5.
- [14:12](https://www.youtube.com/watch?v=VMj-3S1tku0&t=852s) With several inputs each one has its own slope. For d equal to a times b plus c, the slopes are b, a and 1.
- [19:09](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1149s) A Value object wraps a number and overloads add and multiply, so each operation returns a new Value that remembers its children and the operation that made it.
- [32:10](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1930s) Manual backpropagation: set the output gradient to 1, then at each node multiply the incoming gradient by the local derivative. Nudging inputs by h gives a numerical gradient check.
- [37:50](https://www.youtube.com/watch?v=VMj-3S1tku0&t=2270s) The chain rule is the crux. An add node routes the gradient to both inputs unchanged, and a multiply node gives each input the other input's value times the incoming gradient.
- [51:10](https://www.youtube.com/watch?v=VMj-3S1tku0&t=3070s) One optimisation step: move every input a small step along its gradient and the output goes up (in the example it moves from minus 8 to about minus 7.3).
- [52:52](https://www.youtube.com/watch?v=VMj-3S1tku0&t=3172s) A neuron computes tanh of the weighted sum of its inputs plus a bias. Its derivative is one minus tanh squared, written in terms of the output.
- [1:09:02](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4142s) Each operation stores a small backward closure that chains the output gradient into its inputs. Leaf nodes have an empty one.
- [1:17:32](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4652s) backward() runs those closures in reverse topological order, so a node only passes gradients on after everything downstream has delivered its share.
- [1:22:28](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4948s) A variable used more than once breaks plain assignment. The fix is to accumulate with plus equals, which is the multivariate chain rule.
- [1:27:05](https://www.youtube.com/watch?v=VMj-3S1tku0&t=5225s) Composite operations such as tanh can be built from exp, add, power and division and give identical gradients. The level at which you write backward rules is your choice.
- [1:39:31](https://www.youtube.com/watch?v=VMj-3S1tku0&t=5971s) PyTorch gives the same numbers through tensors with requires_grad, and its tensors expose data and grad just like micrograd.
- [1:43:55](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6235s) A neuron, a layer and an MLP are plain Python classes. The 3, 4, 4, 1 MLP has 41 parameters.
- [1:51:04](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6664s) A tiny four example dataset and a squared error loss turn how good the net is into one number.
- [2:01:12](https://www.youtube.com/watch?v=VMj-3S1tku0&t=7272s) Gradient descent steps against the gradient. Too large a step overshoots, and forgetting to zero the gradients is a subtle bug that this tiny problem happens to hide.
- [2:14:03](https://www.youtube.com/watch?v=VMj-3S1tku0&t=8043s) Summary: a forward pass, a loss, backward, and an update, repeated. GPT trains the same way at far larger scale with cross entropy loss on next token prediction.
- [2:16:46](https://www.youtube.com/watch?v=VMj-3S1tku0&t=8206s) The real repo adds ReLU, minibatches, a max margin loss, L2 regularisation and learning rate decay, and PyTorch's own tanh backward lives in separate CPU and GPU kernels.

### Code to remember

The core of the engine: a node that records its inputs and a backward pass over a topological order.

```python
class Value:
    def __init__(self, data, prev=()):
        self.data, self.grad, self._prev = data, 0.0, prev
        self._backward = lambda: None

    def __add__(self, o):
        out = Value(self.data + o.data, (self, o))
        def _backward():
            self.grad += out.grad
            o.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, o):
        out = Value(self.data * o.data, (self, o))
        def _backward():
            self.grad += o.data * out.grad
            o.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

A gradient check against rise over run, using the class above.

```python
def L(a, b, c, f):
    return (a * b + c) * f

h = 1e-6
num = (L(2 + h, -3, 10, -2) - L(2, -3, 10, -2)) / h
a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
L(a, b, c, f).backward()
print(round(num, 3), a.grad)   # 6.0 6.0
```

### Interview questions this prepares you for

- Explain backpropagation to someone who remembers a little calculus.
- Why do gradients have to be accumulated with plus equals rather than assigned?
- What does zero_grad do, and what goes wrong if you forget it?
- How would you check that your backward pass is correct?
- Implement reverse mode autodiff for add, multiply and tanh.
- Why is reverse mode used for training and not forward mode or numerical differences?

### Related study modules

- [Deep learning foundations](#/module/deep-learning?d=deep)
- [ML coding](#/module/ml-coding?d=deep)

## Video 2: Building makemore, the bigram model

[Watch](https://www.youtube.com/watch?v=PaCmpygFfXo&t=0s) (1:57:45). Original title: The spelled out intro to language modeling, building makemore.

Karpathy starts a character level language model that generates new names. He trains a bigram model twice: first by counting how often each character follows another and normalising the counts, then as a one layer neural net with softmax output trained by gradient descent on the negative log likelihood. Both routes land on the same loss, about 2.45, and the same model, which shows that the neural framing is a more flexible way to reach what counting gives directly. Along the way you learn tensors, broadcasting, sampling, the loss for classification, smoothing and its link to regularisation.

### Key ideas

- [1:06](https://www.youtube.com/watch?v=PaCmpygFfXo&t=66s) makemore learns from about 32,000 names and generates new name like strings. It is a character level language model: each name is a sequence of characters and the model predicts the next one.
- [4:24](https://www.youtube.com/watch?v=PaCmpygFfXo&t=264s) One name holds several training examples, one for each position plus one for the end of the name. A bigram model only uses the previous character.
- [12:45](https://www.youtube.com/watch?v=PaCmpygFfXo&t=765s) Counting adjacent pairs into a 2D tensor is the whole training step. Rows are the current character and columns are the next character.
- [20:54](https://www.youtube.com/watch?v=PaCmpygFfXo&t=1254s) Using one special dot token for both start and end shrinks the table to 27 by 27 and removes rows and columns that are always zero.
- [24:02](https://www.youtube.com/watch?v=PaCmpygFfXo&t=1442s) To sample, read the row for the current character, treat it as probabilities, draw the next character with a seeded generator, and stop when the dot comes up.
- [36:17](https://www.youtube.com/watch?v=PaCmpygFfXo&t=2177s) Normalise all rows at once with a keepdim sum and broadcasting. Leaving out keepdim silently normalises the columns instead, a bug that produces no error.
- [50:14](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3014s) The quality of the model is the negative log likelihood: the average of minus the log probability given to each true next character. About 2.45 on the training set, against about 3.30 for a uniform guess.
- [1:00:50](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3650s) Model smoothing adds fake counts so no bigram gets probability zero and an unseen bigram does not cost infinite loss.
- [1:02:57](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3777s) The neural approach: the input is a character index, the output is a probability distribution over the next character, and the loss is the same negative log likelihood.
- [1:10:01](https://www.youtube.com/watch?v=PaCmpygFfXo&t=4201s) Integers are fed to a network as one hot vectors, cast to float. The one hot function returns integers, so the cast matters.
- [1:13:53](https://www.youtube.com/watch?v=PaCmpygFfXo&t=4433s) A linear layer is a matrix multiply that evaluates all neurons on all examples in parallel: 5 examples by 27 inputs times 27 by 27 weights gives 5 by 27 outputs.
- [1:18:46](https://www.youtube.com/watch?v=PaCmpygFfXo&t=4726s) Interpret the outputs as log counts (logits), exponentiate to get counts and normalise rows. Those two steps are the softmax.
- [1:35:49](https://www.youtube.com/watch?v=PaCmpygFfXo&t=5749s) Pluck the probability of each correct next character with integer indexing, take the log, average and negate.
- [1:38:36](https://www.youtube.com/watch?v=PaCmpygFfXo&t=5916s) Backward and update in PyTorch: reset grads, call backward on the loss, then subtract the learning rate times the gradient from the weights.
- [1:42:55](https://www.youtube.com/watch?v=PaCmpygFfXo&t=6175s) Trained on all 228,000 bigrams, gradient descent reaches about the same loss as counting, 2.45, because the model class is the same.
- [1:47:49](https://www.youtube.com/watch?v=PaCmpygFfXo&t=6469s) A one hot input times the weight matrix just selects one row of the weights, which is exactly the table lookup we had before.
- [1:50:18](https://www.youtube.com/watch?v=PaCmpygFfXo&t=6618s) Smoothing has a neural equivalent: a penalty on the squared weights pulls them towards zero, which pulls the probabilities towards uniform.

### Code to remember

Count bigrams, normalise rows with keepdims, and read off the negative log likelihood of a name.

```python
import math
import numpy as np

words = ['emma', 'olivia', 'ava']
chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
V = len(stoi)

N = np.zeros((V, V))
for w in words:
    cs = ['.'] + list(w) + ['.']
    for a, b in zip(cs, cs[1:]):
        N[stoi[a], stoi[b]] += 1

P = (N + 1) / (N + 1).sum(1, keepdims=True)   # smoothed, each row sums to 1

cs = ['.'] + list('emma') + ['.']
logp = [math.log(P[stoi[a], stoi[b]]) for a, b in zip(cs, cs[1:])]
print(round(-sum(logp) / len(logp), 3))        # 1.505
```

Softmax and the vectorised loss.

```python
import numpy as np

logits = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 3.0]])
counts = np.exp(logits)
probs = counts / counts.sum(1, keepdims=True)
ys = np.array([0, 2])
loss = -np.log(probs[np.arange(len(ys)), ys]).mean()
print(probs.sum(1), round(loss, 3))            # [1. 1.] 0.285
```

### Interview questions this prepares you for

- What is a language model, and what loss do you train it with?
- What is the negative log likelihood and why do we use logs?
- What is softmax, and why interpret the network output as log counts?
- What happens if a bigram never appears in the training set, and how do you fix it?
- Explain broadcasting and give an example of a silent broadcasting bug.
- Why does a one hot vector times a weight matrix equal a table lookup?

### Related study modules

- [Deep learning foundations](#/module/deep-learning?d=deep)
- [Transformers and LLMs](#/module/transformers-llms?d=deep)
- [Probability and statistics](#/module/probability-statistics?d=deep)
- [ML coding](#/module/ml-coding?d=deep)

## Video 3: Makemore Part 2, the MLP

[Watch](https://www.youtube.com/watch?v=TCH_1BHY58I&t=0s) (1:15:40). Original title: Building makemore Part 2, MLP.

The count table from the last video cannot take more context, because it grows exponentially with the number of previous characters. Karpathy follows the 2003 paper by Bengio and colleagues instead: each character gets a learned embedding vector, three embeddings are joined, and a tanh hidden layer and a softmax predict the next character. Along the way he introduces the everyday machinery of machine learning: minibatches, a learning rate search, train, dev and test splits, and the difference between underfitting and overfitting. The tuned model reaches a dev loss of about 2.17, against 2.45 for the bigram table on its training set.

### Key ideas

- [0:00](https://www.youtube.com/watch?v=TCH_1BHY58I&t=0s) A count table needs 27 to the power of k rows for k characters of context, which is 729 rows for two and 19,683 for three. Most contexts would have too few counts to be useful.
- [1:48](https://www.youtube.com/watch?v=TCH_1BHY58I&t=108s) Bengio et al. 2003 give every word a learned vector from a shared lookup table, join the vectors for the context, and use a tanh hidden layer and softmax output. Similar words end up close together, so knowledge transfers between them.
- [9:03](https://www.youtube.com/watch?v=TCH_1BHY58I&t=543s) The dataset is a rolling window: the context starts as dots, the next character is the label, and the window slides one step. Five names give 32 examples with block size 3.
- [12:19](https://www.youtube.com/watch?v=TCH_1BHY58I&t=739s) Indexing the embedding table with the whole input tensor gives a 32 by 3 by 2 result in one step. This is the same as a one hot vector times the table.
- [18:35](https://www.youtube.com/watch?v=TCH_1BHY58I&t=1115s) view reshapes a tensor by changing only metadata over one flat storage, so it is free. Concatenation copies memory.
- [32:17](https://www.youtube.com/watch?v=TCH_1BHY58I&t=1937s) Shapes through the net: 32 by 3, then 32 by 3 by 2, then 32 by 6, then 32 by 100, then 32 by 27. With 100 hidden units the model has 3,481 parameters.
- [32:49](https://www.youtube.com/watch?v=TCH_1BHY58I&t=1969s) Use F.cross_entropy on the raw logits. It is faster, has a simpler backward pass, and subtracts the largest logit so it never overflows.
- [37:56](https://www.youtube.com/watch?v=TCH_1BHY58I&t=2276s) Overfitting a single batch of 32 examples is a quick check that the training loop works. The loss cannot reach exactly zero because the same input has several labels.
- [41:25](https://www.youtube.com/watch?v=TCH_1BHY58I&t=2485s) Minibatches make each step cheap. The gradient is noisier, but many cheap steps beat a few exact ones.
- [45:40](https://www.youtube.com/watch?v=TCH_1BHY58I&t=2740s) Find the learning rate by sweeping the exponent from minus 3 to 0 within one run and picking the valley before the loss explodes. About 0.1 works here, with a drop to 0.01 at the end.
- [53:20](https://www.youtube.com/watch?v=TCH_1BHY58I&t=3200s) Split the data into 80 percent train, 10 percent dev and 10 percent test. Tune on dev and touch test only a few times.
- [1:00:49](https://www.youtube.com/watch?v=TCH_1BHY58I&t=3649s) Train and dev loss about equal means underfitting. A bigger hidden layer alone did not help much.
- [1:05:27](https://www.youtube.com/watch?v=TCH_1BHY58I&t=3927s) A plot of the 2 dimensional embeddings shows vowels clustered together, and q and the dot token far from the rest.
- [1:07:16](https://www.youtube.com/watch?v=TCH_1BHY58I&t=4036s) Widening the embedding from 2 to 10 dimensions helped more than widening the hidden layer. The best dev loss was about 2.17.
- [1:13:24](https://www.youtube.com/watch?v=TCH_1BHY58I&t=4404s) Sampling slides a context window: embed, hidden layer, logits, softmax, draw with multinomial, repeat until the dot.

### Code to remember

Build a rolling window dataset, look up embeddings, and check the forward pass shapes.

```python
import numpy as np

def build(words, block=3):
    stoi = {c: i + 1 for i, c in enumerate(sorted(set(''.join(words))))}
    stoi['.'] = 0
    X, Y = [], []
    for w in words:
        ctx = [0] * block
        for ch in w + '.':
            X.append(ctx)
            Y.append(stoi[ch])
            ctx = ctx[1:] + [stoi[ch]]      # slide the window
    return np.array(X), np.array(Y)

X, Y = build(['emma', 'olivia', 'ava', 'isabella', 'sophia'])
rng = np.random.default_rng(0)
C = rng.standard_normal((27, 2))            # embedding table
W1, b1 = rng.standard_normal((6, 100)), rng.standard_normal(100)
W2, b2 = rng.standard_normal((100, 27)), rng.standard_normal(27)
emb = C[X]                                  # (32, 3, 2)
h = np.tanh(emb.reshape(len(X), -1) @ W1 + b1)
logits = h @ W2 + b2
print(X.shape, emb.shape, h.shape, logits.shape)   # (32, 3) (32, 3, 2) (32, 100) (32, 27)
```

A numerically stable cross entropy from logits.

```python
import numpy as np

logits = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 3.0]])
ys = np.array([0, 2])
m = logits.max(1, keepdims=True)                      # subtract the max first
logp = logits - m - np.log(np.exp(logits - m).sum(1, keepdims=True))
print(round(-logp[np.arange(len(ys)), ys].mean(), 3))  # 0.285
```

### Interview questions this prepares you for

- Why does an embedding table help a language model generalise to unseen contexts?
- Why use cross entropy from logits instead of softmax followed by log?
- How do you choose a learning rate?
- What are train, dev and test sets for, and why touch the test set rarely?
- Train and dev loss are almost equal and both are high. What does that mean and what would you try?
- Why train on minibatches?

### Related study modules

- [Deep learning foundations](#/module/deep-learning?d=deep)
- [Transformers and LLMs](#/module/transformers-llms?d=deep)
- [Classical machine learning](#/module/classical-ml?d=deep)
- [ML coding](#/module/ml-coding?d=deep)

## Video 4: Makemore Part 3, activations, gradients, BatchNorm

[Watch](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=0s) (1:55:58). Original title: Building makemore Part 3, Activations and Gradients, BatchNorm.

Karpathy stays with the MLP to build intuition for what happens inside a network while it trains. He fixes a wildly wrong starting loss, then a layer of saturated tanh units, and derives the Kaiming rule for weight scales. He then introduces batch normalisation as the tool that keeps activations well behaved without hand tuning, along with its costs. The video ends by wrapping the code in PyTorch style modules and showing diagnostics: activation and gradient histograms and the update to data ratio.

### Key ideas

- [4:19](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=259s) The expected loss at initialisation for 27 classes is the log of 27, about 3.29. Seeing 27 means the network is confidently wrong, so set the output bias to zero and shrink the output weights.
- [12:59](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=779s) Most hidden activations sat at minus 1 or 1 because the pre-activations spread from about minus 5 to 15. Saturated tanh units pass back almost no gradient, since the local derivative is one minus tanh squared.
- [19:32](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=1172s) A dead neuron never gets a gradient. This applies to tanh, sigmoid and especially ReLU, which has a flat region below zero. Leaky ReLU avoids it.
- [23:56](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=1436s) Scaling W1 by 0.2 and b1 by 0.01 fixed the saturation. Validation loss went 2.17, then 2.13 after the softmax fix, then 2.10.
- [27:53](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=1673s) Multiplying unit Gaussian inputs by unit Gaussian weights widens the output to about the square root of the fan in. Dividing the weights by that square root keeps the standard deviation at 1.
- [31:19](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=1879s) Kaiming initialisation uses a standard deviation of gain over the square root of the fan in. The gain is 1 for linear, the square root of 2 for ReLU and 5 over 3 for tanh.
- [36:16](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=2176s) Residual connections, normalisation layers and better optimisers such as Adam make initialisation much less delicate than it used to be.
- [40:40](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=2440s) Batch normalisation standardises each neuron over the batch, then applies a learned gain and bias. It is differentiable and placed after linear or convolution layers.
- [50:16](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=3016s) It couples the examples in a batch. That adds noise that regularises, and it is also the source of many bugs.
- [54:08](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=3248s) For inference, keep running estimates of the mean and variance with a moving average, so single examples can be processed.
- [1:01:26](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=3686s) A bias in the layer before batch norm is cancelled by the mean subtraction, so set bias to False there.
- [1:04:50](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=3890s) ResNet stacks convolution, batch norm and ReLU blocks, with no bias on the convolutions.
- [1:18:35](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=4715s) The code is wrapped in Linear, BatchNorm1d and Tanh modules with the same API as PyTorch. Running statistics are buffers, not parameters.
- [1:26:51](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=5211s) Plot forward activation histograms and percent saturation. About 5 percent saturation and a standard deviation near 0.65 is healthy for a gain of 5 over 3.
- [1:32:07](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=5527s) A stack of linear layers with no nonlinearity collapses to a single linear layer, and the correct gain there is exactly 1.
- [1:39:55](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=5995s) The update to data ratio should be about 1 in 1000, which is minus 3 on a log10 scale. Far below means the learning rate is too small.
- [1:46:04](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=6364s) With batch norm, forward and backward statistics stay healthy even if the gain is changed, but the learning rate may need retuning.

### Code to remember

Activation statistics through a stack of linear and tanh layers, for three gains.

```python
import numpy as np
rng = np.random.default_rng(0)

def run(gain, layers=5, width=100, n=1000):
    x = rng.standard_normal((n, width))
    stats = []
    for _ in range(layers):
        x = np.tanh(x @ (rng.standard_normal((width, width)) * gain / width ** 0.5))
        stats.append((round(float(x.std()), 2), round(float((np.abs(x) > 0.97).mean()) * 100, 1)))
    return stats

for g in (1.0, 5 / 3, 3.0):
    print(round(g, 2), run(g)[-1])    # (std, percent saturated) in the last layer
```

A batch norm layer with running statistics.

```python
import numpy as np

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps, self.momentum, self.training = eps, momentum, True
        self.gamma, self.beta = np.ones(dim), np.zeros(dim)              # learned
        self.running_mean, self.running_var = np.zeros(dim), np.ones(dim)  # buffers

    def __call__(self, x):
        mean, var = (x.mean(0), x.var(0)) if self.training else (self.running_mean, self.running_var)
        out = self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta
        if self.training:
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mean
            self.running_var = (1 - m) * self.running_var + m * var
        return out

rng = np.random.default_rng(0)
bn = BatchNorm1d(4)
y = bn(rng.standard_normal((32, 4)) * 3 + 2)
print(np.allclose(y.mean(0), 0), y.std(0).round(2))   # True [1. 1. 1. 1.]
bn.training = False
print(bn(np.ones((1, 4))).shape)                      # (1, 4)
```

### Interview questions this prepares you for

- What loss should a freshly initialised classifier show, and what does a much larger value tell you?
- Why do saturated tanh or sigmoid units stop learning, and what is a dead ReLU?
- How should you scale the initial weights of a layer, and where does the square root of the fan in come from?
- Explain batch normalisation, including what happens at inference time.
- Why can you drop the bias before a batch norm layer?
- How would you tell from training plots that your learning rate is too small or too large?

### Related study modules

- [Deep learning foundations](#/module/deep-learning?d=deep)
- [ML coding](#/module/ml-coding?d=deep)
- [Classical machine learning](#/module/classical-ml?d=deep)

## Video 5: Makemore Part 4, backprop ninja

[Watch](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=0s) (1:55:24). Original title: Building makemore Part 4, Becoming a Backprop Ninja.

Karpathy takes the two layer MLP with batch normalisation and computes every gradient by hand, with no loss.backward. He argues that backpropagation is a leaky abstraction: it fails in quiet ways unless you understand what it does inside. First he backpropagates through the whole graph in tiny steps, then he shortcuts cross entropy and batch normalisation with compact formulas worked out on paper. The lecture is meant to be done as an exercise, pausing to try each gradient before he reveals it. The transcript available for this lecture stops near 1:48:30, so the last exercise (putting it together) is described from the chapter list and the video description only.

### Key ideas

- [1:03](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=63s) Backprop is a leaky abstraction. Saturated tanh, dead neurons and exploding or vanishing gradients appear when you do not understand what happens inside.
- [2:06](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=126s) A real bug: clipping the loss value instead of the gradient sets the gradient to zero for outliers, so those examples are silently ignored.
- [3:40](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=220s) Ten years earlier everyone wrote the backward pass by hand and checked it numerically. Karpathy shows his own old Matlab and numpy code.
- [7:20](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=440s) The setup has biases that start as small random numbers, because exact zeros can hide wrong gradients. A cmp helper compares each hand made gradient with PyTorch's.
- [13:01](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=781s) Exercise 1 backpropagates through the loss one tiny step at a time. The mean loss gives minus 1 over n at each correct entry and zero elsewhere.
- [19:00](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=1140s) The log step divides the incoming gradient by the probability, which boosts the gradient on examples the model finds unlikely.
- [21:39](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=1299s) A broadcast in the forward pass becomes a sum in the backward pass over that same dimension, and a sum in the forward pass becomes a broadcast.
- [36:06](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=2166s) The gradient on the max logit is about 1e-9, which is zero in practice, because subtracting the max does not change the probabilities.
- [43:05](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=2585s) For a linear layer, the gradient is the output gradient times the transposed weights, the transposed input times the output gradient, and a column sum for the bias. The shapes force the formula.
- [53:22](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=3202s) Tanh uses one minus the output squared. The batch norm gain and bias gradients sum over the batch dimension.
- [1:05:17](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=3917s) Bessel's correction: dividing the variance by n minus 1 is unbiased for a small batch, and the batch norm paper and PyTorch mix the two versions between training and testing.
- [1:23:14](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=4994s) The embedding table gradient is a scatter add, since the same row can be used many times and its gradients must add up.
- [1:26:31](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=5191s) Exercise 2 replaces all the cross entropy steps with one formula: the softmax probabilities minus 1 at the correct class, divided by n.
- [1:32:57](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=5577s) The cross entropy gradient is a push and pull: it pulls down the wrong classes and pulls up the right one by the same total, and it is zero when the prediction is perfect.
- [1:36:37](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=5797s) Exercise 3 derives a compact backward formula for the whole batch norm layer on paper, using three paths from each input: the mean, the variance and the normalised value.
- [1:50:02](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=6602s) Exercise 4 (from the video description only) trains the whole network with these hand written gradients and no loss.backward, with the same results.

### Code to remember

The cross entropy gradient in one shot.

```python
import numpy as np

logits = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 3.0]])
y = np.array([0, 2])
n = len(y)
p = np.exp(logits - logits.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)                 # softmax probabilities
dlogits = p.copy()
dlogits[np.arange(n), y] -= 1                # subtract 1 at the correct class
dlogits /= n                                 # the loss is a mean over n examples
print(dlogits.round(3).tolist())             # [[-0.17, 0.121, 0.049], [0.035, 0.035, -0.071]]
print(np.abs(dlogits.sum(1)).max() < 1e-12)  # True, every row sums to zero
```

Backward through a linear layer, decided by the shapes.

```python
import numpy as np

rng = np.random.default_rng(0)
h, W2 = rng.standard_normal((32, 64)), rng.standard_normal((64, 27))
dlogits = rng.standard_normal((32, 27))
dh = dlogits @ W2.T          # (32, 27) @ (27, 64) -> (32, 64)
dW2 = h.T @ dlogits          # (64, 32) @ (32, 27) -> (64, 27)
db2 = dlogits.sum(0)         # (27,)
print(dh.shape, dW2.shape, db2.shape)
```

### Interview questions this prepares you for

- Why is backpropagation called a leaky abstraction? Give two failure examples.
- What is the gradient of cross entropy with respect to the logits?
- Derive the backward pass of a linear layer, and say how you remember it.
- A value is broadcast in the forward pass. What happens in the backward pass?
- Why does the gradient with respect to the max logit come out as zero?
- Why does batch norm use n minus 1 for the variance, and what mismatch exists in practice?

### Related study modules

- [Deep learning foundations](#/module/deep-learning?d=deep)
- [ML coding](#/module/ml-coding?d=deep)

## Video 6: Makemore Part 5, building a WaveNet

[Watch](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=0s) (56:22). Original title: Building makemore Part 5, Building a WaveNet.

Karpathy makes the character model deeper and gives it more context, 8 characters instead of 3. Instead of squashing all the characters into one hidden layer, the network fuses them in pairs, level by level, which is the tree like idea from DeepMind's 2016 WaveNet. To build it he adds Embedding, Flatten and Sequential modules, a flatten consecutive layer, and fixes a subtle batch norm bug for three dimensional input. He also shows that convolutions are only an efficient way to slide the same model over a sequence. The validation loss goes from 2.10 to about 1.99, with notes on why the experiments are not yet trustworthy.

### Key ideas

- [0:33](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=33s) More context is the goal. A single hidden layer squashes too much information at once, so the network should fuse characters gradually, as in WaveNet.
- [3:18](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=198s) Batch norm is a strange layer: running statistics learned outside backpropagation, a training flag that causes bugs, and coupling between examples.
- [7:12](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=432s) A noisy loss curve is smoothed by viewing the list as rows of 1,000 values and averaging each row.
- [9:16](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=556s) Embedding, Flatten and Sequential modules turn the model into a container of layers, rebuilding torch.nn. Calling the model on the integer inputs gives the logits.
- [15:30](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=930s) A bug: forgetting evaluation mode. With one example, batch norm in training mode computes a variance of a single number, which is not a number.
- [19:22](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=1162s) Raising the block size from 3 to 8 lowers the validation loss from 2.10 to 2.02, even with the old flat architecture.
- [24:18](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=1458s) Matrix multiply works on the last dimension only. Extra leading dimensions are treated as batch dimensions.
- [30:03](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=1803s) Flatten consecutive groups every n neighbours with a view to shape batch, positions over n, channels times n. This equals explicit concatenation of even and odd positions.
- [35:06](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=2106s) The hierarchical network fuses 2 characters at a time over 3 levels. With about 22,000 parameters it scores 2.029, the same as the flat one at 2.027.
- [38:20](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=2300s) Batch norm for three dimensional input must average over the batch and position dimensions together. The first version kept separate statistics for each position.
- [45:01](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=2701s) After the fix the validation loss improves slightly to 2.022. A bigger network with 76,000 parameters reaches 1.993.
- [47:44](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=2864s) Convolution slides the same small model over every position of the sequence. This is faster and reuses shared intermediate values, but does not change the model.
- [52:28](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=3148s) Building deep networks means lots of documentation, shape juggling, prototyping in a notebook, and then moving code to a repository.
- [54:32](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=3272s) Missing: an experiment harness, validation tracking alongside training loss, and hyperparameter searches. Also not built: residual and skip connections.

### Code to remember

Flatten consecutive characters with a view, and check it equals explicit concatenation.

```python
import numpy as np

def flatten_consecutive(x, n):
    B, T, C = x.shape
    x = x.reshape(B, T // n, C * n)          # a view, no copy
    return x.squeeze(1) if x.shape[1] == 1 else x

e = np.random.default_rng(0).standard_normal((4, 8, 10))
explicit = np.concatenate([e[:, ::2, :], e[:, 1::2, :]], axis=2)
print(flatten_consecutive(e, 2).shape, np.array_equal(explicit, flatten_consecutive(e, 2)))
# (4, 4, 20) True
```

Batch norm statistics for three dimensional input.

```python
import numpy as np

h = np.random.default_rng(0).standard_normal((32, 4, 68))   # batch, position, channels
print(h.mean(0, keepdims=True).shape)          # (1, 4, 68), wrong: one mean per position
print(h.mean((0, 1), keepdims=True).shape)     # (1, 1, 68), right: one mean per channel
```

### Interview questions this prepares you for

- Why fuse characters gradually instead of using one big hidden layer?
- What happens to batch norm if you forget evaluation mode and pass one example?
- How does a matrix multiply treat extra leading dimensions?
- Over which dimensions should batch norm compute statistics for sequence data?
- What does a convolution add, and what does it not change, compared with the hierarchical model?
- Why are the small differences in the lecture's validation losses not trustworthy yet?

### Related study modules

- [Deep learning foundations](#/module/deep-learning?d=deep)
- [ML coding](#/module/ml-coding?d=deep)
- [Transformers and LLMs](#/module/transformers-llms?d=deep)

## Video 7: Let's build GPT from scratch

[Watch](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=0s) (1:56:20). Original title: Let's build GPT, from scratch, in code, spelled out.

Karpathy builds a small GPT, a decoder only transformer, from an empty file and trains it on the Tiny Shakespeare text. He starts from the bigram model of the earlier videos, then adds self attention using a matrix multiply trick, and then multi head attention, a feed forward layer, residual connections, layer norm and dropout. The validation loss falls from 2.5 for the bigram model to 1.48 for a model of about 10 million parameters. He finishes by comparing the code with nanoGPT and with how ChatGPT is trained, including the fine tuning stages the lecture does not build.

### Key ideas

- [1:42](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=102s) A language model completes sequences. GPT means generative pretrained transformer, and the transformer comes from the 2017 paper Attention Is All You Need.
- [9:26](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=566s) A character tokenizer gives 65 symbols for Tiny Shakespeare. Subword tokenizers such as tiktoken give about 50,000 tokens and shorter sequences, so vocabulary size trades off against length.
- [15:32](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=932s) A chunk of 9 characters holds 8 training examples, with contexts from 1 character up to the block size of 8.
- [18:22](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1102s) A batch stacks random chunks, with targets shifted by one. A batch of 4 by 8 holds 32 independent examples.
- [22:17](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1337s) The bigram model is an embedding table whose rows are logits. Cross entropy needs the logits reshaped to two dimensions, and the start loss should be near the log of 65, about 4.17.
- [29:06](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1746s) Generation takes the last position's logits, applies softmax, samples one character and appends it.
- [34:42](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2082s) Training uses AdamW. The bigram model reaches a loss of about 2.5.
- [39:43](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2383s) Loss is estimated by averaging many batches in evaluation mode under no_grad.
- [42:13](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2533s) The trick: averaging the past is a multiplication by a lower triangular matrix of weights, first with loops, then matmul, then masked softmax.
- [1:00:18](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3618s) Token embeddings and position embeddings are added together.
- [1:02:00](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s) Self attention: each token makes a query, a key and a value. Query times key gives affinities, and the output is the weighted sum of values.
- [1:11:38](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4298s) Attention is communication on a directed graph. It has no notion of position and does not mix batch examples.
- [1:14:14](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4454s) Encoder blocks have no mask and decoder blocks have a triangular mask. Cross attention takes keys and values from another source.
- [1:16:56](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4616s) Scaling by the square root of the head size keeps softmax from becoming one hot at the start.
- [1:21:59](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4919s) Multi head attention runs several smaller heads in parallel and concatenates them. Loss falls to about 2.28.
- [1:24:25](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5065s) A feed forward layer lets each token compute on what it gathered. Loss falls to about 2.24.
- [1:26:48](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5208s) Residual connections give gradients a direct path, and the loss falls to about 2.08.
- [1:32:51](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5571s) Layer norm normalises each token's features. It is used before each sublayer (pre norm), and the loss falls to about 2.06.
- [1:37:49](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5869s) Dropout switches off random values during training to fight overfitting.
- [1:39:44](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5984s) Scaling to 6 layers, 6 heads, embedding 384, block 256 gives a loss of 1.48 in about 15 minutes on an A100.
- [1:42:39](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6159s) GPT is decoder only because it has nothing to condition on. The original paper used an encoder and a decoder for translation.
- [1:46:22](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6382s) nanoGPT has the same model with batched heads and GELU, so that OpenAI's weights can be loaded.
- [1:48:53](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6533s) ChatGPT adds fine tuning on top of pretraining: question and answer data, a reward model from rankings, and PPO.

### Code to remember

Average the past with one matrix multiply, then the same with a masked softmax.

```python
import numpy as np

rng = np.random.default_rng(0)
B, T, C = 4, 8, 2
x = rng.standard_normal((B, T, C))

wei = np.tril(np.ones((T, T)))
wei /= wei.sum(1, keepdims=True)          # row t averages tokens 0..t
xbow = wei @ x                            # (T, T) @ (B, T, C) -> (B, T, C)

mask = np.where(np.tril(np.ones((T, T))) == 0, -np.inf, 0.0)
w2 = np.exp(mask)
w2 /= w2.sum(1, keepdims=True)            # softmax of masked zeros
print(np.allclose(wei, w2), np.allclose(xbow, w2 @ x))   # True True
```

One causal self attention head.

```python
import numpy as np

def head(x, Wk, Wq, Wv):
    k, q, v = x @ Wk, x @ Wq, x @ Wv                         # (B, T, hs)
    wei = q @ k.transpose(0, 2, 1) * k.shape[-1] ** -0.5     # (B, T, T), scaled
    wei = np.where(np.tril(np.ones(wei.shape[1:])) == 0, -np.inf, wei)   # no peeking ahead
    wei = np.exp(wei - wei.max(-1, keepdims=True))
    wei /= wei.sum(-1, keepdims=True)                        # softmax over each row
    return wei @ v, wei

rng = np.random.default_rng(1)
x = rng.standard_normal((4, 8, 32))
Wk, Wq, Wv = [rng.standard_normal((32, 16)) / 32 ** 0.5 for _ in range(3)]
out, wei = head(x, Wk, Wq, Wv)
print(out.shape, wei.shape, np.allclose(wei.sum(-1), 1))   # (4, 8, 16) (4, 8, 8) True
```

### Interview questions this prepares you for

- Explain self attention with queries, keys and values.
- Why do we divide the attention scores by the square root of the head size?
- What does the causal mask do, and how do encoder and decoder blocks differ?
- Why do transformers use residual connections and layer norm?
- Why is GPT decoder only, and when would you use cross attention?
- What are the stages of training ChatGPT, and which one does this lecture build?

### Related study modules

- [Transformers and LLMs](#/module/transformers-llms?d=deep)
- [Deep learning foundations](#/module/deep-learning?d=deep)
- [Post training and reasoning](#/module/post-training-reasoning?d=deep)
- [ML coding](#/module/ml-coding?d=deep)

## Video 8: State of GPT

[Watch](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=0s) (42:40). Original title: State of GPT, a talk at Microsoft Build 2023.

This talk has two parts. The first explains how a GPT assistant is trained in four stages: pretraining, supervised fine tuning, reward modelling and reinforcement learning from human feedback. The second explains how to use such assistants well, by thinking of them as token simulators that need room to reason, tools, retrieved context and clear requests. It ends with advice on fine tuning, a workflow for getting the best results, and the limits of these models, such as hallucination and prompt injection. The details of models and releases are as of mid 2023.

### Key ideas

- [0:35](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=35s) Four stages: pretraining, supervised fine tuning, reward modelling and reinforcement learning. Pretraining takes about 99 percent of the compute.
- [2:16](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=136s) The pretraining data is a mixture of sources such as web crawls, GitHub, Wikipedia and books, sampled by proportions and turned into tokens.
- [3:21](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=201s) Parameters do not decide power. LLaMA has 65 billion parameters and 1.4 trillion training tokens, against GPT-3 with 175 billion and 300 billion.
- [5:06](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=306s) Documents are packed into rows with end of text tokens. Every position predicts the next token from all earlier ones.
- [8:23](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=503s) Base models can be fine tuned for tasks, or prompted with a few examples. They complete documents and are not assistants.
- [11:09](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=669s) Supervised fine tuning continues language modelling on tens of thousands of prompt and ideal response pairs written by contractors.
- [12:48](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=768s) Reward modelling trains a model to score answers, using human rankings of several completions of the same prompt.
- [15:04](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=904s) In reinforcement learning, sampled tokens are weighted by the reward, so good completions become more likely.
- [17:16](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=1036s) RLHF models are preferred by people. One reason is that comparing answers is easier than writing them. RLHF models lose some diversity.
- [22:13](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=1333s) A GPT does the same work for every token and has no inner monologue. It needs tokens to think, so spread reasoning over many tokens.
- [25:32](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=1532s) Sampling several answers and voting, asking the model to check itself, and tree search over prompts bring back slower deliberate thinking.
- [30:35](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=1835s) Models imitate, they do not try to succeed, so ask for a strong answer explicitly.
- [31:41](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=1901s) Give the model tools and retrieved context, because the context window is its working memory.
- [34:27](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=2067s) Constraint prompting forces a format such as JSON, and LoRA makes fine tuning cheaper.
- [37:12](https://www.youtube.com/watch?v=bZQun8Y4L2A&t=2232s) First reach top performance with the strongest model and detailed prompts, then optimise cost. Use LLMs in low stakes settings with human oversight.

### Code to remember

The pairwise loss used to train a reward model from rankings.

```python
import math

def pair_loss(r_better, r_worse):
    return math.log1p(math.exp(-(r_better - r_worse)))   # -log sigmoid(difference)

print(round(pair_loss(1.0, -1.2), 3), round(pair_loss(-1.2, 1.0), 3))   # 0.105 2.305
```

A tiny retrieval step: embed chunks, then fetch the best match for a query.

```python
import numpy as np

docs = ["California has about 39.2 million people",
        "Alaska has about 0.74 million people",
        "Python lists are mutable"]
vocab = sorted({w for d in docs for w in d.lower().split()})

def embed(text):
    v = np.array([text.lower().split().count(w) for w in vocab], float)
    return v / (np.linalg.norm(v) + 1e-9)

M = np.stack([embed(d) for d in docs])
sims = M @ embed("how many people live in alaska")
print(sims.round(2).tolist(), docs[int(sims.argmax())])
# [0.29, 0.58, 0.0] Alaska has about 0.74 million people
```

### Interview questions this prepares you for

- Describe the stages of training a GPT assistant and what data each stage uses.
- How is a reward model trained, and how is it used in reinforcement learning from human feedback?
- Why might RLHF work better than supervised fine tuning, and what does it cost?
- Why do language models need tokens to think? Name techniques that help.
- What is retrieval augmented generation and why does it work?
- What are the main limitations of language models in applications?

### Related study modules

- [Post training and reasoning](#/module/post-training-reasoning?d=deep)
- [Transformers and LLMs](#/module/transformers-llms?d=deep)
- [RAG, agents and evals](#/module/rag-agents-evals?d=deep)

## Video 9: Let's build the GPT tokenizer

[Watch](https://www.youtube.com/watch?v=zduSFxRajkE&t=0s) (2:13:35). Original title: Let's build the GPT Tokenizer.

A tokenizer turns text into integers and back, and it is a separate stage of the pipeline with its own training data. Karpathy explains Unicode and UTF-8, then builds byte pair encoding from scratch: count the most common neighbouring pair, merge it into a new token, and repeat. He then covers how GPT-2 and GPT-4 use a regular expression to forbid some merges, special tokens, SentencePiece as used by Llama, and how to pick the vocabulary size. The last hour connects odd model behaviour, such as poor spelling, arithmetic mistakes, trailing space warnings and glitch tokens, back to tokenization.

### Key ideas

- [0:00](https://www.youtube.com/watch?v=zduSFxRajkE&t=0s) Tokens are the atoms of language models. Many strange behaviours, such as bad spelling and weaker non English, trace back to tokenization.
- [5:50](https://www.youtube.com/watch?v=zduSFxRajkE&t=350s) The Tiktokenizer web app shows that spaces belong to tokens, that case matters, and that number splits are arbitrary. A Korean sentence needs many more tokens than the English one.
- [12:17](https://www.youtube.com/watch?v=zduSFxRajkE&t=737s) The GPT-4 tokenizer has about 100,000 tokens against about 50,000 for GPT-2, and it groups Python indentation. The same text takes roughly half the tokens.
- [14:56](https://www.youtube.com/watch?v=zduSFxRajkE&t=896s) Strings are sequences of Unicode code points, about 150,000 of them. UTF-8 turns each into 1 to 4 bytes, and it is the preferred encoding.
- [21:20](https://www.youtube.com/watch?v=zduSFxRajkE&t=1280s) Raw bytes give only 256 tokens, which makes sequences far too long. Byte pair encoding compresses them with a larger vocabulary.
- [23:50](https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s) Byte pair encoding repeatedly merges the most frequent pair of tokens into a new token. A sequence of 11 tokens becomes 9, then 7, then 5.
- [28:35](https://www.youtube.com/watch?v=zduSFxRajkE&t=1715s) get_stats counts adjacent pairs and merge replaces a pair by a new id. Pair (101, 32), the letter e followed by a space, occurs 20 times in the example.
- [34:58](https://www.youtube.com/watch?v=zduSFxRajkE&t=2098s) Training loops over merges until the vocabulary reaches the chosen size. With 20 merges the vocabulary is 276 and the compression ratio is about 1.27.
- [39:20](https://www.youtube.com/watch?v=zduSFxRajkE&t=2360s) The tokenizer is a separate stage with its own training set. Its language mix decides how many tokens each language gets.
- [42:47](https://www.youtube.com/watch?v=zduSFxRajkE&t=2567s) Decoding maps ids to bytes, joins them, and decodes UTF-8 with errors replaced, because not every byte sequence is valid.
- [48:21](https://www.youtube.com/watch?v=zduSFxRajkE&t=2901s) Encoding applies merges in the order they were learned, always taking the pair with the lowest merge index.
- [57:36](https://www.youtube.com/watch?v=zduSFxRajkE&t=3456s) GPT-2 first splits text with a regular expression so that letters, digits and punctuation are never merged together.
- [1:11:38](https://www.youtube.com/watch?v=zduSFxRajkE&t=4298s) The tiktoken library is inference only. GPT-4's pattern is case insensitive, merges at most 3 digits, and handles spaces differently.
- [1:18:26](https://www.youtube.com/watch?v=zduSFxRajkE&t=4706s) Special tokens such as end of text sit outside byte pair encoding. GPT-2 has 256 bytes, 50,000 merges and 1 special token, 50,257 in all.
- [1:28:42](https://www.youtube.com/watch?v=zduSFxRajkE&t=5322s) SentencePiece, used by Llama, runs byte pair encoding on code points with byte fallback and many options.
- [1:43:27](https://www.youtube.com/watch?v=zduSFxRajkE&t=6207s) Larger vocabularies shorten sequences but grow the embedding table and output layer, and make rare tokens undertrained. About 100,000 is typical.
- [1:51:41](https://www.youtube.com/watch?v=zduSFxRajkE&t=6701s) Model quirks come from tokens: long tokens hurt spelling, odd number splits hurt arithmetic, and untrained tokens like SolidGoldMagikarp cause erratic behaviour.
- [2:10:20](https://www.youtube.com/watch?v=zduSFxRajkE&t=7820s) Use tiktoken with the GPT-4 vocabulary if you can. For your own vocabulary, a careful SentencePiece or a future trainable tiktoken style tool.

### Code to remember

Byte pair encoding: count pairs, merge a pair, and the lecture's small example.

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

seq = list("aaabdaaabac")
for pair, new in [(("a", "a"), "Z"), (("a", "b"), "Y"), (("Z", "Y"), "X")]:
    seq = merge(seq, pair, new)
    print(''.join(seq), len(seq))     # ZabdZabac 9, then ZYdZYac 7, then XdXac 5
```

Train, encode and decode.

```python
def train(text, vocab_size):
    ids, merges = list(text.encode('utf-8')), {}
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, 256 + i)
        merges[pair] = 256 + i
    return merges

def encode(text, merges):
    ids = list(text.encode('utf-8'))
    while len(ids) >= 2:
        pair = min(get_stats(ids), key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in merges.items():
        vocab[idx] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode('utf-8', errors='replace')

text = "low lower lowest newer newest wider " * 20
merges = train(text, 276)
print(decode(encode("lower newest", merges), merges))   # lower newest
```

### Interview questions this prepares you for

- Explain the byte pair encoding algorithm and how encode and decode use the merges.
- Why does the GPT-2 tokenizer split text with a regular expression first?
- Why use bytes from UTF-8 rather than Unicode code points?
- How do you choose a vocabulary size, and what are the costs of a very large one?
- Why are language models bad at spelling, arithmetic and some languages? Link each to tokenization.
- What is a special token and how do you add one to a pretrained model?

### Related study modules

- [Transformers and LLMs](#/module/transformers-llms?d=deep)
- [ML coding](#/module/ml-coding?d=deep)
- [RAG, agents and evals](#/module/rag-agents-evals?d=deep)

## Video 10: Let's reproduce GPT-2 (124M)

[Watch](https://www.youtube.com/watch?v=l8pRSuU81PU&t=0s) (4:01:26). Original title: Let's reproduce GPT-2 (124M).

Karpathy rebuilds the 124 million parameter GPT-2 in four stages. First he writes the model and checks it by loading OpenAI's released weights. Then he makes training fast with lower precision, torch.compile and FlashAttention. Next he copies the GPT-2 and GPT-3 training recipe: AdamW settings, gradient clipping, a warmup and cosine learning rate schedule, weight decay, gradient accumulation to a half million token batch, and training on 8 GPUs. Finally he trains on 10 billion tokens of FineWeb-Edu and matches GPT-2 124M on the HellaSwag test. Several known issues, such as torch.compile breaking sampling and unshuffled data, are listed at the end.

### Key ideas

- [0:33](https://www.youtube.com/watch?v=l8pRSuU81PU&t=33s) GPT-2 is a family of four sizes, and the smallest, 124 million parameters, has 12 layers and 768 channels. Larger models give better downstream scores, which is the scaling law idea.
- [3:39](https://www.youtube.com/watch?v=l8pRSuU81PU&t=219s) The target is OpenAI's released checkpoint, loaded through Hugging Face because the original code is TensorFlow. It has a 50,257 by 768 token table and a 1,024 by 768 position table.
- [13:47](https://www.youtube.com/watch?v=l8pRSuU81PU&t=827s) GPT-2 is the original transformer without the encoder and cross attention, with layer norm moved before each sublayer and one more layer norm at the end.
- [18:28](https://www.youtube.com/watch?v=l8pRSuU81PU&t=1108s) A clean residual stream lets gradients flow straight to the input. Attention is a reduce step across tokens and the MLP is a map applied to each token.
- [21:18](https://www.youtube.com/watch?v=l8pRSuU81PU&t=1278s) The MLP uses GELU in its tanh approximation, a smooth ReLU that never has a flat zero region.
- [28:08](https://www.youtube.com/watch?v=l8pRSuU81PU&t=1688s) The own model loads the released weights, copies the tensors, transposes some of them, and generates coherent text.
- [37:02](https://www.youtube.com/watch?v=l8pRSuU81PU&t=2222s) Sampling takes the last position's logits, applies softmax, keeps the top 50 probabilities and draws one token.
- [48:18](https://www.youtube.com/watch?v=l8pRSuU81PU&t=2898s) A batch comes from a buffer of B times T plus 1 tokens, so inputs are the buffer without its last token and targets are the buffer shifted by one.
- [55:03](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3303s) A random model should start with loss near the log of 50,257, about 10.82.
- [1:06:14](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3974s) The token embedding and the output layer share one matrix, which saves about 38.6 million parameters.
- [1:13:47](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4427s) Weights start at standard deviation 0.02, biases at 0, and residual output layers are scaled by 1 over the square root of 2 times the layer count.
- [1:22:18](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4938s) An A100 gives 19.5 TFLOPS in float32, 156 with TF32 and 312 with bfloat16. Training is usually limited by memory movement.
- [1:28:14](https://www.youtube.com/watch?v=l8pRSuU81PU&t=5294s) TF32 is one line and cuts a step from about 1,000 ms to about 330 ms, roughly 50,000 tokens per second. bfloat16 autocast takes it to about 300 ms, roughly 55,000 tokens per second.
- [1:48:15](https://www.youtube.com/watch?v=l8pRSuU81PU&t=6495s) torch.compile fuses operations and removes Python overhead, taking a step from 300 ms to about 130 ms.
- [2:00:18](https://www.youtube.com/watch?v=l8pRSuU81PU&t=7218s) FlashAttention never builds the full attention matrix, using an online softmax. It does more arithmetic yet runs faster, about 96 ms per step.
- [2:06:54](https://www.youtube.com/watch?v=l8pRSuU81PU&t=7614s) Padding the vocabulary from 50,257 to 50,304 gives a nicer number and a small speed up.
- [2:14:55](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8095s) Follow GPT-3: AdamW betas 0.9 and 0.95, gradient norm clipping at 1.0, weight decay 0.1 on matrices only.
- [2:21:06](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8466s) The learning rate warms up over 715 steps to 6e-4, then follows a cosine down to 10 percent.
- [2:34:09](https://www.youtube.com/watch?v=l8pRSuU81PU&t=9249s) Gradient accumulation reaches a batch of 524,288 tokens on small hardware. The loss must be divided by the number of accumulation steps.
- [2:46:52](https://www.youtube.com/watch?v=l8pRSuU81PU&t=10012s) Distributed data parallel runs 8 processes, gives each a different slice of the data, and averages gradients with all reduce.
- [3:10:21](https://www.youtube.com/watch?v=l8pRSuU81PU&t=11421s) Data: 10 billion tokens of FineWeb-Edu in 100 shards of 100 million tokens. One epoch is 19,073 steps and takes about 1.75 hours.
- [3:28:23](https://www.youtube.com/watch?v=l8pRSuU81PU&t=12503s) HellaSwag asks a model to pick the right ending of a sentence. GPT-2 124M scores 29.5 percent, and random is 25 percent.
- [3:43:05](https://www.youtube.com/watch?v=l8pRSuU81PU&t=13385s) The run matches GPT-2 124M on HellaSwag with 10 billion tokens against GPT-2's 100 billion, with caveats about data and contamination.
- [3:56:21](https://www.youtube.com/watch?v=l8pRSuU81PU&t=14181s) llm.c reproduces the same training in pure C and CUDA with identical losses, and faster.

### Code to remember

The learning rate schedule with warmup and cosine decay, and the token budget that goes with it.

```python
import math

max_lr, min_lr, warmup, max_steps = 6e-4, 6e-5, 715, 19073

def get_lr(it):
    if it < warmup:
        return max_lr * (it + 1) / warmup            # linear warmup, never exactly 0
    if it > max_steps:
        return min_lr
    ratio = (it - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)

print([f"{get_lr(i):.2e}" for i in (0, 714, 715, 10000, 19073)])
# ['8.39e-07', '6.00e-04', '6.00e-04', '3.25e-04', '6.00e-05']
print(int(10e9 / 2 ** 19), int(375e6 / 2 ** 19))     # 19073 715
```

Gradient accumulation needs the loss divided by the number of steps.

```python
import numpy as np

rng = np.random.default_rng(0)
X, Y, w = rng.standard_normal((4, 16)), rng.standard_normal(4), rng.standard_normal(16)
grad = lambda X, Y: 2 * X.T @ (X @ w - Y) / len(Y)        # gradient of the mean squared error

full = grad(X, Y)
unscaled = sum(grad(X[i:i + 1], Y[i:i + 1]) for i in range(4))
scaled = sum(grad(X[i:i + 1], Y[i:i + 1]) / 4 for i in range(4))   # loss divided by 4
print(np.allclose(full, unscaled), np.allclose(full, scaled))        # False True
```

### Interview questions this prepares you for

- How does GPT-2 differ from the original transformer?
- Why share the token embedding and the output layer, and what does it save?
- Why scale residual layer initialisation by 1 over the square root of the layer count?
- Why is training memory bound, and how do TF32, bfloat16, torch.compile and FlashAttention each help?
- How do you reach a very large batch size on a small GPU, and what must you change in the loss?
- How does distributed data parallel work, and when do gradients synchronise?

### Related study modules

- [Transformers and LLMs](#/module/transformers-llms?d=deep)
- [Deep learning foundations](#/module/deep-learning?d=deep)
- [Post training and reasoning](#/module/post-training-reasoning?d=deep)
- [ML coding](#/module/ml-coding?d=deep)
