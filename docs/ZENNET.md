# ZenNet Reference (`import zennet as nn`)

ZenNet is ZEN's PyTorch-like deep learning framework written 100% in `.zen` on top of ZenNum arrays.

```zen
import zennet as nn
```
Submodules can also be imported directly (`import zennet.autograd`, `import zennet.nn`, `import zennet.optim`, `import zennet.data`, `import zennet.train`) or accessed without alias (`import zennet`).

## Autograd & `Tensor`

- `nn.tensor(data, requires_grad = false)` / `nn.Tensor(data, requires_grad = false)` — Wrap an array or list as an autograd tensor.
- `nn.zeros(shape, requires_grad = false)`, `nn.ones(...)`, `nn.rand(...)`, `nn.randn(...)` — Create autograd tensors directly.
- `t.backward(grad = none)` — Run reverse-mode automatic differentiation and accumulate `.grad`.
- `t.zero_grad()` — Reset `.grad` to `none`.
- `t.detach()` — Return a copy detached from the computation graph.
- `t.tolist()` / `t.item()` — Extract plain ZEN list or scalar float.
- `nn.no_grad(fn() { ... })` — Run a function without tracking gradients.
- `nn.is_grad_enabled()` / `nn.set_grad_enabled(bool)` — Inspect or toggle gradient tracking.
- Supported differentiable operations (methods & `nn.*` functions + operators `+`, `-`, `*`, `/`, `**`, `@`, unary `-`, indexing/slicing `t[...]`):
  `reshape`, `transpose`, `sum`, `mean`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `clip`, `relu`, `gelu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`, `dropout`, `masked_fill`, `gather`, `embedding`, `layernorm`.

```zen
import zennet as nn
let x = nn.tensor([2.0, 3.0], true)
let y = (x * x + 1.0).sum()
y.backward()
say x.grad   // [4.0, 6.0]
```

## Layers & Models

- `nn.xavier(shape)` / `nn.normal(shape, mean = 0.0, std = 0.02)` — Parameter tensor initializers (`requires_grad = true`).
- `class MyNet(nn.Module)` — Base class providing `parameters()`, `zero_grad()`, `train()`, `eval()`, `save(path)`, `load(path)`, and callable `model(x)`.
- `nn.Linear(in_features, out_features, use_bias = true)` — Dense affine layer ($xW + b$).
- `nn.Embedding(num_embeddings, embedding_dim)` — Token/index lookup embedding table.
- `nn.LayerNorm(normalized_shape, eps = 1e-5)` — Layer normalization over the last dimension.
- `nn.Dropout(p = 0.1)` — Inverted dropout active during `train()` mode.
- Activation layers for `Sequential`: `nn.ReLU()`, `nn.GELU()`, `nn.Tanh()`, `nn.Sigmoid()`.
- `nn.Sequential(layers)` — Chain modules and activations in order.
- `nn.MultiHeadAttention(d_model, n_head, block_size = 128, dropout_p = 0.0)` — Causal/non-causal multi-head self-attention.
- `nn.TransformerBlock(d_model, n_head, block_size = 128, dropout_p = 0.0, mlp_mult = 4)` — Pre-norm transformer block with GELU MLP.
- `nn.GPT(config)` — Decoder-only GPT taking `{"vocab_size", "n_layer", "n_head", "d_model", "block_size", "dropout"}`.

## Losses

- `nn.mse(pred, target)` — Mean squared error loss.
- `nn.cross_entropy(logits, targets)` — Log-softmax negative log-likelihood for 2D `[N, C]` or 3D `[B, T, C]` logits.
- `nn.binary_cross_entropy(pred, target, eps = 1e-7)` — Binary cross-entropy for probabilities in `(0, 1)`.

## Optimizers & Schedulers

- `nn.SGD(params, lr = 0.01, momentum = 0.0, weight_decay = 0.0)` — Stochastic gradient descent with optional momentum and L2 decay.
- `nn.Adam(params, lr = 0.001, betas = [0.9, 0.999], eps = 1e-8, weight_decay = 0.0)` — Adaptive moment estimation.
- `nn.AdamW(params, lr = 0.001, betas = [0.9, 0.999], eps = 1e-8, weight_decay = 0.01)` — Adam with decoupled weight decay.
- `nn.clip_grad_norm(params, max_norm)` — Rescale gradients so total L2 norm is at most `max_norm`.
- `nn.StepLR(optimizer, step_size, gamma = 0.1)` — Decay LR by `gamma` every `step_size` calls to `.step()`.
- `nn.CosineAnnealingLR(optimizer, total_steps, min_lr = 0.0)` — Smooth cosine decay from initial LR to `min_lr`.
- `nn.WarmupLR(optimizer, warmup_steps)` — Linear warmup from `0` to initial LR.

## Data, Tokenizers & Training

- `nn.Dataset(x, y)` — Holds inputs and labels (`size()`, `get(i)`).
- `nn.train_val_split(x, y, val_ratio = 0.2, shuffle = true, seed_val = none)` — Returns `[train_dataset, val_dataset]`.
- `nn.DataLoader(dataset, batch_size = 32, shuffle = true, seed_val = none)` — Yields mini-batches via `.batches()`.
- `nn.CharTokenizer(text = "")` — Character-level tokenizer (`fit`, `encode`, `decode`, `vocab_size`).
- `nn.BPETokenizer(vocab_size = 64)` — Byte-pair encoding tokenizer trained directly from text (`fit`, `encode`, `decode`).
- `nn.text_to_batches(ids, block_size, batch_size)` — Sample next-token prediction `[X, Y]` batches.
- `nn.Trainer(model, optimizer = none, loss_fn = none, lr = 0.01, clip_grad = 1.0, log_every = 10, checkpoint_path = none, save_every = 0)` — Training manager with logging, validation, checkpointing, and resume.
- `nn.fit(model, train_data, steps = 100, ...)` — One-line training helper returning `{"loss_history", "final_loss", "val_loss"}`.
- `nn.generate(model, prompt_ids, max_new_tokens, temperature = 1.0, top_k = none)` — Autoregressive token sampling under `no_grad`.
- `nn.accuracy(logits, targets)` — Classification accuracy fraction in `[0.0, 1.0]`.
- `nn.perplexity(loss_val)` — Language-model perplexity $\exp(\text{loss})$.
