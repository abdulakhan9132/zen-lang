# ZenNum Reference (`import zennum as zn`)

ZenNum is ZEN's N-dimensional array and math library (inspired by NumPy), backed by flat C-order buffers with optional C native kernels and a pure-Python fallback.

```zen
import zennum as zn
```
You can also import without an alias (`import zennum`).

## Creation

- `zn.array(data, dtype = none)` — Create an array from a number, bool, or nested list (`"float64"`, `"float32"`, `"int64"`, `"bool"`).
  ```zen
  let a = zn.array([[1, 2], [3, 4]])
  ```
- `zn.zeros(shape, dtype = "float64")` — Create an array filled with zeros.
  ```zen
  let z = zn.zeros([2, 3])
  ```
- `zn.ones(shape, dtype = "float64")` — Create an array filled with ones.
  ```zen
  let o = zn.ones([2, 2])
  ```
- `zn.full(shape, value, dtype = none)` — Create an array filled with a constant value.
  ```zen
  let f = zn.full([2, 3], 7)
  ```
- `zn.arange(start, stop = none, step = 1, dtype = none)` — Create evenly spaced numbers over an interval.
  ```zen
  let r = zn.arange(0, 10, 2)
  ```
- `zn.linspace(start, stop, num = 50, dtype = "float64")` — Create `num` evenly spaced values from `start` to `stop` inclusive.
  ```zen
  let l = zn.linspace(0.0, 1.0, 5)
  ```
- `zn.eye(n, m = none, dtype = "float64")` — Create an `n x m` identity matrix (square `n x n` if `m` omitted).
  ```zen
  let id = zn.eye(3)
  ```
- `zn.seed(s)` — Set the process-wide shared random seed (`_GLOBAL_RNG`) so `zennum`, `zennet`, and built-ins stay reproducible together.
  ```zen
  zn.seed(42)
  ```
- `zn.rand(shape, dtype = "float64")` — Uniform random numbers in `[0, 1)`.
  ```zen
  let u = zn.rand([2, 3])
  ```
- `zn.randn(shape, dtype = "float64")` — Standard normal random numbers (mean 0, std 1).
  ```zen
  let n = zn.randn([2, 3])
  ```

## Indexing, Slicing & Properties

- Properties: `a.shape`, `a.ndim`, `a.size`, `a.dtype`, `a.T` (transpose).
- Multi-dimensional slicing & boolean masks:
  ```zen
  let a = zn.arange(1, 13).reshape([3, 4])
  say a[1:3, 1:3]
  say a[a > 8]
  ```
- Array methods: `a.tolist()`, `a.copy()`, `a.astype(dtype)`, `a.flatten()`, `a.squeeze(axis = none)`, `a.unsqueeze(axis)`.

## Shape Manipulation & Combining

- `zn.reshape(a, shape)` — Return a view/array with a new shape (`-1` infers one axis).
  ```zen
  let b = zn.reshape(zn.arange(6), [2, 3])
  ```
- `zn.transpose(a, axes = none)` — Reverse dimensions or permute them in order `axes`.
  ```zen
  let t = zn.transpose(zn.ones([2, 3]))
  ```
- `zn.flatten(a)` — Collapse an array into 1D.
  ```zen
  say zn.flatten([[1, 2], [3, 4]])
  ```
- `zn.squeeze(a, axis = none)` — Remove size-1 dimensions.
  ```zen
  say zn.squeeze(zn.ones([1, 3, 1])).shape
  ```
- `zn.unsqueeze(a, axis)` — Insert a size-1 dimension at `axis`.
  ```zen
  say zn.unsqueeze(zn.ones([3]), 0).shape
  ```
- `zn.concat(arrs, axis = 0)` — Join arrays along an existing axis.
  ```zen
  let c = zn.concat([zn.ones([2, 2]), zn.zeros([2, 1])], 1)
  ```
- `zn.stack(arrs, axis = 0)` — Join arrays along a new axis.
  ```zen
  let s = zn.stack([zn.array([1, 2]), zn.array([3, 4])], 0)
  ```
- `zn.split(a, sections, axis = 0)` — Split an array into sub-arrays.
  ```zen
  let parts = zn.split(zn.arange(6), 3)
  ```
- `zn.tril(a, k = 0)` / `zn.triu(a, k = 0)` — Lower / upper triangular part of a 2D+ array (`k` shifts the diagonal; `k = 1` on `zn.triu` gives a causal mask).
  ```zen
  let mask = zn.triu(zn.ones([4, 4], "bool"), 1)
  ```
- `zn.where(cond, x, y)` — Pick elements from `x` where `cond` is true, else `y`.
  ```zen
  say zn.where(zn.array([true, false]), [1, 2], [10, 20])
  ```

## Elementwise Math

- `zn.clip(a, lo, hi)` — Clamp values between `lo` and `hi`.
- `zn.abs(a)` — Elementwise absolute value.
- `zn.sqrt(a)` — Elementwise square root.
- `zn.exp(a)` — Elementwise exponential $e^x$.
- `zn.log(a)` — Elementwise natural logarithm $\ln(x)$.
- `zn.sin(a)` / `zn.cos(a)` / `zn.tanh(a)` — Elementwise trigonometric and hyperbolic functions.
- `zn.pow(a, p)` — Elementwise power (`a ** p` also supported).
- Arithmetic operators `+`, `-`, `*`, `/`, `**`, unary `-`, and comparisons `<`, `<=`, `>`, `>=` broadcast shapes automatically.

## Reductions

- `zn.sum(a, axis = none, keepdims = false)` — Sum elements overall or along an axis.
- `zn.mean(a, axis = none, keepdims = false)` — Arithmetic mean.
- `zn.std(a, axis = none, keepdims = false)` — Standard deviation.
- `zn.var(a, axis = none, keepdims = false)` — Variance.
- `zn.min(a, axis = none, keepdims = false)` / `zn.max(a, axis = none, keepdims = false)` — Minimum / maximum values.
- `zn.argmin(a, axis = none, keepdims = false)` / `zn.argmax(a, axis = none, keepdims = false)` — Index of min / max.
- `zn.sort(a, axis = -1)` — Return sorted copy along `axis`.
- `zn.cumsum(a, axis = none)` — Cumulative sum.

## Linear Algebra & Helpers

- `zn.dot(a, b)` / `zn.matmul(a, b)` / `a @ b` — Vector dot product, 2D matrix multiply, and batched 3D/4D matrix multiply.
- `zn.outer(a, b)` — Outer product of two vectors.
- `zn.norm(a, axis = none, keepdims = false)` — Euclidean L2 norm.
- `zn.gather(table, indices)` / `zn.scatter_add(shape, indices, values)` — Row gather and gradient scatter-add helpers.
- `zn.backend` — Active kernel backend string (`"native-c"` or `"pure-python ..."`).
- **Scientific debugging (`--nan-trace`):** Run `zen run --nan-trace script.zen` to catch any `NaN` or `Inf` inside ZenNum arithmetic, reductions, or `matmul` immediately with error `E014`, op name, shapes, and exact ZEN line.
