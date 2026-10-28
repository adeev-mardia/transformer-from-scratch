"""A minimal reverse-mode automatic differentiation engine, built on NumPy.

This is the "from scratch" substitute for a deep learning framework:
there is no PyTorch, TensorFlow or JAX anywhere in this project. Every
operation a `Tensor` supports records how to compute its own local
gradient, and `Tensor.backward()` walks the computation graph in
reverse topological order, accumulating gradients with the chain rule
by hand, exactly the way backpropagation is defined mathematically.

Only the operations the transformer actually needs are implemented:
matmul, elementwise add/mul, broadcasting-aware sum, reshape/transpose,
embedding lookup, softmax, and a few activation functions. Each `_backward`
closure below *is* the hand-derived gradient for that operation.
"""
from __future__ import annotations

import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "_prev", "_backward", "requires_grad", "_op")

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._prev = set(_children)
        self._backward = lambda: None
        self.requires_grad = requires_grad
        self._op = _op

    # -- helpers -----------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    @staticmethod
    def _unbroadcast(grad, shape):
        """Sum-reduce `grad` down to `shape`, undoing NumPy broadcasting."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape)

    # -- arithmetic ----------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self._unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __neg__(self):
        return self * Tensor(-1.0, requires_grad=False)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** -1.0)

    def __pow__(self, p):
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (p * self.data ** (p - 1)) * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                grad = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += self._unbroadcast(grad, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                grad = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += self._unbroadcast(grad, other.data.shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # -- shape ops -------------------------------------------------------
    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        out = Tensor(np.transpose(self.data, axes), (self,), "transpose")
        inv = np.argsort(axes)

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.transpose(out.grad, inv)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")
        orig_shape = self.data.shape

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.reshape(orig_shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        in_shape = self.data.shape

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis if isinstance(axis, int) else tuple(axis))
                self.grad += np.broadcast_to(g, in_shape)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            n = 1
            for a in axes:
                n *= self.data.shape[a]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # -- activations -----------------------------------------------------
    def relu(self):
        mask = (self.data > 0).astype(np.float64)
        out = Tensor(self.data * mask, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * mask

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def gelu(self):
        # Exact (erf-based) GELU, differentiated by hand.
        x = self.data
        from math import sqrt, pi
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        cdf = 0.5 * (1.0 + _erf(x * inv_sqrt2))
        out_data = x * cdf
        out = Tensor(out_data, (self,), "gelu")
        pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                dgelu = cdf + x * pdf
                self.grad += out.grad * dgelu

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * out.data

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad / self.data

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def softmax(self, axis=-1):
        x = self.data
        x_shift = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x_shift)
        s = e / np.sum(e, axis=axis, keepdims=True)
        out = Tensor(s, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                # d/dx softmax: (grad - sum(grad * s, axis)) * s
                g = out.grad
                dot = np.sum(g * s, axis=axis, keepdims=True)
                self.grad += (g - dot) * s

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def layer_norm(self, gamma: "Tensor", beta: "Tensor", eps: float = 1e-5):
        x = self.data
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        x_hat = (x - mu) / std
        out_data = x_hat * gamma.data + beta.data
        out = Tensor(out_data, (self, gamma, beta), "layernorm")
        D = x.shape[-1]

        def _backward():
            g = out.grad
            if gamma.requires_grad:
                gamma._ensure_grad()
                gamma.grad += Tensor._unbroadcast(np.sum(g * x_hat, axis=tuple(range(g.ndim - 1))), gamma.data.shape)
            if beta.requires_grad:
                beta._ensure_grad()
                beta.grad += Tensor._unbroadcast(np.sum(g, axis=tuple(range(g.ndim - 1))), beta.data.shape)
            if self.requires_grad:
                self._ensure_grad()
                dx_hat = g * gamma.data
                dvar = np.sum(dx_hat * (x - mu) * -0.5 * std ** -3, axis=-1, keepdims=True)
                dmu = np.sum(dx_hat * -1.0 / std, axis=-1, keepdims=True) + dvar * np.mean(
                    -2.0 * (x - mu), axis=-1, keepdims=True
                )
                dx = dx_hat / std + dvar * 2.0 * (x - mu) / D + dmu / D
                self.grad += dx

        out._backward = _backward
        out.requires_grad = True
        return out

    # -- indexing / embedding --------------------------------------------
    def embedding_lookup(self, idx: np.ndarray):
        """self is the embedding table (V, D); idx is an int array of any
        shape. Returns a Tensor of shape idx.shape + (D,)."""
        out_data = self.data[idx]
        out = Tensor(out_data, (self,), "embedding")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def __getitem__(self, key):
        out = Tensor(self.data[key], (self,), "getitem")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                np.add.at(self.grad, key, out.grad) if isinstance(key, (np.ndarray, list)) else None
                if not isinstance(key, (np.ndarray, list)):
                    self.grad[key] += out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # -- backward -----------------------------------------------------
    def backward(self, grad=None):
        """Run reverse-mode autodiff starting from this tensor.

        `grad` is the upstream gradient to seed with (shape must match
        this tensor's shape). If omitted, a scalar tensor is seeded with
        1.0 (the usual `loss.backward()` case); a non-scalar tensor is
        seeded with all-ones, matching the convention that calling
        `.backward()` with no argument means "treat every element of
        this tensor as contributing with unit weight."
        """
        topo = []
        visited = set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        if grad is not None:
            self.grad = np.asarray(grad, dtype=np.float64)
        else:
            self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorized error function via the Abramowitz-Stegun approximation
    (max absolute error ~1.5e-7), used so GELU does not require SciPy."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    x_abs = np.abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs)
    return sign * y


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """logits: Tensor of shape (N, V); targets: int array of shape (N,)."""
    probs = logits.softmax(axis=-1)
    N = targets.shape[0]
    idx = (np.arange(N), targets)
    picked = probs[idx]  # Tensor, shape (N,)
    log_picked = picked.log()
    loss = log_picked.sum() * (-1.0 / N)
    return loss
