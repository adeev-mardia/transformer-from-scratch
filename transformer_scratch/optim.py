"""Adam optimizer, implemented from scratch (no framework optimizer used)."""
from __future__ import annotations

from typing import List

import numpy as np

from .autograd import Tensor


class AdamW:
    """AdamW: Adam with decoupled weight decay (Loshchilov & Hutter)."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 3e-4,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, lr: float = None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            # decoupled weight decay
            p.data -= lr * (update + self.weight_decay * p.data)


def cosine_lr_schedule(step: int, max_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float = 0.1):
    """Linear warmup then cosine decay, a standard GPT-style LR schedule."""
    import math

    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * min_lr_ratio + coeff * (base_lr - base_lr * min_lr_ratio)
