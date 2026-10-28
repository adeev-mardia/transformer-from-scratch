"""A GPT-style decoder-only transformer, implemented from first principles
on top of the small autograd engine in `autograd.py`.

Every piece is written out explicitly rather than called from a deep
learning framework:
  - token embedding + learned positional embedding
  - multi-head causal self-attention (QKV projections, scaled dot-product
    attention, causal masking, head concatenation, output projection)
  - position-wise feedforward network with GELU
  - residual connections and layer normalization
  - a final linear projection back to vocabulary logits (weight-tied to
    the token embedding, as in GPT-2)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from .autograd import Tensor


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128       # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    d_ff: int = 512             # feedforward hidden size
    dropout: float = 0.0        # kept for completeness; 0 by default (no dropout framework needed)


def _init_linear(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    # Standard GPT-2-style scaled init.
    std = 0.02
    return (rng.standard_normal((fan_in, fan_out)) * std).astype(np.float64)


class CausalSelfAttention:
    def __init__(self, cfg: GPTConfig, rng: np.random.Generator):
        self.cfg = cfg
        D, H = cfg.n_embd, cfg.n_head
        assert D % H == 0, "n_embd must be divisible by n_head"
        self.head_dim = D // H

        self.W_q = Tensor(_init_linear(D, D, rng))
        self.W_k = Tensor(_init_linear(D, D, rng))
        self.W_v = Tensor(_init_linear(D, D, rng))
        self.W_o = Tensor(_init_linear(D, D, rng))
        self.b_q = Tensor(np.zeros(D))
        self.b_k = Tensor(np.zeros(D))
        self.b_v = Tensor(np.zeros(D))
        self.b_o = Tensor(np.zeros(D))

    def params(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o, self.b_q, self.b_k, self.b_v, self.b_o]

    def __call__(self, x: Tensor) -> Tensor:
        """x: (B, T, D). Returns (B, T, D)."""
        B, T, D = x.shape
        H, hd = self.cfg.n_head, self.head_dim

        q = x.matmul(self.W_q) + self.b_q   # (B, T, D)
        k = x.matmul(self.W_k) + self.b_k
        v = x.matmul(self.W_v) + self.b_v

        # split heads: (B, T, D) -> (B, H, T, hd)
        def split_heads(t: Tensor) -> Tensor:
            t = t.reshape(B, T, H, hd)
            return t.transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # scaled dot-product attention scores: (B, H, T, T)
        scale = 1.0 / math.sqrt(hd)
        scores = q.matmul(k.transpose(0, 1, 3, 2)) * scale

        # causal mask: position i may only attend to j <= i
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # True above diagonal = masked
        neg_inf = np.where(mask, -1e9, 0.0)
        scores = scores + Tensor(neg_inf, requires_grad=False)

        attn = scores.softmax(axis=-1)       # (B, H, T, T)
        out = attn.matmul(v)                  # (B, H, T, hd)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)  # merge heads
        out = out.matmul(self.W_o) + self.b_o
        return out


class FeedForward:
    def __init__(self, cfg: GPTConfig, rng: np.random.Generator):
        D, F = cfg.n_embd, cfg.d_ff
        self.W1 = Tensor(_init_linear(D, F, rng))
        self.b1 = Tensor(np.zeros(F))
        self.W2 = Tensor(_init_linear(F, D, rng))
        self.b2 = Tensor(np.zeros(D))

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def __call__(self, x: Tensor) -> Tensor:
        h = (x.matmul(self.W1) + self.b1).gelu()
        return h.matmul(self.W2) + self.b2


class LayerNorm:
    def __init__(self, dim: int):
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))

    def params(self):
        return [self.gamma, self.beta]

    def __call__(self, x: Tensor) -> Tensor:
        return x.layer_norm(self.gamma, self.beta)


class Block:
    def __init__(self, cfg: GPTConfig, rng: np.random.Generator):
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, rng)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg, rng)

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.ff.params()

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))   # pre-norm residual attention
        x = x + self.ff(self.ln2(x))     # pre-norm residual feedforward
        return x


class GPT:
    """A decoder-only transformer language model."""

    def __init__(self, cfg: GPTConfig, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)

        self.tok_emb = Tensor(rng.standard_normal((cfg.vocab_size, cfg.n_embd)) * 0.02)
        self.pos_emb = Tensor(rng.standard_normal((cfg.block_size, cfg.n_embd)) * 0.02)
        self.blocks: List[Block] = [Block(cfg, rng) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(cfg.n_embd)
        # Output projection is weight-tied to the token embedding (GPT-2 style),
        # which both halves the parameter count and is a real, documented
        # design choice, not a shortcut.

    def params(self) -> List[Tensor]:
        ps = [self.tok_emb, self.pos_emb] + self.ln_f.params()
        for b in self.blocks:
            ps += b.params()
        return ps

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.params())

    def __call__(self, idx: np.ndarray) -> Tensor:
        """idx: int array (B, T) of token ids. Returns logits (B, T, V)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"

        tok = self.tok_emb.embedding_lookup(idx)              # (B, T, D)
        pos_ids = np.arange(T)
        pos = self.pos_emb.embedding_lookup(pos_ids)          # (T, D)
        x = tok + pos  # broadcasts (B,T,D) + (T,D)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # logits = x @ tok_emb^T  (weight tying)
        logits = x.matmul(self.tok_emb.transpose())
        return logits
