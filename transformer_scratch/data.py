"""Simple batching utilities for next-token-prediction training."""
from __future__ import annotations

from typing import Tuple

import numpy as np


class TokenDataset:
    """Holds a flat sequence of token ids and samples random training
    windows of length `block_size` for next-token prediction."""

    def __init__(self, ids: np.ndarray, block_size: int):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.block_size = block_size
        if len(self.ids) <= block_size + 1:
            raise ValueError("corpus too short for the requested block_size")

    def __len__(self):
        return len(self.ids) - self.block_size

    def get_batch(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        max_start = len(self.ids) - self.block_size - 1
        starts = rng.integers(0, max_start, size=batch_size)
        x = np.stack([self.ids[s : s + self.block_size] for s in starts])
        y = np.stack([self.ids[s + 1 : s + self.block_size + 1] for s in starts])
        return x, y

    def iter_eval_batches(self, batch_size: int, max_batches: int = 20):
        """Deterministic, non-overlapping batches for evaluation."""
        n = len(self)
        step = self.block_size
        starts = list(range(0, n, step))[: batch_size * max_batches]
        for i in range(0, len(starts), batch_size):
            chunk = starts[i : i + batch_size]
            if not chunk:
                continue
            x = np.stack([self.ids[s : s + self.block_size] for s in chunk])
            y = np.stack([self.ids[s + 1 : s + self.block_size + 1] for s in chunk])
            yield x, y
