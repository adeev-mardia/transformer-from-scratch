"""A GPT-style decoder-only transformer implemented from scratch on top of NumPy.

This package contains:
  - tokenizer: a byte-pair-encoding tokenizer trained from raw text
  - autograd: a small reverse-mode automatic differentiation engine
  - model: the transformer architecture (embeddings, attention, blocks)
  - optim: Adam optimizer
  - data: batching utilities
  - train: the training loop / CLI
  - generate: the text generation CLI
"""

__version__ = "0.1.0"
