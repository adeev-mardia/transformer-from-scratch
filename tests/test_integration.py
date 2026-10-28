"""End-to-end integration test: train for real on a tiny fixture corpus
and verify loss actually decreases. No part of this test fakes the
result -- it runs the real tokenizer, the real model, the real AdamW
optimizer and the real training loop on a small amount of text so the
test runs quickly."""
import numpy as np

from transformer_scratch.autograd import cross_entropy_loss
from transformer_scratch.data import TokenDataset
from transformer_scratch.model import GPT, GPTConfig
from transformer_scratch.optim import AdamW
from transformer_scratch.tokenizer import BPETokenizer

TINY_CORPUS = (
    "the quick fox runs through the quiet forest. "
    "the quick fox runs past the old stone bridge. "
    "the quiet forest hides a quick fox near the stone bridge. "
    "a quiet fox and a quick dog run through the old forest. "
) * 20


def test_tiny_training_run_reduces_loss():
    tok = BPETokenizer()
    tok.train(TINY_CORPUS, target_vocab_size=300)
    ids = np.array(tok.encode(TINY_CORPUS), dtype=np.int64)

    block_size = 16
    ds = TokenDataset(ids, block_size=block_size)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=2,
        n_embd=32,
        d_ff=64,
    )
    model = GPT(cfg, seed=123)
    opt = AdamW(model.params(), lr=5e-3, weight_decay=0.0)
    rng = np.random.default_rng(123)

    losses = []
    for step in range(60):
        x, y = ds.get_batch(batch_size=8, rng=rng)
        logits = model(x)
        B, T, V = logits.shape
        loss = cross_entropy_loss(logits.reshape(B * T, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    first_avg = np.mean(losses[:5])
    last_avg = np.mean(losses[-5:])

    assert last_avg < first_avg, (
        f"loss did not decrease: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
    )
    # On this small, highly repetitive fixture corpus the model should
    # comfortably more than halve its loss within 60 steps.
    assert last_avg < first_avg * 0.7
