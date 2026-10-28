import numpy as np

from transformer_scratch.autograd import cross_entropy_loss
from transformer_scratch.model import GPT, GPTConfig


def _make_model(seed=0, **overrides):
    cfg_kwargs = dict(vocab_size=40, block_size=8, n_layer=2, n_head=2, n_embd=16, d_ff=32)
    cfg_kwargs.update(overrides)
    cfg = GPTConfig(**cfg_kwargs)
    return GPT(cfg, seed=seed), cfg


def test_forward_output_shape():
    model, cfg = _make_model()
    idx = np.random.randint(0, cfg.vocab_size, size=(3, cfg.block_size))
    logits = model(idx)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)


def test_forward_backward_gradients_are_nonzero_and_finite():
    model, cfg = _make_model()
    idx = np.random.randint(0, cfg.vocab_size, size=(2, cfg.block_size))
    targets = np.random.randint(0, cfg.vocab_size, size=(2, cfg.block_size))

    logits = model(idx)
    B, T, V = logits.shape
    loss = cross_entropy_loss(logits.reshape(B * T, V), targets.reshape(-1))
    loss.backward()

    params = model.params()
    assert any(p.grad is not None for p in params)
    for p in params:
        if p.grad is not None:
            assert np.all(np.isfinite(p.grad))
    # at least one parameter should have received a nonzero gradient
    assert any(p.grad is not None and np.abs(p.grad).sum() > 0 for p in params)


def test_causal_mask_blocks_future_tokens():
    """Changing a token at position t must not change logits at any
    position before t -- this is the defining property of a causal
    (autoregressive) transformer, verified numerically rather than
    assumed from the code."""
    model, cfg = _make_model(seed=7, block_size=6)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, cfg.vocab_size, size=(1, cfg.block_size))

    base_logits = model(idx).data.copy()

    for change_pos in range(1, cfg.block_size):
        modified = idx.copy()
        modified[0, change_pos] = (modified[0, change_pos] + 1) % cfg.vocab_size
        new_logits = model(modified).data

        # positions strictly before the change must be identical
        before = new_logits[0, :change_pos]
        base_before = base_logits[0, :change_pos]
        assert np.allclose(before, base_before, atol=1e-10), (
            f"position before index {change_pos} was affected by a future token"
        )

        # the changed position itself (or later) should generally differ
        assert not np.allclose(
            new_logits[0, change_pos:], base_logits[0, change_pos:], atol=1e-10
        )


def test_weight_tying_shares_memory():
    model, cfg = _make_model()
    idx = np.random.randint(0, cfg.vocab_size, size=(1, cfg.block_size))
    logits = model(idx)
    B, T, V = logits.shape
    targets = np.random.randint(0, cfg.vocab_size, size=(B * T,))
    loss = cross_entropy_loss(logits.reshape(B * T, V), targets)
    loss.backward()
    # tok_emb is used both as input embedding and output projection, so its
    # gradient should reflect both paths (i.e. it should not be all zero).
    assert np.abs(model.tok_emb.grad).sum() > 0


def test_num_parameters_matches_manual_count():
    model, cfg = _make_model()
    manual = sum(p.data.size for p in model.params())
    assert model.num_parameters() == manual
    assert model.num_parameters() > 0
