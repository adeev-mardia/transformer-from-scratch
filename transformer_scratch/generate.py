"""Text-generation CLI: loads a trained checkpoint and samples text from
a prompt. Temperature, top-k and top-p (nucleus) sampling are all
implemented here by hand using plain NumPy -- no sampling library.
"""
from __future__ import annotations

import argparse

import numpy as np

from .tokenizer import BPETokenizer
from .train import load_checkpoint


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    rng: np.random.Generator = None,
) -> int:
    """logits: 1D array over the vocabulary for the final position."""
    rng = rng or np.random.default_rng()
    logits = logits.astype(np.float64)

    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature

    if top_k and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        kth_value = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth_value, -np.inf, logits)

    probs = softmax_np(logits)

    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        # keep the smallest set of tokens whose cumulative prob >= top_p
        cutoff_idx = np.searchsorted(cumulative, top_p) + 1
        keep = order[:cutoff_idx]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs = probs / probs.sum()

    return int(rng.choice(len(probs), p=probs))


def generate(
    model,
    tok: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.0,
    seed: int = None,
) -> str:
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt)
    if not ids:
        ids = [0]
    block_size = model.cfg.block_size

    for _ in range(max_new_tokens):
        context = ids[-block_size:]
        x = np.array(context, dtype=np.int64)[None, :]
        logits = model(x)  # Tensor (1, T, V)
        last_logits = logits.data[0, -1, :]
        next_id = sample_next_token(
            last_logits, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng
        )
        ids.append(next_id)

    return tok.decode(ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/model.npz")
    parser.add_argument("--tokenizer-path", default="checkpoints/tokenizer.json")
    parser.add_argument("--prompt", default="The lighthouse")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model, cfg, meta = load_checkpoint(args.checkpoint)
    tok = BPETokenizer.load(args.tokenizer_path)

    text = generate(
        model,
        tok,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()
