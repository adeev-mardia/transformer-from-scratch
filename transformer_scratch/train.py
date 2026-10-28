"""Training loop CLI for the from-scratch GPT model.

Trains a real byte-pair tokenizer on the included corpus (if no trained
tokenizer exists yet), then trains the transformer with cross-entropy
next-token-prediction loss, AdamW, a cosine learning-rate schedule, and
periodic checkpointing. No internet access is used or needed: the corpus
is read from `data/corpus.txt`, already included in this repository.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np

from .autograd import cross_entropy_loss
from .data import TokenDataset
from .model import GPT, GPTConfig
from .optim import AdamW, cosine_lr_schedule
from .tokenizer import BPETokenizer


def build_tokenizer(corpus_path: str, tokenizer_path: str, vocab_size: int) -> BPETokenizer:
    if os.path.exists(tokenizer_path):
        print(f"loading existing tokenizer from {tokenizer_path}")
        return BPETokenizer.load(tokenizer_path)

    print(f"training a new BPE tokenizer (vocab_size={vocab_size}) on {corpus_path} ...")
    text = open(corpus_path, "r", encoding="utf-8").read()
    tok = BPETokenizer()
    tok.train(text, target_vocab_size=vocab_size, verbose=True)
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    tok.save(tokenizer_path)
    print(f"saved tokenizer to {tokenizer_path} (actual vocab size: {tok.vocab_size})")
    return tok


def save_checkpoint(path: str, model: GPT, cfg: GPTConfig, step: int, tokenizer_path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {f"param_{i}": p.data for i, p in enumerate(model.params())}
    meta = {
        "cfg": cfg.__dict__,
        "step": step,
        "tokenizer_path": os.path.basename(tokenizer_path),
    }
    np.savez(path, meta=json.dumps(meta), **arrays)


def load_checkpoint(path: str) -> tuple:
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    cfg = GPTConfig(**meta["cfg"])
    model = GPT(cfg, seed=0)
    params = model.params()
    for i, p in enumerate(params):
        p.data = data[f"param_{i}"]
    return model, cfg, meta


def main():
    parser = argparse.ArgumentParser(description="Train the from-scratch GPT model.")
    parser.add_argument("--corpus", default="data/corpus.txt")
    parser.add_argument("--tokenizer-path", default="checkpoints/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=1200)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint", default="checkpoints/model.npz")
    parser.add_argument("--log-csv", default="checkpoints/loss_log.csv")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    tok = build_tokenizer(args.corpus, args.tokenizer_path, args.vocab_size)

    text = open(args.corpus, "r", encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int64)
    n_val = max(1, int(0.05 * len(ids)))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    print(f"corpus: {len(text)} chars -> {len(ids)} tokens "
          f"({len(train_ids)} train / {len(val_ids)} val)")

    train_ds = TokenDataset(train_ids, args.block_size)
    val_ds = TokenDataset(val_ids, args.block_size)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        d_ff=args.d_ff,
    )
    model = GPT(cfg, seed=args.seed)
    print(f"model parameters: {model.num_parameters():,}")

    opt = AdamW(model.params(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    log_f = open(args.log_csv, "w", encoding="utf-8")
    log_f.write("step,train_loss,val_loss,lr,seconds\n")

    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        lr = cosine_lr_schedule(step, args.max_steps, args.lr, args.warmup_steps)

        x, y = train_ds.get_batch(args.batch_size, rng)
        logits = model(x)
        B, T, V = logits.shape
        loss = cross_entropy_loss(logits.reshape(B * T, V), y.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step(lr=lr)
        train_loss_value = float(loss.data)

        # Every Tensor node's `_backward` closure captures the node itself
        # (it needs `out.grad`), which creates a reference cycle for every
        # intermediate activation in the graph. Ordinary refcounting cannot
        # free a cycle, so without an explicit periodic collection the
        # cyclic GC's automatic thresholds fall behind the rate at which a
        # training loop generates large NumPy-backed cycles, and resident
        # memory grows without bound. Dropping the local references and
        # forcing a collection keeps memory flat across arbitrarily many
        # steps.
        del x, y, logits, loss
        if step % 5 == 0:
            gc.collect()

        if step % args.eval_every == 0 or step == 1:
            val_loss = evaluate(model, val_ds, batch_size=args.batch_size, max_batches=4)
            elapsed = time.time() - t0
            print(
                f"step {step:5d}/{args.max_steps} | train_loss {train_loss_value:.4f} | "
                f"val_loss {val_loss:.4f} | lr {lr:.5f} | {elapsed:.1f}s"
            )
            log_f.write(f"{step},{train_loss_value:.6f},{val_loss:.6f},{lr:.6f},{elapsed:.2f}\n")
            log_f.flush()
            save_checkpoint(args.checkpoint, model, cfg, step, args.tokenizer_path)

    save_checkpoint(args.checkpoint, model, cfg, args.max_steps, args.tokenizer_path)
    log_f.close()
    print(f"training complete. checkpoint saved to {args.checkpoint}")


def evaluate(model: GPT, ds: TokenDataset, batch_size: int, max_batches: int) -> float:
    losses = []
    for x, y in ds.iter_eval_batches(batch_size, max_batches):
        logits = model(x)
        B, T, V = logits.shape
        loss = cross_entropy_loss(logits.reshape(B * T, V), y.reshape(-1))
        losses.append(float(loss.data))
        del logits, loss
    gc.collect()
    return float(np.mean(losses)) if losses else float("nan")


if __name__ == "__main__":
    main()
