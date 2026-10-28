# transformer-from-scratch

A GPT-style decoder-only transformer language model, built entirely from
first principles on top of NumPy. There is no PyTorch, TensorFlow, or JAX
anywhere in this repository — the autograd engine, the tokenizer, the
optimizer, and the model itself are all hand-written.

I built this to understand exactly what happens between "text in" and
"text out" in a transformer, down to the gradient computations, rather
than treating `.backward()` as a black box.

## What's in here

- **`autograd.py`** — a minimal reverse-mode automatic differentiation
  engine. Every `Tensor` operation (matmul, add, multiply, broadcasting
  sum, reshape/transpose, embedding lookup, softmax, layer norm, GELU)
  records its own local gradient as a closure, and `Tensor.backward()`
  walks the computation graph in reverse topological order, applying the
  chain rule by hand — exactly how backpropagation is defined
  mathematically.
- **`tokenizer.py`** — a byte-level BPE (byte-pair encoding) tokenizer,
  following the same approach GPT-2 uses: start from raw UTF-8 bytes (so
  anything is encodable, with no "unknown token" needed), count adjacent
  symbol pairs across the corpus, and repeatedly merge the most frequent
  pair until the target vocabulary size is reached. No external
  tokenizer library — merge learning, encoding, and decoding are all
  implemented directly on top of Python's `re` module.
- **`model.py`** — the actual GPT architecture: token + learned
  positional embeddings, multi-head causal self-attention (QKV
  projections, scaled dot-product attention with causal masking, head
  concatenation, output projection), a GELU feedforward block, pre-norm
  residual connections, and a final output projection weight-tied to the
  token embedding (GPT-2 style).
- **`optim.py`** — AdamW (decoupled weight decay) and a cosine
  learning-rate schedule with linear warmup, implemented directly against
  the `Tensor` gradients.
- **`data.py`** — batching utilities for next-token-prediction training.
- **`train.py`** / **`generate.py`** — the training loop and sampling CLI
  (temperature, top-k, and top-p/nucleus sampling, all implemented by
  hand in NumPy).

## Architecture

The default configuration (also what the checkpoint in this repo's
history was trained with) is:

| Hyperparameter      | Value |
|----------------------|-------|
| Layers (`n_layer`)   | 4     |
| Attention heads      | 4     |
| Embedding dim         | 128   |
| Feedforward dim       | 512   |
| Context length (`block_size`) | 64 |
| Vocabulary size       | 800 (learned BPE) |
| Parameters            | ~904k |

Everything runs on float64 NumPy arrays on CPU. This is intentionally a
small model — the point of the project is the from-scratch mechanics,
not scale.

## Installation

Requires Python 3.9+ and NumPy. No GPU, no other dependencies.

```bash
git clone https://github.com/adeev-mardia/transformer-from-scratch.git
cd transformer-from-scratch
pip install -e .
# or, for running the test suite too:
pip install -e ".[dev]"
```

## Usage

### Train

```bash
python -m transformer_scratch.train \
  --max-steps 800 --vocab-size 800 --block-size 64 \
  --n-layer 4 --n-head 4 --n-embd 128 --d-ff 512 \
  --batch-size 32 --lr 2e-3
```

The first run trains a BPE tokenizer on `data/corpus.txt` (a short
collection of original short stories included in this repo) and saves it
to `checkpoints/tokenizer.json`; later runs reuse it. Training writes a
step-by-step loss log to `checkpoints/loss_log.csv` and a model
checkpoint to `checkpoints/model.npz`. Checkpoints are gitignored —
run the command above to reproduce one locally (on this corpus it takes
about 10 minutes on a single CPU core for 800 steps).

### Generate

```bash
python -m transformer_scratch.generate \
  --prompt "The mountain village" \
  --max-new-tokens 150 --temperature 0.9 --top-k 40 --seed 7
```

### Run the tests

```bash
pytest tests/ -v
```

20 tests cover the autograd engine (gradient checks against the hand
derivation for matmul, GELU, softmax, layer norm, cross-entropy, and
embedding accumulation), the tokenizer (round-trip correctness, merge
behavior, fuzz testing), the model (output shapes, causal masking,
weight tying, parameter counts, finite/non-zero gradients end-to-end),
and a small integration test that trains a tiny model for a few steps
and checks the loss actually decreases.

## Training results

Training on the included 35KB corpus for 800 steps (vocab size 800,
batch size 32, cosine LR schedule peaking at 2e-3):

| Step | Train loss | Val loss |
|------|-----------|----------|
| 1    | 6.71      | 6.69     |
| 100  | 4.07      | 4.68     |
| 300  | 1.20      | 5.45     |
| 500  | 0.24      | 6.69     |
| 800  | 0.11      | 7.12     |

Train loss drops steadily and predictably — the model is clearly
learning the structure of the corpus (character/word statistics,
grammar, recurring character names and phrasing). Validation loss
bottoms out early (around step 100) and then climbs, which is the
expected signature of a ~900k-parameter model trained on a corpus of
only about 12k tokens: there just isn't enough data for a model this
size not to memorize it. On an unseen prompt the generations are
noticeably less fluent than on prompts drawn from the training text,
which is the honest version of what this project demonstrates — not a
state-of-the-art language model, but every piece of a real one, written
and verified by hand.

Sample generation from an unseen prompt (`--prompt "The mountain
village" --temperature 0.9 --top-k 40 --seed 7`):

> The mountain village for she was too old to hold on the sounding pole
> year and when she diedy who had once been at all.
>
> "Doesn't it bother you," a younger diver asked her once, "spending so
> much time on things that don't matter to anyone anymore?"
>
> Kiri thought about this while checking her regulator, the question
> settling into her the way questions did when she was about to go
> under, slow and heavy and impossible to answer quickly. "Everything
> mattered to some

It picks up character voice, dialogue formatting, and sentence rhythm
from the corpus, with the kind of small grammatical slips (`diedy`)
you'd expect from a model this size trained on this little data.

## License

MIT — see `LICENSE`.
