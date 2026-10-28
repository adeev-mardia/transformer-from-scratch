import os
import random
import string

from transformer_scratch.tokenizer import BPETokenizer

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")


def _sample_text():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return f.read()


def test_roundtrip_small_vocab():
    text = _sample_text()[:5000]
    tok = BPETokenizer()
    tok.train(text, target_vocab_size=400)
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_roundtrip_full_corpus():
    text = _sample_text()
    tok = BPETokenizer()
    tok.train(text, target_vocab_size=600)
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_roundtrip_unseen_text():
    # Text containing characters/words never seen in training must still
    # round-trip correctly thanks to byte-level fallback.
    text = _sample_text()[:3000]
    tok = BPETokenizer()
    tok.train(text, target_vocab_size=400)

    novel = "Zzyx the qwerty élan café ☃ snowman emoji-free test 12345!!!"
    ids = tok.encode(novel)
    assert tok.decode(ids) == novel


def test_merges_reduce_token_count():
    text = _sample_text()
    raw_byte_length = len(text.encode("utf-8"))

    tok = BPETokenizer()
    tok.train(text, target_vocab_size=1000)
    ids = tok.encode(text)

    # BPE should compress the sequence: meaningfully fewer tokens than
    # raw bytes once real merges have been learned.
    assert len(ids) < raw_byte_length
    assert tok.vocab_size > 256
    assert len(tok.merge_order) == tok.vocab_size - 256


def test_merge_correctness_on_toy_example():
    # A fully controlled toy corpus where we know exactly which pair
    # must be merged first: "ab" repeated many times, "cd" only once.
    text = "ab ab ab ab ab cd"
    tok = BPETokenizer()
    tok.train(text, target_vocab_size=257)  # exactly one merge allowed
    assert len(tok.merge_order) == 1
    a_id, b_id = ord("a"), ord("b")
    assert tok.merge_order[0] == (a_id, b_id)

    ids = tok.encode("ab")
    assert len(ids) == 1
    assert tok.decode(ids) == "ab"


def test_empty_string():
    tok = BPETokenizer()
    tok.train(_sample_text()[:2000], target_vocab_size=300)
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_save_and_load(tmp_path):
    text = _sample_text()[:4000]
    tok = BPETokenizer()
    tok.train(text, target_vocab_size=500)

    path = tmp_path / "tok.json"
    tok.save(str(path))
    loaded = BPETokenizer.load(str(path))

    assert loaded.vocab_size == tok.vocab_size
    sample = "The storm rolled in from the north, and the harbor grew quiet."
    assert loaded.encode(sample) == tok.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


def test_random_fuzz_roundtrip():
    text = _sample_text()
    tok = BPETokenizer()
    tok.train(text, target_vocab_size=700)

    rng = random.Random(42)
    alphabet = string.ascii_letters + string.digits + " .,!?'\n"
    for _ in range(20):
        length = rng.randint(0, 200)
        s = "".join(rng.choice(alphabet) for _ in range(length))
        assert tok.decode(tok.encode(s)) == s
