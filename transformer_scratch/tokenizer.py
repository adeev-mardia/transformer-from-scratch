"""Byte-pair-encoding tokenizer, implemented from scratch.

The algorithm follows the original BPE formulation used by GPT-2:
  1. Start from raw UTF-8 bytes (so any input text can always be encoded,
     even characters never seen during training -- no "unknown" token
     is needed).
  2. Count how often every adjacent pair of symbols occurs across the
     training corpus.
  3. Repeatedly merge the single most frequent pair into a new symbol,
     recording the merge rule, until the desired vocabulary size is
     reached.
  4. To encode new text, apply the learned merges greedily in the order
     they were learned.

No external tokenizer library is used -- merge-rule learning, encoding
and decoding are all implemented here directly.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Tuple

# A GPT-2-style pre-tokenization regex: split text into chunks (words,
# punctuation runs, whitespace runs) before applying BPE within each
# chunk. This keeps merges from crossing word boundaries in ways that
# would hurt generalization. Built on Python's standard `re` module only
# (no third-party dependency) -- letters/digits are matched with ASCII
# classes plus the Latin-1 supplement range, which covers the apostrophe
# variants and accented characters that show up in ordinary English text.
_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-zÀ-ɏ]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _get_pairs(word: Tuple[int, ...]) -> Counter:
    pairs = Counter()
    for i in range(len(word) - 1):
        pairs[(word[i], word[i + 1])] += 1
    return pairs


class BPETokenizer:
    """A from-scratch byte-level BPE tokenizer."""

    def __init__(self) -> None:
        # Base vocabulary: every possible byte value (0-255).
        self.byte_to_id: Dict[int, int] = {b: b for b in range(256)}
        self.id_to_bytes: Dict[int, bytes] = {b: bytes([b]) for b in range(256)}
        # merges: ordered list of ((id1, id2) -> new_id)
        self.merges: Dict[Tuple[int, int], int] = {}
        self.merge_order: List[Tuple[int, int]] = []
        self.vocab_size = 256

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, text: str, target_vocab_size: int, verbose: bool = False) -> None:
        """Learn BPE merge rules from `text` until vocab reaches target size."""
        assert target_vocab_size >= 256, "vocab must be at least 256 (raw bytes)"

        chunks = _SPLIT_PATTERN.findall(text)
        # Represent each chunk as a tuple of byte-ids.
        words: List[List[int]] = [list(chunk.encode("utf-8")) for chunk in chunks]

        num_merges = target_vocab_size - 256
        next_id = 256

        for merge_idx in range(num_merges):
            pair_counts: Counter = Counter()
            for word in words:
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += 1

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                # No more pairs worth merging.
                break

            new_id = next_id
            next_id += 1
            self.merges[best_pair] = new_id
            self.merge_order.append(best_pair)
            self.id_to_bytes[new_id] = (
                self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]
            )

            # Apply the merge to every word that contains the pair.
            new_words: List[List[int]] = []
            for word in words:
                new_words.append(self._merge_word(word, best_pair, new_id))
            words = new_words

            if verbose and merge_idx % 50 == 0:
                merged_str = self.id_to_bytes[new_id].decode("utf-8", errors="replace")
                print(
                    f"merge {merge_idx:4d}/{num_merges}: "
                    f"{best_pair} -> {new_id}  ({merged_str!r}, count={best_count})"
                )

        self.vocab_size = next_id

    @staticmethod
    def _merge_word(word: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        if len(word) < 2:
            return word
        out = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        chunks = _SPLIT_PATTERN.findall(text)
        ids: List[int] = []
        for chunk in chunks:
            word = list(chunk.encode("utf-8"))
            ids.extend(self._encode_word(word))
        return ids

    def _encode_word(self, word: List[int]) -> List[int]:
        word = list(word)
        while len(word) >= 2:
            pairs = _get_pairs(tuple(word))
            # Find the pair in `word` that was learned earliest (lowest
            # merge index acts as priority, matching training order).
            candidate = None
            candidate_rank = None
            for pair in pairs:
                if pair in self.merges:
                    rank = self.merge_order.index(pair) if pair in self.merges else None
                    new_id = self.merges[pair]
                    if candidate_rank is None or new_id < candidate_rank:
                        candidate = pair
                        candidate_rank = new_id
            if candidate is None:
                break
            word = self._merge_word(word, candidate, self.merges[candidate])
        return word

    def decode(self, ids: List[int]) -> str:
        byte_seq = b"".join(self.id_to_bytes[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "merges": [
                [list(pair), new_id] for pair, new_id in self.merges.items()
            ],
            "merge_order": [list(p) for p in self.merge_order],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab_size = data["vocab_size"]
        tok.merge_order = [tuple(p) for p in data["merge_order"]]
        tok.merges = {}
        for pair, new_id in data["merges"]:
            tok.merges[tuple(pair)] = new_id
        # Rebuild id_to_bytes in merge order, since later merges' byte
        # strings depend on earlier merges already being resolved.
        tok.id_to_bytes = {b: bytes([b]) for b in range(256)}
        next_id = 256
        for pair in tok.merge_order:
            new_id = tok.merges[pair]
            tok.id_to_bytes[new_id] = tok.id_to_bytes[pair[0]] + tok.id_to_bytes[pair[1]]
            next_id = max(next_id, new_id + 1)
        return tok
