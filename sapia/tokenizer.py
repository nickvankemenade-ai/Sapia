"""Tokenizers. Token ids must fit in uint16 (the shard dtype)."""
from __future__ import annotations

from typing import Sequence

SMOLLM2_REPO = "HuggingFaceTB/SmolLM2-1.7B"


class SmolLM2Tokenizer:
    """SmolLM2 byte-level BPE (49,152 tokens) via the ``tokenizers`` library.

    Special tokens: ``<|endoftext|>`` = 0 (document boundary), ``<|im_start|>`` = 1 and
    ``<|im_end|>`` = 2 (chat turns). ``encode`` treats special-token strings inside text as
    plain text, so a document that contains "<|im_end|>" cannot forge a marker; use the
    ``*_token`` ids to insert real markers.
    """

    name = "smollm2"

    def __init__(self, repo: str = SMOLLM2_REPO) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_pretrained(repo)
        self._tok.encode_special_tokens = True      # specials in text are encoded as plain text
        self.vocab_size = self._tok.get_vocab_size(with_added_tokens=True)
        self.eot_token = self._require("<|endoftext|>")
        self.im_start_token = self._require("<|im_start|>")
        self.im_end_token = self._require("<|im_end|>")

    def _require(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise ValueError(f"tokenizer has no {token!r} token")
        return tid

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str], num_threads: int = 1) -> list[list[int]]:
        # The tokenizers library parallelises batches itself (RAYON_NUM_THREADS); num_threads
        # is accepted for interface compatibility.
        return [e.ids for e in self._tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=False)


class ByteTokenizer:
    """UTF-8 bytes (0..255) plus EOT = 256. Needs no downloads; used by tests and offline demos."""

    name = "bytes"
    vocab_size = 257
    eot_token = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_batch(self, texts: Sequence[str], num_threads: int = 1) -> list[list[int]]:
        return [list(t.encode("utf-8")) for t in texts]

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(int(i) for i in ids if 0 <= int(i) < 256).decode("utf-8", errors="replace")


TOKENIZERS = {"smollm2": SmolLM2Tokenizer, "bytes": ByteTokenizer}


def get_tokenizer(name: str):
    if name not in TOKENIZERS:
        raise ValueError(f"unknown tokenizer {name!r}; available: {sorted(TOKENIZERS)}")
    tok = TOKENIZERS[name]()
    if tok.vocab_size > 65536:
        raise ValueError(f"tokenizer {name!r} does not fit in uint16")
    return tok
