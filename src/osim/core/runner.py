"""The model-execution boundary. A runner turns scheduled work into next-token samples."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class SamplingParams:
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop_ids: tuple[int, ...] = ()
    ignore_eos: bool = False


@dataclass
class SeqWork:
    """Tokens ``tokens`` occupy positions ``start..start+len(tokens)-1`` of one sequence."""

    req_id: str
    tokens: list[int]
    start: int
    block_table: list[int]
    sample: bool  # True when this chunk reaches the end of the sequence -> return a token
    params: SamplingParams


class ModelRunner(Protocol):
    block_size: int

    def execute(self, work: list[SeqWork]) -> dict[str, int]:
        """Run one forward step: write KV for every chunk, sample where ``sample`` is set."""
        ...


class SimRunner:
    """Deterministic CPU stand-in for a model, with a real (fake) KV cache.

    Token ``t`` written at position ``p`` lands in ``kv[block_table[p // bs]][p % bs]``. The
    next token is a hash of the *entire context read back from that cache*, so any bug in block
    tables, prefix sharing or preemption/recompute changes the output. An optional linear cost
    model (``base_ms + per_token_ms * tokens``) makes step time depend on batch size.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        vocab_size: int = 256,
        eos_id: int | None = None,
        base_ms: float = 0.0,
        per_token_ms: float = 0.0,
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.base_ms, self.per_token_ms = base_ms, per_token_ms
        self.kv = [[-1] * block_size for _ in range(num_blocks)]
        self.tokens_computed = 0  # total tokens pushed through the "model"

    def _ctx_hash(self, table: list[int], upto: int) -> int:
        h = 1469598103934665603
        bs = self.block_size
        for p in range(upto):
            t = self.kv[table[p // bs]][p % bs]
            if t < 0:
                raise RuntimeError(f"KV hole at position {p}")
            h = ((h ^ (t + 1)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

    def execute(self, work: list[SeqWork]) -> dict[str, int]:
        bs = self.block_size
        n = sum(len(w.tokens) for w in work)
        self.tokens_computed += n
        if self.base_ms or self.per_token_ms:
            time.sleep((self.base_ms + self.per_token_ms * n) / 1000)
        out: dict[str, int] = {}
        for w in work:
            for i, t in enumerate(w.tokens):
                p = w.start + i
                self.kv[w.block_table[p // bs]][p % bs] = t
            if w.sample:
                tok = self._ctx_hash(w.block_table, w.start + len(w.tokens)) % self.vocab_size
                if tok == self.eos_id and w.params.ignore_eos:
                    tok = (tok + 1) % self.vocab_size
                out[w.req_id] = tok
        return out
