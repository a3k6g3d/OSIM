"""Continuous-batching scheduler with chunked prefill, prefix reuse and recompute preemption.

Every step builds one mixed batch under a token budget:

1. running sequences in decode (1 token each) - latency-critical, always first;
2. running sequences still prefilling (chunked, so a long prompt cannot stall decodes);
3. waiting sequences, FCFS, admitted while the block pool and ``max_num_seqs`` allow.

When the pool runs dry the most recently admitted sequence is preempted (its blocks are freed
and it is recomputed later; its already-cached prefix blocks make that cheap).
"""

from __future__ import annotations

import enum
import time
from collections import deque
from dataclasses import dataclass, field

from .blocks import ROOT_HASH, BlockPool, block_hash
from .runner import SamplingParams, SeqWork


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_batched_tokens: int = 2048
    max_model_len: int = 4096
    eos_id: int | None = None


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    id: str
    prompt: list[int]
    params: SamplingParams
    arrival: float = field(default_factory=time.monotonic)
    output: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    finish_reason: str | None = None
    blocks: list[int] = field(default_factory=list)
    num_computed: int = 0
    committed: int = 0  # leading blocks registered in the prefix cache
    parent_hash: bytes = ROOT_HASH
    cached_tokens: int = 0  # prompt tokens served from the prefix cache (first admission)
    preemptions: int = 0
    aborted: bool = False

    def __post_init__(self) -> None:
        self.all: list[int] = list(self.prompt)


@dataclass
class Batch:
    work: list[SeqWork]
    reqs: list[Request]
    counts: list[int]

    def __bool__(self) -> bool:
        return bool(self.work)


class Scheduler:
    def __init__(self, pool: BlockPool, cfg: SchedulerConfig):
        self.pool, self.cfg = pool, cfg
        self.bs = pool.block_size
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.preemptions = 0
        self.prefix_hit_tokens = 0
        self.prompt_tokens = 0

    # ------------------------------------------------------------------ requests
    def add(self, req: Request) -> None:
        if not req.prompt:
            raise ValueError("empty prompt")
        need = -(
            -min(len(req.prompt) + req.params.max_new_tokens, self.cfg.max_model_len) // self.bs
        )
        if len(req.prompt) >= self.cfg.max_model_len:
            raise ValueError(f"prompt of {len(req.prompt)} tokens >= max_model_len")
        if need > self.pool.num_blocks:
            raise ValueError(f"request needs {need} KV blocks, pool has {self.pool.num_blocks}")
        self.prompt_tokens += len(req.prompt)
        self.waiting.append(req)

    def abort(self, req_id: str) -> None:
        """Flag for removal; freed at the next ``schedule`` so an in-flight step stays safe."""
        for r in (*self.waiting, *self.running):
            if r.id == req_id:
                r.aborted = True

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------ internals
    def _release(self, req: Request) -> None:
        self.pool.free(req.blocks)
        req.blocks = []
        req.num_computed = req.committed = 0
        req.parent_hash = ROOT_HASH

    def _ensure(self, req: Request, upto: int) -> bool:
        need = -(-upto // self.bs) - len(req.blocks)
        got: list[int] = []
        for _ in range(need):
            b = self.pool.allocate()
            if b is None:
                self.pool.free(got)
                return False
            got.append(b)
        req.blocks += got
        return True

    def _preempt(self, victim: Request) -> None:
        self.running.remove(victim)
        self._release(victim)
        victim.status = Status.WAITING
        victim.preemptions += 1
        self.preemptions += 1
        self.waiting.appendleft(victim)

    def _sweep_aborted(self) -> None:
        for r in [r for r in self.running if r.aborted]:
            self.running.remove(r)
            self._finish(r, "abort")
        for r in [r for r in self.waiting if r.aborted]:
            self.waiting.remove(r)
            self._finish(r, "abort")

    def _finish(self, req: Request, reason: str) -> None:
        self._release(req)
        req.status = Status.FINISHED
        req.finish_reason = reason

    def _work(self, req: Request, n: int) -> SeqWork:
        s = req.num_computed
        return SeqWork(
            req.id,
            req.all[s : s + n],
            s,
            list(req.blocks),
            s + n == len(req.all),
            req.params,
        )

    # ------------------------------------------------------------------ one step
    def schedule(self) -> Batch:
        self._sweep_aborted()
        budget = self.cfg.max_batched_tokens
        picked: dict[str, tuple[Request, int]] = {}

        def unpick(r: Request) -> None:
            nonlocal budget
            if r.id in picked:
                budget += picked.pop(r.id)[1]

        # decodes first, then in-progress prefills
        snapshot = sorted(self.running, key=lambda r: len(r.all) - r.num_computed > 1)
        for req in snapshot:
            if req.status is not Status.RUNNING or budget <= 0:
                continue
            n = min(len(req.all) - req.num_computed, budget)
            while not self._ensure(req, req.num_computed + n):
                victim = self.running[-1]
                unpick(victim)
                self._preempt(victim)
                if victim is req:
                    break
            if req.status is not Status.RUNNING:
                continue
            picked[req.id] = (req, n)
            budget -= n

        while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            req = self.waiting[0]
            if not req.blocks:
                blocks, parent = self.pool.match_prefix(req.all)
                req.blocks, req.parent_hash = blocks, parent
                req.committed = len(blocks)
                req.num_computed = len(blocks) * self.bs
            n = min(len(req.all) - req.num_computed, budget)
            if not self._ensure(req, req.num_computed + n):
                self._release(req)  # keep FCFS: wait for memory rather than skip ahead
                break
            if req.preemptions == 0 and req.cached_tokens == 0:
                req.cached_tokens = req.num_computed
                self.prefix_hit_tokens += req.num_computed
            self.waiting.popleft()
            req.status = Status.RUNNING
            self.running.append(req)
            picked[req.id] = (req, n)
            budget -= n

        reqs = [r for r, _ in picked.values()]
        counts = [n for _, n in picked.values()]
        return Batch([self._work(r, n) for r, n in zip(reqs, counts, strict=True)], reqs, counts)

    def update(self, batch: Batch, sampled: dict[str, int]) -> list[tuple[Request, int | None]]:
        """Apply a step's results. Returns (request, new token or None) for every request that
        emitted a token or finished; finished requests have ``finish_reason`` set."""
        events: list[tuple[Request, int | None]] = []
        bs = self.bs
        for req, n in zip(batch.reqs, batch.counts, strict=True):
            if req.status is not Status.RUNNING:
                continue
            req.num_computed += n
            while (req.committed + 1) * bs <= req.num_computed:
                i = req.committed
                h = block_hash(req.parent_hash, req.all[i * bs : (i + 1) * bs])
                req.blocks[i] = self.pool.commit(req.blocks[i], h)
                req.parent_hash = h
                req.committed += 1
            tok = sampled.get(req.id)
            if tok is None:
                continue
            req.output.append(tok)
            req.all.append(tok)
            reason = None
            if not req.params.ignore_eos and tok == self.cfg.eos_id:
                reason = "stop"
            elif tok in req.params.stop_ids:
                reason = "stop"
            elif (
                len(req.output) >= req.params.max_new_tokens
                or len(req.all) >= self.cfg.max_model_len
            ):
                reason = "length"
            if reason:
                self.running.remove(req)
                self._finish(req, reason)
            events.append((req, tok))
        return events
