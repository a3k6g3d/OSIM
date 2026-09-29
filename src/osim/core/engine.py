"""Async front-end: many concurrent callers, one scheduler loop, one runner thread."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .blocks import BlockPool
from .runner import ModelRunner
from .scheduler import Request, SamplingParams, Scheduler, SchedulerConfig


@dataclass
class EngineStats:
    steps: int = 0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    prefix_hit_tokens: int = 0
    preemptions: int = 0
    running: int = 0
    waiting: int = 0
    free_blocks: int = 0


class AsyncEngine:
    def __init__(
        self,
        runner: ModelRunner,
        num_blocks: int,
        cfg: SchedulerConfig | None = None,
        enable_prefix_cache: bool = True,
    ):
        self.runner = runner
        self.pool = BlockPool(num_blocks, runner.block_size, enable_prefix_cache)
        self.scheduler = Scheduler(self.pool, cfg or SchedulerConfig())
        self._queues: dict[str, asyncio.Queue[tuple[int | None, str | None]]] = {}
        self._ids = itertools.count()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="osim-runner")
        self._steps = 0
        self._generated = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._pool.shutdown(wait=False, cancel_futures=True)

    def stats(self) -> EngineStats:
        s = self.scheduler
        return EngineStats(
            self._steps,
            self._generated,
            s.prompt_tokens,
            s.prefix_hit_tokens,
            s.preemptions,
            len(s.running),
            len(s.waiting),
            self.pool.num_free,
        )

    async def generate(
        self, prompt: list[int], params: SamplingParams | None = None
    ) -> AsyncIterator[tuple[int, str | None]]:
        """Yield (token, finish_reason); finish_reason is set on the final token only."""
        await self.start()
        rid = f"req-{next(self._ids)}"
        req = Request(rid, list(prompt), params or SamplingParams())
        q: asyncio.Queue[tuple[int | None, str | None]] = asyncio.Queue()
        self.scheduler.add(req)  # raises ValueError for requests that can never fit
        self._queues[rid] = q
        self._wake.set()
        done = False
        try:
            while True:
                tok, reason = await q.get()
                if tok is None:
                    done = True
                    if reason and reason.startswith("error"):
                        raise RuntimeError(reason)
                    return
                done = reason is not None
                yield tok, reason
                if done:
                    return
        finally:
            self._queues.pop(rid, None)
            if not done:
                self.scheduler.abort(rid)

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self.scheduler.has_work():
                batch = self.scheduler.schedule()
                if not batch:
                    await asyncio.sleep(0)  # only aborted requests were pending
                    continue
                try:
                    sampled = await loop.run_in_executor(
                        self._pool, self.runner.execute, batch.work
                    )
                except Exception as e:  # runner failure: fail everything in flight
                    for r in list(self.scheduler.running):
                        self.scheduler.abort(r.id)
                        q = self._queues.get(r.id)
                        if q:
                            q.put_nowait((None, f"error: {e}"))
                    continue
                self._steps += 1
                for req, tok in self.scheduler.update(batch, sampled):
                    q = self._queues.get(req.id)
                    if q and tok is not None:
                        self._generated += 1
                        q.put_nowait((tok, req.finish_reason))
