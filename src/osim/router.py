"""Replica pool: prefix-affinity routing, least-loaded fallback, failover, circuit breaking."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .backends.base import Backend, BackendError

PREFIX_CHARS = 512  # how much of the prompt identifies a shared KV/radix-cache prefix


@dataclass
class Replica:
    backend: Backend
    inflight: int = 0
    failures: int = 0
    open_until: float = 0.0  # circuit breaker: skip until this monotonic time

    @property
    def id(self) -> str:
        return self.backend.id

    def available(self, now: float) -> bool:
        return now >= self.open_until


@dataclass
class PoolStats:
    requests: int = 0
    errors: int = 0
    rejected: int = 0
    latency_sum: float = 0.0
    prefix_hits: int = 0  # requests routed by affinity
    per_replica: dict[str, int] = field(default_factory=dict)


def prefix_key(payload: dict[str, Any]) -> str:
    """Stable key for the shared prompt prefix (system prompt + leading turns)."""
    if "messages" in payload:
        chunks = []
        for m in payload["messages"]:
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if p.get("type") == "text")
            chunks.append(f"{m.get('role')}:{c}")
        text = "\n".join(chunks)
    else:
        p = payload.get("prompt", "")
        text = p if isinstance(p, str) else str(p)
    return text[:PREFIX_CHARS]


def _score(key: str, replica_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{key}|{replica_id}".encode()).digest()[:8], "big")


class PoolOverloaded(Exception):
    pass


class ReplicaPool:
    def __init__(
        self,
        model: str,
        replicas: list[Backend],
        backend_model: str | None = None,
        max_concurrency: int = 256,
        queue_timeout: float = 30.0,
        affinity_slack: int = 4,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 10.0,
    ):
        if not replicas:
            raise ValueError("pool needs at least one replica")
        self.model = model
        self.backend_model = backend_model or model
        self.replicas = [Replica(b) for b in replicas]
        self._sem = asyncio.Semaphore(max_concurrency)
        self.queue_timeout = queue_timeout
        self.affinity_slack = affinity_slack
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown = breaker_cooldown
        self.stats = PoolStats()

    # -- selection ---------------------------------------------------------------------------
    def pick(self, key: str, exclude: set[str] | None = None) -> Replica | None:
        now = time.monotonic()
        cands = [r for r in self.replicas if r.available(now) and r.id not in (exclude or set())]
        if not cands:
            return None
        least = min(cands, key=lambda r: r.inflight)
        best = max(cands, key=lambda r: _score(key, r.id))  # rendezvous hashing
        if best.inflight <= least.inflight + self.affinity_slack:
            self.stats.prefix_hits += 1
            return best
        return least

    def _ok(self, r: Replica) -> None:
        r.failures = 0

    def _fail(self, r: Replica) -> None:
        r.failures += 1
        if r.failures >= self.breaker_threshold:
            r.open_until = time.monotonic() + self.breaker_cooldown
            r.failures = 0

    @contextlib.asynccontextmanager
    async def _slot(self) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(self._sem.acquire(), self.queue_timeout)
        except asyncio.TimeoutError:
            self.stats.rejected += 1
            raise PoolOverloaded(self.model) from None
        try:
            yield
        finally:
            self._sem.release()

    def _prep(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "model": self.backend_model}

    # -- request paths -----------------------------------------------------------------------
    async def request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        key, body = prefix_key(payload), self._prep(payload)
        tried: set[str] = set()
        start = time.monotonic()
        self.stats.requests += 1
        async with self._slot():
            last: BackendError | None = None
            for _ in range(len(self.replicas)):
                r = self.pick(key, tried)
                if r is None:
                    break
                tried.add(r.id)
                r.inflight += 1
                try:
                    out = await r.backend.forward(path, body)
                except BackendError as e:
                    last = e
                    if e.retryable:
                        self._fail(r)
                        continue
                    self.stats.errors += 1
                    raise
                finally:
                    r.inflight -= 1
                self._ok(r)
                self.stats.latency_sum += time.monotonic() - start
                self.stats.per_replica[r.id] = self.stats.per_replica.get(r.id, 0) + 1
                return out
            self.stats.errors += 1
            raise last or BackendError(503, f"no healthy replica for {self.model}")

    async def stream(self, path: str, payload: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """Failover happens only before the first chunk; after that the stream is committed."""
        key, body = prefix_key(payload), self._prep(payload)
        tried: set[str] = set()
        start = time.monotonic()
        self.stats.requests += 1
        async with self._slot():
            last: BackendError | None = None
            for _ in range(len(self.replicas)):
                r = self.pick(key, tried)
                if r is None:
                    break
                tried.add(r.id)
                r.inflight += 1
                gen = r.backend.forward_stream(path, body)
                try:
                    try:
                        first = await gen.__anext__()  # type: ignore[attr-defined]
                    except StopAsyncIteration:
                        self._ok(r)
                        return
                    except BackendError as e:
                        last = e
                        if e.retryable:
                            self._fail(r)
                            continue
                        self.stats.errors += 1
                        raise
                    self._ok(r)
                    self.stats.per_replica[r.id] = self.stats.per_replica.get(r.id, 0) + 1
                    yield first
                    async for chunk in gen:
                        yield chunk
                    self.stats.latency_sum += time.monotonic() - start
                    return
                finally:
                    r.inflight -= 1
                    await gen.aclose()  # type: ignore[attr-defined]
            self.stats.errors += 1
            raise last or BackendError(503, f"no healthy replica for {self.model}")

    async def health(self) -> dict[str, bool]:
        res = await asyncio.gather(*(r.backend.healthy() for r in self.replicas))
        return {r.id: ok for r, ok in zip(self.replicas, res, strict=True)}
