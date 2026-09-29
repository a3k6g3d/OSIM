import asyncio

import pytest

from osim.backends import BackendError, EchoBackend
from osim.router import PoolOverloaded, ReplicaPool, prefix_key


class Flaky(EchoBackend):
    def __init__(self, id, status=503):
        super().__init__(id)
        self.status, self.calls = status, 0

    async def forward(self, path, payload):
        self.calls += 1
        raise BackendError(self.status, "boom")

    async def forward_stream(self, path, payload):
        self.calls += 1
        raise BackendError(self.status, "boom")
        yield b""  # pragma: no cover


def msg(text, system="you are helpful"):
    return {
        "model": "m",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
    }


async def test_prefix_affinity_is_sticky():
    pool = ReplicaPool("m", [EchoBackend(f"r{i}") for i in range(4)])
    keys = {prefix_key(msg("q1", system=f"sys{i}")) for i in range(20)}
    picks = {k: pool.pick(k).id for k in keys}
    assert len({pool.pick(k).id for k in keys for _ in range(3)}) > 1  # spreads across replicas
    for k, rid in picks.items():
        assert all(pool.pick(k).id == rid for _ in range(5))  # ...but is stable per prefix


async def test_overloaded_replica_sheds_to_least_loaded():
    pool = ReplicaPool("m", [EchoBackend("a"), EchoBackend("b")], affinity_slack=2)
    k = "same-prefix"
    hot = pool.pick(k)
    hot.inflight = 10
    assert pool.pick(k).id != hot.id


async def test_failover_and_breaker():
    bad, good = Flaky("bad"), EchoBackend("good")
    pool = ReplicaPool("m", [bad, good], breaker_threshold=2, breaker_cooldown=60)
    for i in range(6):
        out = await pool.request("/v1/chat/completions", msg(f"hello {i}", system=f"s{i}"))
        assert out["choices"][0]["message"]["content"].startswith("echo:")
    assert bad.calls <= 2  # breaker opened after threshold, then bad is skipped
    assert pool.stats.per_replica["good"] == 6


async def test_non_retryable_error_propagates():
    pool = ReplicaPool("m", [Flaky("a", status=400), EchoBackend("b")])
    pool.pick = lambda key, exclude=None: pool.replicas[0]  # type: ignore[method-assign]
    with pytest.raises(BackendError) as ei:
        await pool.request("/v1/chat/completions", msg("x"))
    assert ei.value.status == 400


async def test_all_down_raises_503():
    pool = ReplicaPool("m", [Flaky("a"), Flaky("b")])
    with pytest.raises(BackendError) as ei:
        await pool.request("/v1/chat/completions", msg("x"))
    assert ei.value.status == 503


async def test_stream_failover_before_first_byte():
    pool = ReplicaPool("m", [Flaky("a"), EchoBackend("b")])
    chunks = [c async for c in pool.stream("/v1/chat/completions", msg("hi there"))]
    assert chunks[-1] == b"data: [DONE]\n\n" and len(chunks) > 2


async def test_backpressure_rejects_when_saturated():
    gate = asyncio.Event()

    class Slow(EchoBackend):
        async def forward(self, path, payload):
            await gate.wait()
            return await super().forward(path, payload)

    pool = ReplicaPool("m", [Slow("s")], max_concurrency=1, queue_timeout=0.05)
    first = asyncio.create_task(pool.request("/v1/chat/completions", msg("a")))
    await asyncio.sleep(0.01)
    with pytest.raises(PoolOverloaded):
        await pool.request("/v1/chat/completions", msg("b"))
    gate.set()
    await first
    assert pool.stats.rejected == 1


async def test_model_name_rewritten_for_backend():
    seen = {}

    class Spy(EchoBackend):
        async def forward(self, path, payload):
            seen.update(payload)
            return await super().forward(path, payload)

    pool = ReplicaPool("public", [Spy("s")], backend_model="tiny:1b")
    await pool.request("/v1/chat/completions", msg("x") | {"model": "public"})
    assert seen["model"] == "tiny:1b"
