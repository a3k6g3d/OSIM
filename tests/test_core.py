from __future__ import annotations

import asyncio
import random

import pytest

from osim.core import (
    AsyncEngine,
    BlockPool,
    Request,
    SamplingParams,
    Scheduler,
    SchedulerConfig,
    SimRunner,
)

BS = 4


def run_sync(prompts, *, blocks, prefix=True, budget=2048, seqs=64, max_new=12):
    """Drive scheduler+runner directly; return (outputs by index, scheduler, runner)."""
    runner = SimRunner(blocks, BS)
    sch = Scheduler(
        BlockPool(blocks, BS, prefix), SchedulerConfig(max_num_seqs=seqs, max_batched_tokens=budget)
    )
    reqs = [
        Request(f"r{i}", p, SamplingParams(max_new_tokens=max_new)) for i, p in enumerate(prompts)
    ]
    for r in reqs:
        sch.add(r)
    for _ in range(100_000):
        if not sch.has_work():
            break
        batch = sch.schedule()
        assert batch, "scheduler stalled with work pending"
        sch.update(batch, runner.execute(batch.work))
    else:
        raise AssertionError("did not terminate")
    assert all(r.finish_reason == "length" for r in reqs)
    assert sch.pool.num_free == blocks  # nothing leaked
    return [r.output for r in reqs], sch, runner


def prompts(n=12, shared=16, seed=0):
    rnd = random.Random(seed)
    head = [rnd.randrange(256) for _ in range(shared)]
    return [head + [rnd.randrange(256) for _ in range(rnd.randrange(1, 30))] for _ in range(n)]


def test_outputs_invariant_to_scheduling_choices():
    ps = prompts()
    ref, _, _ = run_sync(ps, blocks=400, prefix=False)
    for kw in (
        dict(blocks=400, prefix=True),
        dict(blocks=400, prefix=True, budget=7),  # heavy chunked prefill
        dict(blocks=400, prefix=True, budget=7, seqs=2),
    ):
        out, _, _ = run_sync(ps, **kw)
        assert out == ref, kw


def test_prefix_cache_skips_compute():
    ps = prompts(n=8, shared=32)
    _, s_off, r_off = run_sync(ps, blocks=400, prefix=False, seqs=1)
    _, s_on, r_on = run_sync(ps, blocks=400, prefix=True, seqs=1)
    assert s_off.prefix_hit_tokens == 0
    assert s_on.prefix_hit_tokens >= 7 * 32  # everyone after the first reuses the head
    assert r_on.tokens_computed < r_off.tokens_computed - 7 * 32 + 1


def test_preemption_under_memory_pressure_keeps_outputs_exact():
    ps = prompts(n=10, shared=8)
    ref, _, _ = run_sync(ps, blocks=400, prefix=False)
    for prefix in (True, False):
        out, sch, _ = run_sync(ps, blocks=22, prefix=prefix)  # far too small for all at once
        assert sch.preemptions > 0
        assert out == ref


def test_unfittable_request_rejected():
    sch = Scheduler(BlockPool(2, BS), SchedulerConfig())
    with pytest.raises(ValueError):
        sch.add(Request("x", list(range(20)), SamplingParams(max_new_tokens=4)))
    with pytest.raises(ValueError):
        sch.add(Request("y", [], SamplingParams()))


def test_block_pool_refcount_and_lru():
    p = BlockPool(4, 2)
    toks = [1, 2, 3, 4, 5]
    a = [p.allocate(), p.allocate()]
    from osim.core.blocks import ROOT_HASH, block_hash

    h0 = block_hash(ROOT_HASH, toks[:2])
    h1 = block_hash(h0, toks[2:4])
    p.commit(a[0], h0)
    p.commit(a[1], h1)
    blocks, _ = p.match_prefix(toks)
    assert blocks == a and p.ref(a[0]) == 2
    p.free(blocks)
    p.free(a)
    assert p.num_free == 4
    # uncached blocks are handed out before cached ones
    assert {p.allocate(), p.allocate()}.isdisjoint(a)
    # once forced to evict, the cache entry is dropped
    p.allocate()
    p.allocate()
    assert p.match_prefix(toks)[0] == []


def test_abort_frees_blocks():
    runner = SimRunner(64, BS)
    sch = Scheduler(BlockPool(64, BS), SchedulerConfig())
    r = Request("a", list(range(10)), SamplingParams(max_new_tokens=50))
    sch.add(r)
    b = sch.schedule()
    sch.update(b, runner.execute(b.work))
    sch.abort("a")
    assert not sch.schedule()
    assert r.finish_reason == "abort" and sch.pool.num_free == 64


async def test_async_engine_concurrent_streams_match_reference():
    ps = prompts(n=16)
    ref, _, _ = run_sync(ps, blocks=400, prefix=False)
    eng = AsyncEngine(SimRunner(400, BS), 400, SchedulerConfig(max_batched_tokens=64))

    async def one(p):
        toks, last = [], None
        async for t, reason in eng.generate(p, SamplingParams(max_new_tokens=12)):
            toks.append(t)
            last = reason
        return toks, last

    res = await asyncio.gather(*(one(p) for p in ps))
    await eng.stop()
    assert [t for t, _ in res] == ref
    assert all(r == "length" for _, r in res)
    assert eng.stats().prefix_hit_tokens > 0


async def test_async_engine_client_disconnect_aborts():
    eng = AsyncEngine(SimRunner(64, BS, per_token_ms=0.2), 64)
    g = eng.generate(list(range(8)), SamplingParams(max_new_tokens=200))
    await g.__anext__()
    await g.aclose()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if not eng.scheduler.has_work():
            break
    await eng.stop()
    assert eng.stats().free_blocks == 64


async def test_native_backend_through_gateway_and_bench():
    import httpx

    from osim.config import ModelConfig, OsimConfig
    from osim.manager import Manager
    from osim.server import create_app

    cfg = OsimConfig(
        models=[
            ModelConfig(name="nat", source="hf:x/y", engine="native", options={"num_blocks": 512})
        ]
    )
    mgr = Manager(cfg)
    await mgr.start()
    try:
        app = create_app(mgr)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            r = await c.post(
                "/v1/chat/completions",
                json={
                    "model": "nat",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
            )
            assert r.status_code == 200, r.text
            assert r.json()["usage"]["completion_tokens"] == 5
    finally:
        await mgr.stop()


async def test_bench_against_live_native_server():
    import socket

    import uvicorn

    from osim.bench import run_bench
    from osim.config import ModelConfig, OsimConfig
    from osim.manager import Manager
    from osim.server import create_app

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    mgr = Manager(OsimConfig(models=[ModelConfig(name="nat", source="hf:x/y", engine="native")]))
    await mgr.start()
    srv = uvicorn.Server(
        uvicorn.Config(create_app(mgr), host="127.0.0.1", port=port, log_level="error")
    )
    task = asyncio.create_task(srv.serve())
    try:
        while not srv.started:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        res = await run_bench(
            f"http://127.0.0.1:{port}", "nat", requests=12, concurrency=4,
            prompt_words=40, shared_words=30, max_tokens=8,
        )  # fmt: skip
        assert res["ok"] == 12 and res["output_tok_per_s"] > 0
    finally:
        srv.should_exit = True
        await task
        await mgr.stop()
