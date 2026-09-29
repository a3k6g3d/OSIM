"""Load generator for any OpenAI-compatible server (OSIM, vLLM, SGLang, Ollama's /v1).

Same workload, same client, same metrics for every target - the only fair way to compare
engines. Token counts are streamed SSE chunks that carry content (one per token on all four
servers unless they batch chunks), so treat absolute tok/s as comparable across servers, not exact.
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass

import httpx


@dataclass
class Sample:
    ok: bool
    ttft: float = 0.0
    e2e: float = 0.0
    tokens: int = 0
    itl: tuple[float, ...] = ()


def make_prompts(n: int, prompt_words: int, shared_words: int, seed: int = 0) -> list[str]:
    """``shared_words`` of every prompt are one common prefix (a system prompt / RAG context)."""
    rnd = random.Random(seed)
    vocab = [f"w{i}" for i in range(5000)]

    def words(k: int) -> str:
        return " ".join(rnd.choice(vocab) for _ in range(k))

    head = words(shared_words)
    return [f"{head} {words(prompt_words - shared_words)}" for _ in range(n)]


async def _one(client: httpx.AsyncClient, url: str, body: dict, headers: dict) -> Sample:
    t0 = time.perf_counter()
    last = t0
    ttft = 0.0
    n = 0
    itl: list[float] = []
    try:
        async with client.stream("POST", url, json=body, headers=headers) as r:
            if r.status_code != 200:
                await r.aread()
                return Sample(False)
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                ch = json.loads(line[5:]).get("choices") or [{}]
                d = ch[0].get("delta", {})
                if not (d.get("content") or ch[0].get("text")):
                    continue
                now = time.perf_counter()
                if n == 0:
                    ttft = now - t0
                else:
                    itl.append(now - last)
                last = now
                n += 1
    except (httpx.HTTPError, json.JSONDecodeError):
        return Sample(False)
    return Sample(n > 0, ttft, time.perf_counter() - t0, n, tuple(itl))


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


async def run_bench(
    base_url: str,
    model: str,
    requests: int = 64,
    concurrency: int = 16,
    prompt_words: int = 256,
    shared_words: int = 0,
    max_tokens: int = 64,
    api_key: str | None = None,
    seed: int = 0,
) -> dict:
    prompts = make_prompts(requests, prompt_words, shared_words, seed)
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    url = base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=600) as client:

        async def go(p: str) -> Sample:
            async with sem:
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "stream": True,
                }
                return await _one(client, url, body, headers)

        t0 = time.perf_counter()
        samples = await asyncio.gather(*(go(p) for p in prompts))
        wall = time.perf_counter() - t0
    ok = [s for s in samples if s.ok]
    itl = [x for s in ok for x in s.itl]
    ttft = [s.ttft for s in ok]
    toks = sum(s.tokens for s in ok)
    return {
        "requests": requests,
        "ok": len(ok),
        "concurrency": concurrency,
        "wall_s": round(wall, 3),
        "req_per_s": round(len(ok) / wall, 2),
        "output_tok_per_s": round(toks / wall, 1),
        "ttft_ms": {
            "mean": round(1000 * statistics.fmean(ttft), 1) if ttft else 0,
            "p50": round(1000 * _pct(ttft, 0.5), 1),
            "p99": round(1000 * _pct(ttft, 0.99), 1),
        },
        "itl_ms": {"p50": round(1000 * _pct(itl, 0.5), 2), "p99": round(1000 * _pct(itl, 0.99), 2)},
    }


__all__ = ["Sample", "asdict", "make_prompts", "run_bench"]
