"""OpenAI- and Ollama-compatible HTTP gateway."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from . import __version__
from .backends.base import BackendError
from .manager import Manager
from .router import PoolOverloaded, ReplicaPool

# -- Ollama <-> OpenAI translation -------------------------------------------------------------


def ollama_messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ollama puts base64 images in ``images``; OpenAI wants ``image_url`` content parts."""
    out = []
    for m in messages:
        imgs = m.get("images") or []
        if not imgs:
            out.append({"role": m["role"], "content": m.get("content", "")})
            continue
        parts: list[dict[str, Any]] = [{"type": "text", "text": m.get("content", "")}]
        parts += [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{i}"}} for i in imgs
        ]
        out.append({"role": m["role"], "content": parts})
    return out


_OPTION_MAP = {
    "temperature": "temperature", "top_p": "top_p", "seed": "seed",
    "num_predict": "max_tokens", "stop": "stop", "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}  # fmt: skip


def ollama_options_to_openai(options: dict[str, Any] | None) -> dict[str, Any]:
    return {v: options[k] for k, v in _OPTION_MAP.items() if options and k in options}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sse_json(chunk: bytes) -> list[dict[str, Any] | str]:
    """Decode SSE bytes into JSON payloads (or the literal '[DONE]')."""
    out: list[dict[str, Any] | str] = []
    for line in chunk.decode(errors="replace").splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                out.append("[DONE]")
            elif data:
                out.append(json.loads(data))
    return out


# -- app ---------------------------------------------------------------------------------------


def create_app(manager: Manager) -> FastAPI:
    cfg = manager.cfg.server

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(title="OSIM", version=__version__, lifespan=lifespan)
    app.state.manager = manager

    @app.middleware("http")
    async def auth(request: Request, call_next: Any) -> Any:
        open_paths = ("/health", "/metrics")
        if cfg.api_key and request.url.path not in open_paths:
            if request.headers.get("authorization") != f"Bearer {cfg.api_key}":
                return JSONResponse({"error": {"message": "invalid api key"}}, status_code=401)
        return await call_next(request)

    def pool_for(model: str | None) -> ReplicaPool:
        pool = manager.pools.get(model or "")
        if pool is None:
            raise HTTPException(404, f"model {model!r} not found")
        return pool

    async def guarded(coro: Any) -> Any:
        try:
            return await coro
        except PoolOverloaded as e:
            raise HTTPException(429, f"model {e} is overloaded") from e
        except BackendError as e:
            raise HTTPException(e.status if e.status >= 400 else 502, e.detail) from e

    async def guarded_stream(pool: ReplicaPool, path: str, body: dict[str, Any]) -> Any:
        """Prime the stream so upstream errors surface as proper HTTP errors, not broken SSE."""
        gen = pool.stream(path, body)
        try:
            first = await gen.__anext__()
        except StopAsyncIteration:
            first = b""
        except PoolOverloaded as e:
            raise HTTPException(429, f"model {e} is overloaded") from e
        except BackendError as e:
            raise HTTPException(e.status if e.status >= 400 else 502, e.detail) from e

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                if first:
                    yield first
                async for c in gen:
                    yield c
            finally:
                await gen.aclose()

        return body_iter()

    # -- OpenAI ---------------------------------------------------------------------------
    async def openai_proxy(path: str, body: dict[str, Any]) -> Any:
        pool = pool_for(body.get("model"))
        if body.get("stream"):
            it = await guarded_stream(pool, path, body)
            return StreamingResponse(it, media_type="text/event-stream")
        return JSONResponse(await guarded(pool.request(path, body)))

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        return await openai_proxy("/v1/chat/completions", await request.json())

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        return await openai_proxy("/v1/completions", await request.json())

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Any:
        return await openai_proxy("/v1/embeddings", await request.json())

    @app.get("/v1/models")
    async def models() -> Any:
        data = [
            {
                "id": name, "object": "model", "owned_by": "osim",
                "osim": {
                    "format": s.fmt, "modalities": s.modalities,
                    "context_length": s.context_length, "quantization": s.quantization,
                },
            }
            for name, s in manager.specs.items()
        ]  # fmt: skip
        return {"object": "list", "data": data}

    # -- Ollama ---------------------------------------------------------------------------
    @app.get("/api/tags")
    async def tags() -> Any:
        return {
            "models": [
                {"name": n, "model": n, "modified_at": _now(),
                 "details": {"format": s.fmt, "quantization_level": s.quantization}}
                for n, s in manager.specs.items()
            ]
        }  # fmt: skip

    @app.post("/api/chat")
    async def ollama_chat(request: Request) -> Any:
        req = await request.json()
        body = {
            "model": req.get("model"),
            "messages": ollama_messages_to_openai(req.get("messages", [])),
            "stream": req.get("stream", True),
            **ollama_options_to_openai(req.get("options")),
        }
        return await _ollama_bridge(req, body, "/v1/chat/completions", chat=True)

    @app.post("/api/generate")
    async def ollama_generate(request: Request) -> Any:
        req = await request.json()
        body = {
            "model": req.get("model"),
            "prompt": req.get("prompt", ""),
            "stream": req.get("stream", True),
            **ollama_options_to_openai(req.get("options")),
        }
        if req.get("images"):
            msgs = [{"role": "user", "content": req.get("prompt", ""), "images": req["images"]}]
            if req.get("system"):
                msgs.insert(0, {"role": "system", "content": req["system"]})
            body = {k: v for k, v in body.items() if k != "prompt"}
            body["messages"] = ollama_messages_to_openai(msgs)
            return await _ollama_bridge(req, body, "/v1/chat/completions", chat=False)
        return await _ollama_bridge(req, body, "/v1/completions", chat=False, raw_completion=True)

    async def _ollama_bridge(
        req: dict[str, Any], body: dict[str, Any], path: str, chat: bool,
        raw_completion: bool = False,
    ) -> Any:  # fmt: skip
        pool = pool_for(body.get("model"))
        model = body["model"]

        def piece(choice: dict[str, Any], delta_key: str) -> str:
            if raw_completion:
                return choice.get("text") or ""
            return (choice.get(delta_key) or {}).get("content") or ""

        def frame(text: str, done: bool) -> dict[str, Any]:
            d: dict[str, Any] = {"model": model, "created_at": _now(), "done": done}
            if chat:
                d["message"] = {"role": "assistant", "content": text}
            else:
                d["response"] = text
            if done:
                d["done_reason"] = "stop"
            return d

        if not body.get("stream"):
            out = await guarded(pool.request(path, {**body, "stream": False}))
            choice = out["choices"][0]
            text = piece(choice, "message")
            return JSONResponse(frame(text, True))

        it = await guarded_stream(pool, path, body)

        async def ndjson() -> AsyncIterator[bytes]:
            async for chunk in it:
                for ev in _sse_json(chunk):
                    if ev == "[DONE]" or not isinstance(ev, dict) or not ev.get("choices"):
                        continue
                    text = piece(ev["choices"][0], "delta")
                    if text:
                        yield (json.dumps(frame(text, False)) + "\n").encode()
            yield (json.dumps(frame("", True)) + "\n").encode()

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    # -- ops ------------------------------------------------------------------------------
    @app.get("/health")
    async def health() -> Any:
        report = {n: await p.health() for n, p in manager.pools.items()}
        ok = all(all(v.values()) for v in report.values()) and bool(report)
        return JSONResponse({"status": "ok" if ok else "degraded", "models": report},
                            status_code=200 if ok else 503)  # fmt: skip

    @app.get("/metrics")
    async def metrics() -> Any:
        lines = []
        for name, p in manager.pools.items():
            s = p.stats
            lab = f'{{model="{name}"}}'
            lines += [
                f"osim_requests_total{lab} {s.requests}",
                f"osim_errors_total{lab} {s.errors}",
                f"osim_rejected_total{lab} {s.rejected}",
                f"osim_prefix_affinity_routed_total{lab} {s.prefix_hits}",
                f"osim_request_latency_seconds_sum{lab} {s.latency_sum:.6f}",
                f"osim_inflight{lab} {sum(r.inflight for r in p.replicas)}",
            ]
        return PlainTextResponse("\n".join(lines) + "\n")

    return app
