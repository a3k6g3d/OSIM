"""Serve through OSIM's own engine core (``engine: native``).

Ships with the CPU ``SimRunner`` and a byte tokenizer, so output is deterministic filler, not
language. It exists to run the paged-KV / prefix-cache / continuous-batching stack end to end
behind the real API; a GPU ``ModelRunner`` plugs into the same place.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ..core import AsyncEngine, SamplingParams, SchedulerConfig, SimRunner
from .base import Backend, BackendError
from .echo import _flatten


class NativeBackend(Backend):
    def __init__(
        self,
        id: str,
        model: str,
        num_blocks: int = 4096,
        block_size: int = 16,
        per_token_ms: float = 0.0,
        base_ms: float = 0.0,
        prefix_cache: bool = True,
        **sched: Any,
    ):
        self.id, self.model = id, model
        cfg = SchedulerConfig(**sched)
        runner = SimRunner(num_blocks, block_size, per_token_ms=per_token_ms, base_ms=base_ms)
        self.engine = AsyncEngine(runner, num_blocks, cfg, enable_prefix_cache=prefix_cache)

    async def start(self) -> None:
        await self.engine.start()

    async def stop(self) -> None:
        await self.engine.stop()

    async def healthy(self) -> bool:
        return True

    def _prep(self, payload: dict[str, Any]) -> tuple[list[int], SamplingParams]:
        prompt = list(_flatten(payload).encode("utf-8"))
        max_new = payload.get("max_tokens") or payload.get("max_completion_tokens") or 16
        return prompt, SamplingParams(max_new_tokens=int(max_new), ignore_eos=True)

    async def _tokens(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[tuple[int, str | None], None]:
        prompt, params = self._prep(payload)
        try:
            async for item in self.engine.generate(prompt, params):
                yield item
        except ValueError as e:
            raise BackendError(400, str(e)) from e
        except RuntimeError as e:
            raise BackendError(500, str(e)) from e

    async def forward(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        toks: list[int] = []
        reason = "stop"
        async for t, r in self._tokens(payload):
            toks.append(t)
            reason = r or reason
        text = bytes(toks).decode("utf-8", errors="replace")
        chat = path.endswith("chat/completions")
        choice: dict[str, Any] = {"index": 0, "finish_reason": reason}
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        prompt, _ = self._prep(payload)
        return {
            "id": f"native-{int(time.time() * 1000)}",
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": payload.get("model", self.model),
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(toks),
                "total_tokens": len(prompt) + len(toks),
            },
        }

    async def forward_stream(self, path: str, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        chat = path.endswith("chat/completions")
        gen = self._tokens(payload)
        try:
            async for t, reason in gen:
                text = bytes([t]).decode("latin-1")
                choice: dict[str, Any] = (
                    {"index": 0, "delta": {"content": text}} if chat else {"index": 0, "text": text}
                )
                choice["finish_reason"] = reason
                obj = "chat.completion.chunk" if chat else "text_completion"
                yield f"data: {json.dumps({'object': obj, 'choices': [choice]})}\n\n".encode()
            yield b"data: [DONE]\n\n"
        finally:
            await gen.aclose()  # client gone -> abort the request, free its KV blocks
