"""In-process fake engine for development, CI and demos (no GPU or engine install needed)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from .base import Backend


def _flatten(payload: dict[str, Any]) -> str:
    if "messages" in payload:
        parts = []
        for m in payload["messages"]:
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(
                    p.get("text", "[image]") if p.get("type") == "text" else "[image]" for p in c
                )
            parts.append(f"{c}")
        return "\n".join(parts)
    return str(payload.get("prompt", ""))


class EchoBackend(Backend):
    def __init__(self, id: str = "echo-0", model: str = "echo"):
        self.id = id
        self.model = model
        self.up = True

    async def healthy(self) -> bool:
        return self.up

    def _reply(self, payload: dict[str, Any]) -> str:
        return f"echo: {_flatten(payload).splitlines()[-1] if _flatten(payload) else ''}"

    async def forward(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        text = self._reply(payload)
        chat = path.endswith("chat/completions")
        choice: dict[str, Any] = {"index": 0, "finish_reason": "stop"}
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        n = len(text.split())
        return {
            "id": f"echo-{int(time.time() * 1000)}",
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": payload.get("model", self.model),
            "choices": [choice],
            "usage": {"prompt_tokens": len(_flatten(payload).split()), "completion_tokens": n},
        }

    async def forward_stream(self, path: str, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        chat = path.endswith("chat/completions")
        for word in self._reply(payload).split(" "):
            delta = {"content": word + " "}
            choice = {"index": 0, "delta": delta} if chat else {"index": 0, "text": word + " "}
            chunk = {"object": "chat.completion.chunk", "choices": [choice]}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
