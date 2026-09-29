from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class BackendError(Exception):
    """Upstream failure. ``retryable`` failures may be retried on another replica."""

    def __init__(self, status: int, detail: str, retryable: bool | None = None):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail
        self.retryable = status >= 500 if retryable is None else retryable


class Backend(ABC):
    """One engine replica speaking the OpenAI wire protocol."""

    id: str

    async def start(self) -> None:  # pragma: no cover - default no-op
        return None

    async def stop(self) -> None:  # pragma: no cover - default no-op
        return None

    @abstractmethod
    async def healthy(self) -> bool: ...

    @abstractmethod
    async def forward(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` to ``path`` and return the decoded JSON body."""

    @abstractmethod
    def forward_stream(self, path: str, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        """POST with streaming; yields raw SSE bytes. Must raise BackendError before first byte."""
