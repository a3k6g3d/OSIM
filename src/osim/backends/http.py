"""HTTP backends: talk to an already-running engine, or launch and supervise one."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from ..engines import LaunchPlan
from .base import Backend, BackendError


class HTTPBackend(Backend):
    def __init__(
        self,
        id: str,
        base_url: str,
        health_path: str = "/health",
        api_key: str | None = None,
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.id = id
        self.base_url = base_url.rstrip("/")
        self.health_path = health_path
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            transport=transport,
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
        )

    async def stop(self) -> None:
        await self._client.aclose()

    async def healthy(self) -> bool:
        if self._client.is_closed:
            return False
        try:
            r = await self._client.get(self.health_path, timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def forward(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self._client.post(path, json=payload)
        except httpx.HTTPError as e:
            raise BackendError(503, f"{type(e).__name__}: {e}", retryable=True) from e
        if r.status_code >= 400:
            raise BackendError(r.status_code, r.text[:2000])
        return r.json()  # type: ignore[no-any-return]

    async def forward_stream(self, path: str, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        req = self._client.build_request("POST", path, json=payload)
        try:
            resp = await self._client.send(req, stream=True)
        except httpx.HTTPError as e:
            raise BackendError(503, f"{type(e).__name__}: {e}", retryable=True) from e
        try:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode(errors="replace")[:2000]
                raise BackendError(resp.status_code, body)
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()


class ManagedBackend(HTTPBackend):
    """Launches the engine described by a :class:`LaunchPlan` and waits until it is healthy."""

    def __init__(
        self,
        id: str,
        plan: LaunchPlan,
        host: str,
        port: int,
        workdir: str | Path = ".",
        startup_timeout: float = 900.0,
        log_dir: str | Path | None = None,
    ):
        super().__init__(id, f"http://{host}:{port}", health_path=plan.health_path)
        self.plan = plan
        self.workdir = Path(workdir)
        self.startup_timeout = startup_timeout
        self.log_dir = Path(log_dir) if log_dir else None
        self._proc: asyncio.subprocess.Process | None = None
        self._log: Any = None

    async def start(self) -> None:
        env = {**os.environ, **self.plan.env}
        for rel, content in self.plan.prepare_files.items():
            path = self.workdir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        out: Any = asyncio.subprocess.DEVNULL
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_dir / f"{self.id}.log", "ab")  # noqa: ASYNC230, SIM115
            out = self._log
        self._proc = await asyncio.create_subprocess_exec(
            *self.plan.command, env=env, cwd=self.workdir, stdout=out, stderr=out
        )
        try:
            await self._wait_ready(env)
        except BaseException:
            await self._terminate()
            raise

    async def _wait_ready(self, env: dict[str, str]) -> None:
        deadline = time.monotonic() + self.startup_timeout
        prepared = not self.plan.prepare
        while time.monotonic() < deadline:
            assert self._proc is not None
            if self._proc.returncode is not None:
                raise RuntimeError(f"{self.id}: engine exited with code {self._proc.returncode}")
            if await self.healthy():
                if not prepared:
                    for cmd in self.plan.prepare:
                        p = await asyncio.create_subprocess_exec(*cmd, env=env, cwd=self.workdir)
                        if await p.wait() != 0:
                            raise RuntimeError(f"{self.id}: prepare step failed: {' '.join(cmd)}")
                    prepared = True
                return
            await asyncio.sleep(1.0)
        raise TimeoutError(f"{self.id}: engine not healthy after {self.startup_timeout:.0f}s")

    async def _terminate(self) -> None:
        proc = self._proc
        if proc and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 30)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        if self._log:
            with contextlib.suppress(Exception):
                self._log.close()

    async def stop(self) -> None:
        await self._terminate()
        await super().stop()
