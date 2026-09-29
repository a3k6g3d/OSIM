"""Turn an :class:`OsimConfig` into running replica pools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from . import engines
from .backends import Backend, EchoBackend, HTTPBackend, ManagedBackend
from .config import ModelConfig, OsimConfig
from .formats.ollama_store import OllamaStore
from .models import ModelSpec, resolve
from .router import ReplicaPool


@dataclass
class ModelPlan:
    cfg: ModelConfig
    spec: ModelSpec
    engine: str
    launches: list[engines.LaunchPlan]  # one per replica (empty when attaching to urls)
    ports: list[int]


def plan_models(cfg: OsimConfig) -> list[ModelPlan]:
    store = OllamaStore(cfg.server.ollama_models_dir) if cfg.server.ollama_models_dir else None
    out: list[ModelPlan] = []
    port = cfg.server.engine_base_port
    for m in cfg.models:
        if m.engine == "echo":
            spec = ModelSpec(ref=m.source, fmt="hub", path=m.source)
        else:
            spec = resolve(m.source, store)
        engine = engines.choose_engine(spec, m.engine)
        launches: list[engines.LaunchPlan] = []
        ports: list[int] = []
        if not m.urls and engine != "echo":
            for _ in range(m.replicas):
                launches.append(
                    engines.plan(spec, m.name, engine, cfg.server.engine_host, port, m.options)
                )
                ports.append(port)
                port += 1
        out.append(ModelPlan(m, spec, engine, launches, ports))
    return out


def _health_path(engine: str) -> str:
    return "/" if engine == "ollama" else "/health"


class Manager:
    def __init__(self, cfg: OsimConfig):
        self.cfg = cfg
        self.plans = plan_models(cfg)
        self.pools: dict[str, ReplicaPool] = {}
        self.specs: dict[str, ModelSpec] = {p.cfg.name: p.spec for p in self.plans}
        self._started: list[Backend] = []

    def _backends(self, mp: ModelPlan) -> list[Backend]:
        s, name = self.cfg.server, mp.cfg.name
        if mp.engine == "echo":
            return [EchoBackend(f"{name}-{i}", name) for i in range(mp.cfg.replicas)]
        if mp.cfg.urls:
            return [
                HTTPBackend(f"{name}-{i}", u, health_path=_health_path(mp.engine))
                for i, u in enumerate(mp.cfg.urls)
            ]
        return [
            ManagedBackend(
                f"{name}-{i}", lp, s.engine_host, port, s.workdir, s.startup_timeout, s.log_dir
            )
            for i, (lp, port) in enumerate(zip(mp.launches, mp.ports, strict=True))
        ]

    async def start(self) -> None:
        for mp in self.plans:
            backends = self._backends(mp)
            # Replicas boot in parallel: model load time dominates startup.
            await asyncio.gather(*(b.start() for b in backends))
            self._started += backends
            backend_model = (
                mp.launches[0].backend_model
                if mp.launches
                else (
                    mp.spec.ollama_name if mp.engine == "ollama" and mp.spec.ollama_name else None
                )
            )
            self.pools[mp.cfg.name] = ReplicaPool(
                mp.cfg.name,
                backends,
                backend_model=backend_model,
                max_concurrency=self.cfg.server.max_concurrency,
                queue_timeout=self.cfg.server.queue_timeout,
            )

    async def stop(self) -> None:
        await asyncio.gather(*(b.stop() for b in self._started), return_exceptions=True)
        self._started.clear()
