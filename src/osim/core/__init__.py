"""OSIM's native engine core: paged KV cache, prefix caching, continuous-batching scheduler.

The core is model-agnostic. It decides *what to run each step* (which sequences, how many
tokens, which KV blocks) and hands that to a :class:`~osim.core.runner.ModelRunner`, which owns
the actual forward pass. Shipped runners: ``SimRunner`` (deterministic, CPU, for tests and
scheduler research). A GPU runner implements the same one-method protocol.
"""

from .blocks import BlockPool
from .engine import AsyncEngine, EngineStats
from .runner import ModelRunner, SeqWork, SimRunner
from .scheduler import Request, SamplingParams, Scheduler, SchedulerConfig

__all__ = [
    "AsyncEngine",
    "BlockPool",
    "EngineStats",
    "ModelRunner",
    "Request",
    "SamplingParams",
    "Scheduler",
    "SchedulerConfig",
    "SeqWork",
    "SimRunner",
]
