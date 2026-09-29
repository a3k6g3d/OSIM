from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    source: str  # ollama:<n> | hf:<id> | path
    engine: str = "auto"  # auto | vllm | sglang | ollama | echo
    replicas: int = 1
    urls: list[str] = Field(default_factory=list)  # attach to running engines instead of launching
    options: dict[str, Any] = Field(default_factory=dict)  # tensor_parallel_size, max_model_len...


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    max_concurrency: int = 256
    queue_timeout: float = 30.0
    engine_host: str = "127.0.0.1"
    engine_base_port: int = 9000
    startup_timeout: float = 900.0
    workdir: str = "."
    log_dir: str | None = None
    ollama_models_dir: str | None = None


class OsimConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: list[ModelConfig] = Field(default_factory=list)


def load_config(path: str | Path) -> OsimConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return OsimConfig.model_validate(data)
