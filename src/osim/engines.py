"""Launch planners for the supported engines.

Each planner turns a :class:`ModelSpec` plus generic OSIM options into the exact command line
(and environment) for vLLM, SGLang or Ollama. Planning is pure so it can be tested and shown by
``osim plan`` without any engine installed.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import ModelSpec

ENGINES = ("vllm", "sglang", "ollama", "echo", "native")


@dataclass
class LaunchPlan:
    engine: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    health_path: str = "/health"
    backend_model: str = ""  # model name to put in upstream requests
    prepare: list[list[str]] = field(default_factory=list)  # commands run before the server
    prepare_files: dict[str, str] = field(default_factory=dict)  # path -> content


def choose_engine(spec: ModelSpec, requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in ENGINES:
            raise ValueError(f"unknown engine {requested!r}; choose from {ENGINES}")
        return requested
    # GGUF / Ollama-store weights run on Ollama's llama.cpp runner; HF weights on vLLM.
    return "ollama" if spec.fmt == "gguf" else "vllm"


def _flag(cmd: list[str], name: str, value: Any) -> None:
    if value is None or value is False:
        return
    if value is True:
        cmd.append(name)
    else:
        cmd.extend([name, str(value)])


def plan_vllm(spec: ModelSpec, name: str, host: str, port: int, o: dict[str, Any]) -> LaunchPlan:
    if spec.fmt == "gguf":
        raise ValueError("vLLM GGUF support is experimental; use engine=ollama for GGUF models")
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", spec.path]
    cmd += ["--served-model-name", name, "--host", host, "--port", str(port)]
    _flag(cmd, "--tensor-parallel-size", o.get("tensor_parallel_size"))
    _flag(cmd, "--max-model-len", o.get("max_model_len"))
    _flag(cmd, "--gpu-memory-utilization", o.get("gpu_memory_utilization"))
    _flag(cmd, "--quantization", o.get("quantization"))
    _flag(cmd, "--dtype", o.get("dtype"))
    _flag(cmd, "--max-num-seqs", o.get("max_num_seqs"))
    _flag(cmd, "--enable-prefix-caching", o.get("prefix_caching", True))
    _flag(cmd, "--trust-remote-code", o.get("trust_remote_code"))
    if spec.multimodal:
        cmd += ["--limit-mm-per-prompt", json.dumps({"image": o.get("max_images", 4)})]
    cmd += [str(a) for a in o.get("extra_args", [])]
    return LaunchPlan("vllm", cmd, health_path="/health", backend_model=name)


def plan_sglang(spec: ModelSpec, name: str, host: str, port: int, o: dict[str, Any]) -> LaunchPlan:
    if spec.fmt == "gguf":
        raise ValueError("SGLang does not serve GGUF; use engine=ollama for GGUF models")
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", spec.path]
    cmd += ["--served-model-name", name, "--host", host, "--port", str(port)]
    _flag(cmd, "--tp-size", o.get("tensor_parallel_size"))
    _flag(cmd, "--context-length", o.get("max_model_len"))
    _flag(cmd, "--mem-fraction-static", o.get("gpu_memory_utilization"))
    _flag(cmd, "--quantization", o.get("quantization"))
    _flag(cmd, "--dtype", o.get("dtype"))
    _flag(cmd, "--max-running-requests", o.get("max_num_seqs"))
    _flag(cmd, "--trust-remote-code", o.get("trust_remote_code"))
    if not o.get("prefix_caching", True):
        cmd.append("--disable-radix-cache")
    cmd += [str(a) for a in o.get("extra_args", [])]
    return LaunchPlan("sglang", cmd, health_path="/health", backend_model=name)


def plan_ollama(spec: ModelSpec, name: str, host: str, port: int, o: dict[str, Any]) -> LaunchPlan:
    env = {"OLLAMA_HOST": f"{host}:{port}"}
    if o.get("max_num_seqs"):
        env["OLLAMA_NUM_PARALLEL"] = str(o["max_num_seqs"])
    if o.get("max_model_len"):
        env["OLLAMA_CONTEXT_LENGTH"] = str(o["max_model_len"])
    plan = LaunchPlan("ollama", ["ollama", "serve"], env=env, health_path="/")
    if spec.fmt != "gguf":
        raise ValueError("Ollama serves GGUF weights; convert HF weights or use vllm/sglang")
    if spec.ollama_name and spec.modelfile is None:
        plan.backend_model = spec.ollama_name  # already in the store; nothing to import
        return plan
    # Import GGUF (optionally with a user Modelfile's template/system/parameters).
    backend_model = f"osim-{name}"
    lines = [f"FROM {spec.path}"]
    if spec.projector:
        lines.append(f"FROM {spec.projector}")
    mf = spec.modelfile
    if mf:
        for k, vals in mf.parameters.items():
            lines += [f"PARAMETER {k} {v}" for v in vals]
        if mf.template:
            lines.append(f'TEMPLATE """{mf.template}"""')
        if mf.system:
            lines.append(f'SYSTEM """{mf.system}"""')
        lines += [f"ADAPTER {a}" for a in mf.adapters]
    mf_path = f".osim/{name}.Modelfile"
    plan.prepare_files[mf_path] = "\n".join(lines) + "\n"
    plan.prepare.append(["ollama", "create", backend_model, "-f", mf_path])
    plan.backend_model = backend_model
    return plan


def plan_echo(spec: ModelSpec, name: str, host: str, port: int, o: dict[str, Any]) -> LaunchPlan:
    return LaunchPlan("echo", [], backend_model=name)


PLANNERS: dict[str, Callable[..., LaunchPlan]] = {
    "vllm": plan_vllm,
    "sglang": plan_sglang,
    "ollama": plan_ollama,
    "echo": plan_echo,
    "native": plan_echo,
}


def plan(
    spec: ModelSpec, name: str, engine: str, host: str, port: int, options: dict[str, Any]
) -> LaunchPlan:
    return PLANNERS[engine](spec, name, host, port, options)
