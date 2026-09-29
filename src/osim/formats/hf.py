"""Inspect Hugging Face-style model directories (what vLLM and SGLang load)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_VISION_ARCH_HINTS = ("VL", "Vision", "Llava", "Idefics", "Pixtral", "Paligemma", "Mllama", "Phi3V")
_AUDIO_KEYS = ("audio_config", "audio_tower_config")


@dataclass
class HFModelInfo:
    path: Path
    architectures: list[str] = field(default_factory=list)
    model_type: str | None = None
    context_length: int | None = None
    quantization: str | None = None
    dtype: str | None = None
    modalities: list[str] = field(default_factory=lambda: ["text"])
    has_safetensors: bool = False
    has_gguf: bool = False


def is_hf_dir(path: str | Path) -> bool:
    return (Path(path) / "config.json").is_file()


def inspect_hf_dir(path: str | Path) -> HFModelInfo:
    p = Path(path)
    cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config") or {}
    archs = cfg.get("architectures") or []
    ctx = cfg.get("max_position_embeddings") or text_cfg.get("max_position_embeddings")
    quant = (cfg.get("quantization_config") or {}).get("quant_method")
    mods = ["text"]
    if (
        "vision_config" in cfg
        or (p / "preprocessor_config.json").is_file()
        or any(h in a for a in archs for h in _VISION_ARCH_HINTS)
    ):
        mods.append("image")
    if any(k in cfg for k in _AUDIO_KEYS):
        mods.append("audio")
    return HFModelInfo(
        path=p,
        architectures=archs,
        model_type=cfg.get("model_type"),
        context_length=ctx,
        quantization=quant,
        dtype=cfg.get("torch_dtype") or cfg.get("dtype"),
        modalities=mods,
        has_safetensors=any(p.glob("*.safetensors")),
        has_gguf=any(p.glob("*.gguf")),
    )
