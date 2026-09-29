"""Resolve a user-facing model reference into a normalized :class:`ModelSpec`.

Accepted references:
  ollama:<name>[:tag]   model in the local Ollama store
  hf:<org>/<repo>       Hugging Face hub id (vLLM / SGLang download it themselves)
  /path/to/dir          HF-style directory (config.json + safetensors)
  /path/to/file.gguf    GGUF weights
  /path/to/Modelfile    Ollama Modelfile (FROM is resolved recursively)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .formats import gguf, hf
from .formats.modelfile import Modelfile, load_modelfile
from .formats.ollama_store import OllamaStore


@dataclass
class ModelSpec:
    ref: str
    fmt: str  # "hf" | "gguf" | "hub"
    path: str  # local path, or hub id when fmt == "hub"
    modalities: list[str] = field(default_factory=lambda: ["text"])
    context_length: int | None = None
    quantization: str | None = None
    architecture: str | None = None
    ollama_name: str | None = None  # set when the model already lives in an Ollama store
    projector: str | None = None
    modelfile: Modelfile | None = None

    @property
    def multimodal(self) -> bool:
        return len(self.modalities) > 1


def _from_gguf(ref: str, path: Path, projector: Path | None = None) -> ModelSpec:
    summary = gguf.summarize(gguf.read_gguf_metadata(path))
    mods = ["text"] + (["image"] if projector else [])
    return ModelSpec(
        ref=ref,
        fmt="gguf",
        path=str(path),
        modalities=mods,
        context_length=summary["context_length"],
        quantization=summary["quantization"],
        architecture=summary["architecture"],
        projector=str(projector) if projector else None,
    )


def resolve(ref: str, ollama_store: OllamaStore | None = None) -> ModelSpec:
    if ref.startswith("ollama:"):
        name = ref[len("ollama:") :]
        m = (ollama_store or OllamaStore()).resolve(name)
        spec = _from_gguf(ref, m.weights, m.projector)
        spec.ollama_name = name
        return spec
    if ref.startswith("hf:"):
        return ModelSpec(ref=ref, fmt="hub", path=ref[len("hf:") :])

    p = Path(ref).expanduser()
    if p.is_dir():
        if not hf.is_hf_dir(p):
            raise ValueError(f"{p} is a directory but has no config.json")
        info = hf.inspect_hf_dir(p)
        return ModelSpec(
            ref=ref,
            fmt="hf",
            path=str(p),
            modalities=info.modalities,
            context_length=info.context_length,
            quantization=info.quantization,
            architecture=info.architectures[0] if info.architectures else info.model_type,
        )
    if p.is_file():
        if p.suffix == ".gguf":
            return _from_gguf(ref, p)
        if p.name == "Modelfile" or p.suffix == ".modelfile":
            mf = load_modelfile(p)
            if not mf.from_:
                raise ValueError(f"{p} has no FROM instruction")
            src = mf.from_
            if not src.startswith(("ollama:", "hf:")) and not Path(src).expanduser().exists():
                candidate = p.parent / src
                src = str(candidate) if candidate.exists() else f"ollama:{src}"
            spec = resolve(src, ollama_store)
            spec.ref, spec.modelfile = ref, mf
            return spec
    raise FileNotFoundError(f"cannot resolve model reference {ref!r}")
