"""Read Ollama's on-disk model store (manifests + content-addressed blobs)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REGISTRY = "registry.ollama.ai"
MT = "application/vnd.ollama.image."


def default_store_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    return Path(env) if env else Path.home() / ".ollama" / "models"


@dataclass
class OllamaModel:
    name: str  # canonical "library/llama3.2:3b" style, shown as "llama3.2:3b"
    weights: Path
    projector: Path | None = None
    adapters: list[Path] = field(default_factory=list)
    template: str | None = None
    system: str | None = None
    params: dict = field(default_factory=dict)
    size_bytes: int = 0


def split_ref(ref: str) -> tuple[str, str, str]:
    """'llama3.2:3b' -> (registry, 'library/llama3.2', '3b')."""
    name, _, tag = ref.partition(":")
    tag = tag or "latest"
    parts = name.split("/")
    if len(parts) >= 3:
        return parts[0], "/".join(parts[1:]), tag
    if len(parts) == 2:
        return DEFAULT_REGISTRY, name, tag
    return DEFAULT_REGISTRY, f"library/{name}", tag


class OllamaStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else default_store_dir()

    def _blob(self, digest: str) -> Path:
        return self.root / "blobs" / digest.replace(":", "-")

    def _read_blob_text(self, digest: str) -> str:
        return self._blob(digest).read_text(encoding="utf-8")

    def resolve(self, ref: str) -> OllamaModel:
        registry, repo, tag = split_ref(ref)
        manifest_path = self.root / "manifests" / registry / repo / tag
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Ollama model {ref!r} not found in {self.root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model: OllamaModel | None = None
        extras: dict = {"adapters": [], "projector": None, "template": None, "system": None}
        params: dict = {}
        total = 0
        for layer in manifest.get("layers", []):
            mt, digest = layer["mediaType"], layer["digest"]
            total += int(layer.get("size", 0))
            kind = mt.removeprefix(MT) if mt.startswith(MT) else mt
            if kind == "model":
                model = OllamaModel(name=ref, weights=self._blob(digest))
            elif kind == "projector":
                extras["projector"] = self._blob(digest)
            elif kind == "adapter":
                extras["adapters"].append(self._blob(digest))
            elif kind in ("template", "system"):
                extras[kind] = self._read_blob_text(digest)
            elif kind == "params":
                params = json.loads(self._read_blob_text(digest))
        if model is None:
            raise ValueError(f"manifest for {ref!r} has no model layer")
        model.projector, model.adapters = extras["projector"], extras["adapters"]
        model.template, model.system = extras["template"], extras["system"]
        model.params, model.size_bytes = params, total
        return model

    def list(self) -> list[str]:
        base = self.root / "manifests"
        out: list[str] = []
        if not base.is_dir():
            return out
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = p.relative_to(base).parts  # registry, *repo, tag
                registry, repo, tag = rel[0], "/".join(rel[1:-1]), rel[-1]
                short = (
                    repo.removeprefix("library/")
                    if registry == DEFAULT_REGISTRY
                    else (f"{registry}/{repo}")
                )
                out.append(f"{short}:{tag}")
        return out
