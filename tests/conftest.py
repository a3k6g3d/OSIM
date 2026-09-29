from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest


def write_gguf(path: Path, arch: str = "llama", ctx: int = 8192, ftype: int = 15) -> Path:
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    kvs = [
        (s("general.architecture"), struct.pack("<I", 8) + s(arch)),
        (s("general.name"), struct.pack("<I", 8) + s("tiny")),
        (s(f"{arch}.context_length"), struct.pack("<I", 4) + struct.pack("<I", ctx)),
        (s("general.file_type"), struct.pack("<I", 4) + struct.pack("<I", ftype)),
        (  # big array must be summarized, not kept
            s("tokenizer.ggml.tokens"),
            struct.pack("<I", 9)
            + struct.pack("<I", 4)
            + struct.pack("<Q", 100)
            + b"".join(struct.pack("<I", i) for i in range(100)),
        ),
    ]
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(kvs))
    blob += b"".join(k + v for k, v in kvs)
    path.write_bytes(blob)
    return path


def make_ollama_store(
    root: Path, name: str = "tiny", tag: str = "1b", vision: bool = False
) -> Path:
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    layers = []

    def add(mt: str, data: bytes) -> None:
        d = hashlib.sha256(data).hexdigest()
        (blobs / f"sha256-{d}").write_bytes(data)
        layers.append({"mediaType": mt, "digest": f"sha256:{d}", "size": len(data)})

    tmp = root / "w.gguf"
    write_gguf(tmp)
    add("application/vnd.ollama.image.model", tmp.read_bytes())
    tmp.unlink()
    add("application/vnd.ollama.image.template", b"{{ .Prompt }}")
    add("application/vnd.ollama.image.params", json.dumps({"temperature": 0.3}).encode())
    if vision:
        add("application/vnd.ollama.image.projector", b"proj")
    mdir = root / "manifests" / "registry.ollama.ai" / "library" / name
    mdir.mkdir(parents=True)
    (mdir / tag).write_text(json.dumps({"schemaVersion": 2, "layers": layers}))
    return root


@pytest.fixture
def gguf_file(tmp_path: Path) -> Path:
    return write_gguf(tmp_path / "m.gguf")


@pytest.fixture
def ollama_root(tmp_path: Path) -> Path:
    return make_ollama_store(tmp_path / "ollama")
