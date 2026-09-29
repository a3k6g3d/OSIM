"""Minimal GGUF header reader (metadata only; tensors are never loaded)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"
_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}  # fmt: skip
_STRING, _ARRAY = 8, 9
_MAX_ARRAY_KEPT = 64  # tokenizer vocab arrays are huge; keep only a summary


class GGUFError(ValueError):
    pass


def _read(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GGUFError("unexpected end of file")
    return data


def _string(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", _read(f, 8))
    if n > 1 << 28:
        raise GGUFError("implausible string length")
    return _read(f, n).decode("utf-8", errors="replace")


def _value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        fmt = _SCALARS[vtype]
        return struct.unpack(fmt, _read(f, struct.calcsize(fmt)))[0]
    if vtype == _STRING:
        return _string(f)
    if vtype == _ARRAY:
        (etype,) = struct.unpack("<I", _read(f, 4))
        (count,) = struct.unpack("<Q", _read(f, 8))
        kept = [_value(f, etype) for _ in range(count)]
        if count > _MAX_ARRAY_KEPT:
            return {"__array_len__": count}
        return kept
    raise GGUFError(f"unknown value type {vtype}")


def read_gguf_metadata(path: str | Path) -> dict[str, Any]:
    """Return the GGUF key/value metadata plus ``gguf.version`` and ``gguf.tensor_count``."""
    with open(path, "rb") as f:
        if _read(f, 4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file")
        (version,) = struct.unpack("<I", _read(f, 4))
        if version < 2:
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count, kv_count = struct.unpack("<QQ", _read(f, 16))
        meta: dict[str, Any] = {"gguf.version": version, "gguf.tensor_count": tensor_count}
        for _ in range(kv_count):
            key = _string(f)
            (vtype,) = struct.unpack("<I", _read(f, 4))
            meta[key] = _value(f, vtype)
    return meta


# general.file_type -> quantization label (llama.cpp LLAMA_FTYPE_*)
_FTYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 32: "BF16",
}  # fmt: skip


def summarize(meta: dict[str, Any]) -> dict[str, Any]:
    """Pull the fields OSIM cares about out of raw GGUF metadata."""
    arch = meta.get("general.architecture", "unknown")
    return {
        "architecture": arch,
        "name": meta.get("general.name"),
        "context_length": meta.get(f"{arch}.context_length"),
        "quantization": _FTYPES.get(meta.get("general.file_type", -1)),
    }
