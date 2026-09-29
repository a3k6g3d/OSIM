"""Parser for Ollama ``Modelfile`` documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Modelfile:
    from_: str | None = None
    parameters: dict[str, list[str]] = field(default_factory=dict)
    template: str | None = None
    system: str | None = None
    adapters: list[str] = field(default_factory=list)
    license: str | None = None
    messages: list[tuple[str, str]] = field(default_factory=list)

    def param(self, key: str, default: str | None = None) -> str | None:
        vals = self.parameters.get(key)
        return vals[-1] if vals else default


def _split_instruction(line: str) -> tuple[str, str]:
    parts = line.strip().split(None, 1)
    return parts[0].upper(), parts[1] if len(parts) > 1 else ""


def _take_value(first: str, lines: list[str], i: int) -> tuple[str, int]:
    """Read an argument that may be a triple-quoted multi-line block. Returns (value, next_i)."""
    arg = first.strip()
    for q in ('"""', "'''"):
        if arg.startswith(q):
            body = arg[3:]
            if q in body:
                return body[: body.index(q)], i
            chunks = [body]
            while i < len(lines):
                nxt = lines[i]
                i += 1
                if q in nxt:
                    chunks.append(nxt[: nxt.index(q)])
                    return "\n".join(chunks).strip("\n"), i
                chunks.append(nxt)
            raise ValueError("unterminated triple-quoted block in Modelfile")
    if len(arg) >= 2 and arg[0] == arg[-1] == '"':
        arg = arg[1:-1]
    return arg, i


def parse_modelfile(text: str) -> Modelfile:
    mf = Modelfile()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cmd, arg = _split_instruction(raw)
        if cmd == "FROM":
            mf.from_ = arg.strip()
        elif cmd == "PARAMETER":
            k, _, v = arg.partition(" ")
            mf.parameters.setdefault(k.strip().lower(), []).append(v.strip().strip('"'))
        elif cmd in ("TEMPLATE", "SYSTEM", "LICENSE"):
            val, i = _take_value(arg, lines, i)
            setattr(mf, cmd.lower(), val)
        elif cmd == "ADAPTER":
            mf.adapters.append(arg.strip())
        elif cmd == "MESSAGE":
            role, _, content = arg.partition(" ")
            val, i = _take_value(content, lines, i)
            mf.messages.append((role.lower(), val))
        else:
            raise ValueError(f"unknown Modelfile instruction: {cmd}")
    return mf


def load_modelfile(path: str | Path) -> Modelfile:
    return parse_modelfile(Path(path).read_text(encoding="utf-8"))
