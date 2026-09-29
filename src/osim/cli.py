from __future__ import annotations

import argparse
import json
import shlex
import sys

from .config import OsimConfig, load_config
from .formats.ollama_store import OllamaStore
from .models import resolve


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .manager import Manager
    from .server import create_app

    cfg = load_config(args.config)
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port
    uvicorn.run(create_app(Manager(cfg)), host=cfg.server.host, port=cfg.server.port)
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from .manager import plan_models

    cfg = load_config(args.config)
    for mp in plan_models(cfg):
        print(f"# {mp.cfg.name}: engine={mp.engine} format={mp.spec.fmt} "
              f"modalities={','.join(mp.spec.modalities)}")  # fmt: skip
        if not mp.launches:
            print(f"  attach: {mp.cfg.urls or '(in-process)'}")
        for lp in mp.launches:
            for cmd in lp.prepare:
                print("  prepare:", shlex.join(cmd))
            env = " ".join(f"{k}={shlex.quote(v)}" for k, v in lp.env.items())
            print("  run:", (env + " " if env else "") + shlex.join(lp.command))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    store = OllamaStore(args.ollama_dir) if args.ollama_dir else None
    spec = resolve(args.ref, store)
    d = {k: v for k, v in spec.__dict__.items() if k != "modelfile" and v is not None}
    print(json.dumps(d, indent=2))
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    for name in OllamaStore(args.ollama_dir).list():
        print(f"ollama:{name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="osim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the gateway and its engines")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=_cmd_serve)
    p = sub.add_parser("plan", help="print the engine launch commands without running them")
    p.add_argument("-c", "--config", required=True)
    p.set_defaults(fn=_cmd_plan)
    i = sub.add_parser("inspect", help="show what OSIM sees in a model reference")
    i.add_argument("ref")
    i.add_argument("--ollama-dir")
    i.set_defaults(fn=_cmd_inspect)
    m = sub.add_parser("models", help="list models in the local Ollama store")
    m.add_argument("--ollama-dir")
    m.set_defaults(fn=_cmd_models)
    args = ap.parse_args(argv)
    try:
        return int(args.fn(args))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


__all__ = ["OsimConfig", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
