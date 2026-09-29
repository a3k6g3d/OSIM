"""ManagedBackend against a real child process that speaks the OpenAI protocol over TCP."""

import socket
import sys
import textwrap

import pytest

from osim.backends import ManagedBackend
from osim.engines import LaunchPlan
from osim.router import ReplicaPool

FAKE_ENGINE = textwrap.dedent("""
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"choices": [{"message": {"role": "assistant",
                              "content": "model=" + body["model"]}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers(); self.wfile.write(out)

    HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
""")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def test_launch_serve_stop(tmp_path):
    (tmp_path / "engine.py").write_text(FAKE_ENGINE)
    port = free_port()
    plan = LaunchPlan("fake", [sys.executable, "engine.py", str(port)], backend_model="real-name")
    b = ManagedBackend("m-0", plan, "127.0.0.1", port, workdir=tmp_path, startup_timeout=20)
    await b.start()
    try:
        pool = ReplicaPool("public", [b], backend_model="real-name")
        out = await pool.request("/v1/chat/completions", {"model": "public", "messages": []})
        assert out["choices"][0]["message"]["content"] == "model=real-name"
    finally:
        await b.stop()
    assert b._proc is not None and b._proc.returncode is not None
    assert not await b.healthy()


async def test_engine_crash_on_boot_is_reported(tmp_path):
    plan = LaunchPlan("bad", [sys.executable, "-c", "raise SystemExit(3)"])
    b = ManagedBackend("x", plan, "127.0.0.1", free_port(), workdir=tmp_path, startup_timeout=10)
    with pytest.raises(RuntimeError, match="exited with code 3"):
        await b.start()
    await b.stop()
