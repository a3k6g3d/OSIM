import json
import textwrap

import httpx
import pytest
from fastapi import FastAPI

from osim.backends import HTTPBackend
from osim.config import OsimConfig
from osim.manager import Manager
from osim.router import ReplicaPool
from osim.server import create_app, ollama_messages_to_openai


def cfg(api_key=None) -> OsimConfig:
    return OsimConfig.model_validate(
        {
            "server": {"api_key": api_key},
            "models": [{"name": "echo", "source": "hf:demo/echo", "engine": "echo", "replicas": 2}],
        }
    )


@pytest.fixture
async def client():
    app = create_app(Manager(cfg()))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://t") as c:
            yield c


async def test_models_and_health(client):
    r = await client.get("/v1/models")
    assert [m["id"] for m in r.json()["data"]] == ["echo"]
    assert (await client.get("/health")).json() == {
        "status": "ok",
        "models": {"echo": {"echo-0": True, "echo-1": True}},
    }


async def test_chat_completion(client):
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "echo", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "echo: ping"


async def test_chat_streaming(client):
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "echo", "stream": True, "messages": [{"role": "user", "content": "a b c"}]},
    ) as r:
        body = (await r.aread()).decode()
    events = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    text = "".join(json.loads(e)["choices"][0]["delta"]["content"] for e in events[:-1])
    assert text.strip() == "echo: a b c"


async def test_unknown_model_404(client):
    r = await client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404


async def test_ollama_chat_and_generate(client):
    r = await client.post(
        "/api/chat",
        json={
            "model": "echo",
            "stream": False,
            "options": {"num_predict": 5},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.json()["message"]["content"] == "echo: hi" and r.json()["done"] is True

    r = await client.post("/api/generate", json={"model": "echo", "prompt": "yo", "stream": False})
    assert r.json()["response"] == "echo: yo"

    async with client.stream(
        "POST",
        "/api/chat",
        json={"model": "echo", "messages": [{"role": "user", "content": "x y"}]},
    ) as s:
        lines = [json.loads(line) async for line in s.aiter_lines() if line]
    assert lines[-1]["done"] is True
    assert "".join(f["message"]["content"] for f in lines).strip() == "echo: x y"
    assert (await client.get("/api/tags")).json()["models"][0]["name"] == "echo"


def test_ollama_images_become_image_url_parts():
    out = ollama_messages_to_openai([{"role": "user", "content": "what?", "images": ["QUJD"]}])
    parts = out[0]["content"]
    assert parts[0] == {"type": "text", "text": "what?"}
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


async def test_metrics(client):
    await client.post(
        "/v1/chat/completions",
        json={"model": "echo", "messages": [{"role": "user", "content": "m"}]},
    )
    body = (await client.get("/metrics")).text
    assert 'osim_requests_total{model="echo"} 1' in body


async def test_api_key_enforced():
    app = create_app(Manager(cfg("secret")))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://t") as c:
            assert (await c.get("/v1/models")).status_code == 401
            assert (
                await c.get("/v1/models", headers={"Authorization": "Bearer secret"})
            ).status_code == 200
            assert (await c.get("/health")).status_code == 200


async def test_http_backend_proxies_real_openai_upstream():
    """Gateway -> HTTPBackend -> an OpenAI-style upstream (stands in for vLLM/SGLang/Ollama)."""
    up = FastAPI()
    seen = {}

    @up.get("/health")
    async def h():
        return {}

    @up.post("/v1/chat/completions")
    async def c(req: dict):
        seen.update(req)
        if req.get("stream"):
            from fastapi.responses import StreamingResponse

            async def gen():
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"choices": [{"message": {"role": "assistant", "content": "up"}}]}

    backend = HTTPBackend("u", "http://up", transport=httpx.ASGITransport(up))
    pool = ReplicaPool("pub", [backend], backend_model="llama3:8b")
    assert await backend.healthy()
    out = await pool.request("/v1/chat/completions", {"model": "pub", "messages": []})
    assert out["choices"][0]["message"]["content"] == "up" and seen["model"] == "llama3:8b"
    chunks = [
        c
        async for c in pool.stream(
            "/v1/chat/completions", {"model": "pub", "messages": [], "stream": True}
        )
    ]
    assert b"".join(chunks).endswith(b"data: [DONE]\n\n")  # transport may coalesce chunks


def test_config_and_cli_plan(tmp_path, capsys, gguf_file):
    from osim.cli import main

    p = tmp_path / "osim.yaml"
    p.write_text(
        textwrap.dedent(f"""
        server: {{engine_base_port: 9100}}
        models:
          - name: qwen
            source: {tmp_path}
            engine: sglang
            replicas: 2
            options: {{tensor_parallel_size: 2}}
          - name: small
            source: {gguf_file}
    """)
    )
    (tmp_path / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    assert main(["plan", "-c", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.count("sglang.launch_server") == 2 and "--port 9101" in out
    assert "OLLAMA_HOST=127.0.0.1:9102" in out and "ollama create osim-small" in out
    assert main(["inspect", str(gguf_file)]) == 0
    assert main(["inspect", "/no/such/thing"]) == 1
