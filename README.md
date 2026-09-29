# OSIM

Open-source inference & serving gateway for LLMs and multimodal models. OSIM sits in front of
**vLLM**, **SGLang** and **Ollama**, understands the model files each of them uses, launches and
supervises the engines for you, and exposes **one** endpoint that speaks both the OpenAI API and
the Ollama API.

OSIM does not reimplement kernels or paged attention. It uses those engines for the GPU work and
handles the layer above them: model discovery, engine selection, routing and serving.

## What it does

| Area | Details |
|---|---|
| Model files | Reads Ollama's blob store (`~/.ollama/models`), `Modelfile`s, GGUF headers, and HF/vLLM/SGLang directories (`config.json`, safetensors). Detects context length, quantization, and image/audio modality. |
| Engine selection | `engine: auto` sends HF weights to vLLM and GGUF/Ollama weights to Ollama. Override per model with `vllm`, `sglang` or `ollama`. |
| Engine launch | Generic options (`tensor_parallel_size`, `max_model_len`, `gpu_memory_utilization`, `max_num_seqs`, `prefix_caching`...) are translated into each engine's real flags. Multimodal models get `--limit-mm-per-prompt`. GGUF and Modelfiles are imported via `ollama create`. |
| Routing | Prefix-affinity routing (rendezvous hashing on the prompt prefix, so shared system prompts hit the same KV/radix cache), least-loaded spillover, retry on another replica before the first byte, per-replica circuit breaker. |
| Backpressure | Per-model concurrency cap and queue timeout, returning HTTP 429 rather than piling up. |
| APIs | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` (OpenAI) and `/api/chat`, `/api/generate`, `/api/tags` (Ollama), with SSE and NDJSON streaming. Ollama `images` are converted to OpenAI `image_url` parts. |
| Ops | `/health`, Prometheus-style `/metrics`, optional bearer API key. |

## Quick start

```bash
pip install -e ".[dev]"
osim inspect ollama:llama3.2:3b          # what OSIM sees in a model
osim models                              # list your local Ollama models
osim plan -c examples/osim.yaml          # print the exact engine commands, run nothing
osim serve -c examples/osim.yaml
```

Try it without a GPU using the built-in `echo` engine:

```yaml
models:
  - {name: demo, source: "hf:demo/echo", engine: echo, replicas: 2}
```

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"demo","messages":[{"role":"user","content":"hello"}]}'
```

Existing clients work unchanged: point the OpenAI SDK at `http://host:8000/v1`, or an Ollama
client at `http://host:8000`.

## Model references

`ollama:<name>[:tag]` · `hf:<org>/<repo>` · `/path/to/hf-dir` · `/path/to/model.gguf` · `/path/to/Modelfile`

## Layout

```
src/osim/formats/   gguf, modelfile, ollama_store, hf readers
src/osim/models.py  reference -> ModelSpec
src/osim/engines.py launch planners for vLLM / SGLang / Ollama
src/osim/backends/  HTTP + supervised-process + echo backends
src/osim/router.py  replica pool: affinity, failover, breaker, backpressure
src/osim/server.py  OpenAI + Ollama compatible gateway
```

## Status and limits

* vLLM, SGLang and Ollama must be installed separately (`pip install vllm`, `pip install sglang`,
  or the `ollama` binary). The launch flags were written from those engines' documented CLIs; they
  are unit-tested as command strings but have **not** been run against real GPU engines yet, and
  flags change between releases, so use `extra_args` and `osim plan` to adjust.
* vLLM/SGLang do not serve GGUF and Ollama does not serve HF weights; OSIM refuses those
  combinations rather than guessing.
* The `/api/*` Ollama surface covers chat, generate and tags. Pull/create/delete are not proxied.
* `/api/chat` does not forward tool calls yet.

## Development

```bash
pip install -e ".[dev]" && ruff check . && mypy src && pytest -q
```

Apache-2.0.
