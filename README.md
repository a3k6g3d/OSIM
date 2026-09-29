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

## Native engine core (`engine: native`)

`osim.core` is OSIM's own serving core, the part that decides throughput:

* **Paged KV cache** (`BlockPool`): fixed-size blocks, ref-counted, no fragmentation.
* **Automatic prefix caching**: chained block hashes, LRU eviction, prompts sharing a prefix share physical blocks.
* **Continuous batching scheduler**: decodes first, then chunked prefill, then FCFS admission under a token budget.
* **Recompute preemption** when KV memory runs out, and safe abort on client disconnect.
* **`ModelRunner` protocol**: one method, `execute(work) -> sampled tokens`. A GPU runner plugs in here.

Only the CPU `SimRunner` (deterministic, with a real fake KV cache) ships today. Tests assert the output
is bit-identical with prefix caching on or off, chunked or not, and under heavy preemption, so a bug in
block tables or sharing changes the output and fails CI.

```yaml
models:
  - {name: demo, source: "hf:x/y", engine: native,
     options: {num_blocks: 8192, block_size: 16, max_batched_tokens: 2048, prefix_cache: true}}
```

## Benchmarking (`osim bench`)

Same workload and client against any OpenAI-compatible server, so engines can be compared fairly:

```bash
osim bench --url http://host:8000 --model M --requests 64 --concurrency 16 \
           --prompt-words 1500 --shared-words 1400 --max-tokens 32
```

Point it at OSIM, vLLM, SGLang and Ollama (`/v1`) on the same GPU and model. Reports req/s, output tok/s,
TTFT and inter-token latency p50/p99.

Measured so far (CPU, `SimRunner` with a linear cost model of 2 ms + 0.02 ms/token per step, 64 requests,
16 concurrent, 1400 of 1500 prompt words shared): prefix cache off 135 tok/s and mean TTFT 1997 ms;
on 507 tok/s and 191 ms. This validates the scheduler and cache mechanism. **It says nothing about
speed against vLLM, SGLang or Ollama**, since there is no real model or GPU in it.

## Layout

```
src/osim/core/      paged KV blocks, prefix cache, scheduler, async engine, runner protocol
src/osim/bench.py   OpenAI-compatible load generator
src/osim/formats/   gguf, modelfile, ollama_store, hf readers
src/osim/models.py  reference -> ModelSpec
src/osim/engines.py launch planners for vLLM / SGLang / Ollama
src/osim/backends/  HTTP + supervised-process + echo backends
src/osim/router.py  replica pool: affinity, failover, breaker, backpressure
src/osim/server.py  OpenAI + Ollama compatible gateway
```

## Status and limits

* **OSIM does not yet outperform vLLM/SGLang/Ollama, and no result here shows that.** They win on GPU
  kernels (FlashAttention/FlashInfer, CUDA graphs, quantized GEMMs, speculative decoding, TP/PP).
  The native core has the scheduling and memory architecture but no GPU `ModelRunner` yet. The next
  milestone is a torch/CUDA runner, then `osim bench` head-to-head runs on real hardware.

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
