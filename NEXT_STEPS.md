# Handoff: GPU runner and benchmarks (Windows 11, RTX 5070 12 GB)

Read this first in a new Claude Code desktop session. The previous session ran in a cloud
container with no GPU, so nothing here has run on real hardware yet.

## Goal
OSIM should become a high-performance LLM serving framework that can be compared with vLLM,
SGLang and Ollama. **No result so far shows it is faster than any of them.** Do not claim it
until `osim bench` shows it on this GPU. If OSIM loses, report that.

## What exists (branch `claude/jolly-cannon-38tu3v`, not merged to `main`)
- Gateway over vLLM / SGLang / Ollama: model discovery, engine launch, routing, OpenAI + Ollama APIs.
- `src/osim/core/`: paged KV block pool with prefix cache, continuous-batching scheduler with
  chunked prefill and recompute preemption, async engine, and the `ModelRunner` protocol
  (`execute(work) -> {req_id: token}`).
- Only runner today: `SimRunner` (CPU, fake KV, deterministic filler output).
- `engine: native` backend and `osim bench` (same load generator for any OpenAI-compatible URL).
- 44 tests, ruff and mypy clean, on Linux only.
- Measured only on `SimRunner` with a cost model: prefix cache off 135 tok/s, on 507 tok/s.
  That validates the scheduler mechanism, nothing about real speed.

## Known unknowns
- Never run on Windows. Most likely to break: `ManagedBackend` (process launch/supervision) and paths.
- vLLM/SGLang/Ollama launch flags are tested as command strings only, never against real engines.
- The `.venv` from the cloud session is Linux-only and not in git. Create a new one.

## Work order
1. Check the machine: `nvidia-smi` shows the 5070 and CUDA 12.8+. Python 3.11/3.12, Git.
2. New venv, `pip install -e ".[dev]"`, run `pytest -q`, `ruff check .`, `mypy src`. Fix Windows failures.
3. Install PyTorch with the `cu128` wheels (RTX 50-series/Blackwell needs CUDA 12.8+). Confirm
   `torch.cuda.is_available()` and the device name.
4. Write `TorchRunner` (`src/osim/core/torch_runner.py`) implementing `ModelRunner`:
   load HF safetensors, paged KV tensors sized from `BlockPool`, attention via PyTorch
   `scaled_dot_product_attention` first (FlashAttention/FlashInfer wheels may lag on Blackwell),
   greedy sampling first, then temperature/top-p.
5. Correctness before speed: on `Qwen2.5-1.5B-Instruct`, greedy output must match Hugging Face
   `generate` token for token, with prefix cache on and off, and with chunked prefill.
6. Benchmark on `Qwen2.5-3B-Instruct` (about 6.5 GB bf16, leaves roughly 3-4 GB for KV on 12 GB):
   run `osim bench` against OSIM, then Ollama (native Windows, GGUF of the same model, note the
   quantization difference), then vLLM in WSL2 (no native Windows support). One engine on the
   GPU at a time. Same prompts, concurrency and max tokens for every engine.
7. Only then: CUDA graphs, better kernels, quantization, speculative decoding.

## Rules for results
- Report the exact command, GPU, driver, torch version, model, and dtype/quantization.
- Report losses as plainly as wins. Put numbers in README only with the command that produced them.
- Follow the user's usual workflow for changes: verify (ruff, mypy, pytest), commit, then report
  with evidence (commit hash, outputs). There is no `.exe` packaging in this Python project.

## Paste this to start the new session
> Read NEXT_STEPS.md in this repo (branch claude/jolly-cannon-38tu3v) and start at step 1.
> I'm on Windows 11 with an RTX 5070 12 GB. Do steps 1-3 first and tell me what passes and what fails.
