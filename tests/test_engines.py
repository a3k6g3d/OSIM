import pytest

from osim.engines import choose_engine, plan
from osim.models import ModelSpec, resolve

HF = ModelSpec("m", "hf", "/models/qwen", modalities=["text", "image"])
GG = ModelSpec("g", "gguf", "/m/x.gguf")


def test_choose_engine():
    assert choose_engine(HF) == "vllm"
    assert choose_engine(GG) == "ollama"
    assert choose_engine(HF, "sglang") == "sglang"
    with pytest.raises(ValueError):
        choose_engine(HF, "tgi")


def test_vllm_command():
    lp = plan(
        HF,
        "qwen",
        "vllm",
        "127.0.0.1",
        9000,
        {"tensor_parallel_size": 2, "max_model_len": 4096, "gpu_memory_utilization": 0.9},
    )
    c = lp.command
    assert c[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    for pair in [
        ("--model", "/models/qwen"),
        ("--port", "9000"),
        ("--tensor-parallel-size", "2"),
        ("--served-model-name", "qwen"),
        ("--max-model-len", "4096"),
    ]:
        assert c[c.index(pair[0]) + 1] == pair[1]
    assert "--enable-prefix-caching" in c
    assert c[c.index("--limit-mm-per-prompt") + 1] == '{"image": 4}'


def test_sglang_command():
    lp = plan(
        HF,
        "qwen",
        "sglang",
        "0.0.0.0",
        9001,
        {"tensor_parallel_size": 4, "prefix_caching": False, "extra_args": ["--enable-x"]},
    )
    c = lp.command
    assert c[c.index("--tp-size") + 1] == "4"
    assert "--disable-radix-cache" in c and c[-1] == "--enable-x"


def test_ollama_store_model_needs_no_import(ollama_root):
    from osim.formats.ollama_store import OllamaStore

    spec = resolve("ollama:tiny:1b", OllamaStore(ollama_root))
    lp = plan(spec, "t", "ollama", "127.0.0.1", 9002, {"max_num_seqs": 8})
    assert lp.backend_model == "tiny:1b" and not lp.prepare
    assert lp.env == {"OLLAMA_HOST": "127.0.0.1:9002", "OLLAMA_NUM_PARALLEL": "8"}


def test_ollama_gguf_import_with_modelfile(tmp_path, gguf_file):
    mf = tmp_path / "Modelfile"
    mf.write_text(f'FROM {gguf_file}\nPARAMETER temperature 0.2\nSYSTEM """be brief"""\n')
    lp = plan(resolve(str(mf)), "brief", "ollama", "127.0.0.1", 9003, {})
    assert lp.backend_model == "osim-brief"
    assert lp.prepare == [["ollama", "create", "osim-brief", "-f", ".osim/brief.Modelfile"]]
    body = lp.prepare_files[".osim/brief.Modelfile"]
    assert (
        f"FROM {gguf_file}" in body and "PARAMETER temperature 0.2" in body and "be brief" in body
    )


def test_engine_format_mismatch():
    with pytest.raises(ValueError):
        plan(GG, "g", "vllm", "h", 1, {})
    with pytest.raises(ValueError):
        plan(GG, "g", "sglang", "h", 1, {})
    with pytest.raises(ValueError):
        plan(HF, "m", "ollama", "h", 1, {})
