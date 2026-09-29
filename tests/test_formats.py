import json
from pathlib import Path

import pytest

from osim.formats.gguf import GGUFError, read_gguf_metadata, summarize
from osim.formats.hf import inspect_hf_dir
from osim.formats.modelfile import parse_modelfile
from osim.formats.ollama_store import OllamaStore, split_ref
from osim.models import resolve

from .conftest import make_ollama_store


def test_gguf_metadata(gguf_file):
    meta = read_gguf_metadata(gguf_file)
    assert meta["tokenizer.ggml.tokens"] == {"__array_len__": 100}
    s = summarize(meta)
    assert (s["architecture"], s["context_length"], s["quantization"]) == ("llama", 8192, "Q4_K_M")


def test_gguf_rejects_garbage(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(GGUFError):
        read_gguf_metadata(p)
    p.write_bytes(b"GGUF\x03\x00")
    with pytest.raises(GGUFError):
        read_gguf_metadata(p)


def test_modelfile_parse():
    mf = parse_modelfile(
        '# c\nFROM llama3.2:3b\nPARAMETER temperature 0.7\nPARAMETER stop "<end>"\n'
        'PARAMETER stop "</s>"\nSYSTEM """You are\nhelpful."""\nTEMPLATE """{{ .Prompt }}"""\n'
        "MESSAGE user hi\nADAPTER ./lora.gguf\n"
    )
    assert mf.from_ == "llama3.2:3b"
    assert mf.param("temperature") == "0.7"
    assert mf.parameters["stop"] == ["<end>", "</s>"]
    assert mf.system == "You are\nhelpful."
    assert mf.template == "{{ .Prompt }}"
    assert mf.messages == [("user", "hi")]
    assert mf.adapters == ["./lora.gguf"]


def test_modelfile_errors():
    with pytest.raises(ValueError):
        parse_modelfile("BOGUS x")
    with pytest.raises(ValueError):
        parse_modelfile('SYSTEM """never closed')


def test_split_ref():
    assert split_ref("llama3") == ("registry.ollama.ai", "library/llama3", "latest")
    assert split_ref("me/x:2") == ("registry.ollama.ai", "me/x", "2")
    assert split_ref("host.io/me/x:2") == ("host.io", "me/x", "2")


def test_ollama_store(tmp_path):
    root = make_ollama_store(tmp_path / "o", vision=True)
    st = OllamaStore(root)
    assert st.list() == ["tiny:1b"]
    m = st.resolve("tiny:1b")
    assert m.weights.is_file() and m.projector and m.projector.is_file()
    assert m.template == "{{ .Prompt }}" and m.params == {"temperature": 0.3}
    with pytest.raises(FileNotFoundError):
        st.resolve("nope")


def test_hf_dir_multimodal(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "vision_config": {},
                "max_position_embeddings": 32768,
                "quantization_config": {"quant_method": "awq"},
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"")
    info = inspect_hf_dir(tmp_path)
    assert info.modalities == ["text", "image"] and info.quantization == "awq"
    assert info.has_safetensors and info.context_length == 32768


def test_resolve_variants(tmp_path, gguf_file, ollama_root):
    assert resolve(str(gguf_file)).fmt == "gguf"
    spec = resolve("ollama:tiny:1b", OllamaStore(ollama_root))
    assert spec.ollama_name == "tiny:1b" and spec.quantization == "Q4_K_M"
    assert resolve("hf:org/model").fmt == "hub"
    mf = tmp_path / "Modelfile"
    mf.write_text(f"FROM {gguf_file}\nPARAMETER temperature 0.1\n")
    s = resolve(str(mf))
    assert s.fmt == "gguf" and s.modelfile and s.modelfile.param("temperature") == "0.1"
    mf2 = tmp_path / "sub" / "Modelfile"
    mf2.parent.mkdir()
    mf2.write_text("FROM tiny:1b\n")
    assert resolve(str(mf2), OllamaStore(ollama_root)).ollama_name == "tiny:1b"
    with pytest.raises(FileNotFoundError):
        resolve(str(Path(tmp_path / "missing")))
