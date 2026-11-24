"""Convert the merged Qwen3 model to GGUF and quantize it for laptops (llama.cpp, Ollama, LM Studio).

    python -m training.to_gguf --model ../artifacts/merged/qwen-shield-1.7b --quant Q4_K_M [--ollama qwen-shield:1.7b]

Writes <model>-F16.gguf with the same metadata and tensor names llama.cpp's own converter produces for Qwen3 (gpt2 BPE
tokenizer with the qwen2 pre-tokenizer, NeoX rope so no q/k permutation, tied output embedding), then quantizes it with
llama-quantize. With --ollama the quantized file is registered with the Modelfile that pins the training-time prompt
format (export.TEMPLATE, thinking off, greedy decoding)."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from qwen_shield.assessment import SYSTEM
from training.export import TEMPLATE

QUANTIZE = shutil.which("llama-quantize") or "/usr/local/lib/ollama/llama-quantize"


def vocab(tok, n_vocab: int) -> tuple[list[str], list[int]]:
    """Token list and types exactly as llama.cpp's get_vocab_base builds them for BPE models."""
    rev = {i: t for t, i in tok.vocab.items()}
    added, decoder = tok.get_added_vocab(), tok.added_tokens_decoder
    tokens, types = [], []
    for i in range(n_vocab):
        if i not in rev:
            tokens.append(f"[PAD{i}]")
            types.append(gguf.TokenType.UNUSED)
        elif rev[i] in added:
            special = decoder[i].special or (rev[i].startswith("<|") and rev[i].endswith("|>"))
            tokens.append(rev[i])
            types.append(gguf.TokenType.CONTROL if special else gguf.TokenType.USER_DEFINED)
        else:
            tokens.append(rev[i])
            types.append(gguf.TokenType.NORMAL)
    return tokens, types


def convert(model_dir: Path, out: Path) -> None:
    cfg = json.loads((model_dir / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"]
    w = gguf.GGUFWriter(str(out), "qwen3")
    w.add_name(model_dir.name)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_block_count(n_layers)
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_key_length(cfg["head_dim"])
    w.add_value_length(cfg["head_dim"])
    w.add_rope_freq_base(cfg["rope_theta"] if "rope_theta" in cfg else cfg["rope_parameters"]["rope_theta"])
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    tok = AutoTokenizer.from_pretrained(model_dir)
    tokens, types = vocab(tok, cfg["vocab_size"])
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    gguf.SpecialVocab(model_dir, load_merges=True, n_vocab=len(tokens)).add_to_gguf(w)

    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, n_layers)
    with safe_open(model_dir / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            if cfg.get("tie_word_embeddings") and key == "lm_head.weight":
                continue                                 # llama.cpp reuses token_embd for the output projection
            name = names.get_name(key, try_suffixes=(".weight", ".bias"))
            if name is None:
                raise SystemExit(f"no GGUF name for tensor {key}")
            t = f.get_tensor(key).to(torch.float32).numpy()
            w.add_tensor(name, t.astype(np.float16) if t.ndim > 1 else t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--ollama", default="")
    a = ap.parse_args()
    model = Path(a.model).resolve()
    f16 = model.parent / f"{model.name}-F16.gguf"
    q = model.parent / f"{model.name}-{a.quant}.gguf"
    if not f16.exists():
        convert(model, f16)
    subprocess.run([QUANTIZE, str(f16), str(q), a.quant], check=True, capture_output=True)
    print(json.dumps({"f16_mb": round(f16.stat().st_size / 2**20), "quantized": str(q),
                      "quantized_mb": round(q.stat().st_size / 2**20)}))
    if a.ollama:
        mf = q.with_suffix(".Modelfile")
        mf.write_text(f'FROM ./{q.name}\nTEMPLATE """{TEMPLATE}"""\nSYSTEM """{SYSTEM}"""\n'
                      'PARAMETER temperature 0\nPARAMETER num_ctx 4096\nPARAMETER stop "<|im_end|>"\n')
        subprocess.run(["ollama", "create", a.ollama, "-f", str(mf)], check=True, cwd=q.parent, capture_output=True)
        print("ollama model:", a.ollama)


if __name__ == "__main__":
    main()
