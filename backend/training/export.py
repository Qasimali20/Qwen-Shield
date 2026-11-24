"""Merge the LoRA adapter into the base model and package it for laptops.

    python -m training.export --base /path/to/Qwen3-1.7B --adapter ../artifacts/adapters/shield-v1 \
        --out ../artifacts/merged/qwen-shield-1.7b [--ollama qwen-shield:1.7b-q4_K_M]

The merged folder is a standard Hugging Face model (bf16 safetensors + tokenizer), publishable as is. With --ollama it
is imported into Ollama and quantized to Q4_K_M; the resulting GGUF blob is copied next to the merged folder. The
Modelfile pins the exact serving format the model was trained on: the Qwen3 chat template with thinking disabled,
greedy decoding, and Qwen Shield's system prompt."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_shield.assessment import SYSTEM

# Qwen3 chat template with thinking off, as used in training (an empty think block precedes every answer).
TEMPLATE = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
<think>

</think>

{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}
{{- end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
<think>

</think>

{{ end }}
{{- end }}"""


def merge(base: str, adapter: str, out: Path) -> None:
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    print("merged model saved to", out)


def to_ollama(merged: Path, name: str) -> Path:
    mf = merged / "Modelfile"
    mf.write_text(f'FROM {merged}\nTEMPLATE """{TEMPLATE}"""\nSYSTEM """{SYSTEM}"""\n'
                  'PARAMETER temperature 0\nPARAMETER num_ctx 4096\nPARAMETER stop "<|im_end|>"\n')
    subprocess.run(["ollama", "create", name, "-f", str(mf), "--quantize", "q4_K_M"], check=True)
    manifest = Path(subprocess.run(["ollama", "show", name, "--modelfile"], capture_output=True, text=True,
                                   check=True).stdout.split("FROM ", 1)[1].split("\n", 1)[0].strip())
    gguf = merged.parent / f"{merged.name}-Q4_K_M.gguf"
    shutil.copyfile(manifest, gguf)
    (merged.parent / f"{merged.name}-Q4_K_M.Modelfile").write_text(
        mf.read_text().replace(f"FROM {merged}", f"FROM ./{gguf.name}"))
    print(json.dumps({"ollama_model": name, "gguf": str(gguf), "size_mb": round(gguf.stat().st_size / 2**20)}))
    return gguf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ollama", default="")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not (out / "model.safetensors").exists():
        merge(a.base, a.adapter, out)
    if a.ollama:
        to_ollama(out, a.ollama)


if __name__ == "__main__":
    main()
