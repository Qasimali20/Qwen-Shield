"""The before/after report the console's Results page shows, assembled from the saved result files.

Which file plays which role is declared once here, so the page never guesses from file names. A missing file drops its
row rather than failing the page; every number is read from the saved results, nothing is restated by hand."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# End to end: (row key, label, result file, protection mode inside that file)
END_TO_END = {
    "test": [("none", "No defence", "test-agent36-shield-v1b", "none"),
             ("rules", "Rules only", "test-agent36-shield-v1b", "rules"),
             ("base", "Shield + base Qwen3-1.7B", "test-agent36-shield-base-b", "full"),
             ("finetuned", "Shield + fine-tuned Qwen3-1.7B", "test-agent36-shield-v1b", "full")],
    "test_4b": [("none", "No defence", "test-agent4b-shield-q4-v2", "none"),
                ("rules", "Rules only", "test-agent4b-shield-q4-v2", "rules"),
                ("finetuned", "Shield + fine-tuned, Q4_K_M (laptop)", "test-agent4b-shield-q4-v2", "full"),
                ("bf16", "Shield + fine-tuned, bf16", "test-agent4b-shield-bf16", "full")],
    "dev": [("none", "No defence", "dev-agent36-shield-v1b", "none"),
            ("rules", "Rules only", "dev-agent36-shield-v1b", "rules"),
            ("base", "Shield + base Qwen3-1.7B", "dev-agent36-shield-base-b", "full"),
            ("finetuned", "Shield + fine-tuned Qwen3-1.7B", "dev-agent36-shield-v1b", "full")],
}
# Assessment level: (row key, label, result file, model name inside that file)
ASSESSMENT = [("base", "Base Qwen3-1.7B", "assess-base", "qwen3-1.7b"),
              ("finetuned", "Qwen Shield (bf16)", "assess-shield-v1", "qwen-shield"),
              ("q4", "Qwen Shield Q4_K_M", "assess-shield-v1-q4km-ollama", "qwen-shield:1.7b")]
SLICES = ("all", "unseen_style", "unseen_domain")
LAPTOP = [("laptop", "laptop-i5-cpu"), ("reference", "laptop-bench-node2-cpu")]
TRAINING = "training-shield-v1"
EXTERNAL = "injecagent-bf16"     # public benchmark, see benchmarks/injecagent_eval.py
BASELINE = "baseline-protectai-deberta-v2"   # an existing detector on the same tests, see benchmarks/detector_baseline.py


def _load(results: Path, name: str) -> dict | None:
    f = results / f"{name}.json"
    return json.loads(f.read_text()) if f.is_file() else None


@lru_cache(maxsize=4)
def _build(results: Path, stamp: float) -> dict:
    e2e = {}
    for split, rows in END_TO_END.items():
        out, meta = [], None
        for key, label, name, mode in rows:
            d = _load(results, name)
            if d and mode in d:
                meta = meta or d["meta"]
                out.append({"key": key, "label": label, "summary": d[mode]["summary"],
                            "shield_model": d["meta"]["shield_model"] if mode in ("model", "full") else None})
        if out:
            e2e[split] = {"scenarios": meta["scenarios"], "agent_model": meta["agent_model"], "rows": out}

    assess, examples = [], None
    for key, label, name, model in ASSESSMENT:
        d = _load(results, name)
        if d and model in d.get("models", {}):
            m = d["models"][model]
            examples = examples or m["all"]["n"]
            assess.append({"key": key, "label": label, **{s: m[s] for s in SLICES if s in m}})

    laptop = []
    for key, name in LAPTOP:
        d = _load(results, name)
        if d:
            laptop.append({"key": key, "hardware": d["hardware"], "summary": d["summary"], "mode": d["mode"],
                           "model_load_s": d["model_load_s"], "peak_rss_mb": d["runner_peak_rss_mb"], "at": d["at"]})

    return {"end_to_end": e2e, "assessment": {"examples": examples, "rows": assess}, "laptop": laptop,
            "training": _load(results, TRAINING), "external": _load(results, EXTERNAL), "baseline": _load(results, BASELINE)}


def report(results: Path) -> dict:
    """Cached until any result file changes."""
    stamp = max((f.stat().st_mtime for f in results.glob("*.json")), default=0.0)
    return _build(results, stamp)
