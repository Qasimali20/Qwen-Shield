"""An existing open-source detector on the same tests: protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0).

    hf download protectai/deberta-v3-base-prompt-injection-v2 config.json model.safetensors tokenizer.json \
        tokenizer_config.json special_tokens_map.json added_tokens.json spm.model --local-dir <folder>
    python -m benchmarks.detector_baseline --model <folder> --injecagent <InjecAgent data folder>

It is a text classifier (SAFE / INJECTION), so it is given what it was built for: the untrusted content alone, split
into windows of its 512-token limit, flagged if any window scores INJECTION at the model's default 0.5 threshold. It
cannot name the target or quote the attack sentence, so only detection and false alarms are compared:
  - our 344 held-out documents (data/assess/test.jsonl), the same split as Qwen Shield's numbers;
  - InjecAgent's 1,054 attacks and the same 17-case benign control used for Qwen Shield."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmarks.injecagent_eval import load as load_injecagent

ROOT = Path(__file__).resolve().parents[2]


def rate(n: int, of: int) -> dict:
    return {"value": round(n / of, 4) if of else None, "n": n, "of": of}


class Detector:
    def __init__(self, path: str, device: str):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        self.device, self.inj = device, self.model.config.label2id.get("INJECTION", 1)

    @torch.no_grad()
    def flags(self, texts: list[str], window: int = 510, stride: int = 384, batch: int = 32) -> list[bool]:
        """A text is flagged if any of its 512-token windows is classified INJECTION."""
        owners, chunks = [], []
        for i, t in enumerate(texts):
            ids = self.tok(t, add_special_tokens=False)["input_ids"]
            for s in range(0, max(len(ids) - window, 0) + 1, stride):
                owners.append(i)
                chunks.append(self.tok.decode(ids[s:s + window]))
        flagged = [False] * len(texts)
        for b in range(0, len(chunks), batch):
            enc = self.tok(chunks[b:b + batch], truncation=True, max_length=512, padding=True, return_tensors="pt").to(self.device)
            pred = self.model(**enc).logits.argmax(-1).tolist()
            for o, p in zip(owners[b:b + batch], pred):
                flagged[o] |= p == self.inj
        return flagged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--injecagent", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    det, t0 = Detector(a.model, a.device), time.time()

    rows = [json.loads(line) for line in (ROOT / "data" / "assess" / "test.jsonl").open()]
    f = det.flags([r["content"] for r in rows])
    mal = [x for r, x in zip(rows, f) if r["label"]["classification"] == "malicious"]
    ben = [x for r, x in zip(rows, f) if r["label"]["classification"] == "benign"]
    held_out = {"attacks_caught": rate(sum(mal), len(mal)), "false_alarms": rate(sum(ben), len(ben))}

    inj = load_injecagent(Path(a.injecagent))
    f = det.flags([r["content"] for r in inj])
    att = [x for r, x in zip(inj, f) if r["attack"]]
    ctl = [x for r, x in zip(inj, f) if not r["attack"]]
    injecagent = {"attacks_flagged": rate(sum(att), len(att)), "benign_control_flagged": rate(sum(ctl), len(ctl))}

    out = {"meta": {"detector": "protectai/deberta-v3-base-prompt-injection-v2", "licence": "Apache-2.0",
                    "input": "untrusted content only, 512-token windows, flagged if any window is INJECTION (threshold 0.5)",
                    "device": a.device, "seconds": round(time.time() - t0), "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
           "held_out_344": held_out, "injecagent": injecagent}
    p = ROOT / "backend" / "benchmarks" / "results" / "baseline-protectai-deberta-v2.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
