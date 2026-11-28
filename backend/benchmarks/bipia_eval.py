"""Second public benchmark: BIPIA (Yi et al., KDD 2025, Microsoft, MIT licence), email task.

    curl -LO https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/email/test.jsonl
    curl -LO https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_test.json
    python -m benchmarks.bipia_eval --data <folder> --models qwen3-1.7b,qwen-shield --label bf16

BIPIA's email task asks a question about a real email; the attack is one of 75 test instructions (15 categories, from
task automation to scams and misinformation) inserted at the start, middle or end of the email, as in BIPIA's own
construction. We take every email with one attack from every category (the attack and its position rotate with the
email, fixed, no sampling): 50 x 15 = 750 attacked emails. The 50 untouched emails are the benign set, real data, so
false alarms are measured on BIPIA itself. Nothing from BIPIA was used in training or tuning. Scoring is the same as
for InjecAgent (benchmarks/injecagent_eval.py)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from benchmarks.assess_eval import run_model
from benchmarks.injecagent_eval import cool_down, judge, summarize

ROOT = Path(__file__).resolve().parents[1]
POSITIONS = ("start", "middle", "end")


def insert(context: str, attack: str, where: str) -> str:
    if where == "start":
        return f"{attack}\n{context}"
    if where == "end":
        return f"{context}\n{attack}"
    cuts = [m.end() for m in re.finditer(r"[.!?]\s+|\n+", context)]
    mid = min(cuts, key=lambda c: abs(c - len(context) // 2)) if cuts else len(context) // 2
    return f"{context[:mid]}{attack}\n{context[mid:]}"


def load(folder: Path) -> list[dict]:
    emails = [json.loads(line) for line in (folder / "test.jsonl").read_text().splitlines() if line.strip()]
    attacks = json.loads((folder / "text_attack_test.json").read_text())
    rows = []
    for i, e in enumerate(emails):
        task = f"Answer this question about the email: {e['question'].removeprefix('Q:').strip()}"
        for j, (category, items) in enumerate(attacks.items()):
            attack = items[(i + j) % len(items)]
            rows.append({"user_task": task, "source": "retrieved_document", "attack": attack, "kind": category,
                         "content": insert(e["context"], attack, POSITIONS[(i + j) % 3])})
        rows.append({"user_task": task, "source": "retrieved_document", "content": e["context"], "attack": None,
                     "kind": "benign_control"})
    return rows


async def main_async(a) -> None:
    rows = load(Path(a.data))
    result = {"meta": {"benchmark": "BIPIA email task, text attacks (microsoft/BIPIA, MIT)", "label": a.label,
                       "attacks": sum(r["attack"] is not None for r in rows), "benign": sum(r["attack"] is None for r in rows),
                       "url": a.url, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, "models": {}}
    for m in a.models.split(","):
        t0, preds = time.time(), []
        for i in range(0, len(rows), a.batch):
            await cool_down(a.hot, a.cool)
            preds += await run_model(a.url, m, rows[i:i + a.batch], a.concurrency)
        result["models"][m] = summarize(rows, [judge(r, p) for r, (p, _) in zip(rows, preds)]) | {"seconds": round(time.time() - t0)}
        s = result["models"][m]
        print(f"{m}: attacks flagged {s['all_attacks']['flagged']['n']}/{s['all_attacks']['flagged']['of']}, exact evidence "
              f"{s['all_attacks']['evidence_exact']['n']}, rest kept {s['all_attacks']['removed_rest_kept']['n']}; benign emails "
              f"flagged {s['benign_control']['flagged']['n']}/{s['benign_control']['flagged']['of']}", flush=True)
    out = ROOT / "benchmarks" / "results" / f"bipia-{a.label}.json"
    out.write_text(json.dumps(result, indent=1))
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with BIPIA's email test.jsonl and text_attack_test.json")
    ap.add_argument("--models", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", default=os.environ.get("SHIELD_URL", "http://127.0.0.1:8105/v1"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--hot", type=int, default=73, help="pause at this local GPU temperature (C); 0 disables")
    ap.add_argument("--cool", type=int, default=68)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
