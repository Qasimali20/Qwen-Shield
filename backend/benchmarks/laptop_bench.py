"""Laptop benchmark for the quantized Qwen Shield: accuracy, latency and memory on the machine it runs on.

    python3 laptop_bench.py [--limit 80] [--cpu]

Standard library only, so it runs on a stock laptop next to Ollama. Expects, in its own folder: the Q4_K_M GGUF and its
Modelfile (registered with Ollama on first run), test.jsonl (the held-out split) and schema.json (the assessment
schema, used for constrained decoding). Requests go one at a time, as an agent would send them. Writes
laptop-<host>.json with the hardware, per-request timings and the same metrics as assess_eval."""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import socket
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen-shield:1.7b"


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def hardware() -> dict:
    hw = {"os": f"{platform.system()} {platform.release()}", "arch": platform.machine(), "host": socket.gethostname()}
    if platform.system() == "Darwin":
        hw["cpu"] = sh("sysctl", "-n", "machdep.cpu.brand_string")
        hw["cores"] = sh("sysctl", "-n", "hw.ncpu")
        hw["ram_gb"] = round(int(sh("sysctl", "-n", "hw.memsize") or 0) / 2**30, 1)
    else:
        info = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
        m = re.search(r"model name\s*:\s*(.+)", info)
        names = re.findall(r"Model name:\s*(.+)", sh("lscpu"))
        hw["cpu"] = m.group(1).strip() if m else " + ".join(n.strip() for n in names)
        hw["cores"] = sh("nproc")
        mem = re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text())
        hw["ram_gb"] = round(int(mem.group(1)) / 2**20, 1) if mem else None
    if shutil.which("nvidia-smi"):
        hw["gpu"] = sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")
    hw["ollama"] = sh("ollama", "--version")
    return hw


def call(path: str, body: dict | None = None, timeout: float = 300) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ensure_model() -> None:
    names = [m["name"] for m in call("/api/tags").get("models", [])]
    if MODEL not in names:
        mf = next(HERE.glob("*.Modelfile"))
        print(f"registering {MODEL} from {mf.name} ...", flush=True)
        subprocess.run(["ollama", "create", MODEL, "-f", mf.name], check=True, cwd=HERE)


def runner_rss_mb() -> float:
    """Resident memory of all Ollama processes: the server and its model runner (llama-server under the ollama dir)."""
    out = sh("ps", "-eo", "rss=,args=")
    return round(sum(int(l.split(None, 1)[0]) for l in out.splitlines() if "ollama" in l.lower()) / 1024, 1)


def render(row: dict) -> str:
    return (f"USER TASK:\n{row['user_task'].strip()}\n\nSOURCE: {row['source']}\n\n"
            f"CONTENT:\n<<<\n{row['content'].strip()[:6000]}\n>>>")


def assess(row: dict, schema, cpu: bool) -> tuple[dict | None, dict]:
    # No "think" field: the Modelfile template already renders thinking off exactly as in training, and Ollama's
    # think=false switches to a different prompt rendering that measurably raises false alarms (9/28 vs 0/28 here).
    body = {"model": MODEL, "stream": False, "format": schema,
            "messages": [{"role": "user", "content": render(row)}],
            "options": {"temperature": 0, "num_predict": 400, **({"num_gpu": 0} if cpu else {})}}
    t0 = time.perf_counter()
    r = call("/api/chat", body)
    meta = {"latency_s": round(time.perf_counter() - t0, 3), "prompt_tokens": r.get("prompt_eval_count"),
            "output_tokens": r.get("eval_count"),
            "gen_tok_s": round(r["eval_count"] / (r["eval_duration"] / 1e9), 1) if r.get("eval_duration") else None,
            "prefill_tok_s": round(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9), 1)
            if r.get("prompt_eval_duration") else None}
    try:
        p = json.loads(r["message"]["content"])
        return (p if p.get("classification") in ("benign", "malicious") else None), meta
    except (json.JSONDecodeError, AttributeError):
        return None, meta


def sample(rows: list[dict], limit: int) -> list[dict]:
    """Deterministic, class-balanced like the full split: every k-th row of each class."""
    if limit >= len(rows):
        return rows
    out = []
    for cls in ("malicious", "benign"):
        group = [r for r in rows if r["label"]["classification"] == cls]
        k = round(limit * len(group) / len(rows))
        out += [group[int(i * len(group) / k)] for i in range(k)] if k else []
    return out


def score(rows: list[dict], preds: list[tuple[dict | None, dict]]) -> dict:
    tp = fp = fn = tn = invalid = target_ok = ev_ok = 0
    for r, (p, _) in zip(rows, preds):
        truth = r["label"]["classification"] == "malicious"
        if p is None:
            invalid += 1
            fn += truth
            continue
        pred = p["classification"] == "malicious"
        tp, fp, fn, tn = tp + (truth and pred), fp + (pred and not truth), fn + (truth and not pred), tn + (not truth and not pred)
        if truth and pred:
            target_ok += p.get("target_capability") in (r["label"]["target_capability"], "any")
            ev_ok += any(len(e) >= 8 and e.lower() in r["content"].lower() for e in p.get("evidence", []))
    lat = sorted(m["latency_s"] for _, m in preds)
    pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
    tok = [m["gen_tok_s"] for _, m in preds if m.get("gen_tok_s")]
    return {"n": len(rows), "json_valid": len(rows) - invalid, "malicious_caught": f"{tp}/{tp + fn}",
            "false_alarms": f"{fp}/{fp + tn}", "accuracy": round((tp + tn) / len(rows), 4),
            "target_correct": f"{target_ok}/{tp}", "evidence_located": f"{ev_ok}/{tp}",
            "latency_p50_s": round(statistics.median(lat), 2), "latency_p90_s": pct(0.9), "latency_max_s": lat[-1],
            "gen_tok_s_median": statistics.median(tok) if tok else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80, help="examples to run (the full split is 344)")
    ap.add_argument("--cpu", action="store_true", help="force CPU only, even if a GPU is present")
    a = ap.parse_args()
    schema_file = HERE / "schema.json"
    schema = json.loads(schema_file.read_text()) if schema_file.exists() else "json"
    rows = sample([json.loads(l) for l in open(HERE / "test.jsonl")], a.limit)

    hw = hardware()
    print(json.dumps(hw, indent=1), flush=True)
    ensure_model()
    print("loading model ...", flush=True)
    t0 = time.perf_counter()
    assess(rows[0], schema, a.cpu)                            # warm-up: loads the weights, not counted
    load_s = round(time.perf_counter() - t0, 1)

    preds, peak = [], 0.0
    for i, r in enumerate(rows, 1):
        preds.append(assess(r, schema, a.cpu))
        peak = max(peak, runner_rss_mb())
        p, m = preds[-1]
        ok = p is not None and p["classification"] == r["label"]["classification"]
        print(f"[{i:>3}/{len(rows)}] {m['latency_s']:>6.2f}s  {'ok ' if ok else 'MISS'}  "
              f"truth={r['label']['classification']:<9} pred={p['classification'] if p else 'invalid'}", flush=True)

    s = score(rows, preds)
    result = {"hardware": hw, "mode": "cpu-forced" if a.cpu else "auto", "placement": sh("ollama", "ps"),
              "model_load_s": load_s, "runner_peak_rss_mb": peak, "summary": s,
              "per_request": [{"id": r["id"], **m} for r, (_, m) in zip(rows, preds)],
              "at": time.strftime("%Y-%m-%d %H:%M")}
    out = HERE / f"laptop-{hw['host']}{'-cpu' if a.cpu else ''}.json"
    out.write_text(json.dumps(result, indent=1))
    print("\n" + json.dumps({k: v for k, v in result.items() if k not in ("per_request",)}, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
