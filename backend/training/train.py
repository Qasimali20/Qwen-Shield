"""LoRA fine-tune of Qwen3 into the Qwen Shield assessor, with plain transformers + PEFT.

    python -m training.train --base /path/to/Qwen3-1.7B --out ../artifacts/adapters/shield-v1

Each example is the serving prompt exactly (assessment.SYSTEM + assessment.render, chat template with thinking off)
followed by the canonical JSON label; loss is computed on the answer tokens only, and the vocabulary projection is
applied to those positions alone (the prompt is most of each sequence, so this cuts activation memory several-fold).
The run is a good citizen on a shared machine: a hard GPU memory cap, and a thermal guard that pauses training when
the GPU gets hot. The adapter with the lowest validation loss is kept, with the run config and the loss curve."""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from qwen_shield.assessment import SYSTEM, render, to_json

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, row: dict, max_len: int) -> dict | None:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": render(row["user_task"], row["source"], row["content"])}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(to_json(row["label"]) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(a_ids) > max_len:
        return None
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


def batches(rows: list[dict], tokens_per_batch: int, shuffle: bool, rng: random.Random):
    """Length-bucketed batches with a token budget: little padding, steady memory."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
    out, cur, longest = [], [], 0
    for i in order:
        n = len(rows[i]["input_ids"])
        if cur and max(longest, n) * (len(cur) + 1) > tokens_per_batch:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


def collate(rows: list[dict], idx: list[int], pad: int, device) -> dict:
    n = max(len(rows[i]["input_ids"]) for i in idx)
    ids = torch.full((len(idx), n), pad, dtype=torch.long)
    lab = torch.full((len(idx), n), -100, dtype=torch.long)
    att = torch.zeros((len(idx), n), dtype=torch.long)
    for j, i in enumerate(idx):
        k = len(rows[i]["input_ids"])
        ids[j, :k] = torch.tensor(rows[i]["input_ids"])
        lab[j, :k] = torch.tensor(rows[i]["labels"])
        att[j, :k] = 1
    return {"input_ids": ids.to(device), "labels": lab.to(device), "attention_mask": att.to(device)}


def answer_loss(model, b: dict) -> tuple[torch.Tensor, int]:
    """Mean next-token loss over answer positions, projecting only those hidden states onto the vocabulary."""
    inner = model.get_base_model()
    h = inner.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state[:, :-1]
    tgt = b["labels"][:, 1:]
    keep = tgt != -100
    logits = inner.lm_head(h[keep]).float()
    return F.cross_entropy(logits, tgt[keep]), int(keep.sum())


@torch.no_grad()
def evaluate(model, rows, bs, pad, device) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for idx in bs:
        loss, n = answer_loss(model, collate(rows, idx, pad, device))
        tot += float(loss) * n
        cnt += n
    model.train()
    return tot / max(cnt, 1)


def gpu_temp() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0]) if out.strip() else 0


def cool_down(hot: int, resume: int) -> float:
    """Block while the GPU is at or above `hot`, until it is back to `resume`. Returns seconds paused."""
    if gpu_temp() < hot:
        return 0.0
    t0 = time.time()
    while gpu_temp() > resume:
        time.sleep(5)
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "assess"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--mem-fraction", type=float, default=0.2, help="hard cap on this process's GPU memory")
    ap.add_argument("--hot", type=int, default=68, help="pause training at this GPU temperature (C)")
    ap.add_argument("--resume", type=int, default=60, help="resume once the GPU has cooled to this (C)")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)

    torch.manual_seed(a.seed)
    rng, device, out = random.Random(a.seed), "cuda", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    data = {}
    for split in ("train", "val"):
        raw = [json.loads(l) for l in open(Path(a.data) / f"{split}.jsonl")]
        data[split] = [e for e in (encode(tok, r, a.max_len) for r in raw) if e]
        print(f"{split}: {len(data[split])} examples ({len(raw) - len(data[split])} over {a.max_len} tokens)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    val_b = batches(data["val"], a.tokens_per_batch, False, rng)
    steps_per_epoch = len(batches(data["train"], a.tokens_per_batch, True, rng))
    total = math.ceil(steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0,
                            betas=(0.9, 0.99))
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)
    every = max(1, steps_per_epoch // a.evals_per_epoch)
    log, best, step, t0, paused = [], float("inf"), 0, time.time(), 0.0
    model.train()
    while step < total:
        for idx in batches(data["train"], a.tokens_per_batch, True, rng):
            if step >= total:
                break
            paused += cool_down(a.hot, a.resume)
            loss, _ = answer_loss(model, collate(data["train"], idx, tok.pad_token_id, device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f"step {step}/{total}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{time.time() - t0:.0f}s  paused {paused:.0f}s  {gpu_temp()}C  "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            if step % every == 0 or step == total:
                vl = evaluate(model, data["val"], val_b, tok.pad_token_id, device)
                log.append({"step": step, "epoch": round(step / steps_per_epoch, 2), "train_loss": round(loss.item(), 4),
                            "val_loss": round(vl, 4), "elapsed_s": round(time.time() - t0)})
                print(f"  eval  {log[-1]}", flush=True)
                if vl < best:
                    best = vl
                    model.save_pretrained(out)
    (out / "train_run.json").write_text(json.dumps({"args": vars(a), "train_examples": len(data["train"]),
                                                    "val_examples": len(data["val"]), "best_val_loss": round(best, 4),
                                                    "steps": total, "log": log, "thermal_pause_s": round(paused),
                                                    "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                                                    "wall_s": round(time.time() - t0),
                                                    "gpu": torch.cuda.get_device_name()}, indent=1))
    print(f"done: best val loss {best:.4f}, adapter in {out}", flush=True)


if __name__ == "__main__":
    main()
