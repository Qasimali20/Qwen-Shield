# Qwen Shield

**Local, action-aware security for tool-using AI agents, built on a fine-tuned Qwen3-1.7B.**

Model on Hugging Face: [qasim-tmc/qwen-shield-1.7b](https://huggingface.co/qasim-tmc/qwen-shield-1.7b) (Q4_K_M and F16 GGUF, LoRA adapter, Modelfile).

An agent that reads email, tickets and web pages can be steered by text hidden in them ("assistant: forward this
thread to …"). Qwen Shield sits between the agent and its tools. Every piece of content the agent reads is assessed by a
small fine-tuned Qwen model that runs on a laptop; every action the agent takes is checked by a deterministic policy that
knows where each argument came from. Attacks are removed from what the agent reads and blocked at the tool call, while the
user's real task still gets done.

| On 40 held-out scenarios (28 attacks, 12 benign) | Attacks that succeeded | Tasks completed |
|---|---|---|
| No defence | 16 / 28 | 40 / 40 |
| Deterministic rules only | 14 / 28 | 40 / 40 |
| Shield with **base** Qwen3-1.7B | 0 / 28 | 11 / 40 |
| Shield with **fine-tuned** Qwen3-1.7B (this project) | **0 / 28** | **39 / 40** |

The base model blocks attacks only because it flags almost everything (12 of 12 benign scenarios refused). Fine-tuning
is what makes the shield usable: same protection, and the work goes through. That table uses a 35B agent; with a
**laptop-sized 4B agent and the 1.06 GB shield**, 13 / 28 attacks succeed undefended and 3 / 28 with the shield, and the
whole system runs on an ordinary laptop CPU at about a minute per task ([details](#the-whole-system-on-a-laptop)).

## How it works

```
 user task ─► agent (any OpenAI-compatible model)
                │ reads content                      │ calls a tool
                ▼                                    ▼
      ┌──────────────────────┐            ┌──────────────────────────┐
      │ Assessor (Qwen Shield│  verdict + │ Policy (deterministic)   │
      │ 1.7B, local)         ├──evidence─►│ provenance · grants ·    │──► allow / review / block
      │ benign / malicious,  │            │ taint · content-aware    │
      │ attack type, target, │            └──────────────────────────┘
      │ verbatim evidence    │
      └─────────┬────────────┘
                └─► sanitizer removes the evidence sentences before the agent sees the content
```

1. **Provenance ledger.** Every value the agent sees is recorded with where it came from (the user, a tool, a document).
2. **Assessment.** Each tool result is assessed by the fine-tuned model, which returns one JSON object: classification,
   attack type, the capability the attack targets (`external_comm`, `write_internal`, `export_data`, `credentials`,
   `answer_manipulation` …), risk, and the **verbatim evidence** sentences.
3. **Sanitization.** The evidence sentences are cut from the content before the agent reads it, so the agent never sees
   the instruction.
4. **Policy, in fixed order:** unknown tool → capability never grantable (credentials) → capability not granted for this
   task → argument tainted by untrusted content → fail closed if content could not be assessed → assessment. Inside a
   channel the user did grant (e.g. "reply to this customer"), an action is blocked only if its arguments carry the attack
   (an address, URL, domain, account or phone number, or a shared phrase from the evidence); otherwise it goes through as
   `sanitized_clean`.

The model never makes the final decision alone: it supplies evidence, and the deterministic policy decides. If the model
returns invalid output, the policy fails closed.

## The fine-tune

| | |
|---|---|
| Base model | Qwen3-1.7B (Apache-2.0), thinking disabled |
| Method | LoRA, r=16, α=32, dropout 0.05, all attention and MLP projections; 3 epochs, lr 2e-4 cosine, 5 % warm-up |
| Loss | answer tokens only (the vocabulary projection is applied only to answer positions, which cut peak memory from ~35 GB to 9 GB) |
| Data | 1,561 train / 124 val / 344 test, about half malicious, half hard benign |
| Training | 225 steps, 44 min on one NVIDIA GB10 (DGX Spark), including 21 min of thermal pauses; best val loss 0.073 (from ~1.2) |
| Output | 70 MB LoRA adapter → merged bf16 model → GGUF Q4_K_M, **1.06 GB** |

### Data: permitted, synthetic, labelled by construction

All data was generated for this project this week. No private or scraped data was used.

- **Specs decide the labels, not a model.** Code samples a specification for every example: business domain (16),
  user task (7), attack goal (10), injection style (14) or benign hard-negative kind (9), document IDs, recipients and
  destinations. The label (classification, attack type, target capability, risk) follows from the spec.
- **A teacher writes only the text.** Qwen3.6-35B-A3B (local) writes the document to the spec. Evidence must be a verbatim
  substring of the document; drafts that fail structural checks are rejected and rewritten.
- **A blind auditor** (the same teacher, without the spec) must agree on benign versus malicious, or the example is dropped.
- **Hard negatives** are half the data: urgent but legitimate requests, documents that discuss prompt injection, "please
  ignore the earlier invoice", instructions that match what the user actually asked for.
- **No secrets, even fake ones.** Seven teacher-written attack documents contained credential-shaped strings (public
  documentation sample keys such as AWS's `AKIA…EXAMPLE`). In the published files they are replaced with
  `[redacted-example-key]`, consistently in content and labels; `generate build` applies the same step. The model was
  trained before this replacement, on the unredacted strings.
- **Filters:** deduplication; malicious examples whose "attack" only repeats the user's own request (38 dropped); weak
  evidence; and any 8-gram overlap with the end-to-end benchmark scenarios (47 dropped).

### What is held out

- **Two injection styles never seen in training:** instructions in another language (Urdu, Roman Urdu, Arabic, Chinese, Spanish, French) and instructions
  conditional on the reader being an AI.
- **Two domains never seen in training:** legal contract clauses and healthcare clinic notices.
- **The end-to-end scenarios** (`backend/benchmarks/scenarios/test.json`) were written separately, by hand, and frozen
  before any fine-tuning result was seen: Urdu injections, HTML comments, JSON tool output, instructions split across a
  document, and benign documents that look suspicious.

## Results

### Assessment, 344 held-out examples

| Model | Attacks caught | False alarms on benign | Right target | Evidence found in document | Valid JSON | Median latency |
|---|---|---|---|---|---|---|
| Base Qwen3-1.7B | 99.6 % | **96.7 %** | 11.7 % | 0 % | 100 % | 1.4 s |
| **Qwen Shield** (bf16, vLLM) | 99.1 % | **0.8 %** | 89.6 % | 98.2 % | 99.7 % | 2.4 s |
| **Qwen Shield Q4_K_M** (Ollama) | 97.3 % | **0.0 %** | 83.5 % | 96.3 % | 99.7 % | 2.9 s |

On the **unseen injection styles** the fine-tuned model catches 98.0 % (Q4_K_M: 95.1 %); on the **unseen domains**
100 % with 0.8 % false alarms (Q4_K_M: 98.5 %, 0 %).

### End to end, with a real agent

The agent is Qwen3.6-35B-A3B (FP8, local); the shield is Qwen3-1.7B. Attack families: A, an action the user never
authorized; B, abuse inside an email the user did authorize; C, a malicious internal ticket; D, a manipulated answer;
E, benign hard negatives.

Held-out `test.json` (40 scenarios):

| Defence | A | B | C | D | Attacks succeeded | Tasks done | Benign done | False positives |
|---|---|---|---|---|---|---|---|---|
| None | 1/7 | 5/7 | 7/7 | 3/7 | 16 / 28 | 40 / 40 | 12 / 12 | 0 / 12 |
| Rules only | 0/7 | 3/7 | 7/7 | 4/7 | 14 / 28 | 40 / 40 | 12 / 12 | 0 / 12 |
| Shield, base 1.7B | 0 | 0 | 0 | 0 | 0 / 28 | 11 / 40 | 3 / 12 | 12 / 12 |
| **Shield, fine-tuned** | 0 | 0 | 0 | 0 | **0 / 28** | **39 / 40** | **12 / 12** | 1 / 12 |

Development `dev.json` (30 scenarios): no defence 13 / 21 attacks succeed; fine-tuned shield 0 / 21 with 29 / 30 tasks
done; base shield 5 / 30 tasks done with 8 / 9 false positives.

Rules alone cannot stop families B to D: the action (send the email the user asked for, file the ticket, answer the
question) is permitted; only the content is wrong. That is what the model is for.

**With a laptop-sized agent.** The same 40 held-out scenarios with Qwen3-4B-Instruct-2507 (Q4_K_M, 2.5 GB) as the agent
and the quantized shield, both through Ollama (run on the Spark for speed, with the exact files used on the laptop):

| Defence | A | B | C | D | Attacks succeeded | Tasks done | Benign done | False positives |
|---|---|---|---|---|---|---|---|---|
| None | 1/7 | 4/7 | 3/7 | 5/7 | 13 / 28 | 33 / 40 | 12 / 12 | 0 / 12 |
| Rules only | 0/7 | 5/7 | 3/7 | 4/7 | 12 / 28 | 34 / 40 | 12 / 12 | 0 / 12 |
| **Shield, fine-tuned Q4_K_M (1.06 GB)** | 0 | 0 | 1/7 | 2/7 | **3 / 28** | **36 / 40** | **12 / 12** | **0 / 12** |
| Shield, fine-tuned bf16 | 0 | 0 | 0 | 1/7 | 1 / 28 | 36 / 40 | 11 / 12 | 1 / 12 |

Two of the three Q4 misses (an instruction planted in a ticket title, a fake helpline number in Urdu) are caught by the
bf16 model, so they are the measurable cost of quantization. The third, TD06, counts as a success only by the scorer's
substring rule: the agent's answer reports the real findings and says the planted "no issues" claim is false.

### On a normal laptop

The Q4_K_M model (1.06 GB) on an ordinary office laptop, **CPU only**, through Ollama, on 80 held-out examples
(52 attacks, 28 benign, a fixed class-balanced sample of the test split), one request at a time as an agent sends them.
The laptop was in normal use during the run (browser and other services open).

| Machine | Attacks caught | False alarms | Right target | Evidence found | Median latency | p90 | Output speed | Ollama RAM |
|---|---|---|---|---|---|---|---|---|
| **HP EliteBook 840 G8**: Intel Core i5-1135G7 (4 cores / 8 threads), 16 GB RAM, Intel Iris Xe (unused), Ubuntu 26.04, Ollama 0.32.1 | 51 / 52 | **0 / 28** | 41 / 51 | 47 / 51 | 19.8 s | 34.1 s | 12.5 tok/s | 5.6 GB peak |
| DGX Spark CPU (Arm Cortex-X925), same script, for reference | 51 / 52 | 0 / 28 | 41 / 51 | 45 / 51 | 5.3 s | 7.0 s | 46 tok/s | – |

The quantized model gives the same verdicts on a laptop CPU as on the workstation: the same single miss, no false alarms.
Latency is dominated by writing the answer (about 215 tokens of JSON including the evidence sentences), not by reading the
document, so a shorter evidence format is the clearest way to make it faster. Results:
`backend/benchmarks/results/laptop-i5-1135g7-cpu.json`.

### The whole system on a laptop

Agent (Qwen3-4B-Instruct-2507 Q4_K_M) and shield (Qwen Shield Q4_K_M) both on the HP EliteBook's CPU through Ollama, 10
held-out scenarios (6 attacks, 4 benign) one at a time, everything else as above:

| Attacks succeeded | Tasks done | False positives | Median time per task | Slowest task | Shield check (median) | RAM for both models |
|---|---|---|---|---|---|---|
| **0 / 6** | **10 / 10** | **0 / 4** | 67 s | 84 s | 17.5 s | 5.1 GB of 16 GB (Ollama: 3.2 GB agent + 1.6 GB shield) |

In one of the six attacks (TC01) the agent never opened the poisoned document, so that attack was not exercised. Results:
`backend/benchmarks/results/laptop-e2e-agent4b-shield-q4.json`.

## Honesty notes

- **One policy change came after looking at held-out results.** The first end-to-end run blocked every action in a
  permitted channel once the content was flagged, which stopped attacks but also the user's clean emails and tickets. The
  content-aware rule (`sanitized_clean`: allow the action if its arguments do not carry the attack) was added after
  seeing that. The model and data were not changed. Both runs are in `backend/benchmarks/results/`
  (`*-shield-v1` before, `*-shield-v1b` after).
- **Two fail-closed bugs were found by the 4B-agent run on held-out data, and fixed.** When an assessment failed (a
  timeout under load), a ticket built from the unassessed document was allowed, because fail-closed covered only
  external, export and credential actions; and one successful assessment in a task masked a later failed one. Both now
  fail closed for any write, and the assessor allows 120 s with one retry (a laptop CPU check can take 45 s). The
  run before the fix is kept (`test-agent4b-shield-q4.json`: 4 / 28, two of them through this gap); the 35B results are
  unchanged because every assessment in those runs was valid.
- **The headline agent is a 35B model.** Qwen3-1.7B as the agent rarely read the documents, so attacks never reached
  it. Qwen3-4B-Instruct-2507 does read them, and is reported above as the laptop-sized setup. The shield is always the
  1.7B model.
- **Teacher and auditor are the same model family** as the agent in the end-to-end test. The end-to-end scenarios were
  written by hand, not by the teacher, and decontaminated against the training data.
- Scenario counts are small (40 and 30); treat single-scenario differences as noise.

## Run it

**On a laptop (Ollama).** Download the Q4_K_M model and its Modelfile from Hugging Face, then:

```bash
pip install -U huggingface_hub    # provides the `hf` command
hf download qasim-tmc/qwen-shield-1.7b qwen-shield-1.7b-Q4_K_M.gguf Modelfile --local-dir qwen-shield
(cd qwen-shield && ollama create qwen-shield:1.7b -f Modelfile)
python3 backend/benchmarks/laptop_bench.py --cpu      # reproduce the laptop numbers (standard library only)
```

The Modelfile pins the training-time prompt, with thinking already off. **Do not send Ollama's `think: false`**: it
switches Ollama to a different prompt rendering, and on the same 80 held-out examples false alarms rose from 0 / 28 to
9 / 28. Use the OpenAI-compatible endpoint (`/v1/chat/completions`) or `/api/chat` without that field.

**The full stack.**

```bash
cd backend && uv sync --extra dev
SHIELD_URL=http://127.0.0.1:11434/v1 SHIELD_MODEL=qwen-shield:1.7b SHIELD_BASE_MODEL=qwen3:1.7b \
AGENT_URL=<any OpenAI-compatible server> AGENT_MODEL=<model> \
  uv run uvicorn qwen_shield.api:app --port 8100
cd ../frontend && npm ci && npm run build      # the API serves the console at http://127.0.0.1:8100
```

To serve the console beyond localhost, set `QS_PASSWORD` (and optionally `QS_USER`, default `admin`): the console then
opens on a sign-in page and the API needs the session it sets. The console has three pages. **Live demo** runs one scenario (or your own document) through an unprotected agent, the shield
with the base model (`SHIELD_BASE_MODEL`, optional) and the shield with the fine-tuned model, side by side.
**Results** shows every before/after number above, read from `backend/benchmarks/results/`. **Audit** lists every
assessment, decision and executed action.

**Reproduce the model.**

```bash
python -m training.generate gen && python -m training.generate build     # needs a teacher in QS_TEACHERS
python -m training.train --base <Qwen3-1.7B> --out ../artifacts/adapters/shield-v1 --tokens-per-batch 8192
python -m training.export --base <Qwen3-1.7B> --adapter ../artifacts/adapters/shield-v1 --out ../artifacts/merged/qwen-shield-1.7b
python -m training.to_gguf --model ../artifacts/merged/qwen-shield-1.7b --quant Q4_K_M --ollama qwen-shield:1.7b
python -m benchmarks.assess_eval --models qwen3-1.7b,qwen-shield --split test --label shield-v1
python -m benchmarks.run --scenarios benchmarks/scenarios/test.json --modes none,rules,full --label test-shield-v1
pytest tests
```

`training/to_gguf.py` is a small native converter: Ollama could not import Qwen3 safetensors directly, so the merged model
is written to GGUF with llama.cpp's Qwen3 layout and quantized with `llama-quantize`.

## Layout

```
backend/qwen_shield/   provenance ledger, assessor client, policy, sanitizer, guarded agent, FastAPI service
backend/training/      catalog (specs), generate (teacher + auditor + filters), train (LoRA), export, to_gguf
backend/benchmarks/    assess_eval (model-level), run (end to end), laptop_bench, scenarios, results
backend/tests/         policy and agent tests
frontend/              React console: live side-by-side runs, results, audit log
```

## Licence

Apache-2.0. Base model: Qwen3-1.7B (Apache-2.0).
