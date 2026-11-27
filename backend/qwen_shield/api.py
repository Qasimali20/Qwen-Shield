"""HTTP API for the demo console.

  GET  /api/health                  agent and shield model reachability
  GET  /api/scenarios               the benchmark scenarios (dev and held-out test)
  POST /api/runs                    run a task under a protection mode and shield model; streams events (SSE)
  GET  /api/report                  the before/after report: end to end, assessment, laptop, training
  GET  /api/benchmarks              saved end-to-end benchmark results (summaries)
  GET  /api/benchmarks/{label}      one result set, per scenario
  GET  /api/audit                   recent security decisions

Models are local OpenAI-compatible servers configured by environment: AGENT_URL / AGENT_MODEL for the agent,
SHIELD_URL / SHIELD_MODEL for the fine-tuned assessor, and optionally SHIELD_BASE_MODEL (same server) for the untuned
base model, so the console can show both side by side.

Set QS_PASSWORD to require a login (HTTP Basic, user QS_USER, default "qwen") for every page and API call; do this
before serving the console beyond localhost. Unset, the console is open, as for a local demo."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import secrets
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qwen_shield.agent import LLM, run_task
from qwen_shield.assessment import Assessor
from qwen_shield.report import report as build_report
from qwen_shield.tools import CAPABILITIES

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_SETS = {"test": ROOT / "benchmarks" / "scenarios" / "test.json", "dev": ROOT / "benchmarks" / "scenarios" / "dev.json"}
RESULTS = ROOT / "benchmarks" / "results"
AUDIT = Path(os.environ.get("QS_AUDIT", ROOT / "runs" / "audit.jsonl"))
DIST = ROOT.parent / "frontend" / "dist"
AUDITED = {"assessment", "verdict", "blocked", "executed", "sanitized"}
CFG = {"agent_url": os.environ.get("AGENT_URL", "http://127.0.0.1:11434/v1"), "agent_model": os.environ.get("AGENT_MODEL", "qwen3:1.7b"),
       "shield_url": os.environ.get("SHIELD_URL", "http://127.0.0.1:11434/v1"), "shield_model": os.environ.get("SHIELD_MODEL", "qwen3:1.7b"),
       "shield_base_model": os.environ.get("SHIELD_BASE_MODEL", "")}
Shield = Literal["finetuned", "base"]


class State:
    agent: LLM
    shields: dict[str, Assessor]
    recent: deque[dict] = deque(maxlen=500)
    runs = asyncio.Semaphore(int(os.environ.get("QS_MAX_RUNS", "4")))   # bound concurrent model load


@asynccontextmanager
async def lifespan(app: FastAPI):
    State.agent = LLM(CFG["agent_url"], CFG["agent_model"])
    State.shields = {"finetuned": Assessor(CFG["shield_url"], CFG["shield_model"])}
    if CFG["shield_base_model"]:
        State.shields["base"] = Assessor(CFG["shield_url"], CFG["shield_base_model"])
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    yield
    await State.agent.aclose()
    for a in State.shields.values():
        await a.aclose()


app = FastAPI(title="Qwen Shield", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")


def _authorized(header: str, user: str, password: str) -> bool:
    """HTTP Basic credentials match, compared in constant time."""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        u, _, pw = base64.b64decode(value, validate=True).decode().partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    return secrets.compare_digest(u.encode(), user.encode()) & secrets.compare_digest(pw.encode(), password.encode())


@app.middleware("http")
async def require_login(request: Request, call_next):
    password = os.environ.get("QS_PASSWORD", "")
    if password and not _authorized(request.headers.get("authorization", ""), os.environ.get("QS_USER", "qwen"), password):
        return PlainTextResponse("Login required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Qwen Shield"'})
    return await call_next(request)


_SCENARIO_CACHE: dict[str, tuple[float, list[dict]]] = {}


def _scenarios() -> list[dict]:
    out = []
    for name, f in SCENARIO_SETS.items():
        mtime = f.stat().st_mtime
        if _SCENARIO_CACHE.get(name, (None,))[0] != mtime:
            _SCENARIO_CACHE[name] = (mtime, [{**s, "set": name} for s in json.loads(f.read_text())])
        out += _SCENARIO_CACHE[name][1]
    return out


@app.get("/api/health")
async def health() -> dict:
    async def up(url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3, trust_env=False) as c:
                return (await c.get(f"{url.rstrip('/')}/models")).status_code == 200
        except httpx.HTTPError:
            return False
    agent_ok, shield_ok = await asyncio.gather(up(CFG["agent_url"]), up(CFG["shield_url"]))
    return {"agent": {"ok": agent_ok, "model": CFG["agent_model"]}, "shield": {"ok": shield_ok, "model": CFG["shield_model"]},
            "base_shield": {"ok": shield_ok, "model": CFG["shield_base_model"]} if CFG["shield_base_model"] else None}


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    return [{k: s[k] for k in ("id", "set", "family", "kind", "user_task", "granted", "documents", "goal", "task")} for s in _scenarios()]


class RunRequest(BaseModel):
    mode: Literal["none", "rules", "model", "full"] = "full"
    shield: Shield = "finetuned"
    scenario_id: str | None = Field(default=None, max_length=20)
    user_task: str | None = Field(default=None, max_length=2000)
    documents: dict[str, str] = Field(default_factory=dict, max_length=10)
    granted: list[Literal[CAPABILITIES]] = Field(default_factory=list)          # type: ignore[valid-type]


@app.post("/api/runs")
async def runs(req: RunRequest) -> StreamingResponse:
    if req.scenario_id:
        s = next((x for x in _scenarios() if x["id"] == req.scenario_id), None)
        if s is None:
            raise HTTPException(404, "unknown scenario")
        task, docs, granted = s["user_task"], s["documents"], s["granted"]
    elif req.user_task:
        task, docs, granted = req.user_task, req.documents, req.granted
    else:
        raise HTTPException(422, "give a scenario_id or a user_task")
    assessor = State.shields.get(req.shield)
    if assessor is None:
        raise HTTPException(422, f"shield '{req.shield}' is not configured (set SHIELD_BASE_MODEL)")

    async def stream():
        async with State.runs:
            with AUDIT.open("a") as audit:
                async for e in run_task(State.agent, assessor, user_task=task, documents=docs,
                                        granted=frozenset(granted), mode=req.mode):
                    e = {**e, "shield": req.shield} if e["kind"] == "task" else e
                    if e["kind"] in AUDITED:
                        State.recent.append(e)
                        audit.write(json.dumps(e) + "\n")
                    yield f"data: {json.dumps(e)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/report")
def report() -> dict:
    return build_report(RESULTS)


@app.get("/api/benchmarks")
def benchmarks() -> list[dict]:
    out = []
    for f in sorted(RESULTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        d = json.loads(f.read_text())
        if "agent_model" not in d.get("meta", {}):           # assessment, laptop and training files have other shapes
            continue
        out.append({"label": f.stem, "meta": d["meta"], "summary": {m: v["summary"] for m, v in d.items() if not m.startswith("_") and m != "meta"}})
    return out


@app.get("/api/benchmarks/{label}")
def benchmark(label: str) -> dict:
    f = RESULTS / f"{label}.json"
    if not f.is_file() or f.parent != RESULTS:
        raise HTTPException(404, "unknown benchmark")
    d = json.loads(f.read_text())
    d.pop("_events", None)
    return d


@app.get("/api/audit")
def audit(limit: int = 100) -> list[dict]:
    return list(State.recent)[-min(limit, 500):][::-1]


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        return FileResponse(DIST / "index.html")
