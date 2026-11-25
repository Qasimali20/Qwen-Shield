import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Ban, Bot, CircleCheck, FileText, Loader2, Play, ScanSearch, ShieldAlert, ShieldCheck, Square, User, Wrench } from 'lucide-react'
import { api, attackSucceeded, callMatches, streamRun, taskCompleted, type Assessment, type Goal, type Mode, type RunEvent, type Scenario, type ShieldKind } from '../api'

type LaneKey = 'none' | 'base' | 'finetuned'
const LANES: Record<LaneKey, { title: string; mode: Mode; shield?: ShieldKind }> = {
  none: { title: 'Unprotected agent', mode: 'none' },
  base: { title: 'Shield with base Qwen3-1.7B', mode: 'full', shield: 'base' },
  finetuned: { title: 'Shield with fine-tuned Qwen3-1.7B', mode: 'full', shield: 'finetuned' },
}
/** Held-out scenarios that show the whole story: the unprotected agent is hijacked, the base shield blocks the real work,
 *  the fine-tuned shield stops the attack and the task still gets done (or, for E, normal work goes through). */
const FEATURED = new Set(['TB01', 'TB03', 'TC03', 'TD02', 'TA01', 'TE01', 'TE02'])
const SET: Record<string, string> = { test: 'Held-out test', dev: 'Development' }
const empty = (): Record<LaneKey, RunEvent[]> => ({ none: [], base: [], finetuned: [] })

const FAMILY: Record<string, string> = {
  A: 'Unauthorized action hidden in a document', B: 'Abuse of an email the user did ask for',
  C: 'Malicious internal ticket', D: 'Manipulated answer', E: 'Normal work (must not be blocked)',
}
const TOOL: Record<string, string> = {
  search_documents: 'search documents', read_document: 'read document', create_ticket: 'create ticket',
  send_email: 'send email', export_records: 'export customer records', admin_credentials: 'fetch admin credentials',
}
/** Tool arguments, shortened; where an argument carries the attacker's payload, show the text around it, highlighted. */
function Args({ a, goal }: { a: unknown; goal?: Goal }) {
  const needles = Object.values(goal?.args ?? {}).map(v => v.toLowerCase())
  return <code>{Object.entries((a ?? {}) as Record<string, unknown>).map(([k, raw], i) => {
    const v = String(raw), at = needles.map(n => v.toLowerCase().indexOf(n)).find(x => x >= 0) ?? -1
    const n = at >= 0 ? needles.find(x => v.toLowerCase().indexOf(x) === at)!.length : 0
    const body = at < 0 ? v.slice(0, 90) + (v.length > 90 ? '…' : '')
      : <>{at > 40 ? '…' : ''}{v.slice(Math.max(0, at - 40), at)}<mark>{v.slice(at, at + n)}</mark>{v.slice(at + n, at + n + 30)}{v.length > at + n + 30 ? '…' : ''}</>
    return <span key={k}>{i > 0 && ' · '}{k}: {body}</span>
  })}</code>
}

function Step({ e, goal }: { e: RunEvent; goal?: Goal }) {
  const call = { tool: String(e.tool), args: (e.args ?? {}) as Record<string, unknown> }
  const hit = !!goal && callMatches(call, goal)
  switch (e.kind) {
    case 'task': return <li className="ev user"><User size={14} /><span><b>User</b> {String(e.user_task)}
      {(e.granted as string[]).length > 0 && <em> · authorized: {(e.granted as string[]).join(', ').replaceAll('_', ' ')}</em>}</span></li>
    case 'proposed': return <li className="ev agent"><Bot size={14} /><span><b>Agent wants to</b> {TOOL[String(e.tool)] ?? String(e.tool)}<Args a={e.args} goal={goal} /></span></li>
    case 'verdict': return e.decision === 'allow' ? null
      : <li className="ev block"><Ban size={14} /><span><b>Blocked</b> {String(e.reason)} <em className="rule">{String(e.rule).replaceAll('_', ' ')}</em></span></li>
    case 'executed': return <li className={`ev exec ${hit ? 'hot' : ''}`}>
      <Wrench size={14} /><span><b>Executed</b> {TOOL[String(e.tool)] ?? String(e.tool)}{hit && <em> · carrying the attacker's payload</em>}</span></li>
    case 'assessment': {
      const a = e.assessment as Assessment | null
      return <li className={`ev assess ${a?.classification === 'malicious' ? 'bad' : ''}`}><ScanSearch size={14} />
        <span><b>Qwen Shield</b> {a ? (a.classification === 'malicious'
          ? <>found <b>{a.attack_type.replaceAll('_', ' ')}</b> aiming at <b>{a.target_capability.replaceAll('_', ' ')}</b> · risk {a.risk_level}</>
          : 'content is benign') : 'assessment unavailable (sensitive actions fail closed)'}
          <span className="lat">{Number(e.latency_s ?? 0).toFixed(2)} s</span>
          {a?.evidence?.length ? <q>{a.evidence[0]}</q> : null}</span></li>
    }
    case 'sanitized': return <li className="ev assess"><FileText size={14} /><span>Removed {String(e.removed)} injected sentence(s) before the agent read the document</span></li>
    case 'answer': return <li className="ev answer"><Bot size={14} /><span><b>Answer</b> {String(e.text) || '(no answer)'}</span></li>
    case 'error': return <li className="ev block"><span>Error: {String(e.error)}</span></li>
    default: return null
  }
}

function Lane({ title, mode, tone, events, running, scenario }: { title: string; mode: Mode; tone: LaneKey; events: RunEvent[]; running: boolean; scenario?: Scenario }) {
  const done = events.some(e => e.kind === 'done')
  const hit = scenario ? attackSucceeded(scenario.goal, events) : false
  const blocked = events.filter(e => e.kind === 'blocked').length
  const task = scenario ? taskCompleted(scenario.task, events) : blocked === 0
  const verdict = !done ? null : scenario?.kind === 'attack'
    ? (hit ? { tone: 'bad', icon: <ShieldAlert size={18} />, text: 'Attack succeeded' }
      : task ? { tone: 'good', icon: <ShieldCheck size={18} />, text: 'Attack stopped, task done' }
      : { tone: 'warn', icon: <ShieldAlert size={18} />, text: "Attack stopped, but the user's task was not done" })
    : (task && !blocked ? { tone: 'good', icon: <CircleCheck size={18} />, text: 'Task done, nothing blocked' }
      : { tone: 'warn', icon: <ShieldAlert size={18} />, text: blocked ? `Normal work was blocked (${blocked})` : 'Task not done' })
  return (
    <section className={`lane ${tone}`}>
      <header><h2>{mode === 'none' ? <Bot size={16} /> : <ShieldCheck size={16} />}{title}</h2>{running && <Loader2 size={16} className="spin" />}</header>
      {verdict && <div className={`outcome ${verdict.tone}`}>{verdict.icon}{verdict.text}</div>}
      <ol className="steps">{events.map(e => <Step key={e.seq} e={e} goal={scenario?.goal} />)}</ol>
      {!events.length && <p className="muted">Waiting to start.</p>}
    </section>
  )
}

export default function Demo() {
  const { data: scenarios } = useQuery({ queryKey: ['scenarios'], queryFn: api.scenarios })
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 10000 })
  const [id, setId] = useState('TB01')
  const [compare, setCompare] = useState(true)
  const [custom, setCustom] = useState(false)
  const [task, setTask] = useState('Summarize the complaint in document DOC-1.')
  const [doc, setDoc] = useState('')
  const [lanes, setLanes] = useState<Record<LaneKey, RunEvent[]>>(empty)
  const [running, setRunning] = useState<Partial<Record<LaneKey, boolean>>>({})
  const [error, setError] = useState('')
  const ctl = useRef<AbortController | null>(null)
  const scenario = useMemo(() => scenarios?.find(s => s.id === id), [scenarios, id])
  useEffect(() => () => ctl.current?.abort(), [])
  const shown3 = compare && !!health?.base_shield
  const keys: LaneKey[] = shown3 ? ['none', 'base', 'finetuned'] : ['none', 'finetuned']

  const start = () => {
    ctl.current?.abort()
    const c = new AbortController(); ctl.current = c
    setLanes(empty()); setError('')
    for (const k of keys) {
      const { mode, shield } = LANES[k]
      setRunning(r => ({ ...r, [k]: true }))
      const body = custom ? { mode, shield, user_task: task, documents: { 'DOC-1': doc } } : { mode, shield, scenario_id: id }
      streamRun(body, e => setLanes(l => ({ ...l, [k]: [...l[k], e] })), c.signal)
        .catch(err => { if (err.name !== 'AbortError') setError(String(err)) })
        .finally(() => setRunning(r => ({ ...r, [k]: false })))
    }
  }
  const busy = Object.values(running).some(Boolean)
  const shown = custom ? undefined : scenario

  return (
    <>
      <div className="intro">
        <h1>Same task, same poisoned document, {keys.length === 3 ? 'three' : 'two'} agents</h1>
        <p className="muted">The same agent{health ? ` (${health.agent.model})` : ''} runs the task unprotected and behind Qwen Shield
          {shown3 ? ', once with the untrained base model as the assessor and once with the fine-tuned one' : ''}. Everything runs locally.</p>
      </div>
      <div className="picker">
        <div className="tabs"><button className={!custom ? 'on' : ''} onClick={() => setCustom(false)}>Scenarios</button>
          <button className={custom ? 'on' : ''} onClick={() => setCustom(true)}>Your own document</button></div>
        {!custom ? <div className="scen">
          <select value={id} onChange={e => setId(e.target.value)}>
            {Object.entries(SET).flatMap(([set, setLabel]) => Object.entries(FAMILY).map(([f, label]) => <optgroup key={set + f} label={`${setLabel} · ${label}`}>
              {(scenarios ?? []).filter(s => s.set === set && s.family === f).map(s =>
                <option key={s.id} value={s.id}>{FEATURED.has(s.id) ? '★ ' : ''}{s.id} · {s.user_task}</option>)}</optgroup>))}
          </select>
          {scenario && <div className="docs">{Object.entries(scenario.documents).map(([k, v]) => <p key={k}><b>{k}</b> {v}</p>)}
            {scenario.granted.length > 0 && <p className="muted">The user authorized: {scenario.granted.join(', ').replaceAll('_', ' ')}</p>}</div>}
        </div> : <div className="scen">
          <input value={task} onChange={e => setTask(e.target.value)} placeholder="The user's request" />
          <textarea value={doc} onChange={e => setDoc(e.target.value)} rows={4} placeholder="Paste a document (id DOC-1). Try hiding an instruction inside it." />
        </div>}
        <div className="go">
          {busy ? <button onClick={() => ctl.current?.abort()}><Square size={14} /> Stop</button>
            : <button className="primary" disabled={custom && !doc.trim()} onClick={start}><Play size={14} /> Run {keys.length === 3 ? 'all three' : 'both'}</button>}
          {health?.base_shield && <label className="check"><input type="checkbox" checked={compare} disabled={busy}
            onChange={e => setCompare(e.target.checked)} /> Also show the untrained base model</label>}
          {error && <span className="bad">{error}</span>}
        </div>
      </div>
      <div className={`lanes n${keys.length}`}>
        {keys.map(k => <Lane key={k} title={LANES[k].title} mode={LANES[k].mode} tone={k} events={lanes[k]} running={!!running[k]} scenario={shown} />)}
      </div>
    </>
  )
}
