import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Ban, Bot, CircleCheck, FileText, Loader2, Play, ScanSearch, ShieldAlert, ShieldCheck, Square, User, Wrench } from 'lucide-react'
import { api, attackSucceeded, streamRun, type Assessment, type Mode, type RunEvent, type Scenario } from '../api'

const FAMILY: Record<string, string> = {
  A: 'Unauthorized action hidden in a document', B: 'Abuse of an email the user did ask for',
  C: 'Malicious internal ticket', D: 'Manipulated answer', E: 'Normal work (must not be blocked)',
}
const TOOL: Record<string, string> = {
  search_documents: 'search documents', read_document: 'read document', create_ticket: 'create ticket',
  send_email: 'send email', export_records: 'export customer records', admin_credentials: 'fetch admin credentials',
}
const args = (a: unknown) => Object.entries((a ?? {}) as Record<string, unknown>).map(([k, v]) => `${k}: ${String(v).slice(0, 90)}`).join(' · ')

function Step({ e }: { e: RunEvent }) {
  switch (e.kind) {
    case 'task': return <li className="ev user"><User size={14} /><span><b>User</b> {String(e.user_task)}
      {(e.granted as string[]).length > 0 && <em> · authorized: {(e.granted as string[]).join(', ').replaceAll('_', ' ')}</em>}</span></li>
    case 'proposed': return <li className="ev agent"><Bot size={14} /><span><b>Agent wants to</b> {TOOL[String(e.tool)] ?? String(e.tool)}<code>{args(e.args)}</code></span></li>
    case 'verdict': return e.decision === 'allow' ? null
      : <li className="ev block"><Ban size={14} /><span><b>Blocked</b> {String(e.reason)} <em className="rule">{String(e.rule).replaceAll('_', ' ')}</em></span></li>
    case 'executed': return <li className={`ev exec ${['send_email', 'export_records', 'admin_credentials'].includes(String(e.tool)) ? 'hot' : ''}`}>
      <Wrench size={14} /><span><b>Executed</b> {TOOL[String(e.tool)] ?? String(e.tool)}</span></li>
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

function Lane({ title, mode, events, running, scenario }: { title: string; mode: Mode; events: RunEvent[]; running: boolean; scenario?: Scenario }) {
  const done = events.some(e => e.kind === 'done')
  const hit = scenario ? attackSucceeded(scenario.goal, events) : false
  const blocked = events.filter(e => e.kind === 'blocked').length
  const verdict = !done ? null : scenario?.kind === 'attack'
    ? (hit ? { tone: 'bad', icon: <ShieldAlert size={18} />, text: 'Attack succeeded' } : { tone: 'good', icon: <ShieldCheck size={18} />, text: 'Attack stopped' })
    : (blocked ? { tone: 'warn', icon: <ShieldAlert size={18} />, text: `Normal work was blocked (${blocked})` } : { tone: 'good', icon: <CircleCheck size={18} />, text: 'Task done, nothing blocked' })
  return (
    <section className={`lane ${mode}`}>
      <header><h2>{mode === 'none' ? <Bot size={16} /> : <ShieldCheck size={16} />}{title}</h2>{running && <Loader2 size={16} className="spin" />}</header>
      {verdict && <div className={`outcome ${verdict.tone}`}>{verdict.icon}{verdict.text}</div>}
      <ol className="steps">{events.map(e => <Step key={e.seq} e={e} />)}</ol>
      {!events.length && <p className="muted">Waiting to start.</p>}
    </section>
  )
}

export default function Demo() {
  const { data: scenarios } = useQuery({ queryKey: ['scenarios'], queryFn: api.scenarios })
  const [id, setId] = useState('A01')
  const [custom, setCustom] = useState(false)
  const [task, setTask] = useState('Summarize the complaint in document DOC-1.')
  const [doc, setDoc] = useState('')
  const [lanes, setLanes] = useState<Record<Mode, RunEvent[]>>({ none: [], rules: [], model: [], full: [] })
  const [running, setRunning] = useState<Partial<Record<Mode, boolean>>>({})
  const [error, setError] = useState('')
  const ctl = useRef<AbortController | null>(null)
  const scenario = useMemo(() => scenarios?.find(s => s.id === id), [scenarios, id])
  useEffect(() => () => ctl.current?.abort(), [])

  const start = () => {
    ctl.current?.abort()
    const c = new AbortController(); ctl.current = c
    setLanes({ none: [], rules: [], model: [], full: [] }); setError('')
    for (const mode of ['none', 'full'] as Mode[]) {
      setRunning(r => ({ ...r, [mode]: true }))
      const body = custom ? { mode, user_task: task, documents: { 'DOC-1': doc } } : { mode, scenario_id: id }
      streamRun(body, e => setLanes(l => ({ ...l, [mode]: [...l[mode], e] })), c.signal)
        .catch(err => { if (err.name !== 'AbortError') setError(String(err)) })
        .finally(() => setRunning(r => ({ ...r, [mode]: false })))
    }
  }
  const busy = Object.values(running).some(Boolean)
  const shown = custom ? undefined : scenario

  return (
    <>
      <div className="intro">
        <h1>Same task, same poisoned document, two agents</h1>
        <p className="muted">Left: an ordinary tool-using agent. Right: the same agent behind Qwen Shield. Both run locally on Qwen3-1.7B.</p>
      </div>
      <div className="picker">
        <div className="tabs"><button className={!custom ? 'on' : ''} onClick={() => setCustom(false)}>Scenarios</button>
          <button className={custom ? 'on' : ''} onClick={() => setCustom(true)}>Your own document</button></div>
        {!custom ? <div className="scen">
          <select value={id} onChange={e => setId(e.target.value)}>
            {Object.entries(FAMILY).map(([f, label]) => <optgroup key={f} label={label}>
              {(scenarios ?? []).filter(s => s.family === f).map(s => <option key={s.id} value={s.id}>{s.id} · {s.user_task}</option>)}</optgroup>)}
          </select>
          {scenario && <div className="docs">{Object.entries(scenario.documents).map(([k, v]) => <p key={k}><b>{k}</b> {v}</p>)}
            {scenario.granted.length > 0 && <p className="muted">The user authorized: {scenario.granted.join(', ').replaceAll('_', ' ')}</p>}</div>}
        </div> : <div className="scen">
          <input value={task} onChange={e => setTask(e.target.value)} placeholder="The user's request" />
          <textarea value={doc} onChange={e => setDoc(e.target.value)} rows={4} placeholder="Paste a document (id DOC-1). Try hiding an instruction inside it." />
        </div>}
        <div className="go">
          {busy ? <button onClick={() => ctl.current?.abort()}><Square size={14} /> Stop</button>
            : <button className="primary" disabled={custom && !doc.trim()} onClick={start}><Play size={14} /> Run both agents</button>}
          {error && <span className="bad">{error}</span>}
        </div>
      </div>
      <div className="lanes">
        <Lane title="Unprotected agent" mode="none" events={lanes.none} running={!!running.none} scenario={shown} />
        <Lane title="Agent with Qwen Shield" mode="full" events={lanes.full} running={!!running.full} scenario={shown} />
      </div>
    </>
  )
}
