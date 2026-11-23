import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type Rate } from '../api'

const MODE: Record<string, string> = {
  none: 'No defence', rules: 'Rules only', model: 'Model only', full: 'Qwen Shield (full)',
  base: 'Base Qwen3-1.7B', finetuned: 'Fine-tuned',
}
const FAMILY: Record<string, string> = { A: 'Unauthorized action', B: 'Abused email', C: 'Malicious ticket', D: 'Manipulated answer' }
const pct = (r?: Rate) => (r && r.value != null ? `${Math.round(r.value * 100)}%` : '–')

/** A rate as a bar, where `good` says which direction is better. */
function Bar({ r, good }: { r?: Rate; good: 'low' | 'high' }) {
  const v = r?.value ?? 0
  const bad = good === 'low' ? v > 0 : v < 1
  return <div className="rbar" title={r ? `${r.n} of ${r.of}` : ''}>
    <span className="track"><i className={bad ? 'b' : 'g'} style={{ width: `${Math.max(v * 100, 1)}%` }} /></span>
    <span className="num"><b>{pct(r)}</b> <small>{r ? `${r.n}/${r.of}` : ''}</small></span></div>
}

export function Results() {
  const { data } = useQuery({ queryKey: ['benchmarks'], queryFn: api.benchmarks, refetchInterval: 10000 })
  const [label, setLabel] = useState('')
  if (!data) return <p className="muted">Loading…</p>
  if (!data.length) return <p className="muted">No benchmark has been run yet.</p>
  const b = data.find(x => x.label === label) ?? data[0]
  const modes = Object.keys(b.summary)
  const families = Object.keys(b.summary[modes[0]].attack_success_by_family)
  return (
    <>
      <div className="intro"><h1>Measured results</h1>
        <p className="muted">Every number comes from what actually executed in the mock tools, not from what a model said.
          {' '}{b.meta.scenarios} scenarios · agent {b.meta.agent_model} · shield {b.meta.shield_model} · {b.meta.at}</p></div>
      {data.length > 1 && <select value={b.label} onChange={e => setLabel(e.target.value)}>{data.map(x => <option key={x.label} value={x.label}>{x.label}</option>)}</select>}
      <table className="res">
        <thead><tr><th>Protection</th><th>Attacks that succeeded</th>{families.map(f => <th key={f}>{FAMILY[f] ?? f}</th>)}
          <th>Normal tasks completed</th><th>False alarms</th><th>Valid JSON</th><th>Check time (p50)</th></tr></thead>
        <tbody>{modes.map(m => { const s = b.summary[m]; return (
          <tr key={m}><td><b>{MODE[m] ?? m}</b></td><td><Bar r={s.attack_success} good="low" /></td>
            {families.map(f => <td key={f} className="mono">{pct(s.attack_success_by_family[f])}</td>)}
            <td><Bar r={s.benign_task_completion} good="high" /></td><td><Bar r={s.false_positive} good="low" /></td>
            <td className="mono">{s.json_validity.of ? pct(s.json_validity) : '–'}</td>
            <td className="mono">{s.assessment_latency_p50_s != null ? `${s.assessment_latency_p50_s.toFixed(2)} s` : '–'}</td></tr>) })}</tbody>
      </table>
    </>
  )
}

export function Audit() {
  const { data } = useQuery({ queryKey: ['audit'], queryFn: api.audit, refetchInterval: 3000 })
  return (
    <>
      <div className="intro"><h1>Audit trail</h1><p className="muted">Every assessment, decision and executed action, newest first.</p></div>
      <table className="res"><thead><tr><th>Time</th><th>Run</th><th>Event</th><th>Detail</th></tr></thead>
        <tbody>{(data ?? []).map(e => <tr key={`${e.run}-${e.seq}`}>
          <td className="mono">{new Date(e.t * 1000).toLocaleTimeString()}</td><td className="mono">{e.run}</td>
          <td className={e.kind === 'blocked' || (e.kind === 'verdict' && e.decision !== 'allow') ? 'bad' : ''}>{e.kind}</td>
          <td className="small">{e.kind === 'verdict' ? `${String(e.tool)} → ${String(e.decision)} (${String(e.rule)}): ${String(e.reason)}`
            : e.kind === 'assessment' ? JSON.stringify(e.assessment) : `${String(e.tool ?? '')} ${JSON.stringify(e.args ?? e.removed ?? '')}`}</td></tr>)}</tbody></table>
    </>
  )
}
