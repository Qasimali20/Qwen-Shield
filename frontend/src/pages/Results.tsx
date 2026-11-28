import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Cpu, FlaskConical, Globe, Info, Laptop, ShieldCheck, TrendingDown } from 'lucide-react'
import { api, type AssessMetrics, type Baseline, type External, type Rate, type Report, type Training } from '../api'

const MODE: Record<string, string> = { none: 'No defence', rules: 'Rules only', model: 'Model only', full: 'Qwen Shield (full)' }
const FAMILY: Record<string, string> = { A: 'Unauthorized action', B: 'Abused email', C: 'Malicious ticket', D: 'Manipulated answer' }
const SLICE: Record<string, string> = { all: 'All 344 held-out examples', unseen_style: 'Injection styles never seen in training', unseen_domain: 'Domains never seen in training' }
const SPLIT: Record<string, string> = { test: 'Held-out · larger agent', test_4b: 'Held-out · 4B agent (laptop-sized)', dev: 'Development' }
const pct = (r?: Rate | null, digits = 0) => (r && r.value != null ? `${(r.value * 100).toFixed(digits)}%` : '–')
const frac = (r?: Rate) => (r ? `${r.n}/${r.of}` : '')

/** Status of a rate: good, warning or bad, by fixed bands, so 99% recall reads as good and 28% task completion as bad. */
const tone = (v: number, good: 'low' | 'high') =>
  good === 'high' ? (v >= 0.9 ? 'g' : v >= 0.6 ? 'w' : 'b') : (v <= 0.05 ? 'g' : v <= 0.25 ? 'w' : 'b')

/** A rate as a bar; colour says good or bad (with the number always printed beside it), length says how much. */
function Bar({ r, good }: { r?: Rate; good: 'low' | 'high' }) {
  const v = r?.value ?? 0
  return <div className="rbar" title={r ? `${r.n} of ${r.of}` : ''}>
    <span className="track"><i className={tone(v, good)} style={{ width: `${Math.max(v * 100, 1)}%` }} /></span>
    <span className="num"><b>{pct(r)}</b> <small>{frac(r)}</small></span></div>
}

function Hero({ r }: { r: Report }) {
  const t = r.end_to_end.test?.rows ?? []
  const row = (k: string) => t.find(x => x.key === k)?.summary
  const none = row('none'), base = row('base'), ft = row('finetuned')
  const a = (k: string) => r.assessment.rows.find(x => x.key === k)?.all
  const lap = r.laptop.find(l => l.key === 'laptop')
  const tiles = [
    { label: 'Attacks that succeeded', before: frac(none?.attack_success), after: frac(ft?.attack_success), note: 'held-out scenarios, no defence → Qwen Shield' },
    { label: 'Tasks still completed', before: frac(base?.task_completion), after: frac(ft?.task_completion), note: 'shield with base model → fine-tuned model' },
    { label: 'False alarms on benign content', before: pct(a('base')?.false_positive_rate, 1), after: pct(a('finetuned')?.false_positive_rate, 1), note: `${r.assessment.examples ?? ''} held-out documents, base → fine-tuned` },
  ]
  return (
    <div className="hero">
      {tiles.map(x => <div className="tile" key={x.label}>
        <span className="tl">{x.label}</span>
        <span className="tv"><s>{x.before}</s><b>{x.after}</b></span>
        <span className="tn">{x.note}</span></div>)}
      {lap && <div className="tile">
        <span className="tl">On an ordinary laptop, CPU only</span>
        <span className="tv"><b>{lap.summary.false_alarms}</b><small>false alarms</small></span>
        <span className="tn">{lap.summary.malicious_caught} attacks caught · {lap.hardware.cpu.replace(/\(R\)|\(TM\)|11th Gen|@.*$/g, '').trim()}</span></div>}
    </div>
  )
}

function EndToEnd({ r }: { r: Report }) {
  const splits = Object.keys(r.end_to_end)
  const [split, setSplit] = useState(splits[0] ?? 'test')
  const e = r.end_to_end[split]
  if (!e) return null
  const families = Object.keys(e.rows[0].summary.attack_success_by_family)
  return (
    <section className="card">
      <header><h2><ShieldCheck size={16} /> End to end: a real agent doing real tasks</h2>
        <div className="seg">{splits.map(s => <button key={s} className={s === split ? 'on' : ''} onClick={() => setSplit(s)}>
          {SPLIT[s] ?? s} ({r.end_to_end[s].scenarios})</button>)}</div></header>
      <p className="muted">Every number is what actually executed in the mock tools, not what a model said.
        {split !== 'dev' && ' The held-out scenarios were written by hand and frozen before any fine-tuning result was seen.'}
        {split === 'test_4b' && ' Here every model is laptop-sized: a 4B agent and the quantized shield, both through Ollama.'}</p>
      <div className="scroll"><table className="res">
        <thead><tr><th>Protection</th><th>Attacks that succeeded</th>{families.map(f => <th key={f}>{FAMILY[f] ?? f}</th>)}
          <th>All tasks completed</th><th>Benign blocked (false positives)</th><th>Check time (p50)</th></tr></thead>
        <tbody>{e.rows.map(x => { const s = x.summary; return (
          <tr key={x.key} className={x.key === 'finetuned' ? 'hl' : ''}><td><b>{x.label}</b></td><td><Bar r={s.attack_success} good="low" /></td>
            {families.map(f => <td key={f} className="mono">{frac(s.attack_success_by_family[f])}</td>)}
            <td><Bar r={s.task_completion} good="high" /></td><td><Bar r={s.false_positive} good="low" /></td>
            <td className="mono">{s.assessment_latency_p50_s != null ? `${s.assessment_latency_p50_s.toFixed(2)} s` : '–'}</td></tr>) })}</tbody>
      </table></div>
      <p className="note"><Info size={14} /> Rules alone cannot stop abused emails, malicious tickets or manipulated answers: the action itself
        is one the user asked for, only the content is wrong. The base model stops attacks only by blocking nearly everything.</p>
    </section>
  )
}

function Assessment({ r }: { r: Report }) {
  const [slice, setSlice] = useState<'all' | 'unseen_style' | 'unseen_domain'>('all')
  const cols: { k: keyof AssessMetrics; label: string; good: 'low' | 'high' }[] = [
    { k: 'malicious_recall', label: 'Attacks caught', good: 'high' }, { k: 'false_positive_rate', label: 'False alarms', good: 'low' },
    { k: 'target_accuracy', label: 'Right target', good: 'high' }, { k: 'evidence_located', label: 'Evidence found verbatim', good: 'high' },
    { k: 'json_validity', label: 'Valid JSON', good: 'high' }]
  return (
    <section className="card">
      <header><h2><FlaskConical size={16} /> The model alone: before and after fine-tuning</h2>
        <div className="seg">{Object.keys(SLICE).map(s => <button key={s} className={s === slice ? 'on' : ''} onClick={() => setSlice(s as typeof slice)}>
          {s === 'all' ? 'All' : s === 'unseen_style' ? 'Unseen styles' : 'Unseen domains'}</button>)}</div></header>
      <p className="muted">{SLICE[slice]}. The evidence sentences are what the shield cuts out of a document before the agent reads it.</p>
      <div className="scroll"><table className="res">
        <thead><tr><th>Model</th>{cols.map(c => <th key={c.k}>{c.label}</th>)}<th>Latency (p50)</th></tr></thead>
        <tbody>{r.assessment.rows.map(x => { const m = x[slice]; return (
          <tr key={x.key} className={x.key === 'finetuned' ? 'hl' : ''}><td><b>{x.label}</b></td>
            {cols.map(c => <td key={c.k}>{m && (m[c.k] as Rate)?.of ? <Bar r={m[c.k] as Rate} good={c.good} /> : <span className="muted">–</span>}</td>)}
            <td className="mono">{m?.latency_p50_s != null ? `${m.latency_p50_s.toFixed(1)} s` : '–'}</td></tr>) })}</tbody>
      </table></div>
    </section>
  )
}

function LaptopCard({ r }: { r: Report }) {
  if (!r.laptop.length) return null
  return (
    <section className="card">
      <header><h2><Laptop size={16} /> On a normal laptop</h2></header>
      <p className="muted">The Q4_K_M model through Ollama, CPU only, on 80 held-out examples, one request at a time. The laptop
        was in everyday use during the run.</p>
      <div className="laps">{r.laptop.map(l => { const s = l.summary; return (
        <div key={l.key} className={`lap ${l.key}`}>
          <h3>{l.key === 'laptop' ? <Laptop size={15} /> : <Cpu size={15} />}{l.key === 'laptop' ? 'Ordinary laptop, CPU only' : 'Workstation CPU (reference)'}</h3>
          <p className="spec">{l.hardware.cpu} · {l.hardware.cores} threads · {Math.ceil(l.hardware.ram_gb / 2) * 2} GB RAM · {l.hardware.os.split(' ')[0]}</p>
          <dl>
            <div><dt>Attacks caught</dt><dd>{s.malicious_caught}</dd></div>
            <div><dt>False alarms</dt><dd>{s.false_alarms}</dd></div>
            <div><dt>Right target</dt><dd>{s.target_correct}</dd></div>
            <div><dt>Median check</dt><dd>{s.latency_p50_s.toFixed(1)} s</dd></div>
            <div><dt>Slow case (p90)</dt><dd>{s.latency_p90_s.toFixed(1)} s</dd></div>
            <div><dt>Output speed</dt><dd>{s.gen_tok_s_median?.toFixed(0) ?? '–'} tok/s</dd></div>
          </dl></div>) })}</div>
    </section>
  )
}

/** Validation loss over training: one series, so no legend; points carry their own tooltips. */
function LossChart({ t }: { t: Training }) {
  const W = 640, H = 180, P = { l: 40, r: 16, t: 12, b: 26 }
  const pts = t.log.filter((p, i) => i === t.log.length - 1 || t.log[i + 1].epoch - p.epoch > 0.05)   // drop near-duplicate final evals
  const xs = pts.map(p => p.epoch), ys = pts.map(p => p.val_loss)
  const xMax = Math.max(...xs), yMax = Math.max(...ys) * 1.15
  const x = (v: number) => P.l + (v / xMax) * (W - P.l - P.r)
  const y = (v: number) => H - P.b - (v / yMax) * (H - P.t - P.b)
  const ticks = [0, yMax / 2, yMax].map(v => +v.toFixed(2))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Validation loss by epoch">
      {ticks.map(v => <g key={v}><line x1={P.l} x2={W - P.r} y1={y(v)} y2={y(v)} className="grid" />
        <text x={P.l - 6} y={y(v) + 4} textAnchor="end">{v.toFixed(2)}</text></g>)}
      {[1, 2, 3].filter(e => e <= xMax).map(e => <text key={e} x={x(e)} y={H - 8} textAnchor={e === Math.floor(xMax) ? 'end' : 'middle'}>epoch {e}</text>)}
      <polyline points={pts.map(p => `${x(p.epoch)},${y(p.val_loss)}`).join(' ')} className="line" />
      {pts.map(p => <circle key={p.step} cx={x(p.epoch)} cy={y(p.val_loss)} r={4.5} className="dot">
        <title>{`step ${p.step} · epoch ${p.epoch} · validation loss ${p.val_loss}`}</title></circle>)}
    </svg>
  )
}

function TrainingCard({ t }: { t: Training }) {
  const min = (s: number) => `${Math.round(s / 60)} min`
  return (
    <section className="card">
      <header><h2><TrendingDown size={16} /> How it was trained</h2></header>
      <div className="train">
        <dl>
          <div><dt>Method</dt><dd>LoRA r={t.args.rank}, α={t.args.alpha}, all projections</dd></div>
          <div><dt>Data</dt><dd>{t.train_examples.toLocaleString()} train · {t.val_examples} validation</dd></div>
          <div><dt>Steps</dt><dd>{t.steps} ({t.args.epochs} epochs, lr {t.args.lr})</dd></div>
          <div><dt>Time</dt><dd>{min(t.wall_s)} on one NVIDIA GPU, incl. {min(t.thermal_pause_s)} cooling pauses</dd></div>
          <div><dt>Peak GPU memory</dt><dd>{t.peak_gpu_mem_gib} GiB (loss on answer tokens only)</dd></div>
          <div><dt>Best validation loss</dt><dd>{t.best_val_loss}</dd></div>
        </dl>
        <figure><figcaption>Validation loss</figcaption><LossChart t={t} /></figure>
      </div>
      <p className="note"><Info size={14} /> Labels come from the generation spec, not from a model: a larger Qwen teacher only writes the
        text, a blind auditor must agree, and anything overlapping the benchmark scenarios is removed.</p>
    </section>
  )
}

function AllRuns() {
  const { data } = useQuery({ queryKey: ['benchmarks'], queryFn: api.benchmarks })
  const [label, setLabel] = useState('')
  if (!data?.length) return null
  const b = data.find(x => x.label === label) ?? data[0]
  const modes = Object.keys(b.summary)
  return (
    <details className="card raw"><summary>Every saved end-to-end run ({data.length})</summary>
      <select value={b.label} onChange={e => setLabel(e.target.value)}>{data.map(x => <option key={x.label} value={x.label}>{x.label}</option>)}</select>
      <p className="muted">{b.meta.scenarios} scenarios · {b.meta.at}</p>
      <div className="scroll"><table className="res">
        <thead><tr><th>Protection</th><th>Attacks succeeded</th><th>Tasks completed</th><th>False positives</th><th>Valid JSON</th></tr></thead>
        <tbody>{modes.map(m => { const s = b.summary[m]; return <tr key={m}><td><b>{MODE[m] ?? m}</b></td>
          <td><Bar r={s.attack_success} good="low" /></td><td><Bar r={s.task_completion} good="high" /></td>
          <td><Bar r={s.false_positive} good="low" /></td><td className="mono">{s.json_validity.of ? pct(s.json_validity) : '–'}</td></tr> })}</tbody>
      </table></div>
    </details>
  )
}

function ExternalCard({ x }: { x: External }) {
  const rows = [['qwen3-1.7b', 'Base Qwen3-1.7B'], ['qwen-shield', 'Qwen Shield (fine-tuned)']] as const
  return (
    <section className="card">
      <header><h2><Globe size={16} /> On a public benchmark: InjecAgent</h2></header>
      <p className="muted">{x.meta.attacks.toLocaleString('en-US')} attacks hidden in what tools return to an agent (Zhan et al., ACL Findings 2024),
        never used in training. The benchmark has no benign cases, so the false-alarm column uses the same {x.meta.benign_control} tool
        outputs with the attack replaced by an ordinary sentence, a control we added.</p>
      <div className="scroll"><table className="res">
        <thead><tr><th>Model</th><th>Attacks flagged</th><th>Exact attack sentence quoted</th><th>Attack cut, rest of output kept</th><th>Innocent control flagged</th></tr></thead>
        <tbody>{rows.filter(([k]) => x.models[k]).map(([k, label]) => { const m = x.models[k]; return (
          <tr key={k} className={k === 'qwen-shield' ? 'hl' : ''}><td><b>{label}</b></td>
            <td><Bar r={m.all_attacks.flagged} good="high" /></td><td><Bar r={m.all_attacks.evidence_exact} good="high" /></td>
            <td><Bar r={m.all_attacks.removed_rest_kept} good="high" /></td><td><Bar r={m.benign_control.flagged} good="low" /></td></tr>) })}</tbody>
      </table></div>
      <p className="note"><Info size={14} /> InjecAgent's tool outputs are single-line JSON, so the sentence-level sanitizer usually withholds the
        whole output rather than only the attack. Cutting just the attacked JSON field is the next step.</p>
    </section>
  )
}

function BaselineCard({ b, r }: { b: Baseline; r: Report }) {
  const ours = r.assessment.rows.find(x => x.key === 'finetuned')?.all
  const ext = r.external?.models['qwen-shield']
  const bip = r.bipia?.models['qwen-shield']
  const HARM = ['Misinformation & Propaganda', 'Scams & Fraud', 'Marketing & Advertising']
  const sum = (rs: (Rate | undefined)[]): Rate | undefined => rs.every(Boolean)
    ? (({ n, of }) => ({ n, of, value: of ? n / of : null }))(rs.reduce((t, x) => ({ n: t.n + x!.n, of: t.of + x!.of }), { n: 0, of: 0 })) : undefined
  const rows: [string, Rate | undefined, Rate | undefined, 'low' | 'high'][] = [
    ['Held-out documents: attacks caught', b.held_out_344.attacks_caught, ours?.malicious_recall, 'high'],
    ['Held-out documents: false alarms', b.held_out_344.false_alarms, ours?.false_positive_rate, 'low'],
    ['InjecAgent: attacks caught', b.injecagent.attacks_flagged, ext?.all_attacks.flagged, 'high'],
    ['InjecAgent control: false alarms', b.injecagent.benign_control_flagged, ext?.benign_control.flagged, 'low'],
    ['BIPIA: harmful attacks caught', sum(HARM.map(k => b.bipia.by_category[k])), sum(HARM.map(k => bip?.[k]?.flagged)), 'high'],
    ['BIPIA: real emails wrongly flagged', b.bipia.benign_flagged, bip?.benign_control.flagged, 'low']]
  return (
    <section className="card">
      <header><h2><ShieldCheck size={16} /> Against an existing detector</h2></header>
      <p className="muted">The widely used open-source classifier ProtectAI DeBERTa v2 (Apache-2.0) on the same tests, given the untrusted
        content in 512-token windows, its intended use. It catches blunt overrides but misses attacks written as ordinary business text,
        and it cannot quote the attack sentence. BIPIA is Microsoft's public benchmark (email task, 750 attacked and 50 real emails).</p>
      <div className="scroll"><table className="res">
        <thead><tr><th>Same test</th><th>ProtectAI DeBERTa v2</th><th>Qwen Shield</th></tr></thead>
        <tbody>{rows.map(([label, theirs, mine, good]) => (
          <tr key={label}><td><b>{label}</b></td><td><Bar r={theirs} good={good} /></td><td><Bar r={mine} good={good} /></td></tr>))}</tbody>
      </table></div>
    </section>
  )
}

export function Results() {
  const { data, error } = useQuery({ queryKey: ['report'], queryFn: api.report })
  if (error) return <p className="bad">{String(error)}</p>
  if (!data) return <p className="muted">Loading…</p>
  return (
    <>
      <div className="intro"><div className="eyebrow">Results</div><h1>Fine-tuning Qwen3-1.7B into a security assessor</h1>
        <p className="muted">Before and after, on data and scenarios the model never saw in training.</p></div>
      <Hero r={data} />
      <EndToEnd r={data} />
      <Assessment r={data} />
      {data.external && <ExternalCard x={data.external} />}
      {data.baseline && <BaselineCard b={data.baseline} r={data} />}
      <div className="two"><LaptopCard r={data} />{data.training && <TrainingCard t={data.training} />}</div>
      <section className="card honest"><header><h2><Info size={16} /> Honesty notes</h2></header><ul>
        <li>One policy rule came after looking at held-out results: inside a channel the user granted, an action is now blocked only if its
          arguments carry the attack. The model and data were not changed. Both runs are saved (<code>*-shield-v1</code> before,
          <code>*-shield-v1b</code> after).</li>
        <li>The headline agent is a larger Qwen model. A 1.7B agent rarely read the documents, so attacks never reached it. The shield is the 1.7B model.</li>
        <li>A sanitizer bug was found with the Pakistan examples and fixed: Urdu, Arabic and Chinese full stops were not sentence breaks. Rerunning the five held-out scenarios that contain them changed no reported number.</li>
        <li>Known false alarm: an ordinary all-Urdu exam notice (Pakistan example PK03) is flagged.</li>
        <li>Scenario counts are small (40 and 30); treat single-scenario differences as noise.</li></ul></section>
      <AllRuns />
    </>
  )
}

export function Audit() {
  const { data } = useQuery({ queryKey: ['audit'], queryFn: api.audit, refetchInterval: 3000 })
  return (
    <>
      <div className="intro"><div className="eyebrow">Audit</div><h1>Audit trail</h1><p className="muted">Every assessment, decision and executed action, newest first.</p></div>
      <div className="scroll"><table className="res"><thead><tr><th>Time</th><th>Run</th><th>Event</th><th>Detail</th></tr></thead>
        <tbody>{(data ?? []).map(e => <tr key={`${e.run}-${e.seq}`}>
          <td className="mono">{new Date(e.t * 1000).toLocaleTimeString()}</td><td className="mono">{e.run}</td>
          <td className={e.kind === 'blocked' || (e.kind === 'verdict' && e.decision !== 'allow') ? 'bad' : ''}>{e.kind}</td>
          <td className="small">{e.kind === 'verdict' ? `${String(e.tool)} → ${String(e.decision)} (${String(e.rule)}): ${String(e.reason)}`
            : e.kind === 'assessment' ? JSON.stringify(e.assessment) : `${String(e.tool ?? '')} ${JSON.stringify(e.args ?? e.removed ?? '')}`}</td></tr>)}</tbody></table></div>
    </>
  )
}
