import { StrictMode, useEffect, useState, type FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider, useQuery, useQueryClient } from '@tanstack/react-query'
import { BarChart3, LogIn, LogOut, ScrollText, ShieldCheck, Swords } from 'lucide-react'
import { api, setSignedOutHandler } from './api'
import Demo from './pages/Demo'
import { Audit, Results } from './pages/Results'
import './styles.css'

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } })

function Status() {
  const { data } = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 10000 })
  if (!data) return null
  return <div className="status">
    <span className={data.agent.ok ? 'up' : 'down'}>agent</span>
    <span className={data.shield.ok ? 'up' : 'down'}>shield</span>
    {data.base_shield && <span className={data.base_shield.ok ? 'up' : 'down'}>base model</span>}
  </div>
}

function SignIn({ onDone }: { onDone: () => void }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true); setError(null)
    try {
      const err = await api.login(username, password)
      if (err) { setError(err); setPassword('') } else onDone()
    } catch {
      setError('Cannot reach the Qwen Shield server. Check that it is running.')
    } finally { setBusy(false) }
  }
  return (
    <div className="signin">
      <aside className="signin-hero">
        <Brand />
        <div>
          <h1>Stop AI agents from obeying <span className="grad">hidden instructions.</span></h1>
          <p>A fine-tuned Qwen3-1.7B finds the sentence an attacker planted in a document, cuts it out, and lets the real work through.
            It runs on your own hardware.</p>
        </div>
        <div className="signin-stats">
          <div><b>0 / 28</b><span>attacks got through on held-out agent scenarios</span></div>
          <div><b>92.8%</b><span>of 1,054 attacks caught on the public InjecAgent benchmark</span></div>
          <div><b>CPU only</b><span>the whole system runs on an ordinary laptop</span></div>
        </div>
      </aside>
      <form onSubmit={submit} className="signin-card">
        <h2>Sign in</h2>
        <p className="muted">Open the Qwen Shield console.</p>
        <label htmlFor="user">Username</label>
        <input id="user" autoComplete="username" autoFocus value={username} onChange={e => setUsername(e.target.value)} />
        <label htmlFor="pass">Password</label>
        <input id="pass" type="password" autoComplete="current-password" value={password} onChange={e => setPassword(e.target.value)} />
        {error && <p className="bad" role="alert">{error}</p>}
        <button className="primary" type="submit" disabled={busy || !username || !password}><LogIn size={15} /> {busy ? 'Signing in…' : 'Sign in'}</button>
      </form>
    </div>
  )
}

function Brand() {
  return <div className="brand"><span className="mark"><ShieldCheck size={18} /></span><span className="word">Qwen Shield</span></div>
}

function Gate() {
  const qc = useQueryClient()
  const { data, isLoading, refetch } = useQuery({ queryKey: ['session'], queryFn: api.session, retry: false })
  useEffect(() => setSignedOutHandler(() => { qc.setQueryData(['session'], (s: unknown) => s && { ...(s as object), signed_in: false }) }), [qc])
  if (isLoading) return null
  if (data && !data.signed_in) return <SignIn onDone={async () => { await refetch(); qc.invalidateQueries() }} />
  const signOut = async () => { await api.logout(); qc.clear(); await refetch() }
  return <App signOut={data?.required ? signOut : undefined} />
}

function App({ signOut }: { signOut?: () => void }) {
  return (
    <div className="shell">
      <header className="top">
        <Brand />
        <nav>
          <NavLink to="/" end><Swords size={15} /> Live demo</NavLink>
          <NavLink to="/results"><BarChart3 size={15} /> Results</NavLink>
          <NavLink to="/audit"><ScrollText size={15} /> Audit</NavLink>
        </nav>
        <Status />
        {signOut && <button className="signout" onClick={signOut} title="Sign out"><LogOut size={15} /> Sign out</button>}
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Demo />} />
          <Route path="/results" element={<Results />} />
          <Route path="/audit" element={<Audit />} />
        </Routes>
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode><QueryClientProvider client={qc}><BrowserRouter><Gate /></BrowserRouter></QueryClientProvider></StrictMode>,
)
