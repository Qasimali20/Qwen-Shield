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
    <span className={data.agent.ok ? 'up' : 'down'}>agent · {data.agent.model}</span>
    <span className={data.shield.ok ? 'up' : 'down'}>shield · {data.shield.model}</span>
    {data.base_shield && <span className={data.base_shield.ok ? 'up' : 'down'}>base · {data.base_shield.model}</span>}
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
      <form onSubmit={submit} className="signin-card">
        <div className="brand"><ShieldCheck size={26} /> Qwen Shield</div>
        <p className="muted">Local security for AI agents. Sign in to open the console.</p>
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
        <div className="brand"><ShieldCheck size={20} /> Qwen Shield <small>local security for AI agents</small></div>
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
