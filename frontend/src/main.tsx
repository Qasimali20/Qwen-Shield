import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { BarChart3, ScrollText, ShieldCheck, Swords } from 'lucide-react'
import { api } from './api'
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

function App() {
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
  <StrictMode><QueryClientProvider client={qc}><BrowserRouter><App /></BrowserRouter></QueryClientProvider></StrictMode>,
)
