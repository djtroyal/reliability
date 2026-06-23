// Shared primitives for the Reliability Testing tool components.
import { useState } from 'react'
import { Play } from 'lucide-react'
import InfoLabel from '../shared/InfoLabel'

export const inputCls = 'w-full text-xs border border-gray-300 rounded px-2 py-1.5 font-mono focus:outline-none focus:ring-1 focus:ring-blue-400'
export const labelCls = 'block text-xs font-medium text-gray-700 mb-1'
export const btnCls = 'flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-xs font-medium py-2 rounded transition-colors'
export const PLOT_CFG = { responsive: true, displayModeBar: true } as const
export const plotBase = {
  margin: { t: 30, r: 20, b: 45, l: 55 },
  paper_bgcolor: 'white', plot_bgcolor: 'white',
}

export function detail(e: unknown, fb: string): string {
  return (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || fb
}

export function fmtNum(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return '—'
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(2)
  return v.toFixed(2)
}

export function Card({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`rounded-lg border p-3 ${accent ? 'bg-blue-50 border-blue-200' : 'bg-white border-gray-200'}`}>
      <p className="text-xs text-gray-500">{label}</p>
      <p className={`text-lg font-semibold ${accent ? 'text-blue-700' : 'text-gray-900'}`}>{value}</p>
    </div>
  )
}

export function Field({ label, tip, value, onChange, type = 'number' }: {
  label: string; tip?: string; value: string; onChange: (v: string) => void; type?: string
}) {
  return (
    <div>
      {tip ? <InfoLabel tip={tip}>{label}</InfoLabel> : <label className={labelCls}>{label}</label>}
      <input type={type} step="any" value={value} onChange={e => onChange(e.target.value)} className={inputCls} />
    </div>
  )
}

export function Select({ label, tip, value, onChange, options }: {
  label: string; tip?: string; value: string; onChange: (v: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <div>
      {tip ? <InfoLabel tip={tip}>{label}</InfoLabel> : <label className={labelCls}>{label}</label>}
      <select value={value} onChange={e => onChange(e.target.value)} className={inputCls}>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  )
}

export function ToolLayout({ intro, controls, err, loading, onRun, runLabel, results }: {
  intro: string; controls: React.ReactNode; err: string | null; loading: boolean
  onRun: () => void; runLabel: string; results: React.ReactNode
}) {
  return (
    <div className="flex flex-1 overflow-hidden">
      <div className="w-80 flex-shrink-0 bg-white border-r border-gray-200 overflow-y-auto p-4 flex flex-col gap-3">
        <p className="text-xs text-gray-500 leading-snug">{intro}</p>
        {controls}
        {err && <p className="text-xs text-red-600 bg-red-50 p-2 rounded">{err}</p>}
        <button onClick={onRun} disabled={loading} className={btnCls}><Play size={12} /> {loading ? 'Working...' : runLabel}</button>
      </div>
      <div className="flex-1 overflow-y-auto p-6">
        {results ?? (
          <div className="h-full flex items-center justify-center text-gray-400 text-sm">
            Enter inputs and click {runLabel}.
          </div>
        )}
      </div>
    </div>
  )
}

/** Generic sub-tab container: a horizontal tab bar + the active tool's component. */
export interface ToolDef { id: string; label: string; render: () => React.ReactNode }
export function ToolTabs({ tools, initial }: { tools: ToolDef[]; initial?: string }) {
  const [active, setActive] = useState(initial ?? tools[0]?.id)
  const current = tools.find(t => t.id === active) ?? tools[0]
  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <div className="flex items-stretch gap-1 bg-gray-50 border-b border-gray-200 px-3 overflow-x-auto">
        {tools.map(t => (
          <button key={t.id} onClick={() => setActive(t.id)}
            className={`px-3 py-1.5 text-xs font-medium whitespace-nowrap border-b-2 transition-colors ${
              active === t.id ? 'border-blue-600 text-blue-700' : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}>{t.label}</button>
        ))}
      </div>
      {current?.render()}
    </div>
  )
}
