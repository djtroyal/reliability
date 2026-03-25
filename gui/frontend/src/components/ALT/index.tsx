import { useState } from 'react'
import Plot from 'react-plotly.js'
import { Play, Download } from 'lucide-react'
import FileUpload from '../shared/FileUpload'
import ResultsTable from '../shared/ResultsTable'
import { fitALT, ALTFitResponse } from '../../api/client'

const ALL_MODELS = [
  'Weibull_Exponential','Weibull_Eyring','Weibull_Power',
  'Normal_Exponential','Normal_Eyring','Normal_Power',
  'Lognormal_Exponential','Lognormal_Eyring','Lognormal_Power',
  'Exponential_Exponential','Exponential_Eyring','Exponential_Power',
]

export default function ALT() {
  const [failureText, setFailureText] = useState('')
  const [stressText, setStressText] = useState('')
  const [useLevelStress, setUseLevelStress] = useState('')
  const [selectedModels, setSelectedModels] = useState<string[]>(ALL_MODELS)
  const [sortBy, setSortBy] = useState('AICc')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ALTFitResponse | null>(null)

  const parseNumbers = (text: string) =>
    text.split(/[\s,\n]+/).map(Number).filter(n => !isNaN(n))

  const handleCSV = (failures: number[]) => {
    setFailureText(failures.join(', '))
  }

  const toggleModel = (m: string) =>
    setSelectedModels(prev =>
      prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m])

  const run = async () => {
    const failures = parseNumbers(failureText)
    const stresses = parseNumbers(stressText)
    if (failures.length < 4) { setError('At least 4 failure times required.'); return }
    if (failures.length !== stresses.length) { setError('Failures and stresses must have equal length.'); return }
    const useLevel = parseFloat(useLevelStress)
    setError(null)
    setLoading(true)
    try {
      const res = await fitALT({
        failures,
        failure_stress: stresses,
        use_level_stress: isNaN(useLevel) ? undefined : useLevel,
        models_to_fit: selectedModels.length < ALL_MODELS.length ? selectedModels : undefined,
        sort_by: sortBy,
      })
      setResult(res)
    } catch (e: unknown) {
      setError((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Error running ALT analysis.')
    } finally {
      setLoading(false)
    }
  }

  const downloadCSV = () => {
    if (!result) return
    const keys = Object.keys(result.results[0] || {})
    const header = keys.join(',') + '\n'
    const rows = result.results.map(r => keys.map(k => r[k] ?? '').join(',')).join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'alt_results.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  const lifePlotData = (() => {
    if (!result?.life_stress_plot) return []
    const p = result.life_stress_plot
    const traces: Plotly.Data[] = [
      {
        x: p.line_stress as Plotly.Datum[],
        y: p.line_life.filter(Boolean) as Plotly.Datum[],
        mode: 'lines', name: 'Life-Stress model',
        line: { color: '#3b82f6', width: 2 },
      } as Plotly.Data,
      {
        x: p.scatter_stress as Plotly.Datum[],
        y: p.scatter_life as Plotly.Datum[],
        mode: 'markers', name: 'Observed median life',
        marker: { color: '#ef4444', size: 8, symbol: 'circle' },
      } as Plotly.Data,
    ]
    if (p.use_level_stress && p.use_level_life) {
      traces.push({
        x: [p.use_level_stress, p.use_level_stress] as Plotly.Datum[],
        y: [0, p.use_level_life] as Plotly.Datum[],
        mode: 'lines', name: `Use level (S=${p.use_level_stress})`,
        line: { color: '#10b981', width: 1.5, dash: 'dot' },
      } as Plotly.Data)
    }
    return traces
  })()

  const tableColumns = (result?.results[0]
    ? Object.keys(result.results[0]).map(k => ({ key: k, label: k }))
    : [])

  return (
    <div className="flex h-[calc(100vh-57px)]">
      {/* Left panel */}
      <div className="w-72 flex-shrink-0 bg-white border-r border-gray-200 overflow-y-auto p-4 flex flex-col gap-4">
        <FileUpload onData={handleCSV} label="Upload CSV (columns: value, type[F/S])" />

        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">
            Failure times <span className="text-gray-400">(comma separated)</span>
          </label>
          <textarea
            value={failureText}
            onChange={e => setFailureText(e.target.value)}
            className="w-full h-20 text-xs border border-gray-300 rounded p-2 font-mono resize-none focus:outline-none focus:ring-1 focus:ring-blue-400"
            placeholder="1000, 800, 500, 300..."
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">
            Stress values <span className="text-gray-400">(one per failure time)</span>
          </label>
          <textarea
            value={stressText}
            onChange={e => setStressText(e.target.value)}
            className="w-full h-20 text-xs border border-gray-300 rounded p-2 font-mono resize-none focus:outline-none focus:ring-1 focus:ring-blue-400"
            placeholder="350, 350, 400, 400..."
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">
            Use-level stress <span className="text-gray-400">(optional)</span>
          </label>
          <input
            type="number"
            value={useLevelStress}
            onChange={e => setUseLevelStress(e.target.value)}
            className="w-full text-sm border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-400"
            placeholder="e.g. 300"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">Sort by</label>
          <select
            value={sortBy}
            onChange={e => setSortBy(e.target.value)}
            className="w-full text-sm border border-gray-300 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-400"
          >
            <option value="AICc">AICc</option>
            <option value="BIC">BIC</option>
          </select>
        </div>

        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-xs font-medium text-gray-700">Models</label>
            <div className="flex gap-1">
              <button onClick={() => setSelectedModels(ALL_MODELS)} className="text-xs text-blue-600 hover:underline">All</button>
              <span className="text-gray-300">|</span>
              <button onClick={() => setSelectedModels([])} className="text-xs text-gray-500 hover:underline">None</button>
            </div>
          </div>
          <div className="flex flex-col gap-1 max-h-48 overflow-y-auto">
            {ALL_MODELS.map(m => (
              <label key={m} className="flex items-center gap-2 text-xs text-gray-700 cursor-pointer">
                <input type="checkbox" checked={selectedModels.includes(m)}
                  onChange={() => toggleModel(m)} className="rounded text-blue-600" />
                {m}
              </label>
            ))}
          </div>
        </div>

        {error && <p className="text-xs text-red-600 bg-red-50 p-2 rounded">{error}</p>}

        <button
          onClick={run}
          disabled={loading}
          className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-sm font-medium py-2 rounded transition-colors"
        >
          <Play size={14} />
          {loading ? 'Running...' : 'Run ALT Analysis'}
        </button>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {result ? (
          <>
            <div className="bg-white border-b border-gray-200 px-4 py-2 flex items-center justify-between">
              <p className="text-sm text-gray-600">
                Best model: <span className="font-semibold text-green-700">{result.best_model}</span>
              </p>
              <button onClick={downloadCSV}
                className="flex items-center gap-1 text-xs text-gray-500 hover:text-blue-600 border border-gray-200 px-2 py-1 rounded">
                <Download size={12} /> Export CSV
              </button>
            </div>
            <div className="flex-1 overflow-hidden flex">
              {/* Results table */}
              <div className="w-96 flex-shrink-0 border-r border-gray-200 overflow-y-auto p-3">
                <ResultsTable
                  columns={tableColumns}
                  rows={result.results as Record<string, unknown>[]}
                  rowKey="Model"
                />
              </div>
              {/* Life-stress plot */}
              <div className="flex-1 p-4">
                {lifePlotData.length > 0 ? (
                  <Plot
                    data={lifePlotData}
                    layout={{
                      title: `${result.best_model} — Life vs Stress`,
                      xaxis: { title: 'Stress', gridcolor: '#e5e7eb' },
                      yaxis: { title: 'Characteristic Life', gridcolor: '#e5e7eb' },
                      margin: { t: 40, r: 20, b: 50, l: 70 },
                      paper_bgcolor: 'white', plot_bgcolor: 'white',
                    } as any}
                    config={{ responsive: true }}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400 text-sm">
                    No life-stress plot available (set a use-level stress for full plot)
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400">
            <div className="text-center">
              <p className="text-lg font-medium">No results yet</p>
              <p className="text-sm mt-1">Enter failure times + stresses and click Run</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
