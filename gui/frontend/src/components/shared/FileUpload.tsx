import { useRef } from 'react'
import Papa from 'papaparse'
import { Upload } from 'lucide-react'

interface Props {
  onData: (failures: number[], rightCensored: number[]) => void
  label?: string
}

/**
 * Parses a CSV with columns: value, type
 * type = 'F' (failure) or 'S' (suspension/right-censored)
 * If no type column, all rows treated as failures.
 */
export default function FileUpload({ onData, label }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (file: File) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: ({ data }) => {
        const failures: number[] = []
        const censored: number[] = []
        const keys = Object.keys(data[0] || {})
        const valueKey = keys.find(k => /value|time|t|failure/i.test(k)) || keys[0]
        const typeKey = keys.find(k => /type|status|cens/i.test(k))

        for (const row of data) {
          const val = parseFloat(row[valueKey])
          if (isNaN(val)) continue
          const type = typeKey ? row[typeKey]?.trim().toUpperCase() : 'F'
          if (type === 'S' || type === 'C' || type === '0') {
            censored.push(val)
          } else {
            failures.push(val)
          }
        }
        onData(failures, censored)
      },
    })
  }

  return (
    <div>
      {label && <p className="text-xs text-gray-500 mb-1">{label}</p>}
      <button
        onClick={() => inputRef.current?.click()}
        className="flex items-center gap-2 px-3 py-2 text-sm border border-dashed border-gray-300 rounded hover:border-blue-400 hover:bg-blue-50 transition-colors text-gray-600"
      >
        <Upload size={14} />
        Upload CSV
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={e => {
          const file = e.target.files?.[0]
          if (file) handleFile(file)
          e.target.value = ''
        }}
      />
    </div>
  )
}
