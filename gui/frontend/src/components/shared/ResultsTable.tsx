interface Column {
  key: string
  label: string
  format?: (v: unknown) => string
}

interface Props {
  columns: Column[]
  rows: Record<string, unknown>[]
  highlightFirst?: boolean
  onRowClick?: (row: Record<string, unknown>) => void
  selectedRow?: string
  rowKey?: string
}

const fmt = (v: unknown): string => {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return isFinite(v) ? v.toPrecision(5) : '∞'
  return String(v)
}

export default function ResultsTable({
  columns, rows, highlightFirst = true, onRowClick, selectedRow, rowKey,
}: Props) {
  if (!rows.length) return null

  return (
    <div className="overflow-x-auto rounded border border-gray-200">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-gray-50 border-b border-gray-200">
            {columns.map(col => (
              <th key={col.key} className="px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap">
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const key = rowKey ? String(row[rowKey]) : String(i)
            const isSelected = selectedRow === key
            const isFirst = highlightFirst && i === 0
            return (
              <tr
                key={i}
                onClick={() => onRowClick?.(row)}
                className={`border-b border-gray-100 last:border-0 transition-colors ${
                  isSelected ? 'bg-blue-50' :
                  isFirst ? 'bg-green-50' : 'bg-white'
                } ${onRowClick ? 'cursor-pointer hover:bg-gray-50' : ''}`}
              >
                {columns.map(col => (
                  <td key={col.key} className="px-3 py-2 text-gray-800 whitespace-nowrap">
                    {col.format ? col.format(row[col.key]) : fmt(row[col.key])}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
