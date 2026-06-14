import { api } from './client'

export type ChartType = 'i_mr' | 'xbar_r' | 'xbar_s' | 'p' | 'np' | 'c' | 'u'

export interface ChartRequest {
  chart: ChartType
  data: number[] | number[][]
  sizes?: number[]
}

export interface Violation {
  index: number
  value: number
  rule: number
  description: string
}

export interface SubChart {
  name: string
  points: number[]
  indices: number[]
  labels: (number | string)[]
  cl: number | number[]
  ucl: number | number[]
  lcl: number | number[]
  violations: Violation[]
}

export interface ChartResponse {
  chart: string
  sigma?: number
  center?: number
  subcharts: SubChart[]
}

export async function computeChart(req: ChartRequest): Promise<ChartResponse> {
  const res = await api.post<ChartResponse>('/spc/chart', req)
  return res.data
}
