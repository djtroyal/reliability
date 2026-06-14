/**
 * Client-side simulated-data generators (#12). Seeded so results are
 * reproducible. Used by the shared DataGenerator UI and any module that wants
 * to populate its inputs with a representative example dataset.
 */

/** Mulberry32 — small, fast, seedable PRNG returning floats in [0,1). */
export function makeRng(seed?: number): () => number {
  let a = (seed == null || isNaN(seed)) ? (Math.random() * 2 ** 32) >>> 0 : seed >>> 0
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Standard normal via Box–Muller. */
function randn(rng: () => number): number {
  let u = 0, v = 0
  while (u === 0) u = rng()
  while (v === 0) v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

export interface GenDistOption {
  value: string
  label: string
  params: { key: string; label: string; default: number; min?: number; max?: number; step?: number }[]
}

/** Distributions offered by the generator, with their parameter definitions. */
export const GEN_DISTRIBUTIONS: GenDistOption[] = [
  { value: 'normal', label: 'Normal', params: [
    { key: 'mu', label: 'Mean (μ)', default: 100, step: 1 },
    { key: 'sigma', label: 'Std dev (σ)', default: 15, min: 0, step: 1 },
  ] },
  { value: 'uniform', label: 'Uniform', params: [
    { key: 'low', label: 'Low', default: 0, step: 1 },
    { key: 'high', label: 'High', default: 100, step: 1 },
  ] },
  { value: 'weibull', label: 'Weibull', params: [
    { key: 'eta', label: 'Scale (η)', default: 1000, min: 0, step: 10 },
    { key: 'beta', label: 'Shape (β)', default: 2, min: 0.01, step: 0.1 },
  ] },
  { value: 'exponential', label: 'Exponential', params: [
    { key: 'mean', label: 'Mean', default: 1000, min: 0, step: 10 },
  ] },
  { value: 'lognormal', label: 'Lognormal', params: [
    { key: 'mu', label: 'Log-mean (μ)', default: 5, step: 0.1 },
    { key: 'sigma', label: 'Log-std (σ)', default: 0.5, min: 0.01, step: 0.05 },
  ] },
  { value: 'poisson', label: 'Poisson', params: [
    { key: 'lambda', label: 'Rate (λ)', default: 5, min: 0, step: 1 },
  ] },
  { value: 'binomial', label: 'Binomial', params: [
    { key: 'n', label: 'Trials (n)', default: 20, min: 1, step: 1 },
    { key: 'p', label: 'Prob (p)', default: 0.3, min: 0, max: 1, step: 0.05 },
  ] },
]

/** Generate `n` samples from the named distribution. */
export function generateColumn(
  dist: string, params: Record<string, number>, n: number, seed?: number,
): number[] {
  const rng = makeRng(seed)
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    switch (dist) {
      case 'normal':
        out.push(params.mu + params.sigma * randn(rng)); break
      case 'uniform':
        out.push(params.low + (params.high - params.low) * rng()); break
      case 'weibull':
        out.push(params.eta * Math.pow(-Math.log(1 - rng()), 1 / params.beta)); break
      case 'exponential':
        out.push(-params.mean * Math.log(1 - rng())); break
      case 'lognormal':
        out.push(Math.exp(params.mu + params.sigma * randn(rng))); break
      case 'poisson': {
        // Knuth's algorithm.
        const L = Math.exp(-params.lambda); let k = 0, p = 1
        do { k++; p *= rng() } while (p > L)
        out.push(k - 1); break
      }
      case 'binomial': {
        let s = 0
        for (let j = 0; j < params.n; j++) if (rng() < params.p) s++
        out.push(s); break
      }
      default:
        out.push(rng())
    }
  }
  return out
}

/** Round generated values to a reasonable number of significant figures. */
export function roundSamples(values: number[], integer = false): number[] {
  if (integer) return values.map(v => Math.round(v))
  return values.map(v => {
    const a = Math.abs(v)
    if (a >= 1000) return Math.round(v)
    if (a >= 1) return Math.round(v * 100) / 100
    return Math.round(v * 1e4) / 1e4
  })
}
