import { useState, useCallback } from 'react'
import SolutionDisplay from './SolutionDisplay'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const FETCH_TIMEOUT_MS = 30000 // 30 second timeout

const EXAMPLES = {
  2: [
    ['3', '2', '8'],
    ['1', '-1', '-1'],
  ],
  3: [
    ['2', '1', '-1', '8'],
    ['-3', '-1', '2', '-11'],
    ['-2', '1', '2', '-3'],
  ],
  4: [
    ['2', '1', '-1', '2', '5'],
    ['4', '5', '-3', '6', '9'],
    ['-2', '5', '-2', '6', '4'],
    ['4', '11', '-4', '8', '2'],
  ],
  5: [
    ['1', '2', '-1', '0', '1', '5'],
    ['2', '1', '1', '-1', '0', '6'],
    ['-1', '1', '2', '1', '-1', '2'],
    ['0', '-1', '1', '2', '1', '3'],
    ['1', '0', '-1', '1', '2', '1'],
  ],
  6: [
    ['2', '1', '-1', '0', '1', '-1', '8'],
    ['1', '-1', '2', '1', '0', '1', '5'],
    ['-1', '2', '1', '-1', '1', '0', '7'],
    ['0', '1', '-1', '2', '-1', '1', '4'],
    ['1', '0', '1', '-1', '2', '1', '6'],
    ['-1', '1', '0', '1', '-1', '2', '3'],
  ],
  7: [
    ['1', '1', '1', '1', '1', '1', '1', '7'],
    ['1', '-1', '1', '-1', '1', '-1', '1', '1'],
    ['1', '2', '1', '2', '1', '2', '1', '10'],
    ['2', '1', '2', '1', '2', '1', '2', '11'],
    ['1', '0', '1', '0', '1', '0', '1', '4'],
    ['0', '1', '0', '1', '0', '1', '0', '3'],
    ['1', '1', '0', '0', '1', '1', '0', '4'],
  ],
  8: [
    ['1', '0', '1', '0', '1', '0', '1', '0', '4'],
    ['0', '1', '0', '1', '0', '1', '0', '1', '4'],
    ['1', '1', '0', '0', '1', '1', '0', '0', '4'],
    ['1', '0', '1', '0', '0', '0', '1', '0', '3'],
    ['0', '1', '0', '1', '0', '0', '0', '1', '3'],
    ['1', '1', '1', '1', '0', '0', '0', '0', '4'],
    ['1', '0', '0', '0', '1', '1', '1', '1', '4'],
    ['0', '1', '1', '1', '1', '1', '1', '1', '7'],
  ],
}

function emptyMatrix(n) {
  return Array.from({ length: n }, () => Array(n + 1).fill(''))
}

export default function MatrixSolver() {
  const [size, setSize] = useState(3)
  const [matrix, setMatrix] = useState(() => emptyMatrix(3))
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  /* ── size change ──────────────────────────────────────────────────────── */
  const handleSizeChange = useCallback((n) => {
    setSize(n)
    setMatrix(emptyMatrix(n))
    setResult(null)
    setError('')
  }, [])

  /* ── cell edit ────────────────────────────────────────────────────────── */
  const handleCell = useCallback((row, col, val) => {
    setMatrix((prev) => {
      const next = prev.map((r) => [...r])
      next[row][col] = val === '' ? '0' : val
      return next
    })
  }, [])

  /* ── load example ─────────────────────────────────────────────────────── */
  const handleExample = useCallback(() => {
    const ex = EXAMPLES[size]
    if (ex) {
      setMatrix(ex.map((r) => [...r]))
      setResult(null)
      setError('')
    }
  }, [size])

  /* ── reset ────────────────────────────────────────────────────────────── */
  const handleReset = useCallback(() => {
    setMatrix(emptyMatrix(size))
    setResult(null)
    setError('')
  }, [size])

  /* ── solve ────────────────────────────────────────────────────────────── */
  const handleSolve = useCallback(async () => {
    // Validate & parse inputs
    const parsed = []
    for (let i = 0; i < size; i++) {
      const row = []
      for (let j = 0; j <= size; j++) {
        const raw = matrix[i]?.[j] ?? ''
        const num = raw=== '' ? 0 : parseFloat(raw)
        if (isNaN(num)) {
          setError(
            `Row ${i + 1}, column ${j + 1} is empty or invalid. Fill in all cells before solving.`,
          )
          return
        }
        row.push(num)
      }
      parsed.push(row)
    }

    setLoading(true)
    setError('')
    setResult(null)

    // Set up abort controller for timeout
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)

    try {
      const res = await fetch(`${API}/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ matrix: parsed }),
        signal: controller.signal,
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data.detail ?? 'The solver returned an error.')
      } else {
        setResult(data)
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setError(
          `Request timeout after ${FETCH_TIMEOUT_MS / 1000}s. Backend may be unresponsive.`,
        )
      } else {
        setError(
          `Cannot reach the backend at ${API}. ` +
            'Make sure the Python server is running (uvicorn main:app --reload).',
        )
      }
    } finally {
      clearTimeout(timeoutId)
      setLoading(false)
    }
  }, [size, matrix])

  /* ── render ───────────────────────────────────────────────────────────── */
  return (
    <div className="page">
      {/* ── Page header ─────────────────────────────────────────────── */}
      <header className="page-header">
        <h1>Gaussian Elimination Solver</h1>
        <p>
          Solve systems of linear equations using{' '}
          <strong>Gaussian elimination with partial pivoting</strong> and{' '}
          <strong>back-substitution</strong>
        </p>
      </header>

      <main className="page-body">
        {/* ── Size selector ───────────────────────────────────────────── */}
        <section className="card">
          <h2>Step 1 — Choose System Size</h2>
          <p className="hint">Select the number of equations and unknowns (n × n system).</p>
          <div className="size-strip">
            {[2, 3, 4, 5, 6, 7, 8].map((n) => (
              <button
                key={n}
                className={`size-btn${size === n ? ' active' : ''}`}
                onClick={() => handleSizeChange(n)}
              >
                {n} × {n}
              </button>
            ))}
          </div>
        </section>

        {/* ── Matrix input ─────────────────────────────────────────────── */}
        <section className="card">
          <div className="card-row">
            <div>
              <h2>Step 2 — Enter Augmented Matrix [A | b]</h2>
              <p className="hint">
                {size} equation{size > 1 ? 's' : ''} &nbsp;·&nbsp; {size} unknown
                {size > 1 ? 's' : ''} &nbsp;·&nbsp; Last column is the constant vector{' '}
                <strong>b</strong>
              </p>
            </div>
            {EXAMPLES[size] && (
              <button className="btn-ghost" onClick={handleExample}>
                Load Example
              </button>
            )}
          </div>

          <div className="matrix-scroll">
            <table className="mtable">
              <thead>
                <tr>
                  <th className="th-label" />
                  {Array.from({ length: size }, (_, j) => (
                    <th key={j} className="th-col">
                      x<sub>{j + 1}</sub>
                    </th>
                  ))}
                  <th className="th-sep">|</th>
                  <th className="th-col th-b">b</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: size }, (_, i) => (
                  <tr key={i}>
                    <td className="td-label">R{i + 1}</td>
                    {Array.from({ length: size + 1 }, (_, j) => (
                      <td key={j} className={j === size ? 'td-b' : 'td-a'}>
                        <input
                          type="number"
                          step="any"
                          value={matrix[i]?.[j] ?? ''}
                          onChange={(e) => handleCell(i, j, e.target.value)}
                          placeholder="0"
                          aria-label={
                            j === size
                              ? `b${i + 1}`
                              : `a${i + 1}${j + 1}`
                          }
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ── Action buttons ──────────────────────────────────────── */}
          <div className="action-row">
            <button className="btn-solve" onClick={handleSolve} disabled={loading}>
              {loading ? 'Solving…' : '▶  Solve'}
            </button>
            <button className="btn-reset" onClick={handleReset}>
              Reset
            </button>
          </div>
        </section>

        {/* ── Error ───────────────────────────────────────────────────── */}
        {error && (
          <div className="error-box" role="alert">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* ── Solution ────────────────────────────────────────────────── */}
        {result && <SolutionDisplay result={result} />}
      </main>
    </div>
  )
}
