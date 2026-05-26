/* ─── MatrixView ─────────────────────────────────────────────────────────── */
function MatrixView({ matrix, n, pivotCell = null, highlightRows = [] }) {
  if (!matrix || matrix.length === 0) return null;

  const fmt = (v) => {
    const num = typeof v === 'number' ? v : parseFloat(v);
    if (Number.isInteger(num)) return String(num);
    const s = num.toFixed(4);
    // trim trailing zeros after decimal
    return s.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  };

  // Map of row index → highlight type ('pivot' | 'active')
  const rowHighlight = {};
  highlightRows.forEach(({ idx, type }) => { rowHighlight[idx] = type; });

  return (
    <div className="matrix-view">
      <table>
        <thead>
          <tr>
            {Array.from({ length: n }, (_, j) => (
              <th key={j}>
                x<sub>{j + 1}</sub>
              </th>
            ))}
            <th className="mv-bh">b</th>
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i} className={rowHighlight[i] ? `mv-row-${rowHighlight[i]}` : ''}>
              {row.map((val, j) => {
                const isPivot = pivotCell && pivotCell.row === i && pivotCell.col === j;
                const cls = isPivot ? 'mv-pivot-cell' : j === n ? 'mv-bv' : '';
                return <td key={j} className={cls}>{fmt(val)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ─── Step badge labels ───────────────────────────────────────────────────── */
const STEP_LABEL = {
  initial: 'Initial Matrix',
  warning: 'Ill-Conditioned Matrix',
  pivot: 'Pivot (Row Swap)',
  elimination: 'Row Elimination',
  upper_triangular: 'Upper Triangular',
  back_substitution: 'Back Substitution',
};

/* ─── SolutionDisplay ────────────────────────────────────────────────────── */
export default function SolutionDisplay({ result }) {
  const { solution, variables, steps, n } = result;

  return (
    <div className="solution-section">
      {/* Final answer box */}
      <div className="final-answer">
        <h2>Solution</h2>
        <div className="answer-grid">
          {variables.map((v, i) => (
            <div key={i} className="answer-item">
              <span className="av">{v}</span>
              <span className="aeq">=</span>
              <span className="aval">{solution[i]}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Step-by-step working */}
      <div className="steps-block">
        <h2>Step-by-Step Working</h2>
        {steps.map((step, idx) => (
          <div key={idx} className={`step-card sc-${step.type}`}>
            <div className="step-head">
              <span className={`badge b-${step.type}`}>
                {STEP_LABEL[step.type] ?? step.type}
              </span>
              <span className="step-desc">{step.description}</span>
            </div>

            {step.matrix && (
              <MatrixView
                matrix={step.matrix}
                n={n}
                pivotCell={
                  step.pivot_row != null && step.pivot_col != null
                    ? { row: step.pivot_row, col: step.pivot_col }
                    : null
                }
                highlightRows={
                  step.type === 'pivot'
                    ? [{ idx: step.pivot_row, type: 'pivot' }]
                    : step.type === 'elimination'
                    ? [
                        { idx: step.pivot_row, type: 'pivot' },
                        { idx: step.active_row, type: 'active' },
                      ]
                    : []
                }
              />
            )}

            {step.back_steps && (
              <div className="bs-list">
                {step.back_steps.map((bs, i) => (
                  <div key={i} className="bs-row">
                    <span className="bs-var">{bs.variable}</span>
                    <span className="bs-eq">=</span>
                    <span className="bs-val">{bs.value}</span>
                    <code className="bs-formula">{bs.description}</code>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
